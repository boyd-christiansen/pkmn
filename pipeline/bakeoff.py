"""Frontier-model bake-off runner.

Picks one or more cached matches from `parsed_data/bo3.jsonl` (or bo1.jsonl)
and runs every available provider through `synthesize_turn` for each turn.
Reports per-provider cost, tool-call rate, CoT length, action match-rate,
and wall-clock. Saves each provider's full SFT output to a separate JSONL
file (one file per provider, rows from all matches accumulated).

Usage:
    .venv/bin/python bakeoff.py                              # first match only (smoke test)
    .venv/bin/python bakeoff.py --limit 5                    # first 5 matches (real eval)
    .venv/bin/python bakeoff.py --providers openai,anthropic
    .venv/bin/python bakeoff.py --match-id bo3-gen9vgc2026regibo3-2590204993

Resumability:
    Re-runs are safe. Rows already in `bakeoff_<provider>.jsonl` (keyed by
    `(match_id, game_index, turn)`) are skipped — useful if the run dies
    partway. Use a fresh `--output-dir` if you want to overwrite.

Environment variables required (each provider only runs if its key is set):
    OPENAI_API_KEY     — OpenAI provider (default: gpt-5.5; override TEACHER_MODEL_OPENAI)
    ANTHROPIC_API_KEY  — Anthropic provider (default: claude-sonnet-4-6; override TEACHER_MODEL_ANTHROPIC)
    GOOGLE_API_KEY     — Google provider (default: gemini-3.1-pro-preview; override TEACHER_MODEL_GOOGLE)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import aiohttp
import click

import canonical_priors  # noqa: F401  — imported for symmetry with master_pipeline
import damage_inferencer
import threat_matrix
from master_pipeline import (
    DEFAULT_BO3_INPUT,
    DEFAULT_PARSED_DATA_DIR,
)
from action_extraction import (
    extract_p1_actions,
    flip_match_to_winner,
    slot_action,
)
from prompt_formatting import (
    format_p1_inferred_spreads_block,
    format_p1_team_block,
    format_user_prompt,
)
from team_reconstruction import (
    brought_species_keys_for_game,
    reconstruct_p1_team,
    reconstruct_p2_species,
    team_sheets_for_match,
)
from teacher import (
    ProviderResult,
    TeacherProvider,
    render_system_prompt,
    render_system_prompt_bo3,
)

DEFAULT_CALC_BASE_URL = "http://localhost:3000"


def _build_providers(selected: set[str]) -> list[TeacherProvider]:
    """Instantiate the requested providers, skipping any whose API key is unset."""
    providers: list[TeacherProvider] = []

    if "openai" in selected:
        if os.environ.get("OPENAI_API_KEY"):
            from teacher import OpenAIProvider
            providers.append(OpenAIProvider())
        else:
            click.echo("[skip] openai — OPENAI_API_KEY not set", err=True)

    if "anthropic" in selected:
        if os.environ.get("ANTHROPIC_API_KEY"):
            from teacher import AnthropicProvider
            providers.append(AnthropicProvider())
        else:
            click.echo("[skip] anthropic — ANTHROPIC_API_KEY not set", err=True)

    if "google" in selected:
        if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
            from teacher import GoogleProvider
            providers.append(GoogleProvider())
        else:
            click.echo("[skip] google — GOOGLE_API_KEY (or GEMINI_API_KEY) not set", err=True)

    return providers


def _normalize_action(action: dict[str, Any] | None) -> str:
    """Cheap, comparable string repr for action diffs across providers."""
    if not action:
        return "<missing>"
    parts = []
    for slot in ("slot_1", "slot_2"):
        a = action.get(slot, {})
        atype = a.get("action_type", "?")
        if atype == "move":
            mv = (a.get("move") or "").lower().replace(" ", "")
            tg = a.get("target") or ""
            parts.append(f"move:{mv}@{tg}")
        elif atype == "switch":
            parts.append(f"switch:{(a.get('switch_to') or '').lower()}")
        else:
            parts.append(atype)
    return " | ".join(parts)


def _extract_submit_args(messages: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    """Return the full submit_decision args = {pre_tool_thought, action: {...}}."""
    if not messages:
        return None
    for m in messages:
        for tc in (m.get("tool_calls") or []):
            if tc.get("function", {}).get("name") == "submit_decision":
                try:
                    return json.loads(tc["function"]["arguments"])
                except (KeyError, json.JSONDecodeError):
                    return None
    return None


def _extract_action_from_messages(messages: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    """Return just the inner action dict = {slot_1, slot_2} for diff comparison."""
    args = _extract_submit_args(messages)
    return (args or {}).get("action")


# =============================================================================
# Resumability — load already-done rows so re-runs skip them
# =============================================================================


def _output_path(output_dir: Path, provider_name: str) -> Path:
    return output_dir / f"bakeoff_{provider_name}.jsonl"


def _load_seen_keys(out_path: Path) -> tuple[set[tuple[str, int, int]], int]:
    """Return (seen_keys, row_count) for an existing per-provider output file.

    Used to skip already-done rows on rerun. Robust to malformed lines.
    """
    seen: set[tuple[str, int, int]] = set()
    if not out_path.exists():
        return seen, 0
    count = 0
    with out_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                seen.add((rec["match_id"], int(rec["game_index"]), int(rec["turn"])))
                count += 1
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return seen, count


# =============================================================================
# Per-match orchestration — mutates per-provider state
# =============================================================================


def _init_match_knowledge(
    match: dict[str, Any],
) -> tuple[dict[str, dict[str, dict[str, dict[str, int]]]], dict[str, Any]]:
    """Return (per-side KnowledgeState init, derived match metadata).

    KnowledgeStates reset every match — each match is an independent
    (P1 team, P2 team) — but the rest of `provider_state` accumulates.
    """
    games = match.get("games") or []
    match_format = match.get("format", "bo1")
    team_sheets = team_sheets_for_match(games) if match_format == "bo3" else None

    p1_team_recon = reconstruct_p1_team(games)
    if team_sheets:
        p1_species = [s["species"] for s in team_sheets["p1"]]
        p2_species = [s["species"] for s in team_sheets["p2"]]
    else:
        p1_species = list(p1_team_recon.keys())
        p2_species = reconstruct_p2_species(games)

    bo1_system_prompt = (
        render_system_prompt(format_p1_team_block(p1_team_recon))
        if not team_sheets
        else None
    )

    return {
        "p1_species": p1_species,
        "p2_species": p2_species,
    }, {
        "match_format": match_format,
        "team_sheets": team_sheets,
        "bo1_system_prompt": bo1_system_prompt,
        "p1_team_recon": p1_team_recon,
    }


async def _bakeoff_one_match(
    match_record: dict[str, Any],
    providers: list[TeacherProvider],
    *,
    provider_state: dict[str, dict[str, Any]],
    output_paths: dict[str, Path],
    seen_keys: dict[str, set[tuple[str, int, int]]],
    calc_base_url: str,
    format_id: str,
    aiohttp_session: aiohttp.ClientSession,
) -> None:
    """Run every turn through every provider for ONE match. Mutates
    `provider_state` (totals + rows) and appends to per-provider output
    files as rows complete (resumable).
    """

    match_record = flip_match_to_winner(match_record)
    games = match_record.get("games") or []
    if not games:
        click.echo("no games in match — bailing")
        return

    init, meta = _init_match_knowledge(match_record)
    match_format = meta["match_format"]
    team_sheets = meta["team_sheets"]
    bo1_system_prompt = meta["bo1_system_prompt"]

    # Reset per-match KnowledgeStates while preserving rolling totals.
    for p in providers:
        provider_state[p.name]["p1_knowledge"] = damage_inferencer.init_knowledge(init["p1_species"])
        provider_state[p.name]["p2_knowledge"] = damage_inferencer.init_knowledge(init["p2_species"])

    match_id = match_record.get("match_id", "unknown")
    click.echo(f"\n=== match {match_id} ({match_format}, {len(games)} game(s)) ===")

    for game_idx, game in enumerate(games):
        snapshots = game.get("snapshots") or []

        if team_sheets:
            brought = brought_species_keys_for_game(game)
            system_prompt = render_system_prompt_bo3(
                p1_sheet=team_sheets["p1"],
                p2_sheet=team_sheets["p2"],
                p1_brought=brought,
            )
        else:
            system_prompt = bo1_system_prompt

        for i in range(len(snapshots) - 1):
            snap_pre = snapshots[i]
            snap_post = snapshots[i + 1]
            events_stream = snap_pre.get("events") or []
            turn = int(snap_pre.get("turn", 0))

            human_action_dict = extract_p1_actions(snap_pre, snap_post, events_stream)
            if human_action_dict is None:
                # Always advance knowledge even on skipped turns so the
                # next turn's threat matrix isn't stale.
                damage_events = damage_inferencer.events_to_damage_events(events_stream)
                if damage_events:
                    for p in providers:
                        try:
                            await damage_inferencer.update_knowledge(
                                snap_pre, snap_post, damage_events,
                                provider_state[p.name]["p1_knowledge"],
                                provider_state[p.name]["p2_knowledge"],
                                session=aiohttp_session, base_url=calc_base_url,
                            )
                        except Exception:
                            pass
                continue

            human_action = {
                "slot_1": human_action_dict.get("a", slot_action("pass")),
                "slot_2": human_action_dict.get("b", slot_action("pass")),
            }
            human_norm = _normalize_action(human_action)

            click.echo(f"--- game {game_idx} turn {turn} (human: {human_norm}) ---")

            # Per-provider call (sequential to avoid rate-limit cross-talk).
            for p in providers:
                state = provider_state[p.name]
                key = (match_id, game_idx, turn)
                if key in seen_keys[p.name]:
                    click.echo(f"  {p.name:10s} (skipped — already in output)")
                    continue

                try:
                    tm_text = await threat_matrix.generate_threat_matrix(
                        snap_pre, "p1",
                        state["p1_knowledge"], state["p2_knowledge"],
                        format_id=format_id,
                        session=aiohttp_session,
                        base_url=calc_base_url,
                    )
                except Exception as e:
                    click.echo(f"  {p.name:10s} threat_matrix failed: {e}")
                    continue

                inferred = format_p1_inferred_spreads_block(snap_pre, state["p1_knowledge"])
                user_prompt = format_user_prompt(
                    snap_pre, tm_text,
                    p1_inferred_block=inferred,
                    snapshots_so_far=snapshots,
                    current_idx=i,
                    prior_games=games[:game_idx],
                    game_index=game_idx,
                    total_games_in_series=len(games),
                    match_format=match_format,
                )

                res: ProviderResult = await p.synthesize_turn(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    human_action=human_action,
                    calc_url=f"{calc_base_url}/calc",
                    aiohttp_session=aiohttp_session,
                )

                submit_args = _extract_submit_args(res.messages)
                action = (submit_args or {}).get("action")
                action_match = _normalize_action(action) == human_norm

                state["totals"]["calc_calls"] += res.calc_calls
                state["totals"]["iterations"] += res.iterations
                state["totals"]["input_tokens"] += res.input_tokens
                state["totals"]["output_tokens"] += res.output_tokens
                state["totals"]["cost_usd"] += res.cost_usd
                state["totals"]["elapsed_seconds"] += res.elapsed_seconds
                state["totals"]["turns_attempted"] += 1
                if res.messages is not None:
                    state["totals"]["turns_succeeded"] += 1
                if action_match:
                    state["totals"]["actions_matched"] += 1

                cot_chars = 0
                if submit_args and isinstance(submit_args.get("pre_tool_thought"), str):
                    cot_chars = len(submit_args["pre_tool_thought"])
                state["totals"]["cot_chars_total"] += cot_chars

                err_str = f"  ERROR: {res.error}" if res.error else ""
                click.echo(
                    f"  {p.name:10s} calc={res.calc_calls}  iter={res.iterations}  "
                    f"in={res.input_tokens:5d} out={res.output_tokens:4d}  "
                    f"${res.cost_usd:.4f}  {res.elapsed_seconds:5.1f}s  "
                    f"cot={cot_chars:4d}ch  match={action_match}{err_str}"
                )

                if res.messages is not None:
                    row = {
                        "match_id": match_id,
                        "game_index": game_idx,
                        "turn": turn,
                        "format_id": format_id,
                        "messages": res.messages,
                    }
                    state["rows"].append(row)
                    # Append immediately for resumability.
                    with output_paths[p.name].open("a") as f:
                        f.write(json.dumps(row) + "\n")
                    seen_keys[p.name].add(key)

            # Update knowledge for all providers using the actual events.
            damage_events = damage_inferencer.events_to_damage_events(events_stream)
            if damage_events:
                for p in providers:
                    try:
                        await damage_inferencer.update_knowledge(
                            snap_pre, snap_post, damage_events,
                            provider_state[p.name]["p1_knowledge"],
                            provider_state[p.name]["p2_knowledge"],
                            session=aiohttp_session, base_url=calc_base_url,
                        )
                    except Exception:
                        pass


# =============================================================================
# Aggregate summary
# =============================================================================


def _print_summary(
    providers: list[TeacherProvider],
    provider_state: dict[str, dict[str, Any]],
    *,
    n_matches: int,
    output_paths: dict[str, Path],
) -> None:
    click.echo(f"\n=== bake-off summary ({n_matches} match{'es' if n_matches != 1 else ''}) ===")
    click.echo(
        f"{'provider':12s} {'rows':>5s} {'match%':>7s} {'calc/turn':>10s} "
        f"{'$/row':>8s} {'avg cot':>8s} {'wall':>7s}"
    )
    for p in providers:
        s = provider_state[p.name]
        rows = int(s["totals"]["turns_succeeded"])
        attempted = int(s["totals"]["turns_attempted"])
        match_rate = (s["totals"]["actions_matched"] / attempted * 100) if attempted else 0.0
        calc_per = (s["totals"]["calc_calls"] / attempted) if attempted else 0.0
        cost_per = (s["totals"]["cost_usd"] / rows) if rows else 0.0
        avg_cot = int(s["totals"]["cot_chars_total"] // rows) if rows else 0
        wall = s["totals"]["elapsed_seconds"]
        click.echo(
            f"{p.name:12s} {rows:5d} {match_rate:6.1f}% {calc_per:10.2f} "
            f"${cost_per:7.4f} {avg_cot:8d} {wall:6.1f}s"
        )
        click.echo(f"  → {output_paths[p.name]}  ({len(s['rows'])} rows added this run)")


# =============================================================================
# CLI
# =============================================================================


@click.command()
@click.option("--input", "input_path",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=str(DEFAULT_BO3_INPUT),
              show_default=True)
@click.option("--match-id", default=None,
              help="Match-id substring to select. Single-match mode (mutually exclusive with --limit).")
@click.option("--limit", type=int, default=None,
              help="Run on the first N matches in --input. Default: 1 (single-match smoke run).")
@click.option("--providers", "providers_csv", default="openai,anthropic,google", show_default=True,
              help="Comma-separated subset of {openai,anthropic,google} to run.")
@click.option("--output-dir",
              type=click.Path(file_okay=False, path_type=Path),
              default=str(DEFAULT_PARSED_DATA_DIR),
              show_default=True)
@click.option("--calc-base-url", default=DEFAULT_CALC_BASE_URL, show_default=True)
@click.option("--format-id", default=None,
              help="Override the format_id derived from input filename.")
def cli(input_path, match_id, limit, providers_csv, output_dir, calc_base_url, format_id):
    """Run a head-to-head bake-off across frontier teacher models on N matches."""
    if match_id and limit is not None:
        raise click.UsageError("--match-id and --limit are mutually exclusive.")

    selected = {p.strip() for p in providers_csv.split(",") if p.strip()}
    providers = _build_providers(selected)
    if not providers:
        click.echo("FATAL: no providers available (no API keys set)", err=True)
        sys.exit(1)

    records = []
    with input_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        click.echo(f"FATAL: no records in {input_path}", err=True)
        sys.exit(1)

    if match_id:
        # Single-match by substring match.
        matched = [r for r in records if match_id in r["match_id"]]
        if not matched:
            click.echo(f"FATAL: no record matched id substring {match_id!r}", err=True)
            sys.exit(1)
        target_records = matched[:1]
    else:
        # Default: limit=1 (single-match smoke run); otherwise take first N.
        n = limit if limit is not None else 1
        target_records = records[:n]

    derived_format = format_id or ("gen9vgc2026regibo3" if "bo3" in input_path.stem else "gen9vgc2026regi")

    asyncio.run(_run(target_records, providers, Path(output_dir), calc_base_url, derived_format))


async def _run(
    target_records: list[dict[str, Any]],
    providers: list[TeacherProvider],
    output_dir: Path,
    calc_base_url: str,
    format_id: str,
) -> None:
    timeout = aiohttp.ClientTimeout(total=180, connect=10)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-provider state — totals accumulate across all matches; KnowledgeStates
    # are reset per match by `_bakeoff_one_match`.
    provider_state: dict[str, dict[str, Any]] = {}
    output_paths: dict[str, Path] = {}
    seen_keys: dict[str, set[tuple[str, int, int]]] = {}
    for p in providers:
        provider_state[p.name] = {
            "p1_knowledge": {},  # (re)initialized per match
            "p2_knowledge": {},
            "rows": [],
            "totals": defaultdict(float),
        }
        output_paths[p.name] = _output_path(output_dir, p.name)
        keys, count = _load_seen_keys(output_paths[p.name])
        seen_keys[p.name] = keys
        if count > 0:
            click.echo(f"[resume] {p.name}: {count} rows already in {output_paths[p.name].name}")

    click.echo(
        f"providers: {[p.name + '/' + p.model for p in providers]}  "
        f"matches: {len(target_records)}"
    )

    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Health check
        try:
            async with session.get(f"{calc_base_url}/health") as r:
                if r.status != 200:
                    raise RuntimeError(f"health check returned {r.status}")
        except Exception as e:
            raise click.ClickException(
                f"calc_microservice not reachable at {calc_base_url}/health: {e}\n"
                f"  Start it with:  cd calc_microservice && npm run dev"
            )

        for record in target_records:
            await _bakeoff_one_match(
                record, providers,
                provider_state=provider_state,
                output_paths=output_paths,
                seen_keys=seen_keys,
                calc_base_url=calc_base_url,
                format_id=format_id,
                aiohttp_session=session,
            )

    _print_summary(
        providers, provider_state,
        n_matches=len(target_records),
        output_paths=output_paths,
    )


if __name__ == "__main__":
    cli()
