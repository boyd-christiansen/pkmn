"""Frontier-model bake-off runner.

Picks one cached match from `parsed_data/bo3.jsonl` (or bo1.jsonl) and runs
every available provider through `synthesize_turn` for each turn. Reports
per-provider cost, tool-call rate, CoT length, action match-rate, and
wall-clock. Saves each provider's full SFT output to a separate JSONL for
manual inspection.

Usage:
    .venv/bin/python bakeoff.py
    .venv/bin/python bakeoff.py --providers openai,anthropic
    .venv/bin/python bakeoff.py --match-id bo3-gen9vgc2026regibo3-2590204993

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
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import aiohttp
import click

import canonical_priors
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


async def _bakeoff_one_match(
    match_record: dict[str, Any],
    providers: list[TeacherProvider],
    *,
    output_dir: Path,
    calc_base_url: str,
    format_id: str,
    aiohttp_session: aiohttp.ClientSession,
) -> None:
    """Run every turn through every provider; print + save results."""

    # Same orchestration steps as master_pipeline.process_match.
    match_record = flip_match_to_winner(match_record)
    games = match_record.get("games") or []
    if not games:
        click.echo("no games in match — bailing")
        return

    match_format = match_record.get("format", "bo1")
    team_sheets = team_sheets_for_match(games) if match_format == "bo3" else None

    p1_team_recon = reconstruct_p1_team(games)
    if team_sheets:
        p1_species = [s["species"] for s in team_sheets["p1"]]
        p2_species = [s["species"] for s in team_sheets["p2"]]
    else:
        p1_species = list(p1_team_recon.keys())
        p2_species = reconstruct_p2_species(games)

    # Per-provider state — separate KnowledgeStates so they don't share inference history.
    provider_state: dict[str, dict[str, Any]] = {}
    for p in providers:
        provider_state[p.name] = {
            "p1_knowledge": damage_inferencer.init_knowledge(p1_species),
            "p2_knowledge": damage_inferencer.init_knowledge(p2_species),
            "rows": [],
            "totals": defaultdict(float),
        }

    bo1_system_prompt = (
        render_system_prompt(format_p1_team_block(p1_team_recon))
        if not team_sheets
        else None
    )

    match_id = match_record.get("match_id", "unknown")
    click.echo(f"\n=== bake-off on {match_id} ({match_format}, {len(games)} game(s)) ===")
    click.echo(f"providers: {[p.name + '/' + p.model for p in providers]}\n")

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
                continue

            human_action = {
                "slot_1": human_action_dict.get("a", slot_action("pass")),
                "slot_2": human_action_dict.get("b", slot_action("pass")),
            }
            human_norm = _normalize_action({"slot_1": human_action["slot_1"],
                                            "slot_2": human_action["slot_2"]})

            click.echo(f"--- game {game_idx} turn {turn} (human: {human_norm}) ---")

            # Per-provider call (sequential to avoid rate-limit cross-talk).
            for p in providers:
                state = provider_state[p.name]
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

                # CoT — pull from the submit_decision args if present.
                cot_chars = 0
                if submit_args and isinstance(submit_args.get("pre_tool_thought"), str):
                    cot_chars = len(submit_args["pre_tool_thought"])

                err_str = f"  ERROR: {res.error}" if res.error else ""
                click.echo(
                    f"  {p.name:10s} calc={res.calc_calls}  iter={res.iterations}  "
                    f"in={res.input_tokens:5d} out={res.output_tokens:4d}  "
                    f"${res.cost_usd:.4f}  {res.elapsed_seconds:5.1f}s  "
                    f"cot={cot_chars:4d}ch  match={action_match}{err_str}"
                )

                if res.messages is not None:
                    state["rows"].append({
                        "match_id": match_id,
                        "game_index": game_idx,
                        "turn": turn,
                        "format_id": format_id,
                        "messages": res.messages,
                    })

            # Update knowledge for all providers using the actual events.
            damage_events = damage_inferencer.events_to_damage_events(events_stream)
            if damage_events:
                for p in providers:
                    state = provider_state[p.name]
                    try:
                        await damage_inferencer.update_knowledge(
                            snap_pre, snap_post, damage_events,
                            state["p1_knowledge"], state["p2_knowledge"],
                            session=aiohttp_session, base_url=calc_base_url,
                        )
                    except Exception:
                        pass

    # Save per-provider output + print summary.
    output_dir.mkdir(parents=True, exist_ok=True)
    click.echo("\n=== bake-off summary ===")
    click.echo(f"{'provider':12s} {'rows':>5s} {'match%':>7s} {'calc/turn':>10s} {'$/row':>8s} {'avg cot':>8s} {'wall':>6s}")
    for p in providers:
        s = provider_state[p.name]
        rows = int(s["totals"]["turns_succeeded"])
        attempted = int(s["totals"]["turns_attempted"])
        match_rate = (s["totals"]["actions_matched"] / attempted * 100) if attempted else 0.0
        calc_per = (s["totals"]["calc_calls"] / attempted) if attempted else 0.0
        cost_per = (s["totals"]["cost_usd"] / rows) if rows else 0.0
        avg_cot = 0
        if s["rows"]:
            for row in s["rows"]:
                args = _extract_submit_args(row["messages"])
                if args:
                    avg_cot += len(args.get("pre_tool_thought", ""))
            avg_cot //= max(1, len(s["rows"]))

        click.echo(
            f"{p.name:12s} {rows:5d} {match_rate:6.1f}% {calc_per:10.2f} "
            f"${cost_per:7.4f} {avg_cot:8d} {s['totals']['elapsed_seconds']:5.1f}s"
        )

        out_path = output_dir / f"bakeoff_{p.name}.jsonl"
        with out_path.open("w") as f:
            for row in s["rows"]:
                f.write(json.dumps(row) + "\n")
        click.echo(f"  → {out_path}  ({len(s['rows'])} rows)")


@click.command()
@click.option("--input", "input_path",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=str(DEFAULT_BO3_INPUT),
              show_default=True)
@click.option("--match-id", default=None,
              help="Match-id substring to select. Defaults to the first record in --input.")
@click.option("--providers", "providers_csv", default="openai,anthropic,google", show_default=True,
              help="Comma-separated subset of {openai,anthropic,google} to run.")
@click.option("--output-dir",
              type=click.Path(file_okay=False, path_type=Path),
              default=str(DEFAULT_PARSED_DATA_DIR),
              show_default=True)
@click.option("--calc-base-url", default=DEFAULT_CALC_BASE_URL, show_default=True)
@click.option("--format-id", default=None,
              help="Override the format_id derived from input filename.")
def cli(input_path, match_id, providers_csv, output_dir, calc_base_url, format_id):
    """Run a head-to-head bake-off across frontier teacher models on one match."""
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
        records = [r for r in records if match_id in r["match_id"]]
        if not records:
            click.echo(f"FATAL: no record matched id substring {match_id!r}", err=True)
            sys.exit(1)

    target = records[0]
    derived_format = format_id or ("gen9vgc2026regibo3" if "bo3" in input_path.stem else "gen9vgc2026regi")

    asyncio.run(_run(target, providers, output_dir, calc_base_url, derived_format))


async def _run(match_record, providers, output_dir, calc_base_url, format_id):
    timeout = aiohttp.ClientTimeout(total=180, connect=10)
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

        await _bakeoff_one_match(
            match_record, providers,
            output_dir=Path(output_dir),
            calc_base_url=calc_base_url,
            format_id=format_id,
            aiohttp_session=session,
        )


if __name__ == "__main__":
    cli()
