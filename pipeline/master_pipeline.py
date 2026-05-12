"""Orchestrator: scraped replays in -> SFT-ready JSONL out.

Pipeline role:
    Walks through `parsed_data/{bo1,bo3}.jsonl` (produced by replay_parser),
    and for each turn of each match:
      1. relabels sides so the series winner is P1 (`flip_match_to_winner`);
      2. extracts P1's actual two-slot decision from `snap[N].events` +
         the diff to `snap[N+1]` (move / switch / cant_move events, Tera flag);
      3. asks `threat_matrix` to render the dual-track damage envelope;
      4. drives `teacher.synthesize_turn` to elicit a chain-of-thought
         that justifies that exact decision;
      5. writes the resulting OpenAI-fine-tuning conversation to
         `parsed_data/sft_training_data.jsonl`;
      6. filters the same `events` stream for damage observations and
         feeds them to `damage_inferencer.update_knowledge` to tighten
         both KnowledgeStates for the next turn.

    KnowledgeStates start at fully-open `[0, 252]` bounds — the canonical
    priors are used by `threat_matrix` for its Probable track only,
    preserving the Absolute track's strict-math guarantee.

Isolation contract:
    The only file allowed to import from every other pipeline module.
    Everything else is leaf — no cross-imports between siblings.
"""
from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import aiohttp
import click
from tqdm.asyncio import tqdm

import canonical_priors  # noqa: F401  — imported for symmetry; future use in seeding
import damage_inferencer
import threat_matrix
from action_extraction import (
    extract_p1_actions,
    flip_match_to_winner,
    slot_action,
)
from prompt_formatting import (
    format_p1_known_spreads_block,
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
    PRODUCTION_LEAK_RETRIES,
    TeacherProvider,
    detect_oracle_leak,
    render_system_prompt,
    render_system_prompt_bo3,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = Path(__file__).resolve().parent
DEFAULT_PARSED_DATA_DIR = PIPELINE_DIR / "parsed_data"
DEFAULT_BO3_INPUT = DEFAULT_PARSED_DATA_DIR / "bo3.jsonl"
DEFAULT_BO1_INPUT = DEFAULT_PARSED_DATA_DIR / "bo1.jsonl"
DEFAULT_OUTPUT = DEFAULT_PARSED_DATA_DIR / "sft_training_data.jsonl"
DEFAULT_CALC_BASE_URL = "http://localhost:3000"

FORMAT_ID_BY_KIND = {
    "bo1": "gen9vgc2026regi",
    "bo3": "gen9vgc2026regibo3",
}


# ---------------------------------------------------------------------------
# Resumability
# ---------------------------------------------------------------------------


def load_seen_keys(output_path: Path) -> set[tuple[str, int, int]]:
    seen: set[tuple[str, int, int]] = set()
    if not output_path.exists():
        return seen
    with output_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = (rec["match_id"], int(rec["game_index"]), int(rec["turn"]))
                seen.add(key)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return seen


# ---------------------------------------------------------------------------
# Provider + calc-service plumbing
# ---------------------------------------------------------------------------


def _build_teacher(provider: str, model: str | None) -> TeacherProvider:
    """Instantiate the requested provider, validating that its API key is present."""
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise click.ClickException(
                "OPENAI_API_KEY env var is required (or pass --dry-run / --provider <other>)"
            )
        from teacher import OpenAIProvider
        return OpenAIProvider(model=model) if model else OpenAIProvider()
    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise click.ClickException("ANTHROPIC_API_KEY env var is required for --provider anthropic")
        from teacher import AnthropicProvider
        return AnthropicProvider(model=model) if model else AnthropicProvider()
    if provider == "google":
        if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
            raise click.ClickException(
                "GOOGLE_API_KEY (or GEMINI_API_KEY) env var is required for --provider google"
            )
        from teacher import GoogleProvider
        return GoogleProvider(model=model) if model else GoogleProvider()
    raise click.ClickException(f"unknown provider: {provider}")


async def _check_calc_health(session: aiohttp.ClientSession, base_url: str) -> None:
    try:
        async with session.get(f"{base_url}/health") as r:
            if r.status != 200:
                raise RuntimeError(f"health check returned {r.status}")
    except Exception as e:
        raise click.ClickException(
            f"calc_microservice not reachable at {base_url}/health: {e}\n"
            f"  Start it with:  cd calc_microservice && npm run dev"
        )


# ---------------------------------------------------------------------------
# Per-match orchestration
# ---------------------------------------------------------------------------


async def process_match(
    match_record: dict[str, Any],
    *,
    output_path: Path,
    calc_base_url: str,
    teacher: TeacherProvider | None,
    aiohttp_session: aiohttp.ClientSession,
    file_lock: asyncio.Lock,
    format_id: str,
    seen_keys: set[tuple[str, int, int]],
    dry_run: bool,
    model: str,
    leak_retries: int = PRODUCTION_LEAK_RETRIES,
) -> dict[str, int]:
    # Series-winner-as-P1: every SFT example is generated from the perspective
    # of the player who won the series. P2-won matches are relabeled in full.
    match_record = flip_match_to_winner(match_record)

    games = match_record.get("games") or []
    if not games:
        return {"skipped_no_games": 1}

    match_format = match_record.get("format", "bo1")
    team_sheets = team_sheets_for_match(games) if match_format == "bo3" else None

    # Knowledge state seeding — for OTS Bo3, use the full 6-mon team sheets
    # so the threat matrix can reason about the unswitched-in backline too.
    # For CTS Bo1, fall back to whatever the snapshots reveal.
    p1_team_recon = reconstruct_p1_team(games)
    if team_sheets:
        p1_species_universe = [s["species"] for s in team_sheets["p1"]]
        p2_species_universe = [s["species"] for s in team_sheets["p2"]]
    else:
        p1_species_universe = list(p1_team_recon.keys())
        p2_species_universe = reconstruct_p2_species(games)

    # Two parallel KnowledgeStates for P1 + one for P2:
    #   p1_running  — fed turn-by-turn; kept for diagnostics + the inspector.
    #                 NOT used in prompts (the player knows their own team).
    #   p1_final    — computed once below from the full match; surfaces as
    #                 the YOUR SPREADS block AND drives the matrix's P1 side.
    #                 Approximates "the exact spread the player built".
    #   p2_running  — fed turn-by-turn; drives the matrix's P2 side.
    #                 Models the observational asymmetry — we learn about
    #                 the opponent through play.
    p1_running = damage_inferencer.init_knowledge(p1_species_universe)
    p2_running = damage_inferencer.init_knowledge(p2_species_universe)
    p1_final, _ = await damage_inferencer.infer_match_final_bounds(
        games, p1_species_universe, p2_species_universe,
        session=aiohttp_session, base_url=calc_base_url,
    )

    # Bo1 system prompt is stable across all turns of the match.
    bo1_system_prompt = (
        render_system_prompt(format_p1_team_block(p1_team_recon))
        if not team_sheets
        else None
    )

    stats: dict[str, int] = defaultdict(int)
    match_id = match_record.get("match_id", "unknown")

    for game_idx, game in enumerate(games):
        snapshots = game.get("snapshots") or []

        # Bo3 system prompt depends on the brought-4 of THIS game (different
        # selections per game in a series), so render per-game.
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
            key = (match_id, game_idx, turn)
            if key in seen_keys:
                stats["already_done"] += 1
                continue

            human_action_dict = extract_p1_actions(snap_pre, snap_post, events_stream)
            if human_action_dict is None:
                stats["skipped_ambiguous"] += 1
                await _safe_update_knowledge(
                    snap_pre, snap_post, events_stream, p1_running, p2_running,
                    session=aiohttp_session, base_url=calc_base_url,
                )
                continue

            try:
                # Threat matrix gets the asymmetric (p1_final, p2_running) pair:
                # tight bounds for our side (we know our team), loose
                # chronological bounds for the opponent (we learn over time).
                tm_text = await threat_matrix.generate_threat_matrix(
                    snap_pre, "p1", p1_final, p2_running,
                    format_id=format_id,
                    session=aiohttp_session,
                    base_url=calc_base_url,
                )
            except Exception as e:
                stats["skipped_threat_matrix_error"] += 1
                _log_error(f"[{match_id} g{game_idx} t{turn}] threat_matrix failed: {e}")
                await _safe_update_knowledge(
                    snap_pre, snap_post, events_stream, p1_running, p2_running,
                    session=aiohttp_session, base_url=calc_base_url,
                )
                continue

            # YOUR SPREADS surfaces match-final P1 bounds (the player's
            # own-team knowledge stand-in).
            p1_spreads = format_p1_known_spreads_block(snap_pre, p1_final)
            user_prompt = format_user_prompt(
                snap_pre,
                tm_text,
                p1_inferred_block=p1_spreads,
                snapshots_so_far=snapshots,
                current_idx=i,
                prior_games=games[:game_idx],
                game_index=game_idx,
                total_games_in_series=len(games),
                match_format=match_format,
            )
            human_action = {
                "slot_1": human_action_dict.get("a", slot_action("pass")),
                "slot_2": human_action_dict.get("b", slot_action("pass")),
            }

            messages: list[dict[str, Any]] | None = None
            if dry_run:
                messages = _dry_run_messages(system_prompt, user_prompt, human_action)
            else:
                if teacher is None:
                    raise RuntimeError("Teacher provider missing in non-dry-run mode")
                # Leak-retry loop: synthesize, check for ground-truth leakage,
                # retry up to `leak_retries` times on a hit.
                attempts = 0
                while True:
                    try:
                        res = await teacher.synthesize_turn(
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            human_action=human_action,
                            calc_url=f"{calc_base_url}/calc",
                            aiohttp_session=aiohttp_session,
                        )
                        messages = res.messages
                    except Exception as e:
                        stats["skipped_llm_error"] += 1
                        _log_error(f"[{match_id} g{game_idx} t{turn}] teacher LLM failed: {e}")
                        messages = None
                        break
                    if messages is None:
                        if res.error:
                            _log_error(f"[{match_id} g{game_idx} t{turn}] teacher LLM: {res.error}")
                        stats["skipped_llm_error"] += 1
                        break
                    leak = detect_oracle_leak(messages)
                    if not leak:
                        break
                    if attempts >= leak_retries:
                        stats["skipped_persistent_leak"] += 1
                        _log_error(
                            f"[{match_id} g{game_idx} t{turn}] persistent leak after "
                            f"{attempts+1} attempt(s) — phrase {leak!r}; dropping row"
                        )
                        messages = None
                        break
                    attempts += 1
                    stats["leak_retry"] += 1
                    _log_error(
                        f"[{match_id} g{game_idx} t{turn}] leak retry {attempts}/"
                        f"{leak_retries} — phrase {leak!r}"
                    )

            if messages is None:
                # API error, persistent leak, or dry-run-with-no-teacher.
                # Either way, no row gets written; stats already counted.
                if not dry_run:
                    stats.setdefault("skipped_llm_failed", 0)
            else:
                async with file_lock:
                    with output_path.open("a") as f:
                        f.write(json.dumps({
                            "match_id": match_id,
                            "game_index": game_idx,
                            "turn": turn,
                            "format_id": format_id,
                            "messages": messages,
                        }) + "\n")
                seen_keys.add(key)
                stats["written"] += 1

            await _safe_update_knowledge(
                snap_pre, snap_post, events_stream, p1_running, p2_running,
                session=aiohttp_session, base_url=calc_base_url,
            )

    return dict(stats)


async def _safe_update_knowledge(
    snap_pre, snap_post, events_stream, p1_knowledge, p2_knowledge, *, session, base_url
):
    """Filter the new TurnEvent stream for damage observations and feed
    them to the binary-search inferencer. Drops Metronome / Copycat /
    Sketch / Snatch / Me First / Dancer / Instruct / Mirror Move /
    Assist / Nature Power call-throughs (those can hit moves not in the
    user's actual kit) but keeps Sleep Talk (calls own moves only)."""
    damage_events = damage_inferencer.events_to_damage_events(events_stream)
    if not damage_events:
        return
    try:
        await damage_inferencer.update_knowledge(
            snap_pre, snap_post, damage_events, p1_knowledge, p2_knowledge,
            session=session, base_url=base_url,
        )
    except Exception as e:
        _log_error(f"update_knowledge failed: {e}")


def _dry_run_messages(
    system_prompt: str, user_prompt: str, human_action: dict[str, Any]
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "pre_tool_thought": "[DRY RUN — teacher LLM not invoked]",
                    "action": human_action,
                }
            ),
        },
    ]


def _log_error(msg: str) -> None:
    click.echo(msg, err=True)


# ---------------------------------------------------------------------------
# CLI / runner
# ---------------------------------------------------------------------------


def _resolve_format_id(input_path: Path, override: str | None) -> str:
    if override:
        return override
    name = input_path.stem.lower()
    if name in FORMAT_ID_BY_KIND:
        return FORMAT_ID_BY_KIND[name]
    return FORMAT_ID_BY_KIND["bo3"]


def _read_match_records(input_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with input_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


async def run(
    *,
    input_path: Path,
    output_path: Path,
    calc_base_url: str,
    format_id: str,
    limit: int | None,
    concurrency: int,
    dry_run: bool,
    model: str | None,
    provider: str,
    leak_retries: int = PRODUCTION_LEAK_RETRIES,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timeout = aiohttp.ClientTimeout(total=180, connect=10)

    teacher: TeacherProvider | None = None
    if not dry_run:
        teacher = _build_teacher(provider, model)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        await _check_calc_health(session, calc_base_url)

        records = _read_match_records(input_path)
        click.echo(f"loaded {len(records)} match records from {input_path}")

        seen_keys = load_seen_keys(output_path)
        if seen_keys:
            click.echo(f"  resuming: {len(seen_keys)} (match, game, turn) keys already in {output_path.name}")

        if limit is not None:
            records = records[:limit]
            click.echo(f"  --limit {limit}: processing first {len(records)} matches")

        file_lock = asyncio.Lock()
        sem = asyncio.Semaphore(concurrency)

        async def worker(rec):
            async with sem:
                return await process_match(
                    rec,
                    output_path=output_path,
                    calc_base_url=calc_base_url,
                    teacher=teacher,
                    aiohttp_session=session,
                    file_lock=file_lock,
                    format_id=format_id,
                    seen_keys=seen_keys,
                    dry_run=dry_run,
                    model=model,
                    leak_retries=leak_retries,
                )

        results = await tqdm.gather(*(worker(r) for r in records), desc="matches", unit="match")

    totals: dict[str, int] = defaultdict(int)
    for r in results:
        for k, v in r.items():
            totals[k] += v
    click.echo("\n=== summary ===")
    for k in sorted(totals):
        click.echo(f"  {k}: {totals[k]}")


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=str(DEFAULT_BO3_INPUT),
    show_default=True,
    help="Match-records JSONL from replay_parser (parsed_data/bo1.jsonl or bo3.jsonl).",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=str(DEFAULT_OUTPUT),
    show_default=True,
)
@click.option("--calc-base-url", default=DEFAULT_CALC_BASE_URL, show_default=True)
@click.option(
    "--format-id",
    default=None,
    help="Format ID for canonical-priors lookup. Auto-detected from input filename if omitted.",
)
@click.option("--limit", type=int, default=None, help="Process only the first N matches (test batch).")
@click.option("--concurrency", default=1, show_default=True,
              help="Max matches processed in parallel. Keep low (1-3) to respect provider rate limits.")
@click.option("--dry-run", is_flag=True, help="Skip the LLM call; emit a placeholder assistant message.")
@click.option(
    "--provider",
    type=click.Choice(["openai", "anthropic", "google"]),
    default="openai",
    show_default=True,
    help="Which teacher LLM backend to use.",
)
@click.option(
    "--model",
    default=None,
    help="Override the default model id for the chosen provider.",
)
@click.option(
    "--leak-retries", type=int, default=PRODUCTION_LEAK_RETRIES, show_default=True,
    help="Retries per turn when the teacher's CoT contains a ground-truth-leak phrase. "
         "0 = drop on first hit (smoke / measurement). Default in production: 3.",
)
def cli(input_path, output_path, calc_base_url, format_id, limit, concurrency, dry_run,
        provider, model, leak_retries):
    """Generate the SFT training JSONL from parsed replay data."""
    resolved_format = _resolve_format_id(input_path, format_id)
    click.echo(
        f"using format_id={resolved_format}  dry_run={dry_run}  "
        f"provider={provider}  model={model or '(default)'}  leak_retries={leak_retries}"
    )
    asyncio.run(
        run(
            input_path=input_path,
            output_path=output_path,
            calc_base_url=calc_base_url,
            format_id=resolved_format,
            limit=limit,
            concurrency=concurrency,
            dry_run=dry_run,
            model=model,
            provider=provider,
            leak_retries=leak_retries,
        )
    )


if __name__ == "__main__":
    cli()
