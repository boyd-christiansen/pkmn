"""Orchestrator: scraped replays in -> SFT-ready JSONL out.

Pipeline role:
    Walks through `parsed_data/{bo1,bo3}.jsonl` (produced by replay_parser),
    and for each turn of each match:
      1. reconstructs P1's team (revealed item / ability / tera / moves with
         `[UNREVEALED_MOVE]` padding for slots the human never used);
      2. extracts P1's actual two-slot decision from `snap[N].actionLog` +
         the diff to `snap[N+1]` (damage moves, switches, status moves,
         Tera flag);
      3. asks `threat_matrix` to render the dual-track damage envelope;
      4. drives `teacher_llm.synthesize_turn` to elicit a chain-of-thought
         that justifies that exact decision;
      5. writes the resulting OpenAI-fine-tuning conversation to
         `parsed_data/sft_training_data.jsonl`;
      6. feeds the same `actionLog` to `damage_inferencer.update_knowledge`
         to tighten both KnowledgeStates for the next turn.

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
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import aiohttp
import click
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

import canonical_priors
import damage_inferencer
import teacher_llm
import threat_matrix

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
# Reconstruction
# ---------------------------------------------------------------------------


def reconstruct_p1_team(games: list[dict]) -> dict[str, dict[str, Any]]:
    """Forward-scan all snapshots in a match, aggregate revealed P1 info per species.

    Pads each Pokémon's move list to exactly 4 with `"[UNREVEALED_MOVE]"`.
    """
    aggregated: dict[str, dict[str, Any]] = {}

    def _ensure(species: str) -> dict[str, Any]:
        return aggregated.setdefault(
            species,
            {
                "species": species,
                "item": None,
                "ability": None,
                "teraType": None,
                "isTerastallized": False,
                "moves": [],
            },
        )

    for game in games:
        for snap in game.get("snapshots", []):
            for p in snap.get("p1", {}).get("active", []):
                entry = _ensure(p["species"])
                if p.get("item") and not entry["item"]:
                    entry["item"] = p["item"]
                if p.get("ability") and not entry["ability"]:
                    entry["ability"] = p["ability"]
                if p.get("teraType") and not entry["teraType"]:
                    entry["teraType"] = p["teraType"]
                if p.get("isTerastallized"):
                    entry["isTerastallized"] = True
                for mv in p.get("revealedMoves") or []:
                    if mv not in entry["moves"]:
                        entry["moves"].append(mv)
            for b in snap.get("p1", {}).get("bench", []):
                _ensure(b["species"])

    for entry in aggregated.values():
        while len(entry["moves"]) < 4:
            entry["moves"].append("[UNREVEALED_MOVE]")
        entry["moves"] = entry["moves"][:4]

    return aggregated


def reconstruct_p2_species(games: list[dict]) -> list[str]:
    seen: list[str] = []
    for game in games:
        for snap in game.get("snapshots", []):
            for p in snap.get("p2", {}).get("active", []) + snap.get("p2", {}).get("bench", []):
                if p["species"] not in seen:
                    seen.append(p["species"])
    return seen


def _species_key(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum())


def _team_sheets_for_match(games: list[dict]) -> dict[str, list[dict]] | None:
    """Return the first non-null `teamSheets` from any game in the match.

    All games in a Bo3 series carry the same sheet, so we just take the
    earliest one available. None means CTS for the whole match.
    """
    for g in games:
        sheets = g.get("teamSheets")
        if sheets and sheets.get("p1") and sheets.get("p2"):
            return sheets
    return None


def _brought_species_keys_for_game(game: dict) -> set[str]:
    """Species (normalized keys) actually brought by P1 to this single game.

    Derived from the union of P1 active + P1 bench across this game's
    snapshots. In OTS Bo3 the parser already gates P1 bench to broughtSet,
    so this naturally yields the 4 brought.
    """
    out: set[str] = set()
    for snap in game.get("snapshots", []):
        for p in snap.get("p1", {}).get("active", []):
            out.add(_species_key(p["species"]))
        for b in snap.get("p1", {}).get("bench", []):
            out.add(_species_key(b["species"]))
    return out


# ---------------------------------------------------------------------------
# Action extraction
# ---------------------------------------------------------------------------


def _slot_action(
    action_type: str,
    *,
    move: str | None = None,
    target: str | None = None,
    tera: bool | None = None,
    switch_to: str | None = None,
) -> dict[str, Any]:
    return {
        "action_type": action_type,
        "move": move,
        "target": target,
        "tera": tera,
        "switch_to": switch_to,
    }


def extract_p1_actions(
    snap_pre: dict[str, Any],
    snap_post: dict[str, Any],
    action_log: list[dict[str, Any]],
) -> dict[str, dict[str, Any]] | None:
    """Reverse-engineer what each P1 active slot did this turn.

    Returns a dict `{ "a": slot_action, "b": slot_action }` or `None` when
    any slot's action is ambiguous (skip the turn).
    """
    out: dict[str, dict[str, Any]] = {}
    pre_active = {p["slot"]: p for p in snap_pre.get("p1", {}).get("active", [])}
    post_active = {p["slot"]: p for p in snap_post.get("p1", {}).get("active", [])}

    for letter in ("a", "b"):
        pre_p = pre_active.get(letter)
        post_p = post_active.get(letter)
        slot_id = f"p1{letter}"

        if pre_p is None or pre_p.get("fainted"):
            out[letter] = _slot_action("pass")
            continue

        attacker_events = [e for e in action_log if e.get("attacker_slot") == slot_id]
        if attacker_events:
            move_name = attacker_events[0]["move_name"]
            targets = sorted({e["defender_slot"] for e in attacker_events})
            target = targets[0] if len(targets) == 1 else "spread"
            tera = (
                post_p is not None
                and post_p["species"] == pre_p["species"]
                and bool(post_p.get("isTerastallized"))
                and not bool(pre_p.get("isTerastallized"))
            )
            out[letter] = _slot_action("move", move=move_name, target=target, tera=tera)
            continue

        if post_p is not None and pre_p["species"] != post_p["species"]:
            out[letter] = _slot_action("switch", switch_to=post_p["species"])
            continue

        if post_p is not None and post_p["species"] == pre_p["species"]:
            pre_moves = set(pre_p.get("revealedMoves") or [])
            post_moves = set(post_p.get("revealedMoves") or [])
            new_moves = post_moves - pre_moves
            if len(new_moves) == 1:
                tera = (
                    bool(post_p.get("isTerastallized"))
                    and not bool(pre_p.get("isTerastallized"))
                )
                out[letter] = _slot_action(
                    "move", move=next(iter(new_moves)), target="self", tera=tera
                )
                continue

        return None  # ambiguous — skip the whole turn

    return out


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def format_p1_team_block(p1_team: dict[str, dict[str, Any]]) -> str:
    lines: list[str] = []
    for entry in p1_team.values():
        moves_str = " / ".join(entry["moves"])
        item = entry["item"] or "?"
        ability = entry["ability"] or "?"
        tera = entry["teraType"] or "?"
        lines.append(
            f"  - {entry['species']} @ {item}, ability={ability}, tera={tera}\n"
            f"      moves: {moves_str}"
        )
    return "\n".join(lines)


def _summarize_active(p: dict[str, Any]) -> str:
    parts = [
        f"[{p['slot']}] {p['species']}",
        f"HP {p.get('hpPercent', '?')}%",
    ]
    if p.get("status"):
        parts.append(f"status={p['status']}")
    if p.get("item"):
        parts.append(f"item={p['item']}")
    if p.get("ability"):
        parts.append(f"ability={p['ability']}")
    tera_type = p.get("teraType")
    if tera_type:
        if p.get("isTerastallized"):
            parts.append(f"TERA-ACTIVE ({p.get('terastallizedAs') or tera_type})")
        else:
            parts.append(f"tera={tera_type}")
    boosts = p.get("boosts") or {}
    if boosts:
        parts.append("boosts=" + ",".join(f"{k}{v:+d}" for k, v in boosts.items()))
    revealed = p.get("revealedMoves") or []
    if revealed:
        parts.append("revealed=" + ",".join(revealed))
    return "  " + " | ".join(parts)


def _summarize_bench(b: dict[str, Any]) -> str:
    return f"{b['species']}{' (fainted)' if b.get('fainted') else ''}"


def format_user_prompt(snapshot: dict[str, Any], threat_matrix_text: str) -> str:
    f = snapshot.get("field", {})
    field_parts = []
    if f.get("weather"):
        field_parts.append(f"weather={f['weather']}")
    if f.get("terrain"):
        field_parts.append(f"terrain={f['terrain']}")
    field_parts.append(f"P1-tailwind={'YES' if f.get('tailwindP1') else 'no'}")
    field_parts.append(f"P2-tailwind={'YES' if f.get('tailwindP2') else 'no'}")
    field_str = ", ".join(field_parts)

    p1 = snapshot.get("p1", {})
    p2 = snapshot.get("p2", {})

    p1_active_lines = "\n".join(_summarize_active(p) for p in p1.get("active", []))
    p2_active_lines = "\n".join(_summarize_active(p) for p in p2.get("active", []))
    p1_bench = ", ".join(_summarize_bench(b) for b in p1.get("bench", [])) or "(none)"
    p2_bench = ", ".join(_summarize_bench(b) for b in p2.get("bench", [])) or "(none)"

    return (
        f"=== TURN {snapshot.get('turn', '?')} ===\n"
        f"Field: {field_str}\n\n"
        f"YOUR (P1) ACTIVE:\n{p1_active_lines}\n"
        f"YOUR (P1) BENCH: {p1_bench}\n\n"
        f"OPP (P2) ACTIVE:\n{p2_active_lines}\n"
        f"OPP (P2) BENCH: {p2_bench}\n\n"
        f"{threat_matrix_text}"
    )


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
# Match processing
# ---------------------------------------------------------------------------


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


async def process_match(
    match_record: dict[str, Any],
    *,
    output_path: Path,
    calc_base_url: str,
    openai_client: AsyncOpenAI | None,
    aiohttp_session: aiohttp.ClientSession,
    file_lock: asyncio.Lock,
    format_id: str,
    seen_keys: set[tuple[str, int, int]],
    dry_run: bool,
    model: str,
) -> dict[str, int]:
    games = match_record.get("games") or []
    if not games:
        return {"skipped_no_games": 1}

    match_format = match_record.get("format", "bo1")
    team_sheets = _team_sheets_for_match(games) if match_format == "bo3" else None

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

    p1_knowledge = damage_inferencer.init_knowledge(p1_species_universe)
    p2_knowledge = damage_inferencer.init_knowledge(p2_species_universe)

    # Bo1 system prompt is stable across all turns of the match.
    bo1_system_prompt = (
        teacher_llm.render_system_prompt(format_p1_team_block(p1_team_recon))
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
            brought = _brought_species_keys_for_game(game)
            system_prompt = teacher_llm.render_system_prompt_bo3(
                p1_sheet=team_sheets["p1"],
                p2_sheet=team_sheets["p2"],
                p1_brought=brought,
            )
        else:
            system_prompt = bo1_system_prompt
        for i in range(len(snapshots) - 1):
            snap_pre = snapshots[i]
            snap_post = snapshots[i + 1]
            action_log = snap_pre.get("actionLog") or []
            turn = int(snap_pre.get("turn", 0))
            key = (match_id, game_idx, turn)
            if key in seen_keys:
                stats["already_done"] += 1
                continue

            human_action_dict = extract_p1_actions(snap_pre, snap_post, action_log)
            if human_action_dict is None:
                stats["skipped_ambiguous"] += 1
                await _safe_update_knowledge(
                    snap_pre, snap_post, action_log, p1_knowledge, p2_knowledge,
                    session=aiohttp_session, base_url=calc_base_url,
                )
                continue

            try:
                tm_text = await threat_matrix.generate_threat_matrix(
                    snap_pre, "p1", p1_knowledge, p2_knowledge,
                    format_id=format_id,
                    session=aiohttp_session,
                    base_url=calc_base_url,
                )
            except Exception as e:
                stats["skipped_threat_matrix_error"] += 1
                _log_error(f"[{match_id} g{game_idx} t{turn}] threat_matrix failed: {e}")
                await _safe_update_knowledge(
                    snap_pre, snap_post, action_log, p1_knowledge, p2_knowledge,
                    session=aiohttp_session, base_url=calc_base_url,
                )
                continue

            user_prompt = format_user_prompt(snap_pre, tm_text)
            human_action = {
                "slot_1": human_action_dict.get("a", _slot_action("pass")),
                "slot_2": human_action_dict.get("b", _slot_action("pass")),
            }

            messages: list[dict[str, Any]] | None
            if dry_run:
                messages = _dry_run_messages(system_prompt, user_prompt, human_action)
            else:
                if openai_client is None:
                    raise RuntimeError("OpenAI client missing in non-dry-run mode")
                try:
                    messages = await teacher_llm.synthesize_turn(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        human_action=human_action,
                        calc_url=f"{calc_base_url}/calc",
                        model=model,
                        openai_client=openai_client,
                        aiohttp_session=aiohttp_session,
                    )
                except Exception as e:
                    stats["skipped_llm_error"] += 1
                    _log_error(f"[{match_id} g{game_idx} t{turn}] teacher LLM failed: {e}")
                    messages = None

            if messages is None:
                stats["skipped_llm_failed"] += 1
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
                snap_pre, snap_post, action_log, p1_knowledge, p2_knowledge,
                session=aiohttp_session, base_url=calc_base_url,
            )

    return dict(stats)


async def _safe_update_knowledge(
    snap_pre, snap_post, action_log, p1_knowledge, p2_knowledge, *, session, base_url
):
    try:
        events = [damage_inferencer.DamageEvent(**e) for e in action_log]
    except (TypeError, KeyError):
        return
    try:
        await damage_inferencer.update_knowledge(
            snap_pre, snap_post, events, p1_knowledge, p2_knowledge,
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
    model: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timeout = aiohttp.ClientTimeout(total=180, connect=10)

    if not dry_run and not os.environ.get("OPENAI_API_KEY"):
        raise click.ClickException(
            "OPENAI_API_KEY env var is required (or pass --dry-run for the orchestration smoke test)"
        )

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

        openai_client = None if dry_run else AsyncOpenAI()
        file_lock = asyncio.Lock()
        sem = asyncio.Semaphore(concurrency)

        async def worker(rec):
            async with sem:
                return await process_match(
                    rec,
                    output_path=output_path,
                    calc_base_url=calc_base_url,
                    openai_client=openai_client,
                    aiohttp_session=session,
                    file_lock=file_lock,
                    format_id=format_id,
                    seen_keys=seen_keys,
                    dry_run=dry_run,
                    model=model,
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
              help="Max matches processed in parallel. Keep low (1-3) to respect OpenAI rate limits.")
@click.option("--dry-run", is_flag=True, help="Skip the OpenAI call; emit a placeholder assistant message.")
@click.option("--model", default=teacher_llm.DEFAULT_MODEL, show_default=True)
def cli(input_path, output_path, calc_base_url, format_id, limit, concurrency, dry_run, model):
    """Generate the SFT training JSONL from parsed replay data."""
    resolved_format = _resolve_format_id(input_path, format_id)
    click.echo(f"using format_id={resolved_format}  dry_run={dry_run}  model={model}")
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
        )
    )


if __name__ == "__main__":
    cli()
