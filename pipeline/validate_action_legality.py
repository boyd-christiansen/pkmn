"""Corpus-level action-legality validator (Issue 2, Part B).

Pipeline role:
    Scans the parsed corpus for SFT training labels whose action contradicts
    the constraints encoded in their OWN turn state. Because every label is a
    *real human play* (legal by the rules of the live game), a flagged
    violation is never a "bad move" — it is a DATA BUG: the rendered state
    said an action was illegal when in fact it was legal, which would teach a
    student model a false constraint (or, worse, that an illegal action is
    fine). This tool surfaces those.

    Robustly-checkable constraints (state we actually carry):
      • choice-lock   — a Choice-item mon shown `choiceLockedInto X` must play
                        X (or switch). A real play that uses a different move
                        means the lock was rendered wrongly (stale / over-eager
                        `snapshotChoiceLock`).
      • tera-after-used — Terastallizing when the side's `teraUsed` is already
                        set is impossible live; a flag means the teraUsed
                        sticky-state is wrong.
      • moveset-membership (Bo3 OTS) — a move the human used that is absent
                        from the mon's `knownMoves` means the team-sheet decode
                        dropped/mis-normalized a move.

    Not checkable without richer parser state (reported as KNOWN GAPS, not
    scanned): Encore-lock and Disable carry only a boolean (no move name), and
    trapping abilities / `trapped` are not captured at all, so "switched while
    trapped" and "used the disabled/non-encored move" cannot be detected. The
    fix for those is a parser enhancement (capture the encored/disabled move
    id + a trapped flag), tracked in notes/TODO.md.

Isolation contract:
    Read-only over `parsed_data/*.jsonl`. Imports `action_extraction`
    (the same module the orchestrator uses) so the labels validated here are
    byte-identical to the ones synthesis would emit. No calc service, no LLM.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import click

from action_extraction import (
    DEFAULT_MIN_GAME_TURNS,
    extract_p1_actions,
    filter_fragment_games,
    flip_match_to_winner,
)

PIPELINE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUTS = [
    PIPELINE_DIR / "parsed_data" / "bo1.jsonl",
    PIPELINE_DIR / "parsed_data" / "bo3.jsonl",
]


def _move_key(s: str | None) -> str:
    return "".join(c for c in (s or "").lower() if c.isalnum())


def _check_turn(snap_pre: dict[str, Any], actions: dict[str, dict[str, Any]]) -> list[tuple[str, str]]:
    """Return [(violation_type, detail), ...] for one extracted turn."""
    p1 = snap_pre.get("p1", {}) or {}
    tera_used = bool(p1.get("teraUsed"))
    pre_active = {p.get("slot"): p for p in (p1.get("active") or [])}
    out: list[tuple[str, str]] = []

    for letter, act in actions.items():
        mon = pre_active.get(letter)
        if not mon or act.get("action_type") != "move":
            continue
        species = mon.get("species", "?")
        move = act.get("move")

        # 1. Choice lock — a real play can only be the locked move (or a switch,
        #    handled above by action_type != 'move'). A different move ⇒ the
        #    rendered lock is wrong.
        lock = mon.get("choiceLockedInto")
        if lock and _move_key(move) != _move_key(lock):
            out.append(("choice_lock", f"{species}: shown locked into {lock!r}, label plays {move!r}"))

        # 2. Tera after the side already Terastallized.
        if act.get("tera") and tera_used:
            out.append(("tera_after_used", f"{species}: label tera's but teraUsed already set"))

        # 3. OTS moveset membership.
        known = mon.get("knownMoves")
        if known and move and _move_key(move) != "struggle":
            if _move_key(move) not in {_move_key(k) for k in known if k}:
                out.append(("move_not_in_knownset", f"{species}: label move {move!r} ∉ knownMoves {known}"))

    return out


def _scan_file(path: Path, min_game_turns: int) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    turns_scanned = 0
    turns_ambiguous = 0
    matches = 0

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            matches += 1
            rec = flip_match_to_winner(rec)
            games, _ = filter_fragment_games(rec.get("games") or [], min_game_turns)
            for gi, game in enumerate(games):
                snaps = game.get("snapshots") or []
                for i in range(len(snaps) - 1):
                    snap_pre, snap_post = snaps[i], snaps[i + 1]
                    events = snap_pre.get("events") or []
                    actions = extract_p1_actions(snap_pre, snap_post, events)
                    if actions is None:
                        turns_ambiguous += 1
                        continue
                    turns_scanned += 1
                    for vtype, detail in _check_turn(snap_pre, actions):
                        counts[vtype] += 1
                        if len(examples[vtype]) < 8:
                            examples[vtype].append(
                                f"{rec.get('match_id','?')} g{gi} t{snap_pre.get('turn','?')}: {detail}"
                            )

    return {
        "matches": matches,
        "turns_scanned": turns_scanned,
        "turns_ambiguous": turns_ambiguous,
        "counts": dict(counts),
        "examples": {k: v for k, v in examples.items()},
    }


@click.command()
@click.option(
    "--input", "inputs", multiple=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Parsed-data JSONL(s) to scan. Defaults to bo1.jsonl + bo3.jsonl.",
)
@click.option("--min-game-turns", type=int, default=DEFAULT_MIN_GAME_TURNS, show_default=True,
              help="Mirror the synthesis fragment filter so we validate exactly the "
                   "games that become training data. Set 0 to scan everything.")
@click.option("--examples", "n_examples", type=int, default=5, show_default=True,
              help="Max example violations to print per type.")
def cli(inputs: tuple[Path, ...], min_game_turns: int, n_examples: int) -> None:
    """Scan the corpus for labels that contradict their own state's constraints."""
    paths = list(inputs) if inputs else [p for p in DEFAULT_INPUTS if p.exists()]
    if not paths:
        raise click.ClickException("no input files found (pass --input or run replay_parser.py first)")

    grand_counts: Counter[str] = Counter()
    total_turns = 0
    total_ambig = 0
    total_matches = 0

    for path in paths:
        click.echo(f"\n=== scanning {path.name} ===")
        res = _scan_file(path, min_game_turns)
        total_turns += res["turns_scanned"]
        total_ambig += res["turns_ambiguous"]
        total_matches += res["matches"]
        click.echo(
            f"  matches={res['matches']}  turns_scanned={res['turns_scanned']}  "
            f"turns_ambiguous(skipped)={res['turns_ambiguous']}"
        )
        if not res["counts"]:
            click.echo("  ✓ no action-legality violations found")
        for vtype, c in sorted(res["counts"].items(), key=lambda kv: -kv[1]):
            grand_counts[vtype] += c
            rate = c / res["turns_scanned"] if res["turns_scanned"] else 0.0
            click.echo(f"  ✗ {vtype}: {c}  ({rate*100:.4f}% of scanned turns)")
            for ex in res["examples"][vtype][:n_examples]:
                click.echo(f"        {ex}")

    click.echo("\n=== TOTAL ===")
    click.echo(f"  matches={total_matches}  turns_scanned={total_turns}  ambiguous_skipped={total_ambig}")
    if not grand_counts:
        click.echo("  ✓ corpus clean — every label is consistent with its rendered constraints")
    for vtype, c in sorted(grand_counts.items(), key=lambda kv: -kv[1]):
        rate = c / total_turns if total_turns else 0.0
        click.echo(f"  ✗ {vtype}: {c}  ({rate*100:.4f}% of scanned turns)")

    click.echo(
        "\nKNOWN GAPS (not scanned — parser carries no move name / trapped flag):\n"
        "  • Encore-lock: cannot verify the label is the encored move.\n"
        "  • Disable: cannot verify the label avoids the disabled move.\n"
        "  • Trapping (Shadow Tag / Arena Trap / Magnet Pull / partial-trap):\n"
        "    cannot verify a switch was legal — no `trapped` flag is captured.\n"
        "  Fix = parser enhancement (encored/disabled move id + trapped flag).\n"
        "  Until then the action MASK is implicit; these classes rely on the\n"
        "  model reading the ledger's volatile lines rather than a hard mask."
    )


if __name__ == "__main__":
    cli()
