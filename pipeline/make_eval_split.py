"""Match-level, length-stratified train/holdout split for the SFT pilot + eval.

Why this exists:
    The honest test of whether the SFT corpus teaches anything is action-match
    of a *trained* model on turns it never saw. That requires a holdout the
    synthesis run must NOT touch. Two correctness requirements:

      1. **Split by match, never by row.** Turns from the same game share
         board state, team, and outcome — row-level splitting leaks the test
         set into training and inflates the eval. We hold out whole matches.

      2. **Stratify by game length + format.** A holdout that's all short
         games would only measure opening play; a good eval spans short,
         medium, and long games in both Bo1 (CTS) and Bo3 (OTS). We bucket
         matches into length terciles within each format and sample the
         holdout proportionally, then report the per-GAME turn distribution
         so you can eyeball the coverage.

    (Refinement noted, not done here: also de-duping by player/team so the
    model never sees a given team in both splits. Match-level is the standard
    minimum; team-level is a stronger guarantee if opening-memorization shows
    up in the eval.)

Output:
    `parsed_data/eval_split.json` — {"holdout": [...], "train": [...], "meta": {...}}.
    Deterministic (fixed seed) so re-running reproduces the same split.

Isolation contract:
    Read-only over `parsed_data/*.jsonl` + `action_extraction` (same fragment
    filter the synthesis uses, so row counts match). No calc, no LLM, no net.
"""
from __future__ import annotations

import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import click

from action_extraction import DEFAULT_MIN_GAME_TURNS, filter_fragment_games

PIPELINE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUTS = [
    PIPELINE_DIR / "parsed_data" / "bo1.jsonl",
    PIPELINE_DIR / "parsed_data" / "bo3.jsonl",
]
DEFAULT_OUTPUT = PIPELINE_DIR / "parsed_data" / "eval_split.json"
SEED = 42


def _match_profile(rec: dict, min_game_turns: int) -> dict | None:
    """Per-match stats used for stratification. None if the match yields no rows."""
    games, _ = filter_fragment_games(rec.get("games") or [], min_game_turns)
    if not games:
        return None
    game_lens = [max(0, len(g.get("snapshots") or []) - 1) for g in games]  # turn-pairs/game
    total_rows = sum(game_lens)
    if total_rows <= 0:
        return None
    return {
        "match_id": rec.get("match_id"),
        "format": rec.get("format", "bo1"),
        "n_games": len(games),
        "total_rows": total_rows,
        "max_game_len": max(game_lens),
        "game_lens": game_lens,
    }


def _length_bucket(total_rows: int, edges: tuple[float, float]) -> str:
    lo, hi = edges
    if total_rows <= lo:
        return "short"
    if total_rows <= hi:
        return "medium"
    return "long"


@click.command()
@click.option("--input", "inputs", multiple=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Parsed-data JSONL(s). Defaults to bo1.jsonl + bo3.jsonl.")
@click.option("--output", "output_path", type=click.Path(dir_okay=False, path_type=Path),
              default=str(DEFAULT_OUTPUT), show_default=True)
@click.option("--holdout-matches", type=int, default=500, show_default=True,
              help="Target number of matches to hold out (stratified across format × length tercile).")
@click.option("--min-game-turns", type=int, default=DEFAULT_MIN_GAME_TURNS, show_default=True,
              help="Mirror the synthesis fragment filter so row counts line up.")
def cli(inputs, output_path, holdout_matches, min_game_turns):
    """Produce a match-level, length+format-stratified train/holdout split."""
    paths = list(inputs) if inputs else [p for p in DEFAULT_INPUTS if p.exists()]
    if not paths:
        raise click.ClickException("no inputs found (run replay_parser.py first)")

    profiles: list[dict] = []
    for path in paths:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prof = _match_profile(rec, min_game_turns)
                if prof and prof["match_id"]:
                    profiles.append(prof)

    total_matches = len(profiles)
    total_rows = sum(p["total_rows"] for p in profiles)
    click.echo(f"loaded {total_matches} non-fragment matches → ~{total_rows} SFT rows")

    # Length terciles computed PER FORMAT (Bo1 and Bo3 have different length
    # distributions), so "short/medium/long" is meaningful within each.
    by_format: dict[str, list[dict]] = defaultdict(list)
    for p in profiles:
        by_format[p["format"]].append(p)

    rng = random.Random(SEED)
    holdout: list[dict] = []
    strata_report: dict[str, dict[str, int]] = {}

    for fmt, plist in by_format.items():
        rows_sorted = sorted(p["total_rows"] for p in plist)
        t1 = rows_sorted[len(rows_sorted) // 3]
        t2 = rows_sorted[2 * len(rows_sorted) // 3]
        edges = (t1, t2)
        buckets: dict[str, list[dict]] = defaultdict(list)
        for p in plist:
            buckets[_length_bucket(p["total_rows"], edges)].append(p)
        # Proportional allocation of this format's share of the holdout target.
        fmt_share = holdout_matches * (len(plist) / total_matches)
        strata_report[fmt] = {}
        for bucket, bplist in buckets.items():
            take = round(fmt_share * (len(bplist) / len(plist)))
            take = min(take, len(bplist))
            chosen = rng.sample(bplist, take) if take else []
            holdout.extend(chosen)
            strata_report[fmt][bucket] = len(chosen)

    holdout_ids = {p["match_id"] for p in holdout}
    train = [p for p in profiles if p["match_id"] not in holdout_ids]

    # Reporting: confirm the holdout spans game lengths (per-GAME turn counts).
    holdout_game_lens = [gl for p in holdout for gl in p["game_lens"]]
    holdout_rows = sum(p["total_rows"] for p in holdout)

    def _hist(vals):
        c = Counter(min(v, 15) for v in vals)
        return {k: c[k] for k in sorted(c)}

    click.echo(f"\nholdout: {len(holdout)} matches  (~{holdout_rows} rows, "
               f"{100*holdout_rows/total_rows:.1f}% of corpus)")
    click.echo(f"train:   {len(train)} matches  (~{total_rows-holdout_rows} rows)")
    click.echo(f"holdout strata (format × length tercile): {strata_report}")
    if holdout_game_lens:
        click.echo(f"holdout per-GAME length: min={min(holdout_game_lens)} "
                   f"median={int(statistics.median(holdout_game_lens))} "
                   f"max={max(holdout_game_lens)}")
        click.echo(f"holdout game-length histogram (turn-pairs, capped 15+): "
                   f"{_hist(holdout_game_lens)}")

    out = {
        "meta": {
            "seed": SEED,
            "min_game_turns": min_game_turns,
            "total_matches": total_matches,
            "total_rows_est": total_rows,
            "holdout_matches": len(holdout),
            "holdout_rows_est": holdout_rows,
            "strata": strata_report,
        },
        "holdout": sorted(holdout_ids),
        "train": sorted(p["match_id"] for p in train),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(out, f, indent=2)
    click.echo(f"\nwrote {output_path}")


if __name__ == "__main__":
    cli()
