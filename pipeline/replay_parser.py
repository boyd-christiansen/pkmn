"""ETL: raw scraper replays -> per-turn BoardState snapshots, with Bo3 stitching.

Pipeline role:
    First Python stage. Reads raw replay JSONs from data_scraper/, posts each
    `log` to calc_microservice's /parse_log endpoint, stitches Bo3 series, and
    emits one JSONL row per *match* (a Bo1 match is one game; a Bo3 match is
    one to three games played by the same two players within ~30 minutes).

Inputs (CLI defaults):
    Raw replays: ../data_scraper/data/replays/{format_id}/{replay_id}.json
    Parse service: http://localhost:3000/parse_log

Outputs:
    parsed_data/bo1.jsonl       — one match per line (single-game records)
    parsed_data/bo3.jsonl       — one match per line (1–3 game series)
    parsed_data/failures.jsonl  — append-only error log

JSONL row shape:
    {
      "match_id": "bo1-gen9vgc2026regi-2566725666",
      "players":  ["Alice", "Bob"],
      "format":   "bo1" | "bo3",
      "games": [
        { "replay_id": "...", "timestamp": 1738000000, "snapshots": [...] }
      ]
    }

Isolation contract:
    Talks only to calc_microservice's /parse_log. No regex parsing of logs.
    No LLM. No calc. Other modules consume the JSONL output.
"""
from __future__ import annotations

import asyncio
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
import click
from tqdm.asyncio import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCRAPER_DIR = REPO_ROOT / "data_scraper" / "data" / "replays"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "parsed_data"
DEFAULT_PARSE_URL = "http://localhost:3000/parse_log"

BO1_FORMAT_ID = "gen9vgc2026regi"
BO3_FORMAT_ID = "gen9vgc2026regibo3"


@dataclass
class GameMeta:
    replay_id: str
    file_path: Path
    players: tuple[str, str]
    timestamp: int
    formatid: str


@dataclass
class Match:
    match_id: str
    format_label: str  # "bo1" or "bo3"
    players: tuple[str, str]
    games: list[GameMeta]


def _load_meta(path: Path) -> GameMeta | None:
    try:
        data = json.loads(path.read_text())
        players = data["players"]
        if not isinstance(players, list) or len(players) < 2:
            return None
        return GameMeta(
            replay_id=data["id"],
            file_path=path,
            players=(str(players[0]), str(players[1])),
            timestamp=int(data["uploadtime"]),
            formatid=data["formatid"],
        )
    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _finalize_bo3(games: list[GameMeta]) -> Match:
    games.sort(key=lambda g: g.timestamp)
    first = games[0]
    return Match(
        match_id=f"bo3-{first.replay_id}",
        format_label="bo3",
        players=first.players,
        games=games,
    )


def build_matches(scraper_dir: Path, bo3_gap_seconds: int) -> list[Match]:
    """Walk the scraper output, build the list of matches (post-stitching)."""
    matches: list[Match] = []

    bo1_dir = scraper_dir / BO1_FORMAT_ID
    if bo1_dir.exists():
        for path in sorted(bo1_dir.glob("*.json")):
            meta = _load_meta(path)
            if meta is None:
                continue
            matches.append(
                Match(
                    match_id=f"bo1-{meta.replay_id}",
                    format_label="bo1",
                    players=meta.players,
                    games=[meta],
                )
            )

    bo3_dir = scraper_dir / BO3_FORMAT_ID
    if bo3_dir.exists():
        bo3_metas: list[GameMeta] = []
        for path in bo3_dir.glob("*.json"):
            meta = _load_meta(path)
            if meta is not None:
                bo3_metas.append(meta)

        grouped: dict[tuple[str, str], list[GameMeta]] = defaultdict(list)
        for m in bo3_metas:
            key = tuple(sorted((m.players[0].lower(), m.players[1].lower())))
            grouped[key].append(m)

        for games in grouped.values():
            games.sort(key=lambda g: g.timestamp)
            current: list[GameMeta] = []
            for g in games:
                # Split when the gap exceeds the threshold OR we already have a
                # full Bo3 (3 games is the hard ceiling — back-to-back matches
                # between the same players otherwise merge into one giant "series").
                if current and (
                    g.timestamp - current[-1].timestamp > bo3_gap_seconds
                    or len(current) >= 3
                ):
                    matches.append(_finalize_bo3(current))
                    current = []
                current.append(g)
            if current:
                matches.append(_finalize_bo3(current))

    return matches


def load_seen_match_ids(output_dir: Path) -> set[str]:
    seen: set[str] = set()
    for fname in ("bo1.jsonl", "bo3.jsonl"):
        path = output_dir / fname
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                mid = rec.get("match_id")
                if mid:
                    seen.add(mid)
    return seen


async def _post_log(
    session: aiohttp.ClientSession,
    parse_url: str,
    log: str,
) -> dict[str, Any]:
    """Return the full /parse_log response: { snapshots, teamSheets }."""
    async with session.post(parse_url, json={"log": log}) as r:
        if r.status >= 400:
            text = await r.text()
            raise RuntimeError(f"/parse_log {r.status}: {text[:200]}")
        return await r.json()


async def process_match(
    match: Match,
    session: aiohttp.ClientSession,
    parse_url: str,
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    games_out: list[dict[str, Any]] = []
    for g in match.games:
        log = json.loads(g.file_path.read_text())["log"]
        async with sem:
            result = await _post_log(session, parse_url, log)
        games_out.append(
            {
                "replay_id": g.replay_id,
                "timestamp": g.timestamp,
                "snapshots": result.get("snapshots", []),
                "teamSheets": result.get("teamSheets"),  # null in CTS
            }
        )
    return {
        "match_id": match.match_id,
        "players": list(match.players),
        "format": match.format_label,
        "games": games_out,
    }


async def _check_health(session: aiohttp.ClientSession, parse_url: str) -> None:
    health_url = parse_url.rsplit("/", 1)[0] + "/health"
    try:
        async with session.get(health_url) as r:
            if r.status != 200:
                raise RuntimeError(f"health check returned {r.status}")
    except Exception as e:
        raise click.ClickException(
            f"calc_microservice not reachable at {health_url}: {e}\n"
            f"  Start it with:  cd calc_microservice && npm run dev"
        )


async def run(
    scraper_dir: Path,
    output_dir: Path,
    parse_url: str,
    concurrency: int,
    bo3_gap_seconds: int,
    limit: int | None,
    only_format: str | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    timeout = aiohttp.ClientTimeout(total=180, connect=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        await _check_health(session, parse_url)

        click.echo("Building match index...")
        matches = build_matches(scraper_dir, bo3_gap_seconds)
        bo1_count = sum(1 for m in matches if m.format_label == "bo1")
        bo3_count = len(matches) - bo1_count
        click.echo(f"  {len(matches)} matches discovered ({bo1_count} bo1, {bo3_count} bo3)")
        if bo3_count:
            sizes = [len(m.games) for m in matches if m.format_label == "bo3"]
            click.echo(
                f"  bo3 series sizes: 1g={sizes.count(1)} 2g={sizes.count(2)} 3g={sizes.count(3)}"
            )

        if only_format:
            matches = [m for m in matches if m.format_label == only_format]

        seen = load_seen_match_ids(output_dir)
        todo = [m for m in matches if m.match_id not in seen]
        click.echo(f"  {len(todo)} todo ({len(seen)} already in output)")

        if limit is not None:
            todo = todo[:limit]
            click.echo(f"  --limit {limit}: processing first {len(todo)}")

        if not todo:
            return

        sem = asyncio.Semaphore(concurrency)
        write_lock = asyncio.Lock()
        failure_path = output_dir / "failures.jsonl"

        async def worker(match: Match) -> None:
            try:
                record = await process_match(match, session, parse_url, sem)
                out_path = output_dir / f"{match.format_label}.jsonl"
            except Exception as e:
                record = {"match_id": match.match_id, "error": str(e)}
                out_path = failure_path

            async with write_lock:
                with out_path.open("a") as f:
                    f.write(json.dumps(record, separators=(",", ":")) + "\n")

        await tqdm.gather(*(worker(m) for m in todo), desc="parsing matches", unit="match")


@click.command()
@click.option(
    "--scraper-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=str(DEFAULT_SCRAPER_DIR),
    show_default=True,
    help="Root directory containing per-format-id replay subdirectories.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=str(DEFAULT_OUTPUT_DIR),
    show_default=True,
)
@click.option("--parse-url", default=DEFAULT_PARSE_URL, show_default=True)
@click.option("--concurrency", default=8, show_default=True, help="Max in-flight /parse_log requests.")
@click.option(
    "--bo3-gap-minutes",
    default=30,
    show_default=True,
    help="Max gap (minutes) between consecutive games in the same Bo3 series.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Process only the first N matches (test batch mode).",
)
@click.option(
    "--format",
    "only_format",
    type=click.Choice(["bo1", "bo3"]),
    default=None,
    help="Restrict to a single format (test/dev convenience).",
)
def cli(
    scraper_dir: Path,
    output_dir: Path,
    parse_url: str,
    concurrency: int,
    bo3_gap_minutes: int,
    limit: int | None,
    only_format: str | None,
) -> None:
    asyncio.run(
        run(
            scraper_dir=scraper_dir,
            output_dir=output_dir,
            parse_url=parse_url,
            concurrency=concurrency,
            bo3_gap_seconds=bo3_gap_minutes * 60,
            limit=limit,
            only_format=only_format,
        )
    )


if __name__ == "__main__":
    cli()
