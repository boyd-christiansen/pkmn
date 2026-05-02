"""CLI entrypoint: snapshot ladders, enumerate per-user replays, download them."""
from __future__ import annotations

import asyncio
import json
from datetime import date
from pathlib import Path

import click
import httpx
from tqdm.asyncio import tqdm

from ps_client import PSError, fetch_ladder, fetch_replay, list_user_replays

DEFAULT_FORMATS = ["gen9vgc2026regi", "gen9vgc2026regibo3"]


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, separators=(",", ":")))
    tmp.replace(path)


def _append_failure(out_dir: Path, format_id: str, kind: str, key: str, error: str) -> None:
    path = out_dir / f"failures_{format_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps({"kind": kind, "key": key, "error": error}) + "\n")


async def _stage_ladder(
    client: httpx.AsyncClient, format_id: str, out_dir: Path, top_n: int
) -> list[dict]:
    snapshot = out_dir / "ladders" / f"{format_id}_{date.today().isoformat()}.json"
    if snapshot.exists():
        toplist = json.loads(snapshot.read_text())
    else:
        toplist = await fetch_ladder(client, format_id)
        _write_json(snapshot, toplist)
    return toplist[:top_n]


async def _stage_users(
    client: httpx.AsyncClient,
    format_id: str,
    users: list[dict],
    out_dir: Path,
    sem: asyncio.Semaphore,
    refresh: bool,
) -> list[dict]:
    users_dir = out_dir / "users" / format_id

    async def one(user: dict) -> list[dict]:
        userid = user["userid"]
        cache = users_dir / f"{userid}.json"
        if cache.exists() and not refresh:
            return json.loads(cache.read_text())
        async with sem:
            try:
                replays = await list_user_replays(client, userid, format_id)
            except PSError as e:
                _append_failure(out_dir, format_id, "user", userid, str(e))
                return []
        _write_json(cache, replays)
        return replays

    results = await tqdm.gather(
        *(one(u) for u in users), desc=f"[{format_id}] users", unit="user"
    )
    return [r for batch in results for r in batch]


async def _stage_replays(
    client: httpx.AsyncClient,
    format_id: str,
    replay_meta: list[dict],
    out_dir: Path,
    sem: asyncio.Semaphore,
    refetch: bool,
) -> tuple[int, int, int]:
    replays_dir = out_dir / "replays" / format_id
    replays_dir.mkdir(parents=True, exist_ok=True)

    seen: dict[str, dict] = {}
    private_count = 0
    for meta in replay_meta:
        if meta.get("private"):
            private_count += 1
            continue
        seen.setdefault(meta["id"], meta)

    todo = []
    skipped = 0
    for replay_id in seen:
        target = replays_dir / f"{replay_id}.json"
        if target.exists() and not refetch:
            skipped += 1
            continue
        todo.append((replay_id, target))

    async def one(replay_id: str, target: Path) -> bool:
        async with sem:
            try:
                data = await fetch_replay(client, replay_id)
            except PSError as e:
                _append_failure(out_dir, format_id, "replay", replay_id, str(e))
                return False
        _write_json(target, data)
        return True

    if todo:
        results = await tqdm.gather(
            *(one(rid, t) for rid, t in todo),
            desc=f"[{format_id}] replays",
            unit="replay",
        )
        downloaded = sum(1 for r in results if r)
    else:
        downloaded = 0

    return downloaded, skipped, private_count


async def run(
    formats: list[str],
    top_n: int,
    concurrency: int,
    out_dir: Path,
    refresh_users: bool,
    refetch: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    timeout = httpx.Timeout(30.0, connect=10.0)
    limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency)
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(
        timeout=timeout, limits=limits, http2=False, headers={"User-Agent": "pkmn-scraper/0.1"}
    ) as client:
        for format_id in formats:
            click.echo(f"\n=== {format_id} ===")
            users = await _stage_ladder(client, format_id, out_dir, top_n)
            click.echo(f"  ladder: {len(users)} users (top {top_n})")

            replay_meta = await _stage_users(
                client, format_id, users, out_dir, sem, refresh=refresh_users
            )
            click.echo(f"  enumerated {len(replay_meta)} replay refs (with dupes)")

            downloaded, skipped, private = await _stage_replays(
                client, format_id, replay_meta, out_dir, sem, refetch=refetch
            )
            click.echo(
                f"  downloaded={downloaded} skipped_existing={skipped} private_skipped={private}"
            )


@click.command()
@click.option("--top-n", default=500, show_default=True, help="Users to crawl per ladder.")
@click.option("--concurrency", default=8, show_default=True, help="Max concurrent HTTP requests.")
@click.option(
    "--output-dir",
    default="data",
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option(
    "--formats",
    default=",".join(DEFAULT_FORMATS),
    show_default=True,
    help="Comma-separated format IDs.",
)
@click.option(
    "--refresh-users", is_flag=True, help="Re-fetch each user's replay list even if cached."
)
@click.option("--refetch", is_flag=True, help="Re-download replays even if already on disk.")
def cli(
    top_n: int,
    concurrency: int,
    output_dir: Path,
    formats: str,
    refresh_users: bool,
    refetch: bool,
) -> None:
    fmt_list = [f.strip() for f in formats.split(",") if f.strip()]
    asyncio.run(run(fmt_list, top_n, concurrency, output_dir, refresh_users, refetch))


if __name__ == "__main__":
    cli()
