"""Async HTTP helpers for Pokemon Showdown ladder + replay endpoints."""
from __future__ import annotations

import asyncio
import random
from typing import Any

import httpx

LADDER_URL = "https://pokemonshowdown.com/ladder/{format_id}.json"
SEARCH_URL = "https://replay.pokemonshowdown.com/search.json"
REPLAY_URL = "https://replay.pokemonshowdown.com/{replay_id}.json"

PAGE_SIZE = 50
MAX_ATTEMPTS = 3
BASE_BACKOFF = 1.5


class PSError(Exception):
    """Raised when a PS request fails permanently after retries."""


async def _request(client: httpx.AsyncClient, url: str, *, params: dict | None = None) -> Any:
    last_exc: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            r = await client.get(url, params=params)
            if r.status_code == 429 or r.status_code >= 500:
                retry_after = r.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else BASE_BACKOFF * (2 ** (attempt - 1))
                delay += random.uniform(0, 0.5)
                await asyncio.sleep(delay)
                continue
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, ValueError) as e:
            last_exc = e
            if attempt == MAX_ATTEMPTS:
                break
            await asyncio.sleep(BASE_BACKOFF * (2 ** (attempt - 1)) + random.uniform(0, 0.5))
    raise PSError(f"Failed {url} after {MAX_ATTEMPTS} attempts: {last_exc}")


async def fetch_ladder(client: httpx.AsyncClient, format_id: str) -> list[dict]:
    """Return the ladder's `toplist` (sorted by Elo desc)."""
    data = await _request(client, LADDER_URL.format(format_id=format_id))
    return data.get("toplist", [])


async def list_user_replays(
    client: httpx.AsyncClient, userid: str, format_id: str
) -> list[dict]:
    """Paginate /search.json until the response is short or empty."""
    all_replays: list[dict] = []
    page = 1
    while True:
        params = {"user": userid, "format": format_id, "page": page}
        batch = await _request(client, SEARCH_URL, params=params)
        if not isinstance(batch, list) or not batch:
            break
        all_replays.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
        page += 1
    return all_replays


async def fetch_replay(client: httpx.AsyncClient, replay_id: str) -> dict:
    return await _request(client, REPLAY_URL.format(replay_id=replay_id))
