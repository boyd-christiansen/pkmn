# data_scraper

Async Python scraper that captures replay JSONs for the top-500 ladder players
on each of two Pokémon Showdown formats:

- `gen9vgc2026regi` — best-of-1
- `gen9vgc2026regibo3` — best-of-3

This is the upstream of the entire pipeline. Output here is the raw input to
[`pipeline/replay_parser.py`](../pipeline/replay_parser.py).

## Setup

```bash
cd data_scraper
python3 -m venv .venv
.venv/bin/pip install -e .
```

Requires Python ≥3.11. Deps: `httpx`, `tqdm`, `click`.

## Run

```bash
.venv/bin/python scrape.py                       # full crawl, both formats
.venv/bin/python scrape.py --top-n 50            # smaller crawl
.venv/bin/python scrape.py --formats gen9vgc2026regi   # single format
.venv/bin/python scrape.py --concurrency 16      # more parallelism
.venv/bin/python scrape.py --refetch             # re-download all replays
```

| Flag | Default | Notes |
|---|---|---|
| `--top-n` | `500` | Users crawled per ladder. |
| `--concurrency` | `8` | Max in-flight HTTP requests. |
| `--output-dir` | `data` | Where everything is written. |
| `--formats` | both | Comma-separated list. |
| `--refresh-users` | off | Re-fetch each user's replay listing. |
| `--refetch` | off | Re-download replays already on disk. |

The scraper is fully resumable: rerunning with the same args skips any replay
already on disk and reuses cached per-user listings. Failed requests are
written to `data/failures_{format_id}.jsonl` instead of crashing the run.

## Output layout

```
data/
├── ladders/
│   └── {format_id}_{YYYY-MM-DD}.json    # full top-500 snapshot, sorted by Elo desc
├── users/
│   └── {format_id}/
│       └── {userid}.json                # that user's replay metadata (paginated search results)
└── replays/
    └── {format_id}/
        └── {replay_id}.json             # full replay: metadata + pipe-delimited battle log
```

## Pokémon Showdown endpoints used

| Endpoint | Purpose |
|---|---|
| `https://pokemonshowdown.com/ladder/{format_id}.json` | Top-500 list, sorted by Elo desc. |
| `https://replay.pokemonshowdown.com/search.json?user=X&format=Y&page=N` | Paginated replay search for one user (50/page). |
| `https://replay.pokemonshowdown.com/{replay_id}.json` | Full replay JSON (metadata + log). |

There are no documented rate limits. The scraper still uses a shared
semaphore + exponential-backoff retry on 429 / 5xx.

## Current corpus

A complete top-500 crawl produced:

| Format | Unique replays | Disk | Top-500 users with ≥1 saved replay |
|---|---:|---:|---:|
| `gen9vgc2026regi` | 10,997 | 85 MB | 271 / 500 |
| `gen9vgc2026regibo3` | 5,540 | 49 MB | 151 / 500 |
| **Total** | **16,537** | **140 MB** | — |

A meaningful fraction of top-ladder players save no replays publicly (about
half the regi top-500, two-thirds of the bo3 top-500). The corpus is biased
toward replay-savers, not strictly toward the highest Elo. Worth keeping in
mind when reasoning about training-data distribution.

## Architecture

Two files:

- `ps_client.py` — async HTTP helpers (`fetch_ladder`, `list_user_replays`,
  `fetch_replay`) with shared retry/backoff. No file I/O.
- `scrape.py` — Click CLI + three-stage async pipeline (ladder → enumerate
  per-user → download replays). Owns concurrency limit, file I/O, and dedup.

The "stage 3" download phase deduplicates replay IDs across users, since both
opponents in a game appear in each other's replay lists.

## Other endpoints worth exploring

See [`../notes/for_the_future.md`](../notes/for_the_future.md) for further
data-sourcing options if the top-500 corpus turns out to be too small.
