# pkmn

Building the first generalist LLM that plays competitive Pokémon VGC (Gen 9 Reg I,
2v2 doubles) at a professional level. The model relies on chain-of-thought
reasoning, active tool-calling for damage math, and a multi-stage alignment
pipeline (SFT + RLHF) grounded in human expert intuition.

## Repository layout

This is a polyrepo-in-a-folder. Each subdirectory is an atomic component with
its own runtime, deps, and contract — designed so any one can be swapped
without touching the others.

| Directory | Runtime | What it does |
|---|---|---|
| [`data_scraper/`](data_scraper/) | Python 3.11+ | Pulls top-500 ladder users + all their saved replays from Pokémon Showdown. |
| [`calc_microservice/`](calc_microservice/) | Node 20+ / TS | HTTP service wrapping `@smogon/calc` (damage math, `POST /calc`) and `@pkmn/client` (Showdown log → turn snapshots, `POST /parse_log`). |
| [`pipeline/`](pipeline/) | Python 3.11+ | Atomic modules that turn raw replays into SFT-ready conversational training data. `replay_parser.py` is implemented; downstream modules still stubbed. |
| [`notes/`](notes/) | — | Free-form planning notes (data sourcing options, scope decisions, etc). |

## Pipeline overview

```
Pokémon Showdown
      │
      ▼
┌──────────────┐
│ data_scraper │  →  raw replay JSONs on disk
└──────────────┘
      │
      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                              pipeline/                                   │
│                                                                          │
│  replay_parser.py  ─── HTTP /parse_log ─▶  calc_microservice             │
│         │                                                                │
│         ▼  per-turn BoardState[]                                         │
│  threat_matrix.py  ─── HTTP /calc ──────▶  calc_microservice             │
│         │                                                                │
│         ▼                                                                │
│  teacher_llm.py    ─── HTTP ────────────▶  frontier model (e.g. GPT-4o)  │
│         │                                                                │
│         ▼                                                                │
│  master_pipeline.py  →  conversational SFT .jsonl                        │
└──────────────────────────────────────────────────────────────────────────┘
```

Each pipeline module is independently importable and testable. The orchestrator
(`master_pipeline.py`) is the only file allowed to import from all of them.

## End-to-end execution flow

How to run this repo from a clean clone, in order. Each step assumes the
previous step has completed at least once. Every component has its own
`README.md` with full reference docs.

### 1. Data collection — pull raw replays

Pulls top-500 ladder users from each format and downloads every saved replay
they have. Output: ~16K replay JSONs on disk. Idempotent and resumable.

```bash
cd data_scraper
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python scrape.py        # ~30–60 min for the full crawl
```

→ writes to `data_scraper/data/replays/{format_id}/{replay_id}.json`

### 2. Start the mechanics engine

The Node service wraps `@smogon/calc` and `@pkmn/client`. Both `replay_parser`
(below) and the future `threat_matrix` need it running on port 3000.

```bash
cd calc_microservice
npm install
npm run dev                       # listens on http://localhost:3000
```

Verify with `curl http://localhost:3000/health` → `{"status":"ok"}`.

Leave this process running for steps 3 and 4.

### 3. Parse & stitch states

Reads the raw replays from step 1, calls `/parse_log` for each, stitches Bo3
games into series (matched by sorted player pair + chronological order, with a
30-minute gap heuristic and a 3-game ceiling), and writes one JSONL row per
match. Resumable — skips matches already in the output.

```bash
cd pipeline
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python replay_parser.py --limit 10 --format bo3   # quick smoke test first
.venv/bin/python replay_parser.py                            # full run
```

→ writes to `pipeline/parsed_data/{bo1,bo3}.jsonl`

### 4. Generate training data — *pending*

The next stage will turn each turn snapshot into one SFT example: assemble the
threat matrix (queries `/calc` for every plausible attack), drive the teacher
LLM through a tool-calling CoT loop toward the human's known play, and emit a
conversational JSONL row.

```bash
# (not implemented yet)
.venv/bin/python master_pipeline.py
```

→ will write to `pipeline/parsed_data/sft.jsonl`

## Status

| Component | State |
|---|---|
| `data_scraper` | Working. 16,537 replays cached locally across both Reg I formats. |
| `calc_microservice` | Working. `POST /calc` for damage math; `POST /parse_log` returns turn-by-turn snapshots from raw Showdown logs. |
| `pipeline/replay_parser.py` | Working. ETL CLI: walks scraper output, stitches Bo3 series, POSTs each `log` to `/parse_log`, writes one match per line to `parsed_data/{bo1,bo3}.jsonl`. Resumable. |
| `pipeline/threat_matrix.py` | Stub. |
| `pipeline/teacher_llm.py` | Stub. |
| `pipeline/master_pipeline.py` | Stub. |

See each subdirectory's README for setup, contracts, and design notes.
