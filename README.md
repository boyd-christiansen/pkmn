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
| [`calc_microservice/`](calc_microservice/) | Node 20+ / TS | HTTP service wrapping `@smogon/calc` (`POST /calc`), `@pkmn/client` (`POST /parse_log`), and `@pkmn/dex` (`GET /dex/move/:name`). |
| [`pipeline/`](pipeline/) | Python 3.11+ | Atomic modules that turn raw replays into SFT-ready conversational training data. `replay_parser`, `canonical_priors`, `damage_inferencer`, and `threat_matrix` are implemented; `teacher_llm` and `master_pipeline` are still stubs. |
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
│  replay_parser.py     ── /parse_log ──────▶  calc_microservice           │
│         │                                                                │
│         ▼  per-turn snapshots (JSONL)                                    │
│                                                                          │
│  canonical_priors.py  (pure data lookup, no network)                     │
│         │                                                                │
│         ▼                                                                │
│  damage_inferencer.py ── /calc, /dex/move ▶  calc_microservice           │
│         ▲ │   (two-way binary search → tightens                          │
│         │ │    KnowledgeState (p1) AND KnowledgeState (p2) atomically)   │
│         │ ▼                                                              │
│  threat_matrix.py     ── /calc, /dex/move ▶  calc_microservice           │
│         │   (Absolute envelope from KnowledgeStates +                    │
│         │    Probable envelope from canonical_priors,                    │
│         │    flagged [PRIOR CONTRADICTED] when they disagree)            │
│         ▼                                                                │
│  teacher_llm.py       ── HTTP ────────────▶  frontier model              │
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

### 4. Generate training data — *pending* (orchestrator only)

For each turn in each match, the orchestrator (`master_pipeline.py`, still a
stub) will chain four working library modules:

1. **`damage_inferencer.update_knowledge(...)`** — feed the turn's damage
   events into a two-way binary search; tighten **both** `KnowledgeState`s
   atomically.
2. **`canonical_priors.get_probable_spread(...)`** — pure-data lookup of the
   meta spread per species.
3. **`threat_matrix.generate_threat_matrix(...)`** — dual-track text block:
   Absolute envelope (from KnowledgeStates) + Probable envelope (from
   canonical priors), with `[PRIOR CONTRADICTED]` flags.
4. **`teacher_llm.generate(...)`** *(stub)* — drive a frontier model through
   a tool-calling CoT loop toward the human's known play.

> **Blocking sub-task:** the inferencer needs per-turn `DamageEvent` records
> (attacker slot, defender slot, move, hp before/after, crit flag), which
> `/parse_log` doesn't yet emit. The next piece of work is extending the
> Node endpoint to surface protocol events and threading them through
> `replay_parser.py` into `parsed_data/{bo1,bo3}.jsonl`.

Output: one conversational JSONL row per turn.

```bash
# (not implemented yet)
.venv/bin/python master_pipeline.py
```

→ will write to `pipeline/parsed_data/sft.jsonl`

## Status

| Component | State |
|---|---|
| `data_scraper` | Working. 16,537 replays cached locally across both Reg I formats. |
| `calc_microservice` | Working. Three endpoints: `POST /calc` (damage math), `POST /parse_log` (Showdown log → turn snapshots), `GET /dex/move/:name` (move metadata). |
| `pipeline/replay_parser.py` | Working. ETL CLI: walks scraper output, stitches Bo3 series, POSTs each `log` to `/parse_log`, writes one match per line to `parsed_data/{bo1,bo3}.jsonl`. Resumable. |
| `pipeline/canonical_priors.py` | Working. Library: `get_probable_spread(species)` returns the meta spread (curated table for top ~40 species + base-stat heuristic for the rest). Mock — to be replaced by real Smogon usage data. |
| `pipeline/damage_inferencer.py` | Working. Library: dual-state `KnowledgeState` (p1 + p2), two-way binary search per damage event with atomic application. End-to-end blocked on `action_log` production (per-turn protocol events from `/parse_log`). |
| `pipeline/threat_matrix.py` | Working. Library: dual-track output (Absolute envelope from `KnowledgeState`s + Probable envelope from canonical priors), with `[PRIOR CONTRADICTED]` flag when they disagree. |
| `pipeline/teacher_llm.py` | Stub. |
| `pipeline/master_pipeline.py` | Stub. |

See each subdirectory's README for setup, contracts, and design notes.
