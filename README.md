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
| [`calc_microservice/`](calc_microservice/) | Node 20+ / TS | HTTP service wrapping `@smogon/calc` (`POST /calc` with `isCrit` support), `@pkmn/client` (`POST /parse_log` returning per-turn snapshots + `actionLog`), and `@pkmn/dex` (`GET /dex/move/:name`). |
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

### 4. Bootstrap canonical priors *(one-off, ~10 seconds per format)*

Before generating training data, populate the local Smogon Chaos cache so
the threat matrix's Probable track uses real ladder usage data:

```bash
cd pipeline
.venv/bin/python canonical_priors.py --format-id gen9vgc2026regi
.venv/bin/python canonical_priors.py --format-id gen9vgc2026regibo3
```

Walks back from the current month until a 200 OK chaos file is found; saves
to `pipeline/data/smogon_chaos_<format_id>.json`. Run again whenever you want
fresher data. Without this, `canonical_priors` falls back to a curated table
+ heuristic (still works, just less accurate for off-meta species).

### 5. Generate training data — *pending* (orchestrator only)

For each turn in each match, the orchestrator (`master_pipeline.py`, still a
stub) will chain four working library modules:

1. **`damage_inferencer.update_knowledge(snap_pre, snap_post, snap_pre.actionLog, p1_k, p2_k)`**
   — two-way binary search tightens both `KnowledgeState`s atomically, with
   the 508-EV total constraint applied per Pokémon after.
2. **`canonical_priors.get_probable_spread(species, format_id)`** — pure-data
   lookup, hits Smogon chaos data when bootstrapped.
3. **`threat_matrix.generate_threat_matrix(snap, p1_side, p1_k, p2_k, format_id=…)`**
   — dual-track text block: Absolute envelope (from `KnowledgeState`s) +
   Probable envelope (from canonical priors), with `[PRIOR CONTRADICTED]`
   flags.
4. **`teacher_llm.generate(...)`** *(stub)* — drive a frontier model through
   a tool-calling CoT loop toward the human's known play.

All upstream blocking sub-tasks (per-turn `actionLog` production, `isCrit`
threading, EV-budget constraint, real Smogon priors) are now implemented;
only the orchestrator + teacher LLM remain.

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
| `pipeline/replay_parser.py` | Working. ETL CLI; captures per-turn `actionLog` from `/parse_log` straight into `parsed_data/{bo1,bo3}.jsonl`. |
| `pipeline/canonical_priors.py` | Working. Library + bootstrap CLI. Real Smogon Chaos JSON when cached on disk; curated table + heuristic fallback. |
| `pipeline/damage_inferencer.py` | Working. Dual-state, two-way binary search, atomic application, 508-EV constraint pass, crit-aware (via `/calc isCrit`), multi-hit filter. |
| `pipeline/threat_matrix.py` | Working. Dual-track Absolute + Probable output with `[PRIOR CONTRADICTED]` flag. Takes optional `format_id` to drive chaos-backed priors. |
| `pipeline/teacher_llm.py` | Stub. |
| `pipeline/master_pipeline.py` | Stub. |

See each subdirectory's README for setup, contracts, and design notes.
