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
| [`pipeline/`](pipeline/) | Python 3.11+ | Atomic modules that turn raw replays into SFT-ready conversational training data. Stubs only at present. |
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

## Status

| Component | State |
|---|---|
| `data_scraper` | Working. 16,537 replays cached locally across both Reg I formats. |
| `calc_microservice` | Working. `POST /calc` for damage math; `POST /parse_log` returns turn-by-turn snapshots from raw Showdown logs. |
| `pipeline/replay_parser.py` | Stub. Next: thin Python wrapper that POSTs raw logs to `/parse_log` and shapes the response into `BoardState` objects (with Bo3 series stitching). |
| `pipeline/threat_matrix.py` | Stub. |
| `pipeline/teacher_llm.py` | Stub. |
| `pipeline/master_pipeline.py` | Stub. |

See each subdirectory's README for setup, contracts, and design notes.
