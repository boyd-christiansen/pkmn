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
| [`calc_microservice/`](calc_microservice/) | Node 20+ / TS | HTTP service wrapping `@smogon/calc` (`POST /calc`), `@pkmn/client` + `@pkmn/sets` (`POST /parse_log` — per-turn snapshots, `events` stream, and Bo3 OTS `teamSheets`), and `@pkmn/dex` (`GET /dex/move/:name`). |
| [`pipeline/`](pipeline/) | Python 3.11+ | Atomic modules that turn raw replays into SFT-ready conversational training data. Inference modules (`replay_parser`, `canonical_priors`, `damage_inferencer`, `threat_matrix`); orchestration helpers split out of the orchestrator (`team_reconstruction`, `action_extraction`, `prompt_formatting`); the orchestrator + CLI (`master_pipeline`); a `teacher/` sub-package holding the `TeacherProvider` ABC plus OpenAI / Anthropic / Google adapters; and a head-to-head `bakeoff` runner. |
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
│         │    Probable envelope from canonical_priors;                    │
│         │    when the prior is ≥40-EV-clipped, drop the meta             │
│         │    column and tag the line `(off-meta)`)                       │
│         ▼                                                                │
│  teacher/             ── HTTP ────────────▶  frontier model              │
│   (base + openai/anthropic/google adapters behind one ABC)               │
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

### 5. Generate the SFT training dataset

For each turn in each match, the orchestrator chains the library modules:

1. **`flip_match_to_winner(match)`** — relabel sides so the **series
   winner** is always P1 for the rest of the pipeline. Bo3: 2 of 3
   games. Bo1: per-game winner.
2. **`master_pipeline.extract_p1_actions(snap_pre, snap_post, events)`** —
   reverse-engineer P1's two-slot decision from the `events` stream
   (`move` / `switch` / `cant_move` events; `forced_by` filtering for
   intentional vs forced switches). Skip the turn if ambiguous.
3. **`threat_matrix.generate_threat_matrix(snap, "p1", K1, K2, format_id=…)`**
   — dual-track text block: Absolute envelope (from `KnowledgeState`s) +
   Probable envelope (from canonical priors). When the prior is
   ≥40-EV-clipped, the meta column is dropped and the line tagged
   `(off-meta)`.
4. **`format_user_prompt(...)`** — composes the full user prompt:
   board state, GAME-STATE LEDGER (faints / Tera-used / Cumulative
   damage / volatiles / choice locks), TURN-BY-TURN (full prior
   turns of this game), SERIES STATE (Bo3 game ≥ 2; full inlined
   prior-game rollups), YOUR SPREADS (one-sided EV constraints), and
   the threat matrix.
5. **`teacher.synthesize_turn(system, user, human_action)`** —
   provider-agnostic tool-use loop (`TeacherProvider` ABC, with
   OpenAI / Anthropic / Google adapters). The model **must** call
   `calculate_damage` at least once before committing via the
   `submit_decision` tool. The LLM sees the human's play as ground
   truth and writes a Chain-of-Thought that justifies it. Returns
   OpenAI-fine-tuning-format conversation messages (ground-truth
   stripped before save).
6. **`events_to_damage_events(events)` →
   `damage_inferencer.update_knowledge(snap_pre, snap_post,
   damage_events, K1, K2)`** — filter the events stream for
   inference-eligible damage hits (excluding non-own callers like
   Metronome / Copycat / Mirror Move / etc.; Sleep Talk allowed),
   then two-way binary search tightens both `KnowledgeState`s with a
   508-EV constraint pass.

```bash
cd pipeline

# Smoke test (no API key needed, exercises everything except the LLM call):
.venv/bin/python master_pipeline.py --limit 1 --dry-run

# Real run on a single match (OpenAI default):
OPENAI_API_KEY=sk-... .venv/bin/python master_pipeline.py --limit 1

# Pick a different provider / model:
ANTHROPIC_API_KEY=sk-... .venv/bin/python master_pipeline.py \
    --provider anthropic --model claude-sonnet-4-6 --limit 1

# Head-to-head bake-off (runs all configured providers on the same match):
.venv/bin/python bakeoff.py --providers openai,anthropic,google --limit 1
```

→ writes one fine-tuning row per identifiable turn to
`pipeline/parsed_data/sft_training_data.jsonl`. Resumable on rerun
(keyed by `(match_id, game_index, turn)`).

## Status

| Component | State |
|---|---|
| `data_scraper` | Working. 16,537 replays cached locally across both Reg I formats. |
| `calc_microservice` | Working. Three endpoints: `POST /calc` (damage math), `POST /parse_log` (Showdown log → turn snapshots), `GET /dex/move/:name` (move metadata). |
| `pipeline/replay_parser.py` | Working. ETL CLI; captures per-turn `events` (TurnEvent[]) from `/parse_log` straight into `parsed_data/{bo1,bo3}.jsonl`. |
| `pipeline/canonical_priors.py` | Working. Library + bootstrap CLI. Real Smogon Chaos JSON when cached on disk; curated table + heuristic fallback. |
| `pipeline/damage_inferencer.py` | Working. Dual-state, two-way binary search, atomic application, 508-EV constraint pass, crit-aware (via `/calc isCrit`), multi-hit filter, `events_to_damage_events()` filter that excludes non-own-move callers (Metronome / Copycat / Sketch / Snatch / Me First / Dancer / Instruct / Mirror Move / Assist / Nature Power; Sleep Talk allowed). |
| `pipeline/threat_matrix.py` | Working. Dual-track Absolute + Probable output. When canonical priors are ≥40-EV-clipped by the inferred bounds, the Probable column is dropped and the line tagged `(off-meta)`. Optional `format_id` drives chaos-backed priors. |
| `pipeline/teacher/` (sub-package: `base.py` + `openai.py` / `anthropic.py` / `google.py`, re-exported via `__init__.py`) | Working. Provider-agnostic `TeacherProvider` ABC; tool-use loop with `calculate_damage` + `submit_decision`. The model has only one output channel — tool calls — so it can't bypass the calc tool. Two system-prompt templates: Bo1 CTS (Masking Rule + reconstructed team) vs Bo3 OTS (full sheets + ★ brought-flag). Present-tense framing throughout. |
| `pipeline/master_pipeline.py` | Working. CLI orchestrator. `flip_match_to_winner` makes every example come from the series winner's perspective. Three historical-context blocks in the user prompt (GAME-STATE LEDGER / TURN-BY-TURN / SERIES STATE). One-sided EV constraint rendering. Empty-slot annotation. Perspective-aware bench gating (P1: full brought-set, P2: chronological via `seenSpecies`). `--provider {openai,anthropic,google}` flag. Resumable. |
| `pipeline/bakeoff.py` | Working. Runs the same match through multiple providers in lockstep and reports per-row cost, tool-call rate, action-match rate, CoT length, wall-clock. |

See each subdirectory's README for setup, contracts, and design notes.
