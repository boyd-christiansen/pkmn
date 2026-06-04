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
| [`calc_microservice/`](calc_microservice/) | Node 20+ / TS | HTTP service wrapping `@smogon/calc` (`POST /calc`), `@pkmn/client` + `@pkmn/sets` (`POST /parse_log` — per-turn snapshots, `events` stream, and Bo3 OTS `teamSheets`), and `@pkmn/dex` (`GET /dex/move/:name`, `GET /dex/species/:name`). |
| [`pipeline/`](pipeline/) | Python 3.11+ | Atomic modules that turn raw replays into SFT-ready conversational training data. Inference modules (`replay_parser`, `damage_inferencer`, `threat_matrix`); a read-only corpus validator (`validate_action_legality`); orchestration helpers split out of the orchestrator (`team_reconstruction`, `action_extraction`, `prompt_formatting`); the orchestrator + CLI (`master_pipeline` with `--mode {sync,batch,hybrid}`); a [`teacher/`](pipeline/teacher/README.md) sub-package holding the `TeacherProvider` ABC plus OpenAI / Anthropic / Google adapters, a `judge.py` model-judge validator, and a `batch_openai.py` Batch API adapter; a `batch_runner.py` sibling implementing the per-cycle state machine for batch mode; and a head-to-head `bakeoff` runner (OpenAI won — see status below). |
| [`inspector/`](inspector/) | Python 3.11+ (FastAPI) | Local read-only web UI for browsing saved SFT rows + their source data. Splits prompts into structured sections, shows the calc-tool loop step-by-step, and cross-references by `(match_id, game_index, turn)` against `pipeline/parsed_data/`. Listens on port 8001 — doesn't talk to the calc service. |
| [`notes/`](notes/) | — | The project's writing surface. [`pipeline_walkthrough.md`](notes/pipeline_walkthrough.md) is the long-form design doc; [`TODO.md`](notes/TODO.md) is the master tracker for non-shipped work (active workstreams, code-level TODOs, long-horizon plans, data-sourcing options). |

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
│  damage_inferencer.py ── /calc, /dex/move, /dex/species ▶ calc_microsvc  │
│         ▲ │   (two-way binary search → tightens                          │
│         │ │    KnowledgeState (p1) AND KnowledgeState (p2) atomically;   │
│         │ │    `infer_match_final_bounds` runs across the full match     │
│         │ │    once to surface "match-final P1 bounds" in YOUR SPREADS;  │
│         │ │    `update_observed_and_speed` sets per-mon `observed` +     │
│         │ │    a move-order Speed upper-bound)                           │
│         │ ▼                                                              │
│  threat_matrix.py     ── /calc, /dex/move ▶  calc_microservice           │
│         │   (Absolute envelope only — strict provable range from both    │
│         │    sides' inferred KnowledgeState bounds; an unconstrained     │
│         │    stat renders the word `unknown`)                            │
│         ▼                                                                │
│  teacher/             ── HTTP ────────────▶  frontier model              │
│   (base + openai/anthropic/google adapters behind one ABC;               │
│    judge.py for match-level CoT validation;                              │
│    batch_openai.py for OpenAI Batch API submission)                      │
│         │                                                                │
│         ▼                                                                │
│  master_pipeline.py   (sync mode)                                        │
│   batch_runner.py     (batch / hybrid mode — per-iter state machine,     │
│                        per-match resume state in batch_state/*.json)     │
│         │                                                                │
│         ▼  buffer per-match → judge_match_cots → atomic write            │
│                                                                          │
│  parsed_data/sft_training_data.jsonl                                     │
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

### 4. Generate the SFT training dataset

For each turn in each match, the orchestrator chains the library modules:

1. **`flip_match_to_winner(match)`** — relabel sides so the **series
   winner** is always P1 for the rest of the pipeline. Bo3: 2 of 3
   games. Bo1: per-game winner.
2. **`damage_inferencer.infer_match_final_bounds(games, ...)`** —
   one pass across the whole match to compute the tightest
   KnowledgeState the inferencer can extract. Becomes the "match-final
   P1 bounds" used in YOUR SPREADS at every turn (the player knew
   their own spread from day one; this approximates that knowledge
   at training time). Per-turn loop also keeps a chronological
   `p2_running` state for the opponent side.
3. **`master_pipeline.extract_p1_actions(snap_pre, snap_post, events)`** —
   reverse-engineer P1's two-slot decision from the `events` stream
   (`move` / `switch` / `cant_move` events; `forced_by` filtering for
   intentional vs forced switches). Skip the turn if ambiguous.
4. **`threat_matrix.generate_threat_matrix(snap, "p1", p1_final, p2_running)`**
   — Absolute-only text block: the strict provable damage envelope from
   the **asymmetric** (match-final-P1, chronological-P2) KnowledgeState
   pair. An unconstrained stat renders the word `unknown`.
5. **`format_user_prompt(...)`** — composes the full user prompt:
   board state, GAME-STATE LEDGER (faints / Tera-used / Cumulative
   damage / volatiles / choice locks), TURN-BY-TURN (full prior
   turns of this game), SERIES STATE (Bo3 game ≥ 2; full inlined
   prior-game rollups), YOUR SPREADS (match-final P1 bounds via
   `format_p1_known_spreads_block`), and the threat matrix.
6. **`teacher.synthesize_turn(system, user, human_action)`** —
   provider-agnostic tool-use loop (`TeacherProvider` ABC). The
   model decides whether to call `calculate_damage` for hypotheticals
   the matrix doesn't cover (no per-turn minimum), then commits via
   `submit_decision`. The LLM sees the human's play as ground
   truth and writes a Chain-of-Thought that justifies it. Returns
   OpenAI-fine-tuning-format conversation messages (ground-truth
   stripped before save). Output is buffered per-match, not written
   immediately.
7. **`events_to_damage_events(events)` →
   `damage_inferencer.update_knowledge(snap_pre, snap_post,
   damage_events, p1_running, p2_running)`** — filter the events
   stream for inference-eligible damage hits (excluding non-own callers
   like Metronome / Copycat / Mirror Move / etc.; Sleep Talk allowed),
   then two-way binary search tightens both `KnowledgeState`s with a
   508-EV constraint pass.

After all of a match's turns synthesize, the orchestrator runs
**`judge_match_cots(turn_records, ...)`** on the buffered match — one
gpt-5.5 call per match. Flagged turns are re-synthesized through the
same teacher; after `--judge-retries` exhausted, the dropped turns are
discarded and the rest of the match commits atomically to JSONL.

```bash
cd pipeline

# Smoke test (no API key needed, exercises everything except the LLM call):
.venv/bin/python master_pipeline.py --limit 1 --dry-run

# Real run on a single match (sync mode, Gemini default since Plan v8, judge on):
GOOGLE_API_KEY=... .venv/bin/python master_pipeline.py --limit 1

# Production: sync mode at concurrency 8 (batch mode is OpenAI-only):
.venv/bin/python master_pipeline.py --mode sync --concurrency 8

# Hybrid (OpenAI-only — requires --provider openai for the batch portion):
OPENAI_API_KEY=sk-... .venv/bin/python master_pipeline.py \
    --provider openai --mode hybrid --hybrid-sync-n 50

# Resume a crashed batch run (re-polls in-flight batches from --state-dir):
.venv/bin/python master_pipeline.py --provider openai --mode batch --resume

# Pick a different provider / model (sync only for non-OpenAI providers):
ANTHROPIC_API_KEY=sk-... .venv/bin/python master_pipeline.py \
    --provider anthropic --model claude-sonnet-4-6 --limit 1

# Head-to-head bake-off (runs all configured providers on the same match):
.venv/bin/python bakeoff.py --providers openai,anthropic,google --limit 1
```

→ writes one fine-tuning row per identifiable turn to
`pipeline/parsed_data/sft_training_data.jsonl`. Resumable on rerun
(keyed by `(match_id, game_index, turn)`). Per-match atomic commit:
a match either lands complete or not at all.

### 5. (Optional) Inspect the generated SFT data

The inspector is a local read-only web UI that browses every JSONL file
under `pipeline/parsed_data/` — prompts split into structured sections,
the calc-tool loop laid out step-by-step, and cross-references against
the underlying parsed match snapshots. Useful for sanity-checking new
prompt schemas, eyeballing the judge's flagged rows, or comparing two
rows side-by-side.

```bash
cd inspector
python3 -m venv .venv
.venv/bin/pip install -e .
./run.sh                          # → http://localhost:8001
```

The inspector imports nothing from `pipeline/` and never writes back —
purely a viewer. See [`inspector/README.md`](inspector/README.md) for
endpoints, schema-awareness notes, and the "what it doesn't do" list.

## Status

| Component | State |
|---|---|
| `data_scraper` | Working. 16,537 replays cached locally across both Reg I formats. |
| `calc_microservice` | Working. Four endpoints: `POST /calc` (damage math), `POST /parse_log` (Showdown log → turn snapshots), `GET /dex/move/:name` (move metadata incl. `priority`), `GET /dex/species/:name` (base stats / types / weight). |
| `pipeline/replay_parser.py` | Working. ETL CLI; captures per-turn `events` (TurnEvent[]) from `/parse_log` straight into `parsed_data/{bo1,bo3}.jsonl`. |
| `pipeline/damage_inferencer.py` | Working. Dual-state, two-way binary search, atomic application, 508-EV constraint pass, crit-aware (via `/calc isCrit`), multi-hit filter, `events_to_damage_events()` filter that excludes non-own-move callers (Metronome / Copycat / Sketch / Snatch / Me First / Dancer / Instruct / Mirror Move / Assist / Nature Power; Sleep Talk allowed). Plus `infer_match_final_bounds()` for the match-final P1 spreads surfaced in YOUR SPREADS, and `update_observed_and_speed()` (per-mon `observed` flag + move-order Speed upper-bound, base stats via `/dex/species`). |
| `pipeline/threat_matrix.py` | Working. Absolute-only output — the strict provable damage envelope driven by the **asymmetric** (match-final-P1, chronological-P2) knowledge pair. An unconstrained stat renders the word `unknown`. |
| `pipeline/validate_action_legality.py` | Working. Read-only corpus validator. Scans SFT labels for actions that contradict their own turn state (choice-lock, tera-after-used, OTS moveset-membership) — since labels are real human plays, any violation is a data bug. Run `python validate_action_legality.py`. |
| `pipeline/teacher/` (sub-package: `base.py` + `openai.py` / `anthropic.py` / `google.py` / `judge.py` / `batch_openai.py`, re-exported via `__init__.py`) | Working. Provider-agnostic `TeacherProvider` ABC; tool-use loop with `calculate_damage` + `submit_decision`. The model has only one output channel — tool calls — so it can't bypass the calc tool. Two system-prompt templates: Bo1 CTS (Masking Rule + reconstructed team) vs Bo3 OTS (full sheets + ★ brought-flag). Present-tense framing throughout. Plus `judge.py` (match-level model-judge validator) and `batch_openai.py` (`BatchTeacherProvider` ABC + OpenAI Batch API adapter). |
| `pipeline/batch_runner.py` | Working. Per-iteration state machine for `--mode batch`: bundles all in-flight turns at iter=K into one batch upload, runs calc microservice calls synchronously between cycles, persists `BatchWorkItem` state in `batch_state/{match_id}.json`. Supports `--resume` via `active_batch_id` breadcrumbs. Also hosts `_prepare_match_turns()`, the shared sync-and-batch prep helper. |
| `pipeline/master_pipeline.py` | Working. CLI orchestrator with `--mode {sync,batch,hybrid}` dispatcher. `flip_match_to_winner` makes every example come from the series winner's perspective. Three historical-context blocks in the user prompt (GAME-STATE LEDGER / TURN-BY-TURN / SERIES STATE). YOUR SPREADS surfaces match-final P1 bounds (Plan v3). Per-match buffered write + judge integration (Plan v4). Empty-slot annotation. Perspective-aware bench gating. `--provider {openai,anthropic,google}` flag; batch is OpenAI-only in v1. Resumable. |
| `pipeline/bakeoff.py` | Working. Runs the same match through multiple providers in lockstep and reports per-row cost, tool-call rate, action-match rate, CoT length, wall-clock. **Result (May 2026):** Google gemini-3.1-pro and OpenAI gpt-5.5 tied on quality (100% match rate, 0% leak), with Gemini at $0.04/row vs OpenAI's $0.07. **Plan v8 made Gemini the production default** to align with the project's ~$100K GCP credit pool — OpenAI remains a one-flag flip via `--provider openai`. Anthropic claude-sonnet-4-6 produced near-miss meta-leaks in 32% of saved rows (motivated Plan v4's judge layer). |
| `inspector/` | Working. FastAPI + vanilla-JS single-page app on port 8001. Browses every JSONL under `pipeline/parsed_data/`; splits user prompts into 8 logical sections; renders the calc-tool loop step-by-step; pins two rows for side-by-side compare; cross-references rows against the underlying parsed match snapshots. Schema-aware — handles current and legacy prompt formats. Read-only (writes nothing back). |

See each subdirectory's README for setup, contracts, and design notes.
The [`pipeline/teacher/README.md`](pipeline/teacher/README.md) deep-dives
the provider-agnostic teacher sub-package; [`notes/TODO.md`](notes/TODO.md)
tracks everything not yet shipped.
