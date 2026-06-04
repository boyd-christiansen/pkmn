# CLAUDE.md

Project-specific notes for Claude Code agents working in this repo. See
[README.md](README.md) for the full architecture and execution flow.

## TL;DR

A monorepo for training a generalist LLM that plays competitive VGC (Gen 9
Reg I doubles). Four self-contained components under one umbrella:

| Subdir | Runtime | Job |
|---|---|---|
| `data_scraper/` | Python | Pulls top-500 ladder users + replays from Pokémon Showdown. |
| `calc_microservice/` | Node + TS | HTTP wrapper for `@smogon/calc`, `@pkmn/client`, `@pkmn/dex` (3 endpoints). |
| `pipeline/` | Python | Atomic modules + a `teacher/` sub-package (own README) + a `batch_runner.py` sibling for batched runs. Orchestrator is `master_pipeline.py` (`--mode {sync,batch,hybrid}`), which writes the SFT JSONL. See `pipeline/README.md` for the full file map. |
| `inspector/` | Python (FastAPI) | Local read-only web UI on port 8001 that browses saved SFT rows + cross-references them against parsed-match data. Never writes back into `pipeline/`. |
| `notes/` | — | Long-form docs. `pipeline_walkthrough.md` is the design narrative; `TODO.md` is the master tracker for non-shipped work. |

The SFT generation pipeline is **complete end-to-end**. Status of each piece
is tracked in [README.md#status](README.md#status). Non-shipped work
(active workstreams, code-level `# TODO(...)` markers, long-horizon
plans, deferred items) lives in [`notes/TODO.md`](notes/TODO.md) — the
single source of truth. **When adding a new TODO anywhere in the
project, add it there too.**

## Architecture rules (don't violate without a reason)

1. **Each subdirectory is self-contained.** Its own venv / `package.json` /
   deps. No cross-dir imports.
2. **`calc_microservice/` is the only place that knows Showdown formats.**
   Damage math (`@smogon/calc`), protocol parsing (`@pkmn/client`), and dex
   lookups (`@pkmn/dex`) all live behind HTTP. Python never reaches into
   game data directly.
3. **Pipeline modules are leaf-isolated.** Sibling modules in `pipeline/`
   never import from each other. Only `master_pipeline.py` imports from
   the rest. This is what lets us swap the teacher LLM, the calc engine,
   or the spread-inference source without rewriting anything else.

## How to run things

Each subdir has its own README with full setup instructions. Quick reference:

```bash
# 1. Calc microservice (must be running for any pipeline work)
cd calc_microservice && npm run dev   # → http://localhost:3000

# 2. Scraper (one-off; produces data_scraper/data/replays/{format_id}/*.json)
cd data_scraper && .venv/bin/python scrape.py

# 3. Parse + stitch replays → per-turn snapshots + events stream
cd pipeline && .venv/bin/python replay_parser.py
#    (Smogon canonical-priors bootstrap is GONE — Plan v9 removed the meta
#     machinery; `unknown` is the spread fallback now.)

# 4. Generate SFT training JSONL  (set GOOGLE_API_KEY — Gemini is the prod default since Plan v8)
cd pipeline && .venv/bin/python master_pipeline.py
# Smoke-test without the LLM call:
cd pipeline && .venv/bin/python master_pipeline.py --limit 1 --dry-run
# Production: sync mode with Gemini (default). Batch mode is OpenAI-only:
cd pipeline && .venv/bin/python master_pipeline.py --mode sync --concurrency 8
# OpenAI-batch hybrid (OpenAI-only, requires --provider openai); resume with --resume:
cd pipeline && .venv/bin/python master_pipeline.py --provider openai --mode batch --resume
```

## Conventions

### Python (`data_scraper/`, `pipeline/`)
- Python ≥3.11. Async via `asyncio` + `aiohttp`. CLIs via `click` + `tqdm`.
- Type-hint public functions. `from __future__ import annotations` at the
  top of every file.
- Module docstrings spell out the **isolation contract** (what the module
  is and isn't allowed to touch). Match that pattern in new modules.
- Comments are sparse — name things well, explain WHY when it's
  non-obvious.

### TypeScript (`calc_microservice/`)
- Strict mode TypeScript, ESM. `npm run dev` uses `tsx watch`.
- `src/server.ts` is just the Express wiring; logic lives in `calc.ts`,
  `parse_log.ts`, `dex.ts`.

## Big-picture data flow

```
Pokémon Showdown
    │
    ▼ (data_scraper)
data_scraper/data/replays/{format_id}/{replay_id}.json
    │
    ▼ (replay_parser → /parse_log)
pipeline/parsed_data/{bo1,bo3}.jsonl    # per-turn snapshots + events stream
    │
    ▼ (master_pipeline → threat_matrix → /calc, teacher_llm → OpenAI)
    │   per match:
    │     • sync mode:  per-turn synthesize_turn() inline
    │     • batch mode: batch_runner state-machine, one batch cycle per
    │                   tool-loop iter across all matches at once
    │     • hybrid:     first N sync as quality gate, rest via batch
    │   all paths buffer per-match → judge_match_cots → write atomically
    ▼
pipeline/parsed_data/sft_training_data.jsonl   # one fine-tuning example per turn
                                              # (per-match atomic commit)
```

## Known artifacts on disk (gitignored)

- `data_scraper/data/replays/` — ~140 MB, 16,537 cached replay JSONs.
- (`pipeline/data/smogon_chaos_*.json` — REMOVED in Plan v9 with the
  rest of the Smogon meta machinery.)
- `pipeline/parsed_data/{bo1,bo3}.jsonl` — parsed snapshots + events stream
  (TurnEvent discriminated union: move / switch / cant_move / tera / faint /
  item_event).
- `pipeline/parsed_data/sft_training_data.jsonl` — final SFT dataset
  (append-only, resumable; per-match atomic commit so a match either
  shows up complete or not at all).
- `pipeline/batch_state/{match_id}.json` — per-match resume state for
  `--mode batch` / `--mode hybrid`. Holds every WorkItem's
  `api_messages`, `iter`, `status`, and `active_batch_id`. Safe to delete
  between runs; the orchestrator rebuilds from `parsed_data/` if missing.

## Common gotchas

- **Calc service must be running.** Almost everything in `pipeline/` calls
  `http://localhost:3000`. Start it first; both `replay_parser` and
  `master_pipeline` do a `/health` ping and bail with a clear message.
- **No Smogon meta machinery (Plan v9).** `canonical_priors.py` and the
  chaos JSONs were deleted; the threat matrix renders ONLY the Absolute
  envelope (no `| meta` / `(off-meta)` second track), and a spread with
  no observation renders as `unknown` — not a canonical fallback. Don't
  reintroduce a usage-prior into the per-turn prompt.
- **Re-parse if `/parse_log` schema changes.** `replay_parser` is resumable
  — to force regeneration, delete `pipeline/parsed_data/{bo1,bo3}.jsonl`
  and rerun, or pass `--refetch`.
- **`--dry-run` first.** Before burning OpenAI credits on the real
  `master_pipeline` run, verify orchestration with `--dry-run --limit 1`.
- **Multi-hit moves are skipped during inference.** Triple Axel / Bullet
  Seed / Population Bomb expand to multiple `DamageEvent` records (one
  per hit) after `events_to_damage_events` flattens the new event
  schema; the `damage_inferencer` detects and drops them (would need a
  `hits` field on `/calc` to handle properly).
- **Non-own-move callers are filtered for inference and revealed-set
  attribution.** Metronome, Copycat, Sketch, Snatch, Me First, Dancer,
  Instruct, Mirror Move, Assist, Nature Power can all call moves the
  user doesn't own — their hits would corrupt EV inference and
  pollute reconstructed CTS movesets. The Node `NON_OWN_CALLERS` set
  drives both `derivedRevealedMoves` filtering and the
  `events_to_damage_events` filter (`called_via in {None, "Sleep
  Talk"}` only). Sleep Talk is allowed because it only calls own
  moves; Mimic is allowed because it permanently overwrites a slot
  in-battle.
- **Ambiguous turns are skipped silently.** If `extract_p1_actions` can't
  pin a P1 slot's choice (e.g. forced out before acting), the whole
  turn is skipped — no SFT row written, but `update_knowledge` still
  runs on the events stream so we don't lose inference signal.
- **CTS vs OTS format split.** `gen9vgc2026regi` (Bo1) is **Closed Team
  Sheet** — items / abilities / moves hidden until activated, P1 team
  reconstructed by forward-scan with `[UNREVEALED_MOVE]` padding. The Bo1
  system-prompt template carries the Masking Rule. `gen9vgc2026regibo3`
  (Bo3) is **Open Team Sheet** — `|showteam|` decoded by `@pkmn/sets` on
  the Node side; full 6-mon roster + items / abilities / moves / Tera
  type are known from turn 1 for both players. The Bo3 system prompt
  shows both teams in full with ★ markers on P1's brought 4, **no
  Masking Rule**. EVs / IVs / Nature stay hidden in OTS too, so the
  damage + speed inferencer continues to do the heavy spread-bound lifting.
- **Bench rendering: parser is symmetric, perspective is applied in
  Python.** The Node parser emits `bench` for BOTH sides as the full
  pre-scanned brought-set (every species ever switched in across the
  full game), regardless of format. Each side also carries a
  `seenSpecies: string[]` field with the chronological set of species
  that have actually been on field at any turn ≤ current. The
  perspective-aware filtering happens in
  `master_pipeline.format_user_prompt`: P1 bench renders the full
  brought-set (the player knows their own selection from team
  preview), while P2 bench is gated by `seenSpecies` (the player only
  learns the opponent's selection as they switch in). Symmetric
  parser output makes `flip_match_to_winner` clean — no asymmetric
  data to reconstruct after the swap.
- **Six-roster model (Plan v9).** Both SPREADS blocks render the FULL
  known roster per side (all 6 in Bo3 from team sheets; revealed-so-far
  in Bo1), not just the active pair — YOUR SPREADS previously dropped
  the bench. The only spread states are: bounds, `unknown` (the single
  fallback; replaces "(no observations yet)"), and an unknown-brought-
  slot placeholder. Reg I brings 4, but the brought-4 is NEVER in the
  data (only usage is), so when identifiable brings < 4 the BENCH renders
  an explicit placeholder ("brought, never sent" for P1; "identity not
  yet revealed" for P2). Never emit a "not-brought" state. See
  `prompt_formatting._brought_placeholder_lines` + `REG_I_BRING_COUNT`.
- **Field state lives ONLY in the ledger (Plan v9).** The user-prompt
  header no longer carries a `Field:` line; weather / terrain / Trick
  Room / per-side Tailwind / screens render once, in the GAME-STATE
  LEDGER, each as `(N turns left)`. Don't reintroduce a header field
  token — it's a second representation to drift.
- **Speed (move-order) + observed inference (Plan v9).** Damage only
  tightens atk/spa/def/spd/hp; `damage_inferencer.update_observed_and_speed`
  adds (a) an `observed` flag per mon (True once it deals/takes damage
  or reveals move order — the causal definition of NOT-`unknown`), and
  (b) a conservative Spe upper-bound from "moved-after a known mon"
  (Scarf-safe: a Choice Scarf only speeds a mon up, so moved-after ⇒
  genuinely slower). Needs base stats via the new `/dex/species` Node
  endpoint; skips Trick Room / tailwind / sticky-web / paralysis /
  After-You-Quash-Instruct turns; only trusts priority-0 damaging pairs.
  Called from both `_safe_update_knowledge` copies + the match-final
  pass, gated by per-side roster-key sets so a transformed Ditto can't
  pollute a real slot.
- **Action-legality validator (Plan v9).** `validate_action_legality.py`
  scans the corpus for labels that contradict their own state
  (choice-lock, tera-after-used, OTS moveset-membership). Labels are
  real human plays, so any hit is a DATA BUG. It found a small
  choice-lock false-positive (`snapshotChoiceLock` over-eager) + an OTS
  moveset-decode mismatch — both tracked in notes/TODO.md. Encore/Disable
  (boolean only) + trapping (uncaptured) can't be hard-masked yet.
- **Series-winner-as-P1.** Every SFT example is generated from the
  perspective of the player who won the series. `flip_match_to_winner`
  in `master_pipeline.py` rewrites the entire match record (players,
  snapshots, every TurnEvent's slot/side fields, teamSheets) when the
  protocol P2 won. Don't assume "p1" in a saved row corresponds to the
  protocol's p1 — it's whoever won the series.
- **Historical context lives in the user prompt.** Each turn's prompt
  includes a `=== GAME-STATE LEDGER ===` (faints, Tera-used, field +
  pseudo-weather + side conditions with turns-left, on-active
  volatiles, choice locks with display-name-normalized move, recent
  item events, and per-active Cumulative damage taken), a
  `=== TURN-BY-TURN (game N) ===` rollup of every prior turn's events
  this game, and (Bo3 game ≥ 2) a `=== SERIES STATE ===` block whose
  per-prior-game summary now inlines the **full turn-by-turn rollup**
  of that game (an earlier `Notable` heuristic was dropped because it
  was too thin — TODO marker for a future learned summarizer). Built
  by `format_game_state_ledger`, `format_turn_by_turn`,
  `format_series_state` in `master_pipeline.py`.
- **Empty-slot annotation.** When P1 is down to 1 mon (slot vacant,
  no living bench replacement), the prompt explicitly renders
  `[b] (empty — no Pokémon remaining)` so the model doesn't have to
  infer slot vacancy from the absence of an active line.
- **System-prompt tense is present, not historical.** The model is
  trained as if playing live — wording like "you don't yet know
  which 4 they will bring" / "you'll learn that as the battle
  unfolds." Don't reintroduce past-tense framing when editing
  templates in `teacher/base.py`.
- **Per-match atomic commit.** Plan v4 changed the write path: turns
  are buffered in memory per match, then committed atomically as a
  batch after the judge runs. Crash mid-match → nothing for that
  match lands on disk; re-run the match cleanly. Don't expect a
  partial-match JSONL.
- **Two-stage leak filter.** Every CoT runs through both the regex
  (`detect_oracle_leak`) and the model judge (`judge_match_cots`,
  one call per match). The regex is the first line; the judge is the
  long-tail catch. With `--leak-retries 3 --judge-retries 2`
  (production defaults), persistent leaks drop only the offending
  turn — the rest of the match still writes.
- **Judge defaults to the same provider as the teacher (Plan v8).** Set
  via `--judge-provider {google,openai}`. Default `google` matches the
  production teacher (Gemini). Plan v4 originally hard-wired the judge
  to OpenAI for cross-provider consistency; Plan v8 dropped that — the
  cross-provider value wasn't paying for the extra OpenAI dependency
  once we standardized the teacher. Override to `openai` to opt back
  into cross-provider sanity-checking.
- **`--mode batch` is OpenAI-only in v1.** Anthropic Message Batches
  and Vertex Batch (the natural fit now that Gemini is the default
  teacher) are deferred follow-ups; the CLI rejects `--provider
  {anthropic,google}` with `--mode batch`. With ~$100K GCP credits the
  unit-cost saving from batch is no longer a hard constraint, so sync
  mode with Gemini is the production path.
- **Batch latency is unpredictable.** OpenAI's SLA is 24h per cycle.
  Empirically the smoke-test runs we did completed cycles in <2min,
  but a corpus-scale run may see the documented 1–3h band. Plan for
  hours of wall-clock when sizing batches. Use `--resume` to pick up
  in-flight batches after a crash; per-item `active_batch_id`
  breadcrumbs in `batch_state/{match_id}.json` route the recovery.

## Provider-agnostic teacher LLM

(Deep dive in [`pipeline/teacher/README.md`](pipeline/teacher/README.md) —
contract, judge architecture, batch architecture, full bake-off table.
The summary below covers what you need *while coding*.)

The teacher LLM tool-loop goes through a `TeacherProvider` ABC
(`pipeline/teacher/base.py`) with three concrete adapters in the
`pipeline/teacher/` sub-package, all re-exported via `teacher/__init__.py`
so callers write `from teacher import TeacherProvider, OpenAIProvider`:

- `teacher/openai.py` — `gpt-5.5` (default) / `gpt-5.5-pro` / `gpt-5.5-mini` etc. via the OpenAI SDK.
- `teacher/anthropic.py` — `claude-sonnet-4-6` (default) / `claude-opus-4-7` / `claude-haiku-4-5` etc. via the Anthropic SDK.
- `teacher/google.py` — `gemini-3.1-pro-preview` (default) / `gemini-3.1-flash-preview` etc. via the `google-genai` SDK.

Each adapter has the same shape: two tools (`calculate_damage`,
`submit_decision`); `submit_decision` is the only structured-output
channel (no `response_format`); per-call and per-turn timeouts via
`asyncio.wait_for`. The Tool Rule (plan v3) directs `calculate_damage`
toward hypotheticals the threat matrix doesn't already cover — there
is no per-turn minimum.

Pick a provider via `master_pipeline.py --provider {openai,anthropic,google}`
(default: `google` — production switch landed in Plan v8; see below).
Override the model id with `--model gemini-3.1-flash-preview` etc.

For a head-to-head comparison: `python bakeoff.py --providers openai,anthropic,google --limit 5`
runs the same match through each provider and reports per-row cost,
tool-call rate, CoT length, action-match rate, and wall-clock.

### Bake-off result (May 2026) — Gemini chosen for production in Plan v8

| Provider | Match% | Leak rate | $/row | Avg CoT | Notes |
|---|---|---|---|---|---|
| **Google gemini-3.1-pro** | **100.0%** | **0%** | $0.04 | 1027ch | Tied OpenAI on quality at unit-cost ~40% cheaper. **Production default (Plan v8).** |
| OpenAI gpt-5.5 | 100.0% | 0% | $0.07 | 902ch | Concise, consistent. Available via `--provider openai`; required for `--mode batch`. |
| Anthropic claude-sonnet-4-6 | 61.3% | 32% near-miss | $0.09 | 4004ch | Verbose; systematic meta-references in CoT ("the target action" / "training section"). Not used in production. |

Anthropic's near-miss leaks ("the target action", "training section"
substrings) motivated plan v4's two-stage leak filter — the regex
tightened to catch those, and a model-judge layer catches softer
meta-references. See "Plan v4: judge + batch" below.

Plan v8 flipped the production default from OpenAI → Gemini once the
project picked up ~$100K in GCP credits. The bake-off had the two
providers tied on quality (100% / 0%), so the tie-breaker became
economics: GCP credits make Gemini effectively free for both
synthesis AND the downstream fine-tuning + constitutional-learning
inference. OpenAI remains a one-flag-flip away (`--provider openai`)
for cross-provider sanity-checks or when the OpenAI-only batch mode
is needed.

## Plan v4: judge + batch

Two follow-up workstreams shipped on top of the bake-off:

### 1. Match-level model-judge validator

`pipeline/teacher/judge.py` exposes `judge_match_cots(turn_records,
*, client, provider, model=DEFAULT_JUDGE_MODEL) -> JudgeResult`. After
every match's turns are synthesized (sync OR batch), the orchestrator
buffers them, submits all of the match's `pre_tool_thought` CoTs to
the judge in **one** call, and the judge returns
`{flagged_turn_indices, reasons}`. Flagged turns are re-synthesized
via the sync teacher (even in batch mode — batch latency is too high
for retry); after `judge_retries` exhausted (default 2), only the
still-flagged turns get dropped — the rest of the match writes.

Plan v8 added provider dispatch: `--judge-provider {google,openai}`,
defaulting to `google` (matches the production teacher). The OpenAI
judge path is preserved for explicit `--judge-provider openai` runs.

CLI: `--use-judge / --no-judge`, `--judge-provider`, `--judge-model`,
`--judge-retries`.

Cost: one judge call per match. Default Gemini path
(`gemini-3.1-pro-preview`) is ~$0.014/match — equivalent to the OpenAI
gpt-5.5 path it replaced. Both negligible against the per-row
synthesis cost, both effectively free under GCP credits.

### 2. Batch mode + `--mode {sync,batch,hybrid}`

`pipeline/teacher/batch_openai.py` is the SDK plumbing
(`BatchOpenAIProvider`); `pipeline/batch_runner.py` is the state
machine — one OpenAI Batch cycle per tool-loop iteration, with all
in-flight turns at iter=K bundled into one batch upload, calc
microservice calls run synchronously between cycles. Per-match
state files in `pipeline/batch_state/{match_id}.json` make runs
resumable mid-cycle via `--resume` (each item carries an
`active_batch_id` breadcrumb so we can re-poll the right batch).

Hybrid mode runs the first N matches sync as a quality gate. If
`match_rate < min_match_rate` (default 0.95) OR `leak_rate >
max_leak_rate` (default 0.02), the run halts before submitting the
batch portion — better to fail loudly than silently commit thousands
of dollars to a regressed prompt.

Batch is OpenAI-only in v1. Now that Plan v8 made Gemini the
production teacher, a **Vertex Batch adapter is the natural next
workstream** (the `BatchTeacherProvider` ABC is already in place;
the work is one concrete adapter + plumbing). Anthropic Message
Batches sits behind that as a smaller priority. With ~$100K GCP
credits available, batch cost savings are softer than they were
when OpenAI was production default.

CLI: `--mode {sync,batch,hybrid}`, `--hybrid-sync-n`,
`--hybrid-min-match-rate`, `--hybrid-max-leak-rate`,
`--state-dir`, `--poll-interval-seconds`, `--max-cycle-wait-seconds`,
`--resume`.

### Environment variables

Stored in `.env` at the repo root (gitignored):

```
GOOGLE_API_KEY=...          # required for production (Gemini default)
OPENAI_API_KEY=sk-...       # required for --provider openai / --mode batch / --judge-provider openai
ANTHROPIC_API_KEY=sk-ant-... # optional
```

Each pipeline invocation sources `.env` via `set -a && source ../.env && set +a`
(see test commands earlier in the session). Only the providers whose key
is present will actually run; missing-key providers raise a clear error
at startup.

Default models (frontier mid-tier in each family — best cost/quality balance
for our tool-loop use case):

| Provider | Default | Top-tier alternative | Cheap alternative |
|---|---|---|---|
| OpenAI | `gpt-5.5` | `gpt-5.5-pro` | `gpt-5.5-mini` / `gpt-5.5-nano` |
| Anthropic | `claude-sonnet-4-6` | `claude-opus-4-7` | `claude-haiku-4-5` |
| Google | `gemini-3.1-pro-preview` | (same — pro is the top) | `gemini-3.1-flash-preview` / `flash-lite-preview` |

Optional env-var overrides:

```
TEACHER_MODEL_OPENAI=gpt-5.5-pro
TEACHER_MODEL_ANTHROPIC=claude-opus-4-7
TEACHER_MODEL_GOOGLE=gemini-3.1-flash-preview
JUDGE_MODEL=gpt-5.5-mini       # plan v4: judge defaults to gpt-5.5; set to mini if you have access
```

Cost-table placeholders are in `teacher.base.PRICE_PER_M_TOKENS` — confirm
against the provider's pricing page before scaling to the full corpus.

## Where TODOs live

All non-shipped work lives in [`notes/TODO.md`](notes/TODO.md) — active
workstreams, the three open `# TODO(...)` markers in source, Plan v4
follow-ups, long-horizon plans (selection-model SFT, RLHF), inspector
gaps, and data-sourcing options.

When you spot a new TODO anywhere — code comment, design issue, missing
feature — add it to `notes/TODO.md`. This file (`CLAUDE.md`) only
carries the architecture + gotchas you need *while coding*. The
shipped-state docs (`README.md`, `pipeline/README.md`, the walkthrough)
describe what's done, not what's planned.
