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
| `pipeline/` | Python | 6 atomic modules ending in `master_pipeline.py`, which writes the SFT JSONL. |
| `notes/` | — | Free-form planning notes. |

The SFT generation pipeline is **complete end-to-end**. Status of each piece
is tracked in [README.md#status](README.md#status).

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
   or the canonical-priors source without rewriting anything else.

## How to run things

Each subdir has its own README with full setup instructions. Quick reference:

```bash
# 1. Calc microservice (must be running for any pipeline work)
cd calc_microservice && npm run dev   # → http://localhost:3000

# 2. Scraper (one-off; produces data_scraper/data/replays/{format_id}/*.json)
cd data_scraper && .venv/bin/python scrape.py

# 3. Bootstrap canonical priors (one-off per format)
cd pipeline && .venv/bin/python canonical_priors.py --format-id gen9vgc2026regi
                                                    --format-id gen9vgc2026regibo3

# 4. Parse + stitch replays → per-turn snapshots + actionLog
cd pipeline && .venv/bin/python replay_parser.py

# 5. Generate SFT training JSONL  (set OPENAI_API_KEY)
cd pipeline && .venv/bin/python master_pipeline.py
# Smoke-test without the LLM call:
cd pipeline && .venv/bin/python master_pipeline.py --limit 1 --dry-run
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
pipeline/parsed_data/{bo1,bo3}.jsonl    # per-turn snapshots + actionLog
    │
    ▼ (master_pipeline → threat_matrix → /calc, teacher_llm → OpenAI)
pipeline/parsed_data/sft_training_data.jsonl   # one fine-tuning example per turn
```

## Known artifacts on disk (gitignored)

- `data_scraper/data/replays/` — ~140 MB, 16,537 cached replay JSONs.
- `pipeline/data/smogon_chaos_*.json` — Smogon usage data per format
  (regi: 492 species; regibo3: 426 species, both as of 2026-04).
- `pipeline/parsed_data/{bo1,bo3}.jsonl` — parsed snapshots + actionLog.
- `pipeline/parsed_data/sft_training_data.jsonl` — final SFT dataset
  (append-only, resumable).

## Common gotchas

- **Calc service must be running.** Almost everything in `pipeline/` calls
  `http://localhost:3000`. Start it first; both `replay_parser` and
  `master_pipeline` do a `/health` ping and bail with a clear message.
- **Bootstrap canonical priors before generating SFT.** Otherwise
  `threat_matrix` falls back to the curated table + heuristic for the
  Probable track — still works, just less accurate for off-meta species.
- **Re-parse if `/parse_log` schema changes.** `replay_parser` is resumable
  — to force regeneration, delete `pipeline/parsed_data/{bo1,bo3}.jsonl`
  and rerun, or pass `--refetch`.
- **`--dry-run` first.** Before burning OpenAI credits on the real
  `master_pipeline` run, verify orchestration with `--dry-run --limit 1`.
- **Multi-hit moves are skipped during inference.** Triple Axel / Bullet
  Seed / Population Bomb produce one `DamageEvent` per hit; the
  `damage_inferencer` detects and drops them (would need a `hits` field
  on `/calc` to handle properly).
- **Ambiguous turns are skipped silently.** If `extract_p1_actions` can't
  pin a P1 slot's choice (e.g. multiple new revealed moves), the whole
  turn is skipped — no SFT row written, but `update_knowledge` still runs
  on the action_log so we don't lose inference signal.
- **CTS vs OTS format split.** `gen9vgc2026regi` (Bo1) is **Closed Team
  Sheet** — items / abilities / moves hidden until activated, P1 team
  reconstructed by forward-scan with `[UNREVEALED_MOVE]` padding. The Bo1
  system-prompt template carries the Masking Rule. `gen9vgc2026regibo3`
  (Bo3) is **Open Team Sheet** — `|showteam|` decoded by `@pkmn/sets` on
  the Node side; full 6-mon roster + items / abilities / moves / Tera
  type are known from turn 1 for both players. The Bo3 system prompt
  shows both teams in full with ★ markers on P1's brought 4, **no
  Masking Rule**. EVs / IVs / Nature stay hidden in OTS too, so the
  dual-track inferencer continues to do the heavy spread-bound lifting.
- **The Node parser gates P2 bench chronologically in OTS games.** At
  turn 1 of a Bo3 replay, `snapshot.p2.bench` is empty (only the 2
  active have actually appeared); it grows as the opponent's selection
  is revealed via `|switch|`. P1 bench shows the full 4 brought from
  turn 1 (computed via a one-pass pre-scan). Bo1 bench behavior is
  unchanged.
- **Series-winner-as-P1.** Every SFT example is generated from the
  perspective of the player who won the series. `flip_match_to_winner`
  in `master_pipeline.py` rewrites the entire match record (players,
  snapshots, actionLog slot identifiers, teamSheets) when the protocol
  P2 won. Don't assume "p1" in a saved row corresponds to the protocol's
  p1 — it's whoever won the series.

## Provider-agnostic teacher LLM

The teacher LLM tool-loop now goes through a `TeacherProvider` ABC
(`teacher_llm.py`) with three concrete adapters:

- `teacher_openai.py` — `gpt-4o` / `gpt-5` etc. via the OpenAI SDK.
- `teacher_anthropic.py` — `claude-sonnet-4-x` etc. via the Anthropic SDK.
- `teacher_google.py` — `gemini-2.5-pro` etc. via the `google-genai` SDK.

Each adapter implements the same `submit_decision`-tool architecture:
the model **must** call `calculate_damage` at least once before calling
`submit_decision` to commit. There is no `response_format` — tool calls
are the only output channel, which is what fixed the zero-tool-call
regression we saw on the first real run.

Pick a provider via `master_pipeline.py --provider {openai,anthropic,google}`
(default: `openai`). Override the model id with `--model gpt-5` etc.

For a head-to-head comparison: `python bakeoff.py --providers openai,anthropic,google`
runs the same match through each provider and reports per-row cost,
tool-call rate, CoT length, action-match rate, and wall-clock.

### Environment variables

Stored in `.env` at the repo root (gitignored):

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

Each pipeline invocation sources `.env` via `set -a && source ../.env && set +a`
(see test commands earlier in the session). Only the providers whose key
is present will actually run; missing-key providers are skipped with a
warning.

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
```

Cost-table placeholders are in `teacher_llm.PRICE_PER_M_TOKENS` — confirm
against the provider's pricing page before scaling to the full corpus.

## Planned follow-up workstreams (not built yet)

- **Selection-model SFT corpus** — separate dataset for the
  team-preview 4-of-6 pick decision. Sibling module to
  `master_pipeline.py`. Generates `{full p1, full p2, format_meta} →
  {brought 4}` examples per game.
- **Minimax / MCTS distillation for the tool-use loop** — replaces the
  current prompt-driven Alternatives Rule (teacher cherry-picks
  alternatives because it already knows the answer) with a proper
  search step. See `# TODO(rlhf-followup)` in `teacher_llm.py`.
- **Migrate `master_pipeline` default provider to the bake-off winner**
  once empirical results are in. Today's default is `openai/gpt-4o`
  (verified working post-Phase-1 fix); whichever provider wins the
  bake-off becomes the next default.
