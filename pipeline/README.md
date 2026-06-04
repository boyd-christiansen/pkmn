# pipeline

Atomic Python modules that turn raw Pokémon Showdown replays into SFT-ready
conversational training data.

```
pipeline/
├── replay_parser.py          # raw replay JSONs → per-turn snapshot JSONL
├── damage_inferencer.py      # binary-search EV bound inference +
│                             #   infer_match_final_bounds (match-final pass) +
│                             #   update_observed_and_speed (observed flag +
│                             #   move-order Speed upper-bound)
├── threat_matrix.py          # Absolute damage envelope per turn
├── validate_action_legality.py  # read-only corpus validator: SFT labels
│                             #   vs their own turn state (data-bug scanner)
├── team_reconstruction.py    # P1 team / brought-set / sheet helpers
├── action_extraction.py      # winner-flip + extract_p1_actions
├── prompt_formatting.py      # 8-section user-prompt composer
├── master_pipeline.py        # CLI orchestrator with --mode {sync,batch,hybrid}
│                             #   dispatcher (the only file allowed to import
│                             #   from every other module here)
├── batch_runner.py           # Plan v4: batch-mode state machine + shared
│                             #   _prepare_match_turns helper. Per-cycle
│                             #   submissions across all matches; resume via
│                             #   batch_state/{match_id}.json
├── bakeoff.py                # head-to-head provider runner
└── teacher/                  # provider-agnostic teacher LLM
    │                         # — see teacher/README.md for the deep dive
    ├── README.md             # contract + adapters + judge + batch + bake-off
    ├── __init__.py           # re-exports for `from teacher import ...`
    ├── base.py               # TeacherProvider ABC, schemas, prompts,
    │                         #   detect_oracle_leak + extract_pre_tool_thought
    ├── openai.py             # OpenAI adapter (post-bake-off alternative; --provider openai)
    ├── anthropic.py          # Anthropic adapter
    ├── google.py             # Google adapter (production default since Plan v8)
    ├── judge.py              # Plan v4: judge_match_cots — match-level
    │                         #   model-judge validator for CoT hygiene
    └── batch_openai.py       # Plan v4: BatchTeacherProvider ABC +
                              #   BatchOpenAIProvider (OpenAI Batch API)
```

## Setup

```bash
cd pipeline
python3 -m venv .venv
.venv/bin/pip install -e .
```

Deps: `aiohttp`, `tqdm`, `click`. Python ≥3.11.

## Architecture principle

Every module here is **atomic and isolation-respecting**. Each one:

- has a single, narrow responsibility,
- declares its inputs and outputs in its docstring,
- talks to at most one external system (e.g. only the calc service, or only
  the teacher LLM, or no network at all),
- can be swapped without touching the others.

The orchestrator (`master_pipeline.py`) is the **only** file allowed to import
from all the others. The reverse is forbidden — sibling modules never import
each other and never import the orchestrator.

This rule is what lets us swap out e.g. the teacher LLM provider
(OpenAI ↔ Anthropic ↔ Google, all behind the `TeacherProvider` ABC)
or the calc engine without rewriting the whole pipeline.

## Modules

### `replay_parser.py` *(implemented)*

ETL: walks the scraper output, stitches Bo3 series, posts each `log` to
[`calc_microservice`](../calc_microservice/)'s `POST /parse_log` endpoint, and
emits one JSONL row per *match*.

- **In:** raw replay JSONs at `../data_scraper/data/replays/{format_id}/*.json`,
  plus a running calc microservice (defaults to `http://localhost:3000`).
- **Out:** `parsed_data/bo1.jsonl`, `parsed_data/bo3.jsonl`,
  `parsed_data/failures.jsonl` — one match per line.
- **Touches:** the `/parse_log` endpoint only. No regex parsing of logs (the
  `@pkmn/client` Battle state machine on the Node side handles edge cases like
  Zoroark illusion, end-of-turn order, multi-hit moves, forme changes, etc).
- **Stitching rules (Bo3):** group games by sorted player pair → sort by
  `uploadtime` → split a new series whenever consecutive games are >30 min apart
  *or* the current series already has 3 games (the Bo3 ceiling — back-to-back
  matches between the same players otherwise get glued together).
- **Resumable:** existing JSONL is scanned at startup; matches already present
  are skipped. Failed matches go to `failures.jsonl` and are retried on rerun.

#### CLI

```bash
.venv/bin/python replay_parser.py                              # full run, both formats
.venv/bin/python replay_parser.py --limit 10 --format bo3      # 10-match smoke test
.venv/bin/python replay_parser.py --concurrency 16             # more parallelism
.venv/bin/python replay_parser.py --bo3-gap-minutes 45         # tune stitching gap
```

| Flag | Default | Notes |
|---|---|---|
| `--scraper-dir` | `../data_scraper/data/replays` | Root of scraper output. |
| `--output-dir` | `parsed_data/` | Where JSONL + failures go. |
| `--parse-url` | `http://localhost:3000/parse_log` | Mechanics service endpoint. |
| `--concurrency` | `8` | Max in-flight `/parse_log` requests. |
| `--bo3-gap-minutes` | `30` | Series-split threshold. |
| `--limit` | none | Process only the first N matches (test batch). |
| `--format` | both | `bo1` or `bo3` to restrict. |

#### Output row shape

```json
{
  "match_id": "bo3-gen9vgc2026regibo3-2563049793",
  "players":  ["carrotvg", "Yippeewoohoo"],
  "format":   "bo3",
  "games": [
    { "replay_id": "...", "timestamp": 1773978060, "snapshots": [...] },
    { "replay_id": "...", "timestamp": 1773978161, "snapshots": [...] }
  ]
}
```

`snapshots` is the array returned verbatim by `/parse_log` — see the
[calc_microservice README](../calc_microservice/README.md#post-parse_log) for
the per-turn shape.

### `canonical_priors.py` *(removed in Plan v9)*

The Smogon meta machinery is gone. `canonical_priors.py` and the
`pipeline/data/smogon_chaos_*.json` caches were deleted when the threat
matrix dropped its "Probable" / canonical-meta second track. The matrix
now renders the **Absolute** envelope only (the strict provable range
from both sides' inferred `KnowledgeState` bounds); an unconstrained
stat renders the word `unknown` rather than a canonical spread.

### `validate_action_legality.py` *(implemented — Plan v9)*

Read-only corpus validator. Scans the SFT labels for actions that
contradict their own turn state — choice-lock (a Choice-item mon must
repeat its locked move), tera-after-used (can't Terastallize twice),
OTS moveset-membership (a Bo3 move must be on the team sheet). Since
every label is a real human play, any violation is a **data bug**, not
a model error.

```bash
cd pipeline
.venv/bin/python validate_action_legality.py
```

- **Touches:** reads `parsed_data/sft_training_data.jsonl` +
  `parsed_data/{bo1,bo3}.jsonl` only. No network, no LLM, no calc
  service, no sibling imports.

### `damage_inferencer.py` *(implemented — dual-state, two-way)*

Maintains a `KnowledgeState` per side (the orchestrator holds two:
`p1_knowledge`, `p2_knowledge`) — per-species `min_evs` / `max_evs` boxes
in `[0, 252]^6`. Each observed damage event tightens **both** sides at once
via binary search against the calc microservice.

- **In:** turn snapshots, a `list[DamageEvent]` for the turn (slot-based:
  `attacker_slot="p1a"`, `defender_slot="p2b"`, `move_name`,
  `hp_before_pct`, `hp_after_pct`, `is_crit`, `is_ko`), and both
  `KnowledgeState`s.
- **Out:** the same `(p1_knowledge, p2_knowledge)` tuple, mutated in
  place — `min_evs` ratcheted up, `max_evs` ratcheted down per event.
- **Touches:** `POST /calc`, `GET /dex/move/:name`, and
  `GET /dex/species/:name`. No replay parsing, no LLM, no sibling imports.
- **Two-way binary search.** For each non-status event we run six binary
  searches per damage event:
  - DEFENDER `min_def`, `max_def`, `min_hp`, `max_hp`
  - ATTACKER `min_off`, `max_off` (`atk` if physical, `spa` if special)

  Cross-side coupling is handled with interval arithmetic — when
  searching the defender, the attacker is held at its *least restrictive*
  current bound, and vice versa. Symmetric for HP.
- **Atomic application.** All six searches use **pre-update** bounds, then
  results apply at the end. Order-independent.
- **Crits supported.** `event.is_crit=True` is forwarded to `/calc` via the
  `move: { name, isCrit: true }` payload form. The calc applies the 1.5×
  multiplier and bypasses the defender's positive boosts, so the inference
  remains valid for crit observations.
- **Multi-hit filter.** Triple Axel / Bullet Seed / Population Bomb / etc.
  put multiple entries in a single move event's `hits[]` array (typically
  all targeting the same defender). `events_to_damage_events()` flattens
  these into multiple `DamageEvent` objects, then the inferencer counts
  same `(attacker_slot, move_name, defender_slot)` tuples per turn and
  **skips all of them** if the count > 1 — supporting them properly would
  need `/calc` to accept a `hits` field.
- **Caller filter.** `events_to_damage_events()` keeps only `type=="move"`
  events with `called_via in {None, "Sleep Talk"}`. Metronome / Copycat /
  Sketch / Snatch / Me First / Dancer / Instruct hits are excluded —
  those moves may not be in the user's actual kit and would corrupt EV
  bounds.
- **508-EV total constraint.** After the six binary searches, a cheap pass
  enforces the 508-EV usable budget per Pokémon: `max_evs[s] ≤ 508 −
  (sum_of_other_min_evs)`. Once one or two stats are known to be heavily
  invested (e.g. Atk≥252 + Spe≥252 = 504), the other four collapse to ≤4
  in a single pass — solves the HP/Def coupling that binary search alone
  converges on slowly.
- **Fuzzy HP:** ±0.9% tolerance to absorb the 1% rounding on
  spectator-visible HP bars.
- **Skip rules:** Status-category moves, missing slots, multi-hit
  occurrences, and observations with no consistent EV in `[0, 252]` (data
  inconsistency — bounds left untouched).
- **Source of `events`:** `/parse_log` emits per-turn `events` arrays
  (TurnEvent discriminated union) inline with each snapshot. The
  orchestrator pairs `snapshot[N].events` with `snapshot_pre =
  snapshot[N]`, `snapshot_post = snapshot[N+1]`. Use
  `damage_inferencer.events_to_damage_events()` to filter the stream
  for inference-eligible damage observations.
- **`update_observed_and_speed()` (Plan v9).** Two move-order-driven
  derivations per turn:
  - **`observed` flag.** A mon is marked `observed` the moment it
    deals/takes damage or reveals move order — the causal definition of
    when a mon stops rendering as `unknown` in the spread blocks.
  - **Speed upper-bound.** A mon that moved *after* a known mon is
    genuinely slower, so its Speed gets a conservative upper bound from
    move order. Choice-Scarf-safe (the bound only tightens on the slower
    side of an observed ordering). Needs base stats from the new
    `GET /dex/species/:name` endpoint.

### Format split: CTS (Bo1) vs OTS (Bo3)

The pipeline branches on `match_record["format"]`:

- **Bo1 / CTS** (`gen9vgc2026regi`): no `|showteam|` lines in the log.
  `replay_parser.py`'s `games[].teamSheets` is `null`. `master_pipeline`
  uses `reconstruct_p1_team` (forward-scan + `[UNREVEALED_MOVE]` padding)
  and `teacher_llm.render_system_prompt` (Bo1 template + Masking Rule).
  Behavior **unchanged** from before this feature.
- **Bo3 / OTS** (`gen9vgc2026regibo3`): `|showteam|` decoded by
  `@pkmn/sets`'s `Teams.unpackTeam` on the Node side. `games[].teamSheets`
  is `{ p1: OtsPokemonSet[6], p2: OtsPokemonSet[6] }`. `master_pipeline`
  uses `teacher_llm.render_system_prompt_bo3` (full sheets for both
  sides, ★ markers on P1's brought 4, no Masking Rule). KnowledgeStates
  are seeded with all 6 species per side. Active Pokémon snapshots get
  OTS-known `item` / `ability` / `teraType` from turn 1, and a new
  `knownMoves: string[4] | null` field carries the OTS-known full moveset
  — `threat_matrix` prefers `knownMoves` when present (Bo3) and falls
  back to `revealedMoves` (Bo1). VGC OTS does not reveal EVs / IVs /
  Nature, so the dual-state inferencer continues to do the heavy lifting
  on spread bounds.

### `threat_matrix.py` *(implemented — Absolute-only since Plan v9)*

Renders the per-turn damage envelope as a compact human-readable text
block. One track per matchup:

- **Absolute** — the strict mathematical envelope from the live
  `KnowledgeState`s. Wide but provable.

Plan v9 deleted the second "Probable (meta)" track (it depended on the
now-removed `canonical_priors`). There is no `| meta` column and no
`(off-meta)` tag anymore. An unconstrained stat renders the word
`unknown` rather than falling back to a canonical spread.

- **In:** one snapshot, `p1_side` (which side is "us"), and **both**
  `p1_knowledge` and `p2_knowledge`.
- **Out:** a single string, grouped per attacker. Each attacker block
  shows one line per (move, defender) pair (or a single grouped line
  per spread move). Chip moves (max % < 15% across every defender) are
  collapsed into a single footer line per attacker.
- **Touches:** `POST /calc` (2 calls per (move, defender) cell —
  abs-low, abs-high) and `GET /dex/move/:name` (cached, fetched for
  category + target). Status moves filtered out.
- **Volatile state:** status, boosts, weather, terrain, side conditions,
  Tera state — all threaded into every payload from the snapshot.
- **Spread modifier auto-applies.** Doubles spread moves
  (`allAdjacentFoes`, `foeSide`, `allAdjacent`) get the 0.75× modifier
  for free from `@smogon/calc` when `field.gameType: "Doubles"`. Output
  presentation groups them into one `[spread]` line listing every
  defender.

#### Output line format

```
[us Chi-Yu]  (boosts={spe: -1, spa: -2})
  Heat Wave [spread]: Miraidon 11.7%–13.7%, Iron Bundle 47.8%–62.6%  [Iron Bundle: guaranteed 2HKO]
  Overheat → Iron Bundle    85.5%–116.8%  [guaranteed OHKO]
  …plus 1 chip move(s): Snarl
```

### `teacher/` sub-package *(implemented)*

Provider-agnostic tool-calling loop that elicits a chain-of-thought
JUSTIFYING a known human play. The model has only one output channel
— tool calls — so it can't bypass the calc tool; commits via the
`submit_decision` tool.

- **Layout:** `teacher/base.py` carries the `TeacherProvider` ABC, the
  `calculate_damage` and `submit_decision` tool schemas, the system-
  prompt templates, the cost-table, the regex `detect_oracle_leak` +
  shared `extract_pre_tool_thought` helper, and the shared `_call_calc`
  helper. Concrete adapters in `teacher/openai.py`, `teacher/anthropic.py`,
  `teacher/google.py` each implement `synthesize_turn` against their
  SDK's tool-call format. `teacher/judge.py` carries the match-level
  model-judge validator (Plan v4). `teacher/batch_openai.py` carries
  the `BatchTeacherProvider` ABC + `BatchOpenAIProvider` for the
  OpenAI Batch API (Plan v4). `teacher/__init__.py` re-exports the
  entire public surface so call sites use `from teacher import
  TeacherProvider, OpenAIProvider, judge_match_cots,
  BatchOpenAIProvider, ...`.
- **API:** `await provider.synthesize_turn(system_prompt, user_prompt,
  human_action, ...)` → `ProviderResult` with `.messages: list[dict]
  | None`, `.iterations`, `.input_tokens`, `.output_tokens`,
  `.calc_calls`, `.cost_usd`, `.elapsed_seconds`, `.error`.
- **Tools exposed to the LLM:** `calculate_damage` — JSON schema mirrors
  `/calc`'s payload (Pokémon × move × field, with `isCrit` / `evs` / `nature`
  / etc. all optional).
- **Final-output schema:** strict JSON schema lives on the
  `submit_decision` tool's `parameters` (NOT on `response_format`).
  Shape: `{ pre_tool_thought: string, action: { slot_1, slot_2 } }`.
  Each slot has `{action_type: "move"|"switch"|"pass", move?, target?,
  tera?, switch_to?}`. The model has only one output channel — tool
  calls — so it can't bypass `calculate_damage` by emitting a direct
  structured response (this fixed an earlier zero-tool-call regression).
- **Ground-truth handling:** during the API call, the user message has the
  human's play appended as a `=== TRAINING-MODE TARGET ===` suffix
  (rewritten in Plan v3 from `=== EXPERT'S DECISION ===` to discourage
  meta-leaks). The returned messages have that suffix **stripped** —
  saved SFT examples show only board state + threat matrix.
- **Tool loop:** up to 10 iterations (`MAX_TOOL_ITERATIONS`) with a
  5-call upper cap (`MAX_CALC_CALLS_BEFORE_FORCE_SUBMIT`) — past 5 calc
  calls, `tool_choice` forces `submit_decision`. No per-turn minimum on
  calc calls (Plan v3 dropped the iter-0 forced calc; the rewritten
  Tool Rule directs calc at hypotheticals the matrix doesn't cover).
  Per-call timeout 120s + per-turn ceiling 300s via `asyncio.wait_for`.
- **Leak filtering (Plan v3 + v4).** Two stages:
  1. **Regex.** `detect_oracle_leak(messages)` matches "oracle",
     "ground-truth", "the target {is,says,action,field}", "training
     {mode,section,target,example}", etc. — patterns we've observed in
     real bake-off rows. Called by `master_pipeline._synthesize_with_leak_retry`
     on every synthesis; retries up to `--leak-retries` times.
  2. **Model judge.** After all of a match's turns synthesize,
     `judge_match_cots` (gpt-5.5) sees every CoT in one call and
     returns turn indices to retry. Catches softer phrasings the regex
     misses ("clearly the right move", "the data points to"). Bake-off
     audit showed OpenAI + Google produce 0 such near-misses; Anthropic
     produced them in 32% of saved rows — motivating the judge layer.
- **Critical Rules in the system prompt** include the **Masking Rule**
  verbatim from the project spec: `[UNREVEALED_MOVE]` slots are presumed
  suboptimal and not to be reasoned about.
- **Touches:** OpenAI / Anthropic / Google Chat APIs + `/calc` (tool
  execution). No replay parsing, no inference, no canonical-priors imports.

### `teacher/judge.py` *(implemented — Plan v4)*

Match-level CoT hygiene validator. After all of a match's turns are
synthesized (sync or batch), the orchestrator submits every turn's
`pre_tool_thought` to a cheap OpenAI model in **one** call and gets back
turn indices to retry.

- **API:** `await judge_match_cots(turn_records, *, client,
  model=DEFAULT_JUDGE_MODEL) -> JudgeResult`. `turn_records[i]` is a
  dict with `match_id`, `game_idx`, `turn`, `pre_tool_thought` — the
  `turn_idx` field is the 0-based position into the list, which is what
  the judge references in its response. `JudgeResult.flagged_turn_indices`
  is a `list[int]` (empty = clean match); `JudgeResult.reasons[idx]` is
  a short quote-from-CoT explaining each flag.
- **Why match-level not per-row.** Amortizes a fixed system prompt
  across N turns. One call for an 8-turn match costs ~$0.014 with
  gpt-5.5 (~$0.0015 with gpt-5.5-mini when access opens up) versus
  ~$0.04 if we judged each row separately. Also lets the judge see
  cross-turn patterns ("multiple consecutive turns reference the
  training framing").
- **Default model:** `gpt-5.5` (the bake-off-winning teacher model).
  Plan v4 spec'd `gpt-5.5-mini` for the cost win, but the project's
  account doesn't currently have access; falls back to gpt-5.5 with a
  note in the module to set `JUDGE_MODEL=gpt-5.5-mini` once available.
- **Structured output.** The judge call uses
  `response_format=json_schema` with a strict schema requiring
  `{flagged_turns: [{turn_idx, reason}]}`. No partial parses, no
  recovery — if the SDK returns malformed JSON we **fail open**
  (return all rows as if the judge passed) rather than risk losing a
  whole match.
- **Truncation policy.** CoTs longer than 6000ch are truncated with a
  visible marker before going into the judge prompt — keeps judge cost
  predictable regardless of how chatty the teacher got.
- **Prompt design.** `JUDGE_SYSTEM_PROMPT` includes 4 positive examples
  (must flag: "Looking at the target action...", "The training section
  indicates...", etc.) and 3 negative examples (must NOT flag: real
  competent VGC reasoning even when confident). Tuned against actual
  bake-off rows.
- **Touches:** OpenAI Chat Completions API only. No `/calc`, no
  inference, no other teacher provider imports.

### `teacher/batch_openai.py` *(implemented — Plan v4)*

OpenAI Batch API plumbing. Used by `batch_runner.py` to issue one batch
cycle per tool-loop iteration across all in-flight matches at once.

- **`BatchTeacherProvider` ABC:** four methods — `build_request`,
  `submit_batch`, `poll`, `fetch_results` (plus `cancel`). Provider-
  agnostic in spirit; OpenAI is the only concrete adapter in v1.
- **`BatchOpenAIProvider`:**
  - `build_request(custom_id, api_messages, tool_choice) -> dict` —
    renders one JSONL line of the batch upload, mirroring exactly
    what `OpenAIProvider._do_turn` sends synchronously (same tools,
    `parallel_tool_calls=False`, omitted `max_tokens` /
    `temperature` because gpt-5.5 rejects those).
  - `submit_batch(requests) -> batch_id` — uploads the JSONL via
    `client.files.create(..., purpose="batch")` and creates the
    batch with `completion_window="24h"`. Returns the batch id.
  - `poll(batch_id) -> BatchPollStatus` — single status tick.
  - `poll_until_done(batch_id, *, poll_interval_seconds,
    max_wait_seconds)` — convenience wrapper; terminates on
    `completed | failed | expired | cancelled`.
  - `fetch_results(batch_id) -> dict[custom_id, response]` —
    downloads the output file and parses it into a dict keyed by
    `custom_id`. Each value mirrors the shape of a sync chat-completions
    response so the orchestrator's response-handler stays symmetric
    with the sync path.
- **Custom ID encoding.** `"{match_id}::g{game}::t{turn}::iter{cycle}"`.
  The cycle component matters because a single (match, game, turn)
  WorkItem produces multiple batch requests across its tool-loop
  lifetime — one per iter.
- **Architectural constraint (documented in module header).** Batch API
  can't span tool-loop iterations within a single line. So each tool-
  loop iter becomes its own batch cycle; all turns at iter=K bundle
  into one upload; calc microservice calls run synchronously between
  cycles. This is why `batch_runner.py` is a per-cycle state machine.
- **Touches:** OpenAI Files API + Batch API. No `/calc`, no `submit_batch`-
  to-`fetch_results` state on this side (state lives in `batch_runner`).

### `batch_runner.py` *(implemented — Plan v4)*

Sibling of `master_pipeline.py`. Owns the per-iteration batch state
machine. Reuses every leaf module from `pipeline/` directly (same
`damage_inferencer`, `threat_matrix`, `prompt_formatting`, etc.) and the
new `teacher/judge.py` + `teacher/batch_openai.py`.

- **Shared prep:** `_prepare_match_turns(match_record, *, format_id,
  calc_base_url, aiohttp_session, seen_keys) -> tuple[list[TurnPrep],
  stats_dict]`. Pure-compute half of the old `process_match` —
  knowledge state seeding, `infer_match_final_bounds`, per-turn threat
  matrix + prompt formatting + ground-truth injection. No LLM calls.
  Used by both sync mode (`master_pipeline.process_match`) and batch
  mode. Yields a `TurnPrep` per identifiable turn (`match_id`,
  `game_idx`, `turn`, `format_id`, `system_prompt`, `user_prompt`
  (plain), `human_action`, `api_messages` (with ground-truth suffix
  ready to send)).
- **`BatchWorkItem`:** the mutable state of one turn across the state
  machine's lifetime. Serializable to JSON for resume. Status
  transitions: `pending → submitted → {pending, committed, failed}`;
  terminal states `committed → {leak_persistent, judge_flagged_persistent,
  written}`. Carries an `active_batch_id` breadcrumb — non-None while
  the item is waiting on a batch response, so `--resume` can find the
  right batch to re-poll on restart.
- **State persistence:** one JSON file per match in `batch_state/
  {match_id}.json` with the full WorkItem list + the match's
  `format_id`. Atomic via `tmp + rename`. A separate global index
  isn't needed in v1 — recovery scans every state file.
- **`run_batch_for_matches(matches, ...)`** — the orchestrator entry
  point:
  1. **Prep.** For each match, either restore from
     `batch_state/{match_id}.json` (if `--resume`) or call
     `_prepare_match_turns` and write the initial state file.
  2. **Resume preamble.** Drain any in-flight batches from a prior
     crash via `_resume_inflight_batches`: group items with
     `status="submitted"` by `active_batch_id`, re-poll each batch,
     fetch results, apply to the items.
  3. **Cycle loop** (up to `MAX_TOOL_ITERATIONS` cycles).
     - Collect items with `status="pending"` and `iter == cycle`.
     - One batch upload per cycle covering every in-flight match.
     - Submit → persist `active_batch_id` per item → poll → fetch
       → apply (`_apply_batch_response`: append assistant msg, run
       sync `/calc` for any calc tool calls, advance iter or commit).
     - Persist after each cycle.
  4. **Regex leak filter.** `detect_oracle_leak` on every committed
     item; drops `leak_persistent` status.
  5. **Match-level judge** (when `--use-judge`, default on). Calls
     `_run_judge_with_retries` from `master_pipeline` — the same
     helper sync mode uses. Judge re-synthesis falls back to the
     sync teacher (batch latency is too high to be useful for
     retry).
  6. **Atomic per-match write.** Each match's surviving rows go to
     the SFT JSONL under `file_lock`; items mark `written`.
- **Touches:** OpenAI Batch API (via `BatchOpenAIProvider`) + `/calc`
  (between cycles, sync) + OpenAI Chat Completions (for the judge +
  judge re-synthesis fallback).

### `master_pipeline.py` *(implemented)*

Orchestrator. Walks `parsed_data/{bo1,bo3}.jsonl`, generates one SFT example
per identifiable turn, writes to `parsed_data/sft_training_data.jsonl`.

**`--mode {sync,batch,hybrid}` dispatcher (Plan v4).** The top-level CLI
flag picks one of three execution strategies:

- **`sync`** *(default)* — per-turn `synthesize_turn()` inline; one match
  at a time (up to `--concurrency`); per-match buffered write so judge
  can run before commit.
- **`batch`** — dispatches to `batch_runner.run_batch_for_matches`. One
  OpenAI Batch cycle per tool-loop iteration; all in-flight turns
  bundle into one batch upload; per-match resume state in
  `batch_state/{match_id}.json`. OpenAI-only in v1.
- **`hybrid`** — runs the first `--hybrid-sync-n` matches sync as a
  quality gate. If `match_rate ≥ --hybrid-min-match-rate` AND
  `leak_rate ≤ --hybrid-max-leak-rate` after the sync portion, the
  remaining matches go through batch. Otherwise the run halts before
  submitting any batch upload — surfaces quality regressions before
  committing thousands of dollars.

All three modes share the same `_prepare_match_turns` (from
`batch_runner.py`), the same regex leak filter (`detect_oracle_leak`),
the same match-level judge (`_run_judge_with_retries`), and the same
per-match atomic write to JSONL under `file_lock`.

**Series-winner-as-P1.** Every saved SFT example is generated from the
perspective of the player who won the series (Bo3: 2 of 3 games; Bo1:
the per-game winner). `flip_match_to_winner` rewrites the entire match
record — `players[0] ↔ players[1]`, every snapshot's `p1 ↔ p2` and
`tailwindP1 ↔ tailwindP2` (plus `tailwindP*TurnsLeft`), every
`events[i]` slot/side fields (per discriminated-union variant), and
per-game `teamSheets.p1 ↔ teamSheets.p2` — so all downstream code
(action extraction, threat matrix, system prompt rendering) reads "p1"
as the winner from this point on.

Why winner-only: the model is trained to play *correctly*, not to mimic
losing patterns at the same Elo. Including some intra-series losses
(games the series-winner lost) preserves variance — in a Bo3 the
series-winner sometimes drops a game to RNG / matchup, and that
reasoning is still high-quality.

**Match-final P1 spreads in the user prompt (Plan v3, six-roster in
Plan v9).** Each turn's user prompt includes a `=== YOUR SPREADS ===`
block listing every tightened EV bound. **Plan v9 renders the full
known roster per side** — both `=== YOUR SPREADS ===` and `=== OPP
SPREADS (inferred) ===` now show all 6 in Bo3 (from the team sheets)
and the revealed-so-far set in Bo1, rather than only the 2 active mons
(a bug fixed in v9). Bounds are computed once per match via
`damage_inferencer.infer_match_final_bounds` — the tightest bounds the
inferencer can extract from the **entire** match's events. The player
knew their own spreads from day one, and the match-final bound is the
closest approximation available at training time. (At deploy time the
operator surfaces exact spreads from the team-builder JSON.) The render
uses one-sided constraints — `Stat ≤N` when only the upper bound has
narrowed, `Stat ≥N` when only the lower bound has narrowed, the
explicit range when both, or `Stat N` when pinned to a single value.
Fully-open stats roll up into a trailing `, others ?`, and a mon with
no constraints at all renders the single fallback state `unknown`
(Plan v9 replaced the old `(no observations yet)` string). The Spread
Rule in the system prompt tells the model how to reason from ranges
(worst case for survival checks, best case for offensive checks).

**Asymmetric threat matrix (Plan v3).** The threat matrix gets the
`(p1_final, p2_running)` pair: match-final-tight P1 bounds (we know
our own team), chronological-loose P2 bounds (we learn about the
opponent through play). Models the realistic information asymmetry —
our damage ranges tighter on our side, wider on theirs.

**Per-match buffered write (Plan v4).** Turns synthesize into a
`match_buffer` list in memory. After the per-turn loop completes,
`_run_judge_with_retries` runs the match-level judge; flagged turns
re-synthesize through the sync teacher (regardless of `--mode`); after
exhausting `--judge-retries`, only the still-flagged turns drop. The
rest of the match commits atomically to the SFT JSONL.

- **`reconstruct_p1_team(games)`** — forward-scan over every snapshot in
  the match (across all Bo3 games), aggregating revealed `item`, `ability`,
  `teraType`, `isTerastallized`, and `revealedMoves` per P1 species. Pads
  each Pokémon's move list to exactly 4 with `"[UNREVEALED_MOVE]"`.
- **`extract_p1_actions(snap_pre, snap_post, events)`** —
  reverse-engineers each P1 active slot's action from the new TurnEvent
  stream:
  - `move` event with `attacker_slot == "p1a"|"p1b"` and `called_via in
    {None, "Sleep Talk"}` → move action. Target is the single
    `defender_slot` if there's one damage hit, `"spread"` if multiple,
    `"self"` if `hits[]` is empty (status / self-target).
  - `switch` event with `side="p1"` and `forced_by is None` → intentional
    switch. Forced switches (Volt Switch redirect / Eject Button / Roar)
    are NOT the human's choice and are excluded.
  - `cant_move` event for the slot → pass.
  - Tera detected by `isTerastallized` going false→true between snapshots.
  - Slot empty / fainted at `snap_pre` → pass.
  - Returns `None` if no choice event exists for an active non-fainted
    slot (likely forced out before acting) → caller skips the turn.
- **Historical context blocks.** `format_user_prompt` composes three
  prompt sections in addition to the current-frame board state:
  - `=== GAME-STATE LEDGER ===` — faints, Tera-used per side, and
    (Plan v9) **all timed field effects** — weather, terrain, Trick
    Room, per-side Tailwind, screens (Reflect / Light Screen / Aurora
    Veil) — each with a `(N turns left)` count. The per-turn header's
    old `Field: weather=…, P1-tailwind=…` line was removed; field state
    now lives **only** here. Also: on-active volatiles (Substitute /
    Encore / Taunt / …) — Encore renders as "Encore-locked (can only
    repeat its last move, or switch)" and Disable as "has a move
    Disabled (...)" (the parser carries only a boolean, no move name —
    a known gap in `notes/TODO.md`); choice locks (move name normalized
    via the dex); recent item events; and a Cumulative damage row
    showing per-active total damage taken across past turns + turns on
    field. Only-when-active rows: empty rows omit.
  - `=== TURN-BY-TURN (game N) ===` — every prior turn's events as
    indented one-liners. No length cap.
  - `=== SERIES STATE (Bo3, game N of M) ===` — Bo3 game ≥ 2 only.
    For each prior game: header (winner, turns, brought rosters, Tera
    resolutions) followed by the full inlined turn-by-turn rollup.
    `# TODO(token-efficient-series-summary)` flags this for a future
    learned summarizer; raw inlining is the right move until that
    lands.
- **Active slot empty-state.** `_format_actives_with_empty_slots`
  emits an explicit `[b] (empty — no Pokémon remaining)` line when
  P1 is down to 1 mon (slot vacant, no living bench replacement).
  The model doesn't have to infer slot vacancy from active-line
  count + bench fainted-counts.
- **Bench rendering perspective.** The parser emits `bench` for both
  sides as the full pre-scanned brought-set (player's own knowledge);
  `format_user_prompt` filters P2's bench by `seenSpecies`
  (chronological reveal — what the player has actually observed of
  the opponent's selection).
- **Unknown-brought-slot placeholders (Plan v9).** Reg I always brings
  4, but the brought-4 is never in the replay data — only usage is. So
  when fewer than 4 brought mons are identifiable, the BENCH section
  adds explicit placeholders to fill out the 4: the player's own side
  reads `(brought, never sent this game ...)`, the opponent's side
  reads `(brought, identity not yet revealed ...)`. There is no
  "not-brought" state.
- **Process loop** per match:
  1. `reconstruct_p1_team` + gather P2 species from snapshots.
  2. `init_knowledge` for both sides at fully-open `[0, 252]` bounds (the
     inferencer's Absolute envelope stays strict-math — no priors, no
     meta seeding; an unconstrained stat reads `unknown`).
  3. Render system prompt (with team block + Masking Rule).
  4. For each `(snap_pre, snap_post, snap_pre.events)` triple in turn
     order:
     - Extract P1's action; skip turn if ambiguous.
     - Generate the Absolute-only threat matrix.
     - Format user prompt — passes the full snapshot history of this
       game (`snapshots_so_far`, `current_idx`) and prior games
       (`prior_games`) for the new historical context blocks.
     - `await teacher_llm.synthesize_turn(...)`.
     - Append the resulting messages as one row to the SFT JSONL.
     - Filter `events` via `events_to_damage_events()` and
       `await damage_inferencer.update_knowledge(...)` to tighten both
       KnowledgeStates for the next iteration.
- **Resumable:** `(match_id, game_index, turn)` keys already present in the
  output JSONL are skipped on rerun.
- **`--dry-run`:** exercises the entire orchestration except the OpenAI
  call; emits a placeholder assistant message containing the actual human
  action. Useful for verifying the pipeline without spending API credits.

#### CLI

```bash
cd pipeline

# Smoke test (no API key needed):
.venv/bin/python master_pipeline.py --limit 1 --dry-run

# Real run on a single Bo3 match (Gemini default since Plan v8, judge on):
GOOGLE_API_KEY=... .venv/bin/python master_pipeline.py --limit 1

# Production: sync mode at concurrency 8 (Gemini is cheap enough not to need batch):
GOOGLE_API_KEY=... .venv/bin/python master_pipeline.py \
    --mode sync --concurrency 8

# OpenAI hybrid mode — first 50 sync as quality gate, rest via Batch:
OPENAI_API_KEY=sk-... .venv/bin/python master_pipeline.py \
    --provider openai --mode hybrid --hybrid-sync-n 50

# Pure batch mode (OpenAI-only in v1); resume in-flight batches:
OPENAI_API_KEY=sk-... .venv/bin/python master_pipeline.py \
    --provider openai --mode batch --resume

# Pick a different provider (sync only for non-OpenAI in v1):
ANTHROPIC_API_KEY=sk-... .venv/bin/python master_pipeline.py \
    --provider anthropic --limit 1

# Plan v8 — disable the fragment-game filter (default drops games with ≤1 turn-pair):
.venv/bin/python master_pipeline.py --min-game-turns 0
```

##### Synthesis flags

| Flag | Default | Notes |
|---|---|---|
| `--input` | `parsed_data/bo3.jsonl` | Match-records JSONL from replay_parser. |
| `--output` | `parsed_data/sft_training_data.jsonl` | Append-only JSONL; per-match atomic commit. |
| `--calc-base-url` | `http://localhost:3000` | |
| `--format-id` | auto from filename | Identifies the format (CTS Bo1 vs OTS Bo3) for prompt rendering. |
| `--limit` | none | Process first N matches (test batch). |
| `--concurrency` | `1` | Sync mode: max matches in flight. Bump to 8 for production. |
| `--dry-run` | off | Skip the LLM call entirely. Works in sync and batch modes. |
| `--provider` | `google` | Choice of `openai` / `anthropic` / `google`. Batch mode requires `openai`. Production default flipped to `google` in Plan v8. |
| `--model` | per-provider default | OpenAI: `gpt-5.5`. Anthropic: `claude-sonnet-4-6`. Google: `gemini-3.1-pro-preview`. Override with `TEACHER_MODEL_{OPENAI,ANTHROPIC,GOOGLE}` env vars. |
| `--leak-retries` | `3` | Regex-leak retries per turn before the row drops. `0` for smoke / measurement runs. |
| `--min-game-turns` | `2` | Plan v8 — drop games whose snapshot count produces fewer than this many turn-pairs. Default removes ghost games + single-decision sweeps. Set 0 to disable. |

##### Judge flags (Plan v4 + v8 provider dispatch)

| Flag | Default | Notes |
|---|---|---|
| `--use-judge / --no-judge` | `--use-judge` | Toggle the per-match model-judge validator. |
| `--judge-provider` | `google` | Plan v8 — which LLM backend the judge uses. Default tracks the production teacher provider (Gemini). Set `openai` for cross-provider sanity checks. |
| `--judge-model` | `gemini-3.1-pro-preview` | Model the judge calls. Default tracks the judge-provider's default (google → gemini-3.1-pro-preview; openai → gpt-5.5). Override via `JUDGE_MODEL` env var. |
| `--judge-retries` | `2` | Re-synthesis passes after the judge flags a turn. On exhaustion, drops only flagged turns. |

##### Batch / hybrid flags (Plan v4)

| Flag | Default | Notes |
|---|---|---|
| `--mode` | `sync` | One of `sync` / `batch` / `hybrid`. |
| `--state-dir` | `pipeline/batch_state/` | Where per-match `{match_id}.json` resume files live. |
| `--poll-interval-seconds` | `60` | Batch status-poll cadence. |
| `--max-cycle-wait-seconds` | `86400` | Per-cycle SLA. Batch API guarantees 24h. |
| `--resume` | off | Re-use prior state files; re-poll in-flight batches before entering the cycle loop. |
| `--hybrid-sync-n` | `50` | Hybrid: matches to run sync as a quality gate. |
| `--hybrid-min-match-rate` | `0.95` | Hybrid halt threshold: minimum `written / attempted` ratio. |
| `--hybrid-max-leak-rate` | `0.02` | Hybrid halt threshold: maximum `dropped / attempted` ratio. |

### Orchestrator data flow

```
raw replay JSON
    → replay_parser  →  per-turn snapshots (with TurnEvent[] events stream)
    │
    │  per match (sync mode — sketch; batch_runner mirrors but with a per-iter
    │  state machine across all matches in flight):
    │
    │    p1_running = init_knowledge(p1_team)    # chronological (diagnostics only)
    │    p2_running = init_knowledge(p2_team)    # chronological (drives matrix p2)
    │    p1_final, _ = await infer_match_final_bounds(games, ...)
    │      (one full-match offline pass — match-final P1 bounds, surfaced in
    │       YOUR SPREADS at every turn AND drives matrix p1 side)
    │
    │    for each turn in order:
    │      → action_extraction.extract_p1_actions(snap_pre, snap_post, events)
    │        (the human's ground-truth play; skip turn if ambiguous)
    │      → threat_matrix.generate(snap_pre, "p1", p1_final, p2_running)
    │        (Absolute-only provable envelope; asymmetric pair gives realistic
    │         tightness on our side, looseness on theirs; unconstrained stats
    │         render `unknown`)
    │      → prompt_formatting.format_user_prompt(snap_pre, ..., snapshots_so_far,
    │                            current_idx, prior_games, game_index, match_format)
    │        (composes board state + GAME-STATE LEDGER + TURN-BY-TURN +
    │         SERIES STATE + YOUR SPREADS [match-final P1] + threat matrix)
    │      → master_pipeline._synthesize_with_leak_retry(teacher, ...)
    │        (tool-call loop + regex leak filter + retry up to --leak-retries)
    │      → buffer the row + the (system_prompt, user_prompt, human_action) ctx
    │      → damage_events = events_to_damage_events(events)
    │      → damage_inferencer.update_knowledge(snap_pre, snap_post, damage_events,
    │                                            p1_running, p2_running)
    │        (tightens both KnowledgeStates atomically + 508-EV constraint —
    │         p2_running is what next turn's matrix reads; p1_running is just
    │         diagnostics now since p1_final supplanted it)
    │
    │    # End of per-turn loop — match still in memory, nothing on disk yet:
    │
    │    → master_pipeline._run_judge_with_retries(match_buffer, turn_contexts, ...)
    │      (one judge call across all CoTs; flagged turns re-synthesize via the
    │       sync teacher; after --judge-retries, drops only flagged turns)
    │
    │    → atomic per-match write of surviving rows under file_lock
```

## Status

| Module | State |
|---|---|
| `replay_parser.py` | Working. Run `python replay_parser.py --help`. Captures per-turn `events` (TurnEvent[]) from `/parse_log` into the JSONL. |
| `canonical_priors.py` | **Removed in Plan v9.** Deleted alongside the `smogon_chaos_*.json` caches when the threat matrix dropped its Probable / canonical-meta track. |
| `damage_inferencer.py` | Working. Dual-state, two-way binary search, atomic apply, 508-EV constraint pass, crit-aware via `/calc isCrit`, multi-hit filter, `events_to_damage_events()` filter for non-own-move callers (Metronome / Copycat / Sketch / Snatch / Me First / Dancer / Instruct / Mirror Move / Assist / Nature Power excluded; Sleep Talk allowed). Plus `infer_match_final_bounds()` for the match-final P1 spreads, and `update_observed_and_speed()` (Plan v9 — per-mon `observed` flag + move-order Speed upper-bound, base stats via `/dex/species`). |
| `threat_matrix.py` | Working. Absolute-only output (Plan v9 removed the Probable track), driven by the asymmetric `(p1_final, p2_running)` knowledge pair. An unconstrained stat renders the word `unknown`. |
| `validate_action_legality.py` | Plan v9. Read-only corpus validator scanning SFT labels for actions that contradict their own turn state (choice-lock, tera-after-used, OTS moveset-membership). Since labels are real human plays, any violation is a data bug. Run `python validate_action_legality.py`. |
| `teacher/base.py` | Provider-agnostic core: `TeacherProvider` ABC, schemas (incl. `submit_decision` tool), prompt templates (6 rules: Masking/OTS, Tool, Threat-Matrix, Spread, Alternatives, Output), ground-truth stripping, `detect_oracle_leak` regex + `extract_pre_tool_thought` helper. Present-tense system prompts. Plan v3 Tool Rule directs calc at hypotheticals the matrix doesn't cover (no per-turn minimum). Plan v9 split the Alternatives Rule — the live-inference rule 5 no longer mandates evaluating an alternative every turn; the mandatory-alternative obligation moved into the synthesis-only `SYNTHESIS_GROUND_TRUTH_SUFFIX` as a conditional (stripped before save), so at live inference the model commits immediately when the matrix settles the question. `# TODO(rlhf-followup)` still flags the prompt-driven alternative evaluation as temporary. |
| `teacher/openai.py` | OpenAI adapter; default model `gpt-5.5`. Per-call timeout 120s, per-turn ceiling 300s. Bake-off tied at 100% / 0% with Gemini. **Available via `--provider openai`** — also required for `--mode batch`. |
| `teacher/anthropic.py` | Anthropic adapter; default model `claude-sonnet-4-6`. Same tool-loop semantics. Bake-off result: 32% near-miss meta-leak rate; not used in production. |
| `teacher/google.py` | Google adapter; default model `gemini-3.1-pro-preview`. Same tool-loop semantics. Bake-off result: clean (0% leak), cheaper unit-cost than OpenAI ($0.04 vs $0.07/row). **Production default since Plan v8** — economics dominated the bake-off-tied-on-quality result given ~$100K GCP credits. |
| `teacher/judge.py` | Plan v4 + Plan v8 provider dispatch. `judge_match_cots(turn_records, *, client, provider, ...) -> JudgeResult` — one model call per match scoring every CoT for meta-leaks. Structured-output schema (`response_format=json_schema` for OpenAI, `response_schema` for Gemini); fail-open on judge errors. Default provider `google`, default model `gemini-3.1-pro-preview` (~$0.014/match). Set `JUDGE_PROVIDER=openai` / `JUDGE_MODEL=gpt-5.5` to revert. |
| `teacher/batch_openai.py` | Plan v4. `BatchTeacherProvider` ABC + `BatchOpenAIProvider`. `build_request` / `submit_batch` / `poll` / `fetch_results` / `cancel`. OpenAI Batch API: 50% off both input and output, 24h SLA, ~10K-line cap per batch. v1 OpenAI-only; Anthropic / Google batch adapters TODO. |
| `team_reconstruction.py` / `action_extraction.py` / `prompt_formatting.py` | Helper modules split out of the orchestrator so `master_pipeline.py` stays focused on CLI + per-match async loop. `bakeoff.py` and `batch_runner.py` import from these directly rather than from `master_pipeline`. `prompt_formatting.format_p1_known_spreads_block` is the renamed/repurposed Plan v3 YOUR SPREADS renderer. |
| `bakeoff.py` | Head-to-head bake-off runner. `--limit N` covers the first N matches in one invocation with one combined summary; `--match-id <substring>` for a single-match smoke run. Reports per-provider cost, tool-call rate, CoT length, action-match rate. Output is one `bakeoff_<provider>.jsonl` per provider, append-mode, resumable on rerun (skips rows already keyed by `(match_id, game_index, turn)`). May 2026 result: OpenAI gpt-5.5 and Google gemini-3.1-pro tied at 100%/0% quality; Gemini chosen for production in Plan v8 on economics. |
| `batch_runner.py` | Plan v4. `_prepare_match_turns` (shared sync/batch prep) + `BatchWorkItem` dataclass + `run_batch_for_matches` (per-cycle state machine) + `_resume_inflight_batches` (drain in-flight batches on `--resume`). Per-match state in `batch_state/{match_id}.json`. v1 OpenAI-only. |
| `master_pipeline.py` | Working. CLI orchestrator with `--mode {sync,batch,hybrid}` dispatcher. `flip_match_to_winner` makes every SFT example come from the series winner's perspective; match-final P1 spreads in user prompt; asymmetric threat matrix; per-format system prompt branch; three historical-context blocks (`GAME-STATE LEDGER`, `TURN-BY-TURN`, `SERIES STATE`); explicit empty-slot annotation; perspective-aware bench rendering. Per-match buffered write + judge integration (Plan v4). `--provider {openai,anthropic,google}` flag (batch is OpenAI-only). `--dry-run` exercises orchestration without LLM cost. |

## Planned follow-up workstreams

Tracked in [`../notes/TODO.md`](../notes/TODO.md) — the master tracker
for all non-shipped work across the project. That includes the
`# TODO(...)` markers in this directory (`rlhf-followup` in
`teacher/base.py`, `token-efficient-series-summary` in
`prompt_formatting.py`), Plan v4 follow-ups (Anthropic / Google batch
adapters, judge-rate dashboard), and long-horizon plans
(selection-model SFT corpus, RLHF). Pipeline-adjacent additions live
there too — don't fork the list here.
