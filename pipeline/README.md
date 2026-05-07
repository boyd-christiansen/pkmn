# pipeline

Atomic Python modules that turn raw Pokémon Showdown replays into SFT-ready
conversational training data.

```
pipeline/
├── replay_parser.py          # raw replay JSONs → per-turn snapshot JSONL
├── canonical_priors.py       # Smogon chaos data lookup (no network)
├── damage_inferencer.py      # binary-search EV bound inference
├── threat_matrix.py          # dual-track damage envelope per turn
├── team_reconstruction.py    # P1 team / brought-set / sheet helpers
├── action_extraction.py      # winner-flip + extract_p1_actions
├── prompt_formatting.py      # 8-section user-prompt composer
├── master_pipeline.py        # CLI orchestrator (the only file allowed
│                             #   to import from every other module here)
├── bakeoff.py                # head-to-head provider runner
└── teacher/                  # provider-agnostic teacher LLM
    ├── __init__.py           # re-exports for `from teacher import ...`
    ├── base.py               # TeacherProvider ABC, schemas, prompts
    ├── openai.py             # OpenAI adapter
    ├── anthropic.py          # Anthropic adapter
    └── google.py             # Google adapter
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

### `canonical_priors.py` *(implemented — Smogon-backed)*

Returns the *probable* spread (EVs / IVs / nature) for a species under
standard meta play. Backs the **Probable Range** track in
`threat_matrix.py`.

- **API:** `get_probable_spread(species, format_id=None) -> ProbableSpread`.
  The result has a `source` field: `"chaos"` | `"curated"` | `"heuristic"`.
- **Lookup order** (each step falls through on miss):
  1. **Smogon Chaos JSON** at `pipeline/data/smogon_chaos_{format_id}.json` —
     loads the per-species `Spreads` dict and picks the most-used
     `"Nature:hp/atk/def/spa/spd/spe"` entry. Real usage data.
  2. **Curated table** of spreads for the top ~40 Reg I species (Calyrex-Shadow
     → Timid SpA/Spe, Amoonguss → Calm HP/SpD, etc.). Used when chaos
     hasn't been bootstrapped or the species genuinely has 0 usage rows.
  3. **Base-stat heuristic** for anything else (offensive vs bulky vs default).
- **Touches:** at runtime, only the local cache file. Network access is
  isolated to `fetch_chaos(...)` (CLI-invoked).

#### Bootstrap

```bash
cd pipeline
.venv/bin/python canonical_priors.py --format-id gen9vgc2026regi
.venv/bin/python canonical_priors.py --format-id gen9vgc2026regibo3
```

Walks back from the current month until a 200 OK chaos file is found
(default: 12 months back), saves to `pipeline/data/smogon_chaos_<format_id>.json`.
Add `--cutoff 1500` to use the high-ladder cut instead of all rated games.

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
- **Touches:** `POST /calc` and `GET /dex/move/:name`. No replay parsing,
  no LLM, no sibling imports.
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
  Nature, so the dual-track inferencer continues to do the heavy lifting
  on spread bounds.

### `threat_matrix.py` *(implemented — dual-track)*

Renders the per-turn damage envelope as a compact human-readable text
block, with **two damage tracks per matchup**:

- **Absolute** — strict mathematical envelope from the live
  `KnowledgeState`s. Wide but provable.
- **Probable (meta)** — single calc result using each Pokémon's
  `canonical_priors` spread. Narrow, fast, and only as good as the prior.

When the canonical prior is clipped from the inferred KnowledgeState
bounds by **≥ 40 EVs** on any relevant stat, the Probable calc is
**skipped entirely** for that line and only the Absolute envelope is
shown, tagged `(off-meta)`. Avoids surfacing a "meta" range we've
already disproven.

- **In:** one snapshot, `p1_side` (which side is "us"), **both**
  `p1_knowledge` and `p2_knowledge`, and an optional `format_id` (forwarded
  to `canonical_priors` so the Probable track uses real Smogon usage data
  when a chaos cache is bootstrapped for that format).
- **Out:** a single string, grouped per attacker. Each attacker block
  shows one line per (move, defender) pair (or a single grouped line
  per spread move). Off-meta lines drop the Probable column. Chip moves
  (max % < 15% across every defender) are collapsed into a single
  footer line per attacker.
- **Touches:** `POST /calc` (2 calls per (move, defender) cell when
  off-meta — abs-low, abs-high; +1 Probable call when in-meta) and
  `GET /dex/move/:name` (cached, fetched for category + target).
  Status moves filtered out.
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
  Heat Wave [spread]: Miraidon 11.7%–13.7%, Iron Bundle 47.8%–62.6%  [Iron Bundle: guaranteed 2HKO]  (off-meta)
  Overheat → Iron Bundle    85.5%–116.8%  | meta 103.0%–122.7%  [guaranteed OHKO]
  …plus 1 chip move(s): Snarl
```

### `teacher/` sub-package *(implemented)*

Provider-agnostic tool-calling loop that elicits a chain-of-thought
JUSTIFYING a known human play. The model has only one output channel
— tool calls — so it can't bypass the calc tool; commits via the
`submit_decision` tool.

- **Layout:** `teacher/base.py` carries the `TeacherProvider` ABC, the
  `calculate_damage` and `submit_decision` tool schemas, the system-
  prompt templates, the cost-table, and a shared `_call_calc` helper.
  Concrete adapters in `teacher/openai.py`, `teacher/anthropic.py`,
  `teacher/google.py` each implement `synthesize_turn` against their
  SDK's tool-call format. `teacher/__init__.py` re-exports the public
  surface so call sites use `from teacher import TeacherProvider,
  OpenAIProvider`.
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
  human's play appended as an `=== EXPERT'S DECISION ===` suffix. The
  returned messages have that suffix **stripped** — saved SFT examples
  show only board state + threat matrix.
- **Tool loop:** up to 6 iterations (`MAX_TOOL_ITERATIONS`). Each iteration:
  call OpenAI, append assistant message, execute any `calculate_damage`
  tool calls via `aiohttp` to `/calc`, append tool messages, loop. Returns
  on a no-tool-call response (final answer) or `None` on iteration cap.
- **Critical Rules in the system prompt** include the **Masking Rule**
  verbatim from the project spec: `[UNREVEALED_MOVE]` slots are presumed
  suboptimal and not to be reasoned about.
- **Touches:** OpenAI Chat Completions API + `/calc` (tool execution).
  No replay parsing, no inference, no canonical-priors imports.

### `master_pipeline.py` *(implemented)*

Orchestrator. Walks `parsed_data/{bo1,bo3}.jsonl`, generates one SFT example
per identifiable turn, writes to `parsed_data/sft_training_data.jsonl`.

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

**Inferred P1 spreads in the user prompt.** Each turn's user prompt
includes a `=== YOUR SPREADS (inferred) ===` block listing every
tightened EV bound for each active P1 mon. The render uses one-sided
constraints — `Stat ≤N` when only the upper bound has narrowed,
`Stat ≥N` when only the lower bound has narrowed, the explicit range
when both, or `Stat N` when pinned to a single value. Fully-open
stats roll up into a trailing `, others ?`. The Spread Rule in the
system prompt tells the model how to reason from ranges (worst case
for survival checks, best case for offensive checks) and
acknowledges that exact values may also be presented at deploy time.

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
  - `=== GAME-STATE LEDGER ===` — faints, Tera-used per side, field +
    pseudo-weather + side conditions with turns-left (singular/plural
    handled), on-active volatiles (Substitute / Encore / Taunt / …),
    choice locks (move name normalized via the dex), recent item
    events, and a Cumulative damage row showing per-active total
    damage taken across past turns + turns on field. Only-when-active
    rows: empty rows omit.
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
- **Process loop** per match:
  1. `reconstruct_p1_team` + gather P2 species from snapshots.
  2. `init_knowledge` for both sides at fully-open `[0, 252]` bounds (per
     project spec: canonical priors live in the threat-matrix Probable
     track only; the inferencer's Absolute track stays strict-math).
  3. Render system prompt (with team block + Masking Rule).
  4. For each `(snap_pre, snap_post, snap_pre.events)` triple in turn
     order:
     - Extract P1's action; skip turn if ambiguous.
     - Generate threat matrix (uses `format_id` for chaos-backed Probable).
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

# Real run on a single Bo3 match (OpenAI default, gpt-5.5):
OPENAI_API_KEY=sk-... .venv/bin/python master_pipeline.py --limit 1

# Pick a different provider:
ANTHROPIC_API_KEY=sk-... .venv/bin/python master_pipeline.py \
    --provider anthropic --limit 1
```

| Flag | Default | Notes |
|---|---|---|
| `--input` | `parsed_data/bo3.jsonl` | Match-records JSONL from replay_parser. |
| `--output` | `parsed_data/sft_training_data.jsonl` | Append-only JSONL. |
| `--calc-base-url` | `http://localhost:3000` | |
| `--format-id` | auto from filename | Drives chaos-priors lookup. |
| `--limit` | none | Process first N matches (test batch). |
| `--concurrency` | `1` | Keep low to respect provider rate limits. |
| `--dry-run` | off | Skip the LLM call entirely. |
| `--provider` | `openai` | Choice of `openai` / `anthropic` / `google`. |
| `--model` | per-provider default | OpenAI: `gpt-5.5`. Anthropic: `claude-sonnet-4-6`. Google: `gemini-3.1-pro-preview`. Override with `TEACHER_MODEL_{OPENAI,ANTHROPIC,GOOGLE}` env vars. |

### Orchestrator data flow

```
raw replay JSON
    → replay_parser  →  per-turn snapshots (with TurnEvent[] events stream)
    │
    │  per match: K1, K2 = init_knowledge(p1_team), init_knowledge(p2_team)
    │             (full open bounds — canonical priors live in Probable track only)
    │
    │  for each turn in order:
    │    → master_pipeline.extract_p1_actions(snap_pre, snap_post, events)
    │      (the human's ground-truth play; skip turn if ambiguous)
    │    → threat_matrix.generate(snap_pre, "p1", K1, K2, format_id=…)
    │      (dual-track Absolute + Probable; off-meta lines drop Probable)
    │    → format_user_prompt(snap_pre, ..., snapshots_so_far, current_idx,
    │                         prior_games, game_index, match_format)
    │      (composes board state + GAME-STATE LEDGER + TURN-BY-TURN +
    │       SERIES STATE + YOUR SPREADS + threat matrix)
    │    → teacher_llm.synthesize_turn(system, user, human_action)
    │      (tool-call loop, returns OpenAI fine-tuning messages)
    │    → write JSONL row
    │    → damage_events = events_to_damage_events(events)  # filter callers + outcomes
    │    → damage_inferencer.update_knowledge(snap_pre, snap_post, damage_events, K1, K2)
    │      (tightens both KnowledgeStates atomically + 508-EV constraint)
```

## Status

| Module | State |
|---|---|
| `replay_parser.py` | Working. Run `python replay_parser.py --help`. Captures per-turn `events` (TurnEvent[]) from `/parse_log` into the JSONL. |
| `canonical_priors.py` | Working. Library + bootstrap CLI. Reads Smogon Chaos JSON when present; falls back to curated table → heuristic. |
| `damage_inferencer.py` | Working. Dual-state, two-way binary search, atomic apply, 508-EV constraint pass, crit-aware via `/calc isCrit`, multi-hit filter, `events_to_damage_events()` filter for non-own-move callers (Metronome / Copycat / Sketch / Snatch / Me First / Dancer / Instruct / Mirror Move / Assist / Nature Power excluded; Sleep Talk allowed). |
| `threat_matrix.py` | Working. Dual-track Absolute + Probable output. When canonical priors are ≥40-EV-clipped by the inferred bounds, the Probable column is dropped and the line tagged `(off-meta)`. Optional `format_id` drives Smogon-backed priors. |
| `teacher/base.py` | Provider-agnostic core: `TeacherProvider` ABC, schemas (incl. `submit_decision` tool), prompt templates (6 rules: Masking/OTS, Tool, Threat-Matrix, Spread, Alternatives, Output), ground-truth stripping. Present-tense system prompts. `# TODO(rlhf-followup)` flags the prompt-driven alternative evaluation as temporary. |
| `teacher/openai.py` | OpenAI adapter; default model `gpt-5.5`. Forces `calculate_damage` on iter 0, `tool_choice=required` thereafter. Default provider. |
| `teacher/anthropic.py` | Anthropic adapter; default model `claude-sonnet-4-6`. Same tool-loop semantics. |
| `teacher/google.py` | Google adapter; default model `gemini-3.1-pro-preview`. Same tool-loop semantics. |
| `team_reconstruction.py` / `action_extraction.py` / `prompt_formatting.py` | Helper modules split out of the orchestrator so `master_pipeline.py` stays focused on CLI + per-match async loop. `bakeoff.py` imports from these directly rather than from `master_pipeline`. |
| `bakeoff.py` | Head-to-head bake-off runner. Reports per-provider cost, tool-call rate, CoT length, action-match rate. |
| `master_pipeline.py` | Working. `flip_match_to_winner` makes every SFT example come from the series winner's perspective; inferred-spread block (one-sided constraints) in user prompt; per-format system prompt branch; three new historical-context blocks (`GAME-STATE LEDGER` with Cumulative damage row, `TURN-BY-TURN`, `SERIES STATE` with full prior-game rollups); explicit empty-slot annotation for last-Pokémon scenarios; perspective-aware bench rendering (P1: full brought-set, P2: chronological via `seenSpecies`); `--provider {openai,anthropic,google}` flag. `--dry-run` exercises orchestration without LLM cost. |

## Planned follow-up workstreams (TODO)

- **`batch_orchestrator.py` — multi-batch tool-use loop** *(next up after
  the bake-off picks a winner)*. New sibling of `master_pipeline.py` that
  parallelises N turns' tool loops by submitting each loop iteration as
  one batch via OpenAI Batch / Anthropic Message Batches / Vertex batch
  prediction. ~50% cost reduction on the full-corpus run (~$1,150 saved
  on ~$2,300). `master_pipeline.py` would gain a
  `--mode {sync,batch,hybrid}` flag — hybrid runs the first ~1K matches
  sync to validate quality, then batches the rest.
- **Token-efficient series-state summarizer.** `format_series_state`
  currently inlines the full prior-game rollup verbatim, which is
  high-fidelity but verbose. A learned (or careful rule-based)
  summarizer that distills "what mattered for THIS turn's decision"
  would conserve attention without losing decision-relevant signal.
  Tracked as `# TODO(token-efficient-series-summary)` in the function.
- **Selection-model SFT corpus** — separate dataset for the team-preview
  4-of-6 pick decision. Walks the same parsed replays, extracts P1's
  brought set per game, generates one selection example per game with
  `{p1_full_6, p2_full_6, format_meta} → {brought: [4 species]}`. Trains
  as its own model (no tactical play, no tool calls); deployed before
  the turn-play model takes over. Lives in a sibling module, not in
  `master_pipeline.py`.
- **Minimax / MCTS distillation for the tool-use loop** — replaces the
  current prompt-driven alternative evaluation (Alternatives Rule in
  the system prompt) with a proper search step that surfaces
  genuinely-competitive alternative plays for the teacher to articulate
  rejection of. The current approach has the teacher cherry-picking
  weak alternatives because it knows the answer. See the
  `# TODO(rlhf-followup)` in `teacher/base.py`.
