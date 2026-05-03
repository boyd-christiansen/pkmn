# pipeline

Atomic Python modules that turn raw Pokémon Showdown replays into SFT-ready
conversational training data. **All six modules are implemented**:
`replay_parser.py`, `canonical_priors.py`, `damage_inferencer.py`,
`threat_matrix.py`, `teacher_llm.py`, and `master_pipeline.py`.

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

This rule is what lets us swap out e.g. the teacher LLM (GPT-4o → Claude →
o-series) or the calc engine without rewriting the whole pipeline.

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
  emit one `DamageEvent` per hit. The inferencer counts same
  `(attacker_slot, move_name, defender_slot)` tuples per turn and **skips
  all of them** if the count > 1 — supporting them properly would need
  `/calc` to accept a `hits` field and the events to be aggregated upstream.
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
- **Source of `action_log`:** `/parse_log` now emits per-turn `actionLog`
  arrays inline with each snapshot. The orchestrator pairs
  `snapshot[N].actionLog` with `snapshot_pre = snapshot[N]`,
  `snapshot_post = snapshot[N+1]`.

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

If the canonical prior falls outside the Absolute box for any relevant
stat (HP, off-stat, def-stat), the line is flagged
`[PRIOR CONTRADICTED]` so the LLM can spot off-meta opponents.

- **In:** one snapshot, `p1_side` (which side is "us"), **both**
  `p1_knowledge` and `p2_knowledge`, and an optional `format_id` (forwarded
  to `canonical_priors` so the Probable track uses real Smogon usage data
  when a chaos cache is bootstrapped for that format).
- **Out:** a single string, one section for outgoing (us → opp) and one
  for incoming (opp → us).
- **Touches:** `POST /calc` (3 calls per matchup-move: abs-low, abs-high,
  probable) and `GET /dex/move/:name` (cached). Status moves filtered out.
- **Volatile state:** status, boosts, weather, terrain, side conditions,
  Tera state — all threaded into every payload from the snapshot.

#### Output line format

```
[opp Flutter Mane] vs [us Urshifu]:
  moonblast              Absolute: 105.2%–160.8%  |  Probable (meta): 135.4%–150.1%  (guaranteed OHKO)
  shadow ball            Absolute: 47.1%–82.0%   |  Probable (meta): 62.1%–73.5%   (guaranteed 2HKO)  [PRIOR CONTRADICTED]
```

### `teacher_llm.py` *(implemented)*

Drives a frontier OpenAI model through a tool-calling loop, eliciting a
chain-of-thought that JUSTIFIES a known human play. Returns OpenAI-fine-
tuning-ready conversation messages.

- **API:** `await synthesize_turn(system_prompt, user_prompt, human_action,
  *, tools_allowed=True, ...)` → `list[dict] | None`.
- **Tools exposed to the LLM:** `calculate_damage` — JSON schema mirrors
  `/calc`'s payload (Pokémon × move × field, with `isCrit` / `evs` / `nature`
  / etc. all optional).
- **Final-output schema:** strict JSON schema enforced via OpenAI's
  `response_format: json_schema`. Shape: `{ pre_tool_thought: string,
  action: { slot_1, slot_2 } }`. Each slot has
  `{action_type: "move"|"switch"|"pass", move?, target?, tera?, switch_to?}`.
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

- **`reconstruct_p1_team(games)`** — forward-scan over every snapshot in
  the match (across all Bo3 games), aggregating revealed `item`, `ability`,
  `teraType`, `isTerastallized`, and `revealedMoves` per P1 species. Pads
  each Pokémon's move list to exactly 4 with `"[UNREVEALED_MOVE]"`.
- **`extract_p1_actions(snap_pre, snap_post, action_log)`** —
  reverse-engineers each P1 active slot's action:
  - Damage move → look in `action_log` (`attacker_slot == "p1a"|"p1b"`).
    Multiple `defender_slot`s on one attacker = `"spread"` target.
  - Tera detected by `isTerastallized` going false→true between snapshots.
  - Switch detected by species change at the same slot.
  - Status move detected by exactly-1 new entry in `revealedMoves` between
    snapshots when neither damage event nor switch was observed.
  - Returns `None` if any slot is ambiguous → caller skips the turn.
- **Process loop** per match:
  1. `reconstruct_p1_team` + gather P2 species from snapshots.
  2. `init_knowledge` for both sides at fully-open `[0, 252]` bounds (per
     project spec: canonical priors live in the threat-matrix Probable
     track only; the inferencer's Absolute track stays strict-math).
  3. Render system prompt (with team block + Masking Rule).
  4. For each `(snap_pre, snap_post, snap_pre.actionLog)` triple in turn
     order:
     - Extract P1's action; skip turn if ambiguous.
     - Generate threat matrix (uses `format_id` for chaos-backed Probable).
     - Format user prompt (snapshot + threat matrix).
     - `await teacher_llm.synthesize_turn(...)`.
     - Append the resulting messages as one row to the SFT JSONL.
     - `await damage_inferencer.update_knowledge(...)` to tighten both
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

# Real run on a single Bo3 match:
OPENAI_API_KEY=sk-... .venv/bin/python master_pipeline.py --limit 1

# Full run (default model gpt-4o, concurrency 1):
OPENAI_API_KEY=sk-... .venv/bin/python master_pipeline.py
```

| Flag | Default | Notes |
|---|---|---|
| `--input` | `parsed_data/bo3.jsonl` | Match-records JSONL from replay_parser. |
| `--output` | `parsed_data/sft_training_data.jsonl` | Append-only JSONL. |
| `--calc-base-url` | `http://localhost:3000` | |
| `--format-id` | auto from filename | Drives chaos-priors lookup. |
| `--limit` | none | Process first N matches (test batch). |
| `--concurrency` | `1` | Keep low to respect OpenAI rate limits. |
| `--dry-run` | off | Skip the OpenAI call. |
| `--model` | `gpt-4o` | Or `TEACHER_MODEL` env var. |

### Orchestrator data flow

```
raw replay JSON
    → replay_parser  →  per-turn snapshots (with actionLog)
    │
    │  per match: K1, K2 = init_knowledge(p1_team), init_knowledge(p2_team)
    │             (full open bounds — canonical priors live in Probable track only)
    │
    │  for each turn in order:
    │    → master_pipeline.extract_p1_actions(snap_pre, snap_post, action_log)
    │      (the human's ground-truth play; skip turn if ambiguous)
    │    → threat_matrix.generate(snap_pre, "p1", K1, K2, format_id=…)
    │      (dual-track Absolute + Probable, [PRIOR CONTRADICTED] flagged)
    │    → teacher_llm.synthesize_turn(system, user, human_action)
    │      (tool-call loop, returns OpenAI fine-tuning messages)
    │    → write JSONL row
    │    → damage_inferencer.update_knowledge(snap_pre, snap_post, events, K1, K2)
    │      (tightens both KnowledgeStates atomically + 508-EV constraint)
```

## Status

| Module | State |
|---|---|
| `replay_parser.py` | Working. Run `python replay_parser.py --help`. Captures per-turn `actionLog` from `/parse_log` into the JSONL. |
| `canonical_priors.py` | Working. Library + bootstrap CLI. Reads Smogon Chaos JSON when present; falls back to curated table → heuristic. |
| `damage_inferencer.py` | Working. Dual-state, two-way binary search, atomic apply, 508-EV constraint pass, crit-aware via `/calc isCrit`, multi-hit filter. |
| `threat_matrix.py` | Working. Dual-track Absolute + Probable output with `[PRIOR CONTRADICTED]` flag. Optional `format_id` drives Smogon-backed priors. |
| `teacher_llm.py` | Working. OpenAI tool-use loop with `calculate_damage` + structured `{pre_tool_thought, action}` output. Strips ground truth from saved messages. |
| `master_pipeline.py` | Working. Run `python master_pipeline.py --help`. `--dry-run` exercises orchestration without OpenAI cost. |
