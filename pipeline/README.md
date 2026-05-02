# pipeline

Atomic Python modules that turn raw Pokémon Showdown replays into SFT-ready
conversational training data. `replay_parser.py`, `canonical_priors.py`,
`damage_inferencer.py`, and `threat_matrix.py` are implemented; `teacher_llm.py`
and `master_pipeline.py` are still stubs.

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

### `canonical_priors.py` *(implemented)*

Returns the *probable* spread (EVs / IVs / nature) for a species under
standard meta play. Backs the **Probable Range** track in
`threat_matrix.py`. Mock implementation today, will be wired to real
Smogon usage stats later.

- **API:** `get_probable_spread(species, format_id=None) -> ProbableSpread`.
- **Lookup order:**
  1. Hand-curated table of spreads for the top ~40 Reg I species
     (Calyrex-Shadow → Timid SpA/Spe, Urshifu → Adamant Atk/Spe,
     Amoonguss → Calm HP/SpD, etc.).
  2. Base-stat heuristic for species not in the table:
     - `max(atk, spa) ≥ 110` → max attacking stat + Spe, Jolly/Timid if
       fast, Adamant/Modest otherwise.
     - `hp ≥ 95 ∧ spe ≤ 70` → bulky support: max HP + heavier defensive
       side (Calm or Bold).
     - else → mixed offensive default.
  3. Generic balanced base stats for completely unknown species.
- **Touches:** nothing external. Pure data lookup.

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
- **Two-way binary search.** For each non-status, non-crit event we run
  six binary searches per damage event:
  - DEFENDER `min_def`, `max_def`, `min_hp`, `max_hp`
  - ATTACKER `min_off`, `max_off` (`atk` if physical, `spa` if special)

  Cross-side coupling is handled with interval arithmetic — when
  searching the defender, the attacker is held at its *least restrictive*
  current bound, and vice versa. (E.g. to find the largest Def consistent
  with the observation, hold attacker at its `max_off`; to find the
  smallest Def, hold attacker at its `min_off`.) Symmetric for HP.
- **Atomic application.** All six searches use **pre-update** bounds, then
  results apply at the end. This makes the update order-independent —
  no risk of over-tightening one side because the other has already
  shifted.
- **Fuzzy HP:** ±0.9% tolerance to absorb the 1% rounding on
  spectator-visible HP bars.
- **Skip rules:** `is_crit=True` (would need `/calc` `isCrit` support, see
  follow-up), Status-category moves, missing slots, observations with no
  consistent EV in `[0, 252]` (data inconsistency — bound left untouched).
- **Action_log producer (status: not built yet).** `update_knowledge`
  consumes a `list[DamageEvent]` per turn but `/parse_log` doesn't yet
  emit them — extending the Node endpoint to return per-turn protocol
  events (`|move|`, `|-damage|`, `|-crit|`) and threading them through
  `replay_parser.py` is the next blocking task before the orchestrator
  can run end-to-end.

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

- **In:** one snapshot, `p1_side` (which side is "us"), and **both**
  `p1_knowledge` and `p2_knowledge`.
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

### `teacher_llm.py`

Drives a frontier model through a tool-calling loop to synthesise CoT reasoning
*toward* the known label (the play the human actually made).

- **In:** `BoardState`, the player's actual decision (the label), and the
  pre-computed `ThreatMatrix`.
- **Out:** a list of `(role, content)` messages forming a single SFT example
  — system prompt, board-state user turn, assistant CoT (with interleaved calc
  tool calls + results), final action.
- **Touches:** the only file allowed to talk to the frontier LLM. Never parses
  replays or calls the calc service directly.

### `master_pipeline.py`

Orchestrator. Wires everything together:

```
raw replay JSON
    → replay_parser  →  per-turn snapshots
    │
    │  per match: init  K1, K2 = init_knowledge(p1_team), init_knowledge(p2_team)
    │
    │  for each turn (in order):
    │    → damage_inferencer.update_knowledge(snap_pre, snap_post, events, K1, K2)
    │      (tightens BOTH K1 and K2 in place — two-way binary search)
    │    → threat_matrix.generate(snap, p1_side, K1, K2)
    │      (dual-track: Absolute envelope + Probable meta-spread range,
    │       flagged [PRIOR CONTRADICTED] when canonical priors disagree)
    │    → teacher_llm.generate(snap, label, threats, K1, K2)
    │      (CoT messages with tool calls)
    │
    → JSONL row written to disk
```

Owns: file I/O, batching/concurrency, retries on teacher failures, dedup of
seen `(replay_id, turn_idx)` keys, and final dataset formatting (OpenAI /
Anthropic conversational schema).

## Status

| Module | State |
|---|---|
| `replay_parser.py` | Working. Run `python replay_parser.py --help`. |
| `canonical_priors.py` | Working. Library: `get_probable_spread(species)` — mock heuristic until real Smogon data lands. |
| `damage_inferencer.py` | Working. Library: dual-state, two-way binary search. End-to-end blocked on per-turn `action_log` production (see follow-up below). |
| `threat_matrix.py` | Working. Library: dual-track Absolute + Probable output with `[PRIOR CONTRADICTED]` flag. |
| `teacher_llm.py` | Stub. Prompt design + tool-calling loop; the alignment-quality crux of the project. |
| `master_pipeline.py` | Stub. Wires the above five together; build last. |

## Required follow-up: `action_log` from `/parse_log`

`damage_inferencer.update_knowledge` consumes `list[DamageEvent]` per turn,
but `/parse_log` doesn't emit them yet. Building a real orchestrator requires
extending the Node endpoint to surface per-turn protocol events (`|move|`,
`|-damage|`, `|-crit|`, `|-supereffective|`, weather/end-of-turn markers) and
threading them through `replay_parser.py` into `parsed_data/{bo1,bo3}.jsonl`.
That's a separate task — the modules in this PR ship as libraries verified
against synthetic events.
