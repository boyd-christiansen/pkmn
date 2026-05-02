# pipeline

Atomic Python modules that turn raw Pokémon Showdown replays into SFT-ready
conversational training data. `replay_parser.py`, `damage_inferencer.py`, and
`threat_matrix.py` are implemented; `teacher_llm.py` and `master_pipeline.py`
are still stubs.

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

### `damage_inferencer.py` *(implemented)*

Tightens an `OpponentKnowledgeState` (per-Pokémon `min_evs` / `max_evs` boxes
in [0, 252]^6) by binary-searching observed damage events from the calc
microservice. Damage % is monotonically decreasing in defender HP/Def/SpD
EVs, so a textbook binary search works for each stat.

- **In:** turn snapshots, a list of `DamageEvent` records (one per
  damage-dealing hit on a tracked opponent), and the current
  `OpponentKnowledgeState`.
- **Out:** the same dict, mutated in place — `min_evs` ratcheted up,
  `max_evs` ratcheted down where the new observation allows.
- **Touches:** `POST /calc` (binary-search probes) and `GET /dex/move/:name`
  (to identify the move's category and pick Def vs SpD). No LLM, no replay
  parsing, no imports from sibling pipeline modules.
- **Fuzzy HP**: spectator-visible HP percentages are widened by ±0.9% before
  matching against calc rolls, to absorb the 1% rounding on opponent HP bars.
- **Cross-stat coupling**: when bounding one stat, other unknowns are held
  at the *least restrictive* edge of their current bounds, so we never
  over-tighten because of joint uncertainty.
- **KOs**: a KO observation only constrains `max_evs` (the defender can't be
  bulky enough that even max roll wouldn't KO); `min_evs` is left alone.
- **Caveat**: the algorithm assumes the attacker's offensive EVs are known
  (or held constant). If the attacker is also an opponent with wide EV
  bounds, the inference will be conservative or wrong-tight depending on the
  median assumption. The orchestrator should plug in a sensible attacker
  prior (e.g. lock common offensive Pokémon to "max Atk/SpA") before
  invoking this module.

### `threat_matrix.py` *(implemented)*

Renders the per-turn damage envelope as a compact human-readable text block,
ready to inline into an SFT prompt or return as a tool-call response.

- **In:** one snapshot, `p1_side` indicating which side is "us", and the
  `OpponentKnowledgeState`.
- **Out:** a single string, one section for outgoing (us → opp) and one for
  incoming (opp → us). Each line: `move | low <calc range> (KO chance) | high <calc range> (KO chance)`.
- **Touches:** `POST /calc` and `GET /dex/move/:name`. Status moves are
  filtered out before any calc request fires.
- **Volatile state**: every `/calc` payload carries the attacker's and
  defender's `status` and `boosts` from the snapshot (so e.g. an Intimidated
  attacker shows the `-1 Atk` damage hit).
- **Bound semantics**:
  - INCOMING: low uses opp `min_atk` / `min_spa`; high uses opp `max_atk` / `max_spa`.
  - OUTGOING: low uses opp `max_hp` + `max_def` / `max_spd` (bulkiest); high uses opp `min_hp` + `min_def` / `min_spd` (frailest).

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
    → replay_parser              # BoardState[] per turn
    │
    │  for each turn (in order):
    │    → damage_inferencer.update_knowledge(snap_pre, snap_post, events, K)
    │      (tightens K = OpponentKnowledgeState in place)
    │    → threat_matrix.generate(snap, p1_side, K)
    │      (per-turn threat block, low/high envelopes)
    │    → teacher_llm.generate(snap, label, threats, K)
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
| `damage_inferencer.py` | Working. Library module: `update_knowledge(...)` + `init_knowledge(...)`. |
| `threat_matrix.py` | Working. Library module: `await generate_threat_matrix(snapshot, p1_side, knowledge)`. |
| `teacher_llm.py` | Stub. Prompt design + tool-calling loop; the alignment-quality crux of the project. |
| `master_pipeline.py` | Stub. Wires the above four together; build last. |
