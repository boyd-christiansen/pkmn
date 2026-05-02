# pipeline

Atomic Python modules that turn raw Pokémon Showdown replays into SFT-ready
conversational training data. `replay_parser.py` is implemented; the other
three modules are still stubs (docstring + signature only).

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

### `threat_matrix.py`

Evaluates a `BoardState` and queries the calc microservice to summarise the
damage landscape.

- **In:** `BoardState` + URL of [`calc_microservice`](../calc_microservice/).
- **Out:** `ThreatMatrix` — for every (attacker, attacker_move, defender)
  triple on the field, a min/max damage range + KO chance + relevant
  conditional flags (Tera, weather, terrain, screens, items).
- **Touches:** HTTP-calls calc microservice. No replay parsing, no LLM.
- **Consumed two ways:** rendered as a compact text block to inline into the
  SFT prompt, or returned as a tool-call response to the teacher LLM.

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
    → replay_parser.parse(...)              # BoardState[] per turn
    → threat_matrix.evaluate(state, ...)    # per-state threat summary
    → teacher_llm.generate(state, label, threats, ...)
    → JSONL row written to disk
```

Owns: file I/O, batching/concurrency, retries on teacher failures, dedup of
seen `(replay_id, turn_idx)` keys, and final dataset formatting (OpenAI /
Anthropic conversational schema).

## Status

| Module | State |
|---|---|
| `replay_parser.py` | Working. Run `python replay_parser.py --help`. |
| `threat_matrix.py` | Stub. Next up — straightforward now that the per-turn snapshot shape is concrete. |
| `teacher_llm.py` | Stub. Prompt design + tool-calling loop; the alignment-quality crux of the project. |
| `master_pipeline.py` | Stub. Wires the above three together; build last. |
