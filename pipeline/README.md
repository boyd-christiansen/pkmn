# pipeline

Atomic Python modules that turn raw Pokémon Showdown replays into SFT-ready
conversational training data. Currently all stubs — see each module's
docstring for the contract it will implement.

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

### `replay_parser.py`

Stitches Bo3 series and extracts turn-by-turn `BoardState` snapshots from raw
Showdown logs.

- **In:** raw replay JSON dicts (as produced by `data_scraper`).
- **Out:** list of per-turn `BoardState` objects (active Pokémon for both
  players, HP, status, boosts, items, known moves, weather, terrain, side
  conditions, Tera state) plus the actual decision the player made that turn.
- **Touches:** nothing external. Pure deterministic transformation.

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

All four files are stubs with finalised docstrings. Implementation order:

1. `replay_parser.py` — biggest unknown; everything downstream depends on its
   `BoardState` shape.
2. `threat_matrix.py` — straightforward once `BoardState` is concrete.
3. `teacher_llm.py` — prompt design + tool-calling loop; the alignment-quality
   crux of the project.
4. `master_pipeline.py` — last; just wires the above three together.
