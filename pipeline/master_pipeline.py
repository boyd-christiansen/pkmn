"""Orchestrator: replay JSONs in -> SFT-ready JSONL out.

Wires the atomic modules together:

    raw replay JSON
        -> replay_parser.parse(...)            # BoardState[] per turn
        -> threat_matrix.evaluate(state, ...)  # per-state threat summary
        -> teacher_llm.generate(state, label, threats, ...)  # CoT messages
        -> JSONL row written to disk

The orchestrator owns: file I/O, batching/concurrency, retries on teacher
failures, dedup of seen (replay_id, turn_idx) keys, and final dataset
formatting (OpenAI / Anthropic conversational schema).

Isolation contract:
    The only file allowed to import from all other pipeline modules. The
    inverse — pipeline modules importing from master_pipeline — is forbidden.
"""
