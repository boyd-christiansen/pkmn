"""Drive a frontier model through a tool-calling loop to synthesise CoT reasoning.

Inputs:
    A `BoardState`, the player's actual decision that turn (the "label"), and
    the `ThreatMatrix` for that state. The teacher's job is to reason
    *toward* the known label — producing the chain of thought a strong human
    would have used to arrive at the same play.

Outputs:
    A list of `(role, content)` messages forming a single SFT example:
    system prompt + board-state user turn + assistant CoT (with interleaved
    calc tool calls and results) + final action.

Isolation contract:
    The only thing in the pipeline that talks to a frontier LLM. Receives
    pre-computed BoardState + ThreatMatrix; never parses replays or calls
    the calc service directly. Swap teacher model (GPT-4o → Claude → o-series)
    here and nothing else changes.
"""
