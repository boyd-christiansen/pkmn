"""Evaluate a board state and produce a max-damage summary via the calc API.

Inputs:
    A `BoardState` from replay_parser, plus the URL of the calc microservice.

Outputs:
    A `ThreatMatrix` summarising, for every (attacker, attacker_move,
    defender) triple on the field, the min/max damage range, KO chance, and
    relevant conditional flags (Tera, weather, terrain, screens, items).

    Designed for two consumption patterns:
      1. Render as a compact text block to inline into the SFT prompt.
      2. Hand back to the teacher LLM as a tool-call response.

Isolation contract:
    Only HTTP-calls the calc microservice. No replay parsing, no LLM. Given
    the same BoardState + same calc service version, output is deterministic.
"""
