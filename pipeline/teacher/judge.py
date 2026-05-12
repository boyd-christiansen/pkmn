"""Match-level model-judge validator (plan v4, workstream 1).

The v3 regex `detect_oracle_leak` catches the strongest phrases
("training section", "the target action", "expert's decision") but
misses softer meta-references ("clearly the right move", "the data
points to", first-person-knowledge-as-fact constructions). This module
adds a second-line filter that submits an entire match's worth of CoTs
in ONE call to a cheap model (gpt-5.5-mini by default) and gets back a
list of turn indices to retry.

Why per-match, not per-row:
  - Amortizes a fixed prompt overhead across N turns.
  - One call to score 8 turns costs ~$0.0015 vs $0.005 × 8 = $0.04 per
    row. ~25× cheaper.
  - The judge sees more context — if multiple turns in a row reference
    the training framing, that pattern is a stronger signal than each
    in isolation.

The judge is provider-agnostic in spirit but only OpenAI is wired today
(matching the post-bake-off standardization on OpenAI). To swap to a
different judge provider, replace the client kwarg with one of compatible
shape and adjust the structured-output call path.

Contract callers care about (`master_pipeline._run_judge_with_retries`):
  - `judge_match_cots(records, ...) -> JudgeResult` with
    `flagged_turn_indices` and `reasons`.
  - `extract_pre_tool_thought(messages) -> str | None` (re-exported from
    `teacher.base`) for parsing saved rows.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from .base import (
    PRICE_PER_M_TOKENS,
    estimate_cost_usd,
    extract_pre_tool_thought,
)

# -----------------------------------------------------------------------------
# Constants / defaults
# -----------------------------------------------------------------------------

# NOTE: Plan v4 spec'd `gpt-5.5-mini` for ~$0.0015/match, but the project
# account doesn't currently have access to that tier — the OpenAI API
# returns 404 model_not_found. We fall back to `gpt-5.5` (the teacher
# model the bake-off used) which costs ~20x more per token but is still
# only ~$0.04/match — a rounding error against the $0.07/turn synthesis
# cost. If mini access opens up, set `JUDGE_MODEL=gpt-5.5-mini` in the
# env to switch over without a code change.
DEFAULT_JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-5.5")
DEFAULT_JUDGE_RETRIES = 2
DEFAULT_JUDGE_TIMEOUT = 60.0

# Per-call cap on CoT chars rendered into the prompt. The model commits in
# ~500–4000ch of pre_tool_thought; we truncate exceptionally long ones to
# keep judge cost predictable. Truncation marker tells the judge we cut it.
_MAX_COT_CHARS_PER_TURN = 6000
_TRUNC_MARKER = "\n…[truncated for judge]…"


# -----------------------------------------------------------------------------
# Result dataclass
# -----------------------------------------------------------------------------


@dataclass
class JudgeResult:
    """Output of one judge call.

    `flagged_turn_indices` are 0-based positions into the `turn_records`
    list passed to `judge_match_cots` — NOT the original game turn
    numbers. Callers translate back to (game_idx, turn) via the records.

    `reasons[i]` holds a short phrase describing why turn i was flagged
    (for logging / debugging). Always present for every flagged index;
    never present for un-flagged ones.

    `error` is non-None if the judge call failed (network, parse, or
    schema-mismatch). The caller is expected to fail-open: write all
    rows as if the judge had passed them. Better to ship a few possibly-
    leaky rows than to drop a whole match for an infrastructure hiccup.
    """
    flagged_turn_indices: list[int]
    reasons: dict[int, str] = field(default_factory=dict)
    raw_response: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    elapsed_seconds: float = 0.0
    error: str | None = None


# -----------------------------------------------------------------------------
# Structured-output schema (OpenAI response_format json_schema)
# -----------------------------------------------------------------------------

_JUDGE_RESPONSE_SCHEMA: dict[str, Any] = {
    "name": "match_judge_verdict",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "flagged_turns": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "turn_idx": {
                            "type": "integer",
                            "description": "0-based index into the input turn list.",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Short phrase identifying the leak (≤120 chars).",
                        },
                    },
                    "required": ["turn_idx", "reason"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["flagged_turns"],
        "additionalProperties": False,
    },
}


# -----------------------------------------------------------------------------
# Prompt template
# -----------------------------------------------------------------------------
#
# Positive examples are real Anthropic-flagged quotes from the v3 bake-off.
# Negative examples are real OpenAI quotes — we want the judge to LEARN
# that confident, well-reasoned CoT is fine even when it lands on the
# same action the human picked. The danger is meta-awareness, not
# correctness.

JUDGE_SYSTEM_PROMPT = """You are a training-data hygiene auditor for a \
Pokémon VGC teacher model. Each turn record below is the \
`pre_tool_thought` of a single turn — a short chain-of-thought (CoT) \
the teacher LLM wrote BEFORE committing its action.

During synthesis the teacher LLM saw a "TRAINING-MODE TARGET" section \
of the prompt that revealed which action it should commit. The trained \
STUDENT model will NOT see this section. So the pre_tool_thought MUST \
read as if the teacher arrived at its action independently from the \
board state, threat matrix, and calc results.

FLAG a turn if its pre_tool_thought contains ANY meta-reference to:
  • the existence of a target / answer / oracle / expert / ground truth
  • the training framing itself ("training section", "training-mode", \
"target action", "target field", "target move", "the expected play", \
"the indicated move", "as instructed", "as given")
  • first-person knowledge of the correct answer ("I know we should X", \
"given the answer", "the correct play is X", "the answer says X", \
phrased as fact rather than derivation)
  • soft assertions of confidence-without-derivation: "clearly the \
right move is X" / "X is obviously correct" / "the data points to X" \
WHERE X is asserted without being derived from the board state, threat \
matrix, or calc results in the same paragraph

DO NOT FLAG ordinary strategic reasoning, even if confident. The \
following are all FINE:
  • "Calyrex outspeeds and OHKOs after Tera — locking Astral Barrage."
  • "The threat matrix shows Lunala can OHKO with +1; Protect is forced."
  • "Switching Incineroar in absorbs Fake Out and gives Intimidate."

POSITIVE EXAMPLES (must be flagged):
  • "Looking at the target action, we should Trick Room first."
  • "The training section indicates a switch to Incineroar."
  • "Given the expected play, I'll commit Protect on slot 2."
  • "The answer is to Tera Calyrex-Shadow."
  • "The target says we should use Glacial Lance on Kyogre."

NEGATIVE EXAMPLES (must NOT be flagged):
  • "Calyrex-Shadow's Astral Barrage cleanly OHKOs both opposing \
actives at the upper range; Tera Ghost makes Lunala's Moongeist Beam \
a 3HKO. Locking in Astral Barrage + Tera on slot 1, Trick Room on \
slot 2 to flip speed."
  • "Switching Incineroar in absorbs the predicted Fake Out and pivots \
into Intimidate, which the threat matrix shows tips Urshifu's Surging \
Strikes out of 2HKO range. Calyrex stays in for Glacial Lance spread."
  • "Spore on Kyogre — even at full HP Water Spout maxes only ~50% on \
Amoonguss, and putting Kyogre asleep cuts the Trick Room threat \
entirely. Leech Seed from Calyrex chips Kyogre while healing us."

Output STRICT JSON matching the schema. If all turns are clean, return \
an empty `flagged_turns` array. Be conservative — only flag when you \
can quote the specific phrase that violates the rules. Include the \
quoted phrase in the `reason` field."""

JUDGE_USER_TEMPLATE = """Match: {match_id}  ({n_turns} turns)

Each block below is one turn's pre_tool_thought verbatim. Flag any that \
violate the hygiene rules. `turn_idx` in your response refers to the \
0-indexed position in this list (NOT the in-game turn number).

{turn_block}
"""


def _render_turn_block(records: list[dict[str, Any]]) -> str:
    """Render the per-turn CoTs as a single text block for the judge prompt.

    Each turn gets a clearly-bounded section so the judge can refer to
    `turn_idx=K` unambiguously. Long CoTs are truncated to keep the
    judge prompt under O(20K) tokens regardless of how chatty the
    teacher got.
    """
    parts: list[str] = []
    for i, rec in enumerate(records):
        cot = rec.get("pre_tool_thought") or ""
        if len(cot) > _MAX_COT_CHARS_PER_TURN:
            cot = cot[:_MAX_COT_CHARS_PER_TURN] + _TRUNC_MARKER
        gi = rec.get("game_idx", "?")
        tn = rec.get("turn", "?")
        parts.append(f"[turn_idx={i}  game={gi}  turn={tn}]\n{cot}")
    return "\n\n".join(parts)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


async def judge_match_cots(
    turn_records: list[dict[str, Any]],
    *,
    client: AsyncOpenAI,
    model: str = DEFAULT_JUDGE_MODEL,
    timeout: float = DEFAULT_JUDGE_TIMEOUT,
) -> JudgeResult:
    """Score a match's CoTs in one call; return turns that should be retried.

    `turn_records` is a list of dicts, each shaped:
        {"turn_idx": int,         # 0-based position; used by the judge
         "match_id": str,
         "game_idx": int,
         "turn": int,
         "pre_tool_thought": str}

    `turn_idx` ordering must match the list order — we trust the index
    field for logging but use list position for retry routing.

    Returns a `JudgeResult`. On any error the caller is expected to
    fail-open and write all rows as if the judge passed — a single
    judge hiccup must not lose a match's worth of work.
    """
    t0 = time.monotonic()
    if not turn_records:
        # Defensive: no work to do. Don't burn an API call.
        return JudgeResult(flagged_turn_indices=[])

    match_id = turn_records[0].get("match_id", "?")
    turn_block = _render_turn_block(turn_records)
    user_msg = JUDGE_USER_TEMPLATE.format(
        match_id=match_id, n_turns=len(turn_records), turn_block=turn_block,
    )

    # NOTE: The current OpenAI reasoning-class models (gpt-5.5 family)
    # reject `max_tokens` and `temperature` outright — only the legacy
    # chat-completions models accept them. We omit both rather than
    # branch on model name, mirroring what the production teacher
    # adapter in `teacher/openai.py` already does. Determinism comes
    # from the strict JSON schema, not from `temperature=0`.
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": _JUDGE_RESPONSE_SCHEMA,
                },
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return JudgeResult(
            flagged_turn_indices=[],
            elapsed_seconds=time.monotonic() - t0,
            error=f"judge timeout after {timeout}s",
        )
    except Exception as e:  # noqa: BLE001 — fail-open on any client error
        return JudgeResult(
            flagged_turn_indices=[],
            elapsed_seconds=time.monotonic() - t0,
            error=f"{type(e).__name__}: {e}",
        )

    in_tok = (response.usage.prompt_tokens or 0) if response.usage else 0
    out_tok = (response.usage.completion_tokens or 0) if response.usage else 0
    cost = estimate_cost_usd("openai", model, in_tok, out_tok)

    raw = (response.choices[0].message.content or "").strip() if response.choices else ""
    try:
        parsed = json.loads(raw) if raw else {"flagged_turns": []}
    except json.JSONDecodeError as e:
        return JudgeResult(
            flagged_turn_indices=[],
            raw_response=raw,
            input_tokens=in_tok, output_tokens=out_tok, cost_usd=cost,
            elapsed_seconds=time.monotonic() - t0,
            error=f"judge response not valid JSON: {e}",
        )

    flagged_raw = parsed.get("flagged_turns") or []
    n_turns = len(turn_records)
    indices: list[int] = []
    reasons: dict[int, str] = {}
    for entry in flagged_raw:
        idx = entry.get("turn_idx")
        if not isinstance(idx, int) or idx < 0 or idx >= n_turns:
            # Out-of-range index — judge confusion. Skip silently rather
            # than fail the whole match.
            continue
        if idx in reasons:
            # De-dupe; if the judge emitted the same idx twice, keep the
            # first reason and drop the second.
            continue
        indices.append(idx)
        reasons[idx] = (entry.get("reason") or "")[:120]

    return JudgeResult(
        flagged_turn_indices=indices,
        reasons=reasons,
        raw_response=raw,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost_usd=cost,
        elapsed_seconds=time.monotonic() - t0,
        error=None,
    )


__all__ = [
    "DEFAULT_JUDGE_MODEL",
    "DEFAULT_JUDGE_RETRIES",
    "DEFAULT_JUDGE_TIMEOUT",
    "JudgeResult",
    "JUDGE_SYSTEM_PROMPT",
    "JUDGE_USER_TEMPLATE",
    "extract_pre_tool_thought",
    "judge_match_cots",
]
