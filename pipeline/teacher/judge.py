"""Match-level model-judge validator (plan v4, generalized in plan v8).

The v3 regex `detect_oracle_leak` catches the strongest meta-leak
phrases ("training section", "the target action", "expert's decision")
but misses softer references ("clearly the right move", "the data
points to", first-person-knowledge-as-fact constructions). This module
adds a second-line filter that submits an entire match's worth of CoTs
in ONE call to a frontier model and gets back a list of turn indices
to retry.

**Provider dispatch (Plan v8).** Originally OpenAI-only; now
provider-agnostic via the `provider` kwarg. Plan v8 flipped the
production default to Google Gemini to match the new teacher provider
(the May 2026 bake-off had Gemini tied with OpenAI on quality, slightly
cheaper at unit cost, and the project has ~$100K in GCP credits). The
OpenAI judge path is preserved for `--provider openai` backward compat.

Default model: `gemini-3.1-pro-preview` (the bake-off winner). Override
via the `JUDGE_MODEL` env var or the `--judge-model` CLI flag. To
explicitly use OpenAI, pass `provider="openai"` and a `gpt-5.5`-class
model.

Why per-match, not per-row:
  - Amortizes a fixed system prompt across N turns. An 8-turn match
    judges in one call versus 8 separate calls.
  - The judge sees more context — multiple consecutive turns
    referencing the training framing is a stronger signal than each
    in isolation.

Contract callers care about (`master_pipeline._run_judge_with_retries`,
`batch_runner.run_batch_for_matches`):
  - `judge_match_cots(records, *, client, provider, ...) -> JudgeResult`
    with `flagged_turn_indices` and `reasons`.
  - `extract_pre_tool_thought(messages) -> str | None` (re-exported from
    `teacher.base`) for parsing saved rows.

Fail-open contract: on any client error / network failure / malformed
response, the function returns an empty `flagged_turn_indices` and a
non-None `error`. Callers write all rows as if the judge passed. Better
to ship a few possibly-leaky rows than drop a whole match to an infra
hiccup; the regex filter is still the first line of defense.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI
from google import genai
from google.genai import types as genai_types

from .base import (
    PRICE_PER_M_TOKENS,
    estimate_cost_usd,
    extract_pre_tool_thought,
)

# -----------------------------------------------------------------------------
# Constants / defaults
# -----------------------------------------------------------------------------

# Production default = google (the v8 switch). To use OpenAI, pass
# `provider="openai"` explicitly + a gpt-5.5-class model.
DEFAULT_JUDGE_PROVIDER = os.environ.get("JUDGE_PROVIDER", "google")

# Default model tracks the production provider. Gemini's pro tier mirrors
# what the bake-off used (gemini-3.1-pro-preview).
DEFAULT_JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemini-3.1-pro-preview")

# Backup default for the OpenAI judge path. Kept for backward compat with
# Plan v4 callers / docs that mention gpt-5.5. (gpt-5.5-mini would be
# cheaper but the project's OpenAI account doesn't have access; gpt-5.5
# is the working tier.)
DEFAULT_JUDGE_MODEL_OPENAI = "gpt-5.5"

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
# Structured-output schemas (per provider)
# -----------------------------------------------------------------------------

# OpenAI uses `response_format=json_schema` with the wrapper shape
# {name, strict, schema}.
_OPENAI_JUDGE_RESPONSE_SCHEMA: dict[str, Any] = {
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

# Gemini uses `response_schema` (no name/strict wrapper, no
# additionalProperties keyword — the schema is the body of the OpenAPI
# 3.0 object). Same logical shape as the OpenAI version.
_GEMINI_JUDGE_RESPONSE_SCHEMA: dict[str, Any] = {
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
            },
        }
    },
    "required": ["flagged_turns"],
}


# -----------------------------------------------------------------------------
# Prompt template (provider-agnostic)
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


def _parse_flagged(parsed: dict[str, Any], n_turns: int) -> tuple[list[int], dict[int, str]]:
    """Shared post-processing of the judge's structured-output payload.

    Both providers return the same JSON shape under their respective
    structured-output APIs. Drops out-of-range indices and de-dupes
    repeated turn_idx entries; both are silent corrections (we don't
    fail the whole match for judge-side confusion).
    """
    flagged_raw = parsed.get("flagged_turns") or []
    indices: list[int] = []
    reasons: dict[int, str] = {}
    for entry in flagged_raw:
        idx = entry.get("turn_idx")
        if not isinstance(idx, int) or idx < 0 or idx >= n_turns:
            continue
        if idx in reasons:
            continue
        indices.append(idx)
        reasons[idx] = (entry.get("reason") or "")[:120]
    return indices, reasons


# -----------------------------------------------------------------------------
# Public API — provider dispatch
# -----------------------------------------------------------------------------


async def judge_match_cots(
    turn_records: list[dict[str, Any]],
    *,
    client: Any,                                  # AsyncOpenAI | genai.Client
    provider: str = DEFAULT_JUDGE_PROVIDER,
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

    Dispatch:
      • `provider="google"` (default) — uses Gemini's
        `client.aio.models.generate_content()` with
        `response_mime_type="application/json"` + `response_schema`.
        Client is `google.genai.Client`.
      • `provider="openai"` — uses OpenAI's
        `client.chat.completions.create()` with
        `response_format={"type":"json_schema","json_schema":...}`.
        Client is `openai.AsyncOpenAI`.
    """
    if not turn_records:
        # Defensive: no work to do. Don't burn an API call.
        return JudgeResult(flagged_turn_indices=[])
    if provider == "google":
        return await _judge_via_gemini(turn_records, client=client,
                                       model=model, timeout=timeout)
    if provider == "openai":
        return await _judge_via_openai(turn_records, client=client,
                                       model=model, timeout=timeout)
    return JudgeResult(
        flagged_turn_indices=[],
        error=f"unknown judge provider: {provider!r}",
    )


# -----------------------------------------------------------------------------
# OpenAI judge (Plan v4 original; preserved for --provider openai)
# -----------------------------------------------------------------------------


async def _judge_via_openai(
    turn_records: list[dict[str, Any]],
    *,
    client: AsyncOpenAI,
    model: str,
    timeout: float,
) -> JudgeResult:
    t0 = time.monotonic()
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
                    "json_schema": _OPENAI_JUDGE_RESPONSE_SCHEMA,
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

    indices, reasons = _parse_flagged(parsed, len(turn_records))
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


# -----------------------------------------------------------------------------
# Gemini judge (Plan v8 — production default)
# -----------------------------------------------------------------------------


async def _judge_via_gemini(
    turn_records: list[dict[str, Any]],
    *,
    client: genai.Client,
    model: str,
    timeout: float,
) -> JudgeResult:
    t0 = time.monotonic()
    match_id = turn_records[0].get("match_id", "?")
    turn_block = _render_turn_block(turn_records)
    user_msg = JUDGE_USER_TEMPLATE.format(
        match_id=match_id, n_turns=len(turn_records), turn_block=turn_block,
    )

    # Gemini structured-output: response_mime_type="application/json" +
    # response_schema in OpenAPI 3.0 dialect (the same dialect
    # teacher/google.py uses for tool schemas). The system instruction
    # rides on `config`, not as a content message.
    try:
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=model,
                contents=[
                    genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text=user_msg)],
                    )
                ],
                config=genai_types.GenerateContentConfig(
                    system_instruction=JUDGE_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=_GEMINI_JUDGE_RESPONSE_SCHEMA,
                ),
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

    # Token usage on Gemini: usage_metadata.prompt_token_count /
    # candidates_token_count. Fall back gracefully if absent.
    um = getattr(response, "usage_metadata", None)
    in_tok = int(getattr(um, "prompt_token_count", 0) or 0)
    out_tok = int(getattr(um, "candidates_token_count", 0) or 0)
    cost = estimate_cost_usd("google", model, in_tok, out_tok)

    # `response.text` is the SDK's convenience accessor for the
    # primary text part of the first candidate; for JSON mode it
    # returns the model's serialized JSON output.
    raw = (response.text or "").strip() if hasattr(response, "text") else ""
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

    indices, reasons = _parse_flagged(parsed, len(turn_records))
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
    "DEFAULT_JUDGE_MODEL_OPENAI",
    "DEFAULT_JUDGE_PROVIDER",
    "DEFAULT_JUDGE_RETRIES",
    "DEFAULT_JUDGE_TIMEOUT",
    "JudgeResult",
    "JUDGE_SYSTEM_PROMPT",
    "JUDGE_USER_TEMPLATE",
    "extract_pre_tool_thought",
    "judge_match_cots",
]
