"""Google (Gemini) adapter for the teacher LLM tool-loop.

Implements `TeacherProvider` against the Google `google-genai` SDK.
Translates the OpenAI-shaped tool-loop into Gemini's function-calling
content format and converts back to OpenAI-format messages for saving so
JSONL output is comparable across providers.

Tool semantics mirror the OpenAI provider:
  - Tools: calculate_damage, submit_decision (Gemini FunctionDeclarations)
  - On iter 0: tool_config restricts to allowed_function_names=["calculate_damage"]
  - Subsequent iters: tool_config mode=ANY (any tool required)
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import aiohttp
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from .base import (
    CALCULATE_DAMAGE_TOOL,
    DEFAULT_CALC_URL,
    MAX_CALC_CALLS_BEFORE_FORCE_SUBMIT,
    MAX_TOOL_ITERATIONS,
    PER_CALL_TIMEOUT,
    PER_TURN_TIMEOUT,
    ProviderResult,
    SUBMIT_DECISION_TOOL,
    SYNTHESIS_GROUND_TRUTH_SUFFIX,
    TeacherProvider,
    _call_calc,
    estimate_cost_usd,
)

DEFAULT_MODEL_GOOGLE = os.environ.get("TEACHER_MODEL_GOOGLE", "gemini-3.1-pro-preview")

# Vertex quota on a fresh project is tight; 429 RESOURCE_EXHAUSTED (and the
# transient 503 UNAVAILABLE) are expected under concurrent load. Without
# backoff a single 429 kills the whole match (per-match atomic commit → 0
# rows). Exponential backoff + jitter rides them out. Tunable via env so a
# corpus-scale run can widen the budget without a code change.
_BACKOFF_MAX_RETRIES = int(os.environ.get("GEMINI_BACKOFF_RETRIES", "6"))
_BACKOFF_BASE_SECONDS = float(os.environ.get("GEMINI_BACKOFF_BASE", "2.0"))
_BACKOFF_CAP_SECONDS = 60.0
_RETRYABLE_STATUS = (429, 503)


def _is_retryable_api_error(exc: Exception) -> bool:
    """True for rate-limit (429) / transient-unavailable (503) Gemini errors."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code in _RETRYABLE_STATUS:
        return True
    text = str(exc)
    return "RESOURCE_EXHAUSTED" in text or "UNAVAILABLE" in text


# --- Quality levers for the flash teacher -----------------------------------
# Provisioned Throughput for pro-preview is economically absurd ($93.6K/wk), so
# the production teacher is gemini-3.5-flash on (free) Dynamic Shared Quota.
# Flash's raw CoTs run ~40% thinner than pro's; two cheap levers close the gap:
#   1. THINKING BUDGET — flash defaults to shallow thinking. Forcing a budget
#      (~pro's observed ~1.7K thoughts) restores reasoning depth. Note the
#      budget counts toward output, so max_output_tokens must clear budget +
#      the function-call response (the bug that made the first probe emit no
#      submit_decision: budget 4096 == max 4096 left no room for the call).
#   2. FEW-SHOT EXEMPLARS — prepend curated pro turn-analyses so flash imitates
#      the calc-grounding + alternative-evaluation it otherwise drops.
# All env-tunable so pro runs (budget 0 → model default, no exemplars) are
# unaffected unless explicitly opted in.
_THINKING_BUDGET = int(os.environ.get("GEMINI_THINKING_BUDGET", "0"))  # 0=model default; >0 forces depth; -1=dynamic
_MAX_OUTPUT_TOKENS = int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "4096"))
_EXEMPLARS_PATH = Path(__file__).resolve().parent / "synthesis_exemplars.json"


def _load_fewshot_preamble() -> str:
    """Build the multi-shot exemplar preamble, or '' when disabled/absent.

    Exemplars live in synthesis_exemplars.json as [{"user","cot"}]. They MUST
    come from holdout matches (never the training corpus) so a synthesized turn
    can't see a near-neighbor's answer. Toggle off with VGC_FEWSHOT_ENABLED=0.
    """
    if os.environ.get("VGC_FEWSHOT_ENABLED", "1").strip().lower() in ("0", "false", "no"):
        return ""
    try:
        exemplars = json.loads(_EXEMPLARS_PATH.read_text(encoding="utf-8"))
    except OSError:
        return ""
    if not exemplars:
        return ""
    blocks = [
        f"=== EXAMPLE EXPERT ANALYSIS ===\n{ex['user']}\n"
        f"--- Expert's reasoning (match this depth: exact damage %s, disproves the rejected line) ---\n"
        f"{ex['cot']}\n=== END EXAMPLE ==="
        for ex in exemplars
    ]
    return (
        "Below are worked examples of expert turn analysis from UNRELATED games. Match their "
        "analytical depth, their grounding in exact damage percentages, and their habit of "
        "disproving the rejected alternative — then analyze the CURRENT turn the same way. Reason "
        "ONLY about the current board; the examples are for style, not content.\n\n"
        + "\n\n".join(blocks)
        + "\n\n=== CURRENT TURN (analyze this one) ===\n"
    )


_FEWSHOT_PREAMBLE = _load_fewshot_preamble()


def _to_function_declaration(openai_tool: dict[str, Any]) -> dict[str, Any]:
    """OpenAI function-tool schema → Gemini FunctionDeclaration dict.

    Translates JSON Schema 2020-12 (OpenAI's input format) into the
    OpenAPI 3.0 dialect that `google-genai`'s `Tool` accepts. Two
    transforms matter:

    1. Nullable types: `"type": ["string", "null"]` →
       `"type": "string", "nullable": true`. Gemini's schema only allows
       a single string for `type` and uses a separate boolean for
       null-allowance.
    2. Drop `additionalProperties` and other JSON-Schema-only keywords
       Gemini's pydantic validator doesn't recognize.
    """
    fn = openai_tool["function"]
    return {
        "name": fn["name"],
        "description": fn.get("description", ""),
        "parameters": _translate_schema(fn["parameters"]),
    }


# JSON-Schema keywords we silently drop when translating to Gemini's
# OpenAPI 3.0 dialect. They're either unsupported or interpreted
# differently — better to omit than to fail validation.
_DROP_KEYWORDS = frozenset({
    "$schema",
    "additionalProperties",
    "examples",
    "definitions",
})


def _translate_schema(node: Any) -> Any:
    """Recursively walk an OpenAPI/JSON-Schema tree, translating
    nullable-array types, filtering empty enum values, and stripping
    unsupported keywords.
    """
    if isinstance(node, list):
        return [_translate_schema(x) for x in node]
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for k, v in node.items():
        if k in _DROP_KEYWORDS:
            continue
        if k == "type" and isinstance(v, list):
            # JSON Schema 2020-12 nullable array → OpenAPI 3.0 type+nullable.
            non_null = [t for t in v if t != "null"]
            has_null = "null" in v
            if len(non_null) == 1:
                out["type"] = non_null[0]
                if has_null:
                    out["nullable"] = True
            elif not non_null:
                # `["null"]` alone — pretend it's a nullable string.
                out["type"] = "string"
                out["nullable"] = True
            else:
                # Multiple non-null types — Gemini doesn't support oneOf
                # cleanly. Fall back to the first non-null type and mark
                # nullable; downstream we'd need a richer translator if
                # this ever becomes a real case.
                out["type"] = non_null[0]
                out["nullable"] = has_null
        elif k == "enum" and isinstance(v, list):
            # Gemini rejects empty-string enum values. Filter them out;
            # the model can express "no value" by omitting the field
            # entirely. (This matters for our `status` field whose
            # original enum is ["", "brn", "par", "psn", "tox", "slp",
            # "frz"] — the "" entry meant "no status applied", which
            # for Gemini we drop and let the field be implicitly absent.)
            filtered = [x for x in v if x != ""]
            if filtered:
                out["enum"] = filtered
            # If filtering empties out the enum entirely, drop it; the
            # field's `type` constraint still applies.
        else:
            out[k] = _translate_schema(v)
    return out


class GoogleProvider(TeacherProvider):
    name = "google"

    def __init__(
        self,
        model: str | None = None,
        *,
        client: genai.Client | None = None,
    ):
        self.model = model or DEFAULT_MODEL_GOOGLE
        # genai.Client picks GOOGLE_API_KEY (or GEMINI_API_KEY) from env.
        self.client = client or genai.Client()

    async def synthesize_turn(
        self,
        system_prompt: str,
        user_prompt: str,
        human_action: dict[str, Any],
        *,
        calc_url: str = DEFAULT_CALC_URL,
        aiohttp_session: aiohttp.ClientSession | None = None,
    ) -> ProviderResult:
        own_session = aiohttp_session is None
        if own_session:
            aiohttp_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60))

        result = ProviderResult(messages=None)
        t0 = time.monotonic()
        try:
            return await asyncio.wait_for(
                self._do_turn(system_prompt, user_prompt, human_action,
                              calc_url=calc_url, aiohttp_session=aiohttp_session,
                              result=result),
                timeout=PER_TURN_TIMEOUT,
            )
        except asyncio.TimeoutError:
            result.error = f"per-turn timeout after {PER_TURN_TIMEOUT}s"
            return result
        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"
            return result
        finally:
            result.elapsed_seconds = time.monotonic() - t0
            result.cost_usd = estimate_cost_usd(
                self.name, self.model, result.input_tokens, result.output_tokens
            )
            if own_session:
                await aiohttp_session.close()

    async def _do_turn(
        self,
        system_prompt: str,
        user_prompt: str,
        human_action: dict[str, Any],
        *,
        calc_url: str,
        aiohttp_session: aiohttp.ClientSession,
        result: ProviderResult,
    ) -> ProviderResult:
        api_user_content = user_prompt + SYNTHESIS_GROUND_TRUTH_SUFFIX.format(
            ground_truth_json=json.dumps(human_action, indent=2)
        )
        if _FEWSHOT_PREAMBLE:
            api_user_content = _FEWSHOT_PREAMBLE + api_user_content
        # Gemini-native conversation buffer (list of Content)
        contents: list[Any] = [
            genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=api_user_content)],
            )
        ]

        tool = genai_types.Tool(
            function_declarations=[
                _to_function_declaration(CALCULATE_DAMAGE_TOOL),
                _to_function_declaration(SUBMIT_DECISION_TOOL),
            ]
        )

        saved_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        submit_seen = False
        tool_call_id_counter = 0

        for iter_idx in range(MAX_TOOL_ITERATIONS):
            # Two-way tool restriction:
            #   calc_calls >= MAX_CALC_CALLS_BEFORE_FORCE_SUBMIT : force submit_decision
            #   otherwise                                       : open (model picks)
            # iter 0 used to force `calculate_damage` — that's gone now;
            # see the OpenAI adapter for the rationale.
            if result.calc_calls >= MAX_CALC_CALLS_BEFORE_FORCE_SUBMIT:
                allowed = ["submit_decision"]
            else:
                allowed = None

            fc_kwargs: dict[str, Any] = {"mode": genai_types.FunctionCallingConfigMode.ANY}
            if allowed is not None:
                fc_kwargs["allowed_function_names"] = allowed
            tool_config = genai_types.ToolConfig(
                function_calling_config=genai_types.FunctionCallingConfig(**fc_kwargs)
            )

            cfg_kwargs: dict[str, Any] = dict(
                system_instruction=system_prompt,
                tools=[tool],
                tool_config=tool_config,
                max_output_tokens=_MAX_OUTPUT_TOKENS,
            )
            if _THINKING_BUDGET != 0:
                cfg_kwargs["thinking_config"] = genai_types.ThinkingConfig(
                    thinking_budget=_THINKING_BUDGET
                )
            cfg = genai_types.GenerateContentConfig(**cfg_kwargs)
            response = None
            delay = _BACKOFF_BASE_SECONDS
            for attempt in range(_BACKOFF_MAX_RETRIES + 1):
                try:
                    response = await asyncio.wait_for(
                        self.client.aio.models.generate_content(
                            model=self.model, contents=contents, config=cfg),
                        timeout=PER_CALL_TIMEOUT,
                    )
                    break
                except asyncio.TimeoutError:
                    result.error = f"per-call timeout at iter {iter_idx} after {PER_CALL_TIMEOUT}s"
                    return result
                except genai_errors.APIError as e:
                    if not _is_retryable_api_error(e) or attempt >= _BACKOFF_MAX_RETRIES:
                        result.error = f"teacher API error at iter {iter_idx}: {type(e).__name__}: {e}"
                        return result
                    # 429/503 — back off (exponential + jitter) and retry the same turn.
                    await asyncio.sleep(min(delay, _BACKOFF_CAP_SECONDS) + random.uniform(0.0, 1.0))
                    delay *= 2
            if response is None:  # defensive; loop either breaks or returns
                result.error = f"teacher API: no response after retries at iter {iter_idx}"
                return result
            result.iterations += 1

            if response.usage_metadata:
                result.input_tokens += response.usage_metadata.prompt_token_count or 0
                result.output_tokens += response.usage_metadata.candidates_token_count or 0

            # Gemini returns response.candidates[0].content with parts
            if not response.candidates:
                result.error = "no candidates returned"
                return result
            cand_content = response.candidates[0].content
            contents.append(cand_content)

            # Translate to OpenAI format + collect function calls
            openai_assistant: dict[str, Any] = {"role": "assistant"}
            tool_calls_for_openai: list[dict[str, Any]] = []
            text_parts: list[str] = []
            fn_calls: list[Any] = []

            for part in (cand_content.parts or []):
                if part.text:
                    text_parts.append(part.text)
                if part.function_call:
                    fn_calls.append(part.function_call)
                    tool_call_id_counter += 1
                    oai_id = f"call_{tool_call_id_counter:04d}"
                    tool_calls_for_openai.append({
                        "id": oai_id,
                        "type": "function",
                        "function": {
                            "name": part.function_call.name,
                            "arguments": json.dumps(dict(part.function_call.args or {})),
                        },
                    })
                    # stash the OpenAI id alongside the gemini call so we can correlate
                    part.function_call._oai_id = oai_id  # type: ignore[attr-defined]

            if text_parts:
                openai_assistant["content"] = "\n".join(text_parts)
            if tool_calls_for_openai:
                openai_assistant["tool_calls"] = tool_calls_for_openai
            saved_messages.append(openai_assistant)

            if not fn_calls:
                result.error = "no function_call parts (protocol violation)"
                return result

            # Run each function, append responses to BOTH contents
            # (Gemini-native format) and saved_messages (OpenAI format).
            response_parts: list[Any] = []
            for fc in fn_calls:
                if fc.name == "calculate_damage":
                    try:
                        calc_result = await _call_calc(
                            aiohttp_session, calc_url, dict(fc.args or {})
                        )
                        tool_content_dict = calc_result
                    except Exception as e:
                        tool_content_dict = {"error": f"{type(e).__name__}: {e}"}
                    result.calc_calls += 1
                elif fc.name == "submit_decision":
                    submit_seen = True
                    tool_content_dict = {"status": "decision_committed"}
                else:
                    tool_content_dict = {"error": f"unknown tool: {fc.name}"}

                response_parts.append(
                    genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            name=fc.name,
                            response=tool_content_dict,
                        )
                    )
                )
                saved_messages.append({
                    "role": "tool",
                    "tool_call_id": getattr(fc, "_oai_id", "call_unknown"),
                    "content": json.dumps(tool_content_dict),
                })

            contents.append(genai_types.Content(role="function", parts=response_parts))

            if submit_seen:
                result.messages = saved_messages
                return result

        result.error = f"hit max_iterations={MAX_TOOL_ITERATIONS} without submit_decision"
        return result
