"""OpenAI adapter for the teacher LLM tool-loop.

Implements `TeacherProvider` against the OpenAI Chat Completions API.
Tool semantics:
  - tools = [calculate_damage, submit_decision]
  - tool_choice forced to `calculate_damage` on iter 0 (so model can't
    short-circuit to submit), then `required` for subsequent iters.
  - parallel_tool_calls=False — sequential reasoning only.
  - max_retries=8 on the AsyncOpenAI client to absorb 429s.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

import aiohttp
from openai import AsyncOpenAI

from .base import (
    CALCULATE_DAMAGE_TOOL,
    DEFAULT_CALC_URL,
    DEFAULT_MODEL,
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


class OpenAIProvider(TeacherProvider):
    name = "openai"

    def __init__(
        self,
        model: str | None = None,
        *,
        client: AsyncOpenAI | None = None,
        max_retries: int = 8,
    ):
        self.model = model or os.environ.get("TEACHER_MODEL_OPENAI", DEFAULT_MODEL)
        self.client = client or AsyncOpenAI(max_retries=max_retries)

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
            # Outer per-turn wall-clock ceiling — guarantees the whole
            # tool loop can't exceed PER_TURN_TIMEOUT regardless of how
            # many iterations or how slow individual calls are.
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
        api_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": api_user_content},
        ]
        tools = [CALCULATE_DAMAGE_TOOL, SUBMIT_DECISION_TOOL]
        submit_seen = False

        for iter_idx in range(MAX_TOOL_ITERATIONS):
            # Two-way tool_choice:
            #   calc_calls >= MAX_CALC_CALLS_BEFORE_FORCE_SUBMIT : force submit_decision
            #   otherwise                                       : "required" (model picks freely)
            # iter 0 used to force `calculate_damage` — that's gone now.
            # Per the rewritten Tool Rule, the matrix already covers
            # active-vs-active damage and the model may submit immediately
            # if no hypothetical needs calc'ing.
            if result.calc_calls >= MAX_CALC_CALLS_BEFORE_FORCE_SUBMIT:
                tool_choice: Any = {"type": "function", "function": {"name": "submit_decision"}}
            else:
                tool_choice = "required"

            # Per-call timeout — even within the per-turn ceiling, a single
            # API request can't hang for more than PER_CALL_TIMEOUT.
            try:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.model,
                        messages=api_messages,
                        tools=tools,
                        tool_choice=tool_choice,
                        parallel_tool_calls=False,
                    ),
                    timeout=PER_CALL_TIMEOUT,
                )
            except asyncio.TimeoutError:
                result.error = f"per-call timeout at iter {iter_idx} after {PER_CALL_TIMEOUT}s"
                return result
            result.iterations += 1

            if response.usage:
                result.input_tokens += response.usage.prompt_tokens or 0
                result.output_tokens += response.usage.completion_tokens or 0

            msg = response.choices[0].message
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if msg.content is not None:
                assistant_msg["content"] = msg.content
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            api_messages.append(assistant_msg)

            if not msg.tool_calls:
                result.error = "no tool_calls (protocol violation)"
                return result

            for tc in msg.tool_calls:
                name = tc.function.name
                if name == "calculate_damage":
                    try:
                        args = json.loads(tc.function.arguments)
                        calc_result = await _call_calc(aiohttp_session, calc_url, args)
                        tool_content = json.dumps(calc_result)
                    except Exception as e:
                        tool_content = json.dumps({"error": f"{type(e).__name__}: {e}"})
                    api_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_content,
                    })
                    result.calc_calls += 1
                elif name == "submit_decision":
                    submit_seen = True
                    api_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"status": "decision_committed"}),
                    })
                else:
                    api_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"error": f"unknown tool: {name}"}),
                    })

            if submit_seen:
                # Strip the ground-truth suffix from the saved user
                # message before returning — trained model never sees it.
                saved = list(api_messages)
                saved[1] = {"role": "user", "content": user_prompt}
                result.messages = saved
                return result

        result.error = f"hit max_iterations={MAX_TOOL_ITERATIONS} without submit_decision"
        return result
