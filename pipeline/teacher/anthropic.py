"""Anthropic adapter for the teacher LLM tool-loop.

Implements `TeacherProvider` against the Anthropic Messages API. Translates
the OpenAI-shaped tool-loop into Anthropic's content-block format
(`tool_use` / `tool_result`), and converts back to OpenAI-format messages
for saving so JSONL output is comparable across providers.

Tool semantics mirror the OpenAI provider:
  - Tools: calculate_damage, submit_decision
  - On iter 0: tool_choice={"type": "tool", "name": "calculate_damage"} to
    force at least one calc before any submit.
  - Subsequent iters: tool_choice={"type": "any"} (required tool use).
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

import aiohttp
from anthropic import AsyncAnthropic

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

DEFAULT_MODEL_ANTHROPIC = os.environ.get(
    "TEACHER_MODEL_ANTHROPIC", "claude-sonnet-4-6"
)


def _to_anthropic_tool(openai_tool: dict[str, Any]) -> dict[str, Any]:
    """OpenAI function-tool schema → Anthropic tool schema."""
    fn = openai_tool["function"]
    return {
        "name": fn["name"],
        "description": fn.get("description", ""),
        "input_schema": fn["parameters"],
    }


class AnthropicProvider(TeacherProvider):
    name = "anthropic"

    def __init__(
        self,
        model: str | None = None,
        *,
        client: AsyncAnthropic | None = None,
    ):
        self.model = model or DEFAULT_MODEL_ANTHROPIC
        self.client = client or AsyncAnthropic()

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
        anthropic_messages: list[dict[str, Any]] = [
            {"role": "user", "content": api_user_content},
        ]
        tools = [
            _to_anthropic_tool(CALCULATE_DAMAGE_TOOL),
            _to_anthropic_tool(SUBMIT_DECISION_TOOL),
        ]

        # Translation buffer for OpenAI-format saved output.
        saved_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        submit_seen = False

        for iter_idx in range(MAX_TOOL_ITERATIONS):
            # Two-way tool_choice:
            #   calc_calls >= MAX_CALC_CALLS_BEFORE_FORCE_SUBMIT : force submit_decision
            #   otherwise                                       : open ("any" tool)
            # iter 0 used to force `calculate_damage` — that's gone now;
            # see the OpenAI adapter for the rationale.
            if result.calc_calls >= MAX_CALC_CALLS_BEFORE_FORCE_SUBMIT:
                tool_choice: Any = {"type": "tool", "name": "submit_decision"}
            else:
                tool_choice = {"type": "any"}

            try:
                response = await asyncio.wait_for(
                    self.client.messages.create(
                        model=self.model,
                        max_tokens=4096,
                        system=system_prompt,
                        messages=anthropic_messages,
                        tools=tools,
                        tool_choice=tool_choice,
                    ),
                    timeout=PER_CALL_TIMEOUT,
                )
            except asyncio.TimeoutError:
                result.error = f"per-call timeout at iter {iter_idx} after {PER_CALL_TIMEOUT}s"
                return result
            result.iterations += 1

            if response.usage:
                result.input_tokens += response.usage.input_tokens or 0
                result.output_tokens += response.usage.output_tokens or 0

            # Append the assistant turn (Anthropic content-block list)
            # to the Anthropic-native conversation buffer.
            anthropic_messages.append({"role": "assistant", "content": response.content})

            # Translate this assistant message into OpenAI format and
            # accumulate any tool_use blocks for processing.
            openai_assistant: dict[str, Any] = {"role": "assistant"}
            tool_calls_for_openai: list[dict[str, Any]] = []
            text_parts: list[str] = []
            tool_uses: list[Any] = []

            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_uses.append(block)
                    tool_calls_for_openai.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input),
                        },
                    })

            if text_parts:
                openai_assistant["content"] = "\n".join(text_parts)
            if tool_calls_for_openai:
                openai_assistant["tool_calls"] = tool_calls_for_openai
            saved_messages.append(openai_assistant)

            if not tool_uses:
                result.error = "no tool_use blocks (protocol violation)"
                return result

            # Run each tool, append results to BOTH the Anthropic
            # conversation (as a user message with tool_result blocks)
            # and the OpenAI saved messages (as separate role=tool entries).
            tool_results_for_anthropic: list[dict[str, Any]] = []
            for tu in tool_uses:
                if tu.name == "calculate_damage":
                    try:
                        calc_result = await _call_calc(aiohttp_session, calc_url, tu.input)
                        tool_content = json.dumps(calc_result)
                    except Exception as e:
                        tool_content = json.dumps({"error": f"{type(e).__name__}: {e}"})
                    result.calc_calls += 1
                elif tu.name == "submit_decision":
                    submit_seen = True
                    tool_content = json.dumps({"status": "decision_committed"})
                else:
                    tool_content = json.dumps({"error": f"unknown tool: {tu.name}"})

                tool_results_for_anthropic.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": tool_content,
                })
                saved_messages.append({
                    "role": "tool",
                    "tool_call_id": tu.id,
                    "content": tool_content,
                })

            anthropic_messages.append({
                "role": "user",
                "content": tool_results_for_anthropic,
            })

            if submit_seen:
                result.messages = saved_messages
                return result

        result.error = f"hit max_iterations={MAX_TOOL_ITERATIONS} without submit_decision"
        return result
