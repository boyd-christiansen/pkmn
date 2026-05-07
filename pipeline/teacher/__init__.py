"""Provider-agnostic teacher LLM tool-loop.

Re-exports the public surface of `teacher.base` plus each concrete
provider adapter, so callers can `from teacher import TeacherProvider,
OpenAIProvider` without reaching into the sub-modules.

The structure inside the package:
- `teacher.base`       — `TeacherProvider` ABC, schemas, prompt
                         templates, cost table, the legacy
                         `synthesize_turn` orchestrator function,
                         and the `_call_calc` helper that providers
                         share.
- `teacher.openai`     — OpenAI adapter (default).
- `teacher.anthropic`  — Anthropic adapter.
- `teacher.google`     — Google adapter.

Adapters can be imported lazily inside `_build_teacher` if the
relevant SDK isn't installed; this `__init__` performs all three
imports eagerly because all three SDKs are listed in `pyproject.toml`.
"""
from __future__ import annotations

from .base import (
    CALCULATE_DAMAGE_TOOL,
    DEFAULT_CALC_URL,
    DEFAULT_MODEL,
    FINAL_OUTPUT_SCHEMA,
    MAX_TOOL_ITERATIONS,
    PRICE_PER_M_TOKENS,
    ProviderResult,
    SUBMIT_DECISION_TOOL,
    SYNTHESIS_GROUND_TRUTH_SUFFIX,
    TeacherProvider,
    _call_calc,
    estimate_cost_usd,
    render_system_prompt,
    render_system_prompt_bo3,
    synthesize_turn,
)
from .anthropic import DEFAULT_MODEL_ANTHROPIC, AnthropicProvider
from .google import DEFAULT_MODEL_GOOGLE, GoogleProvider
from .openai import OpenAIProvider

__all__ = [
    # Base / shared
    "CALCULATE_DAMAGE_TOOL",
    "DEFAULT_CALC_URL",
    "DEFAULT_MODEL",
    "FINAL_OUTPUT_SCHEMA",
    "MAX_TOOL_ITERATIONS",
    "PRICE_PER_M_TOKENS",
    "ProviderResult",
    "SUBMIT_DECISION_TOOL",
    "SYNTHESIS_GROUND_TRUTH_SUFFIX",
    "TeacherProvider",
    "_call_calc",
    "estimate_cost_usd",
    "render_system_prompt",
    "render_system_prompt_bo3",
    "synthesize_turn",
    # Provider adapters
    "AnthropicProvider",
    "DEFAULT_MODEL_ANTHROPIC",
    "GoogleProvider",
    "DEFAULT_MODEL_GOOGLE",
    "OpenAIProvider",
]
