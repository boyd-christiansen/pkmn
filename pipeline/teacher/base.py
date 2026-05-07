"""Teacher LLM — synthesize a chain-of-thought that reverse-engineers a known play.

Pipeline role:
    The orchestrator (master_pipeline.py) supplies a board state, threat
    matrix, and the human's ground-truth play for one VGC turn. This module
    drives a frontier model through a tool-calling loop with the calc
    microservice, eliciting a Chain-of-Thought that JUSTIFIES the human's
    play. The result is a single fine-tuning example: the same conversation
    that the trained model will eventually produce on its own.

    Ground-truth handling: during the synthesis call the LLM SEES the human
    play (it's appended to the user message). The returned messages have
    that suffix stripped — the saved SFT example shows only the board
    state + threat matrix in the user prompt, so the trained model learns
    to derive the play from scratch.

Provider abstraction:
    `TeacherProvider` is the ABC that concrete frontend adapters implement
    (`teacher_openai.py`, `teacher_anthropic.py`, `teacher_google.py`).
    Each adapter handles SDK specifics + tool-format translation, but
    returns the same OpenAI-format messages so saved JSONL is comparable
    across providers and the bake-off / migration is trivial.

Isolation contract:
    Talks to whichever frontier model the orchestrator's chosen provider
    points at, plus the calc microservice via the `calculate_damage` tool.
    No replay parsing, no inference, no canonical-priors imports.
"""
from __future__ import annotations

import asyncio
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from openai import AsyncOpenAI

DEFAULT_MODEL = os.environ.get("TEACHER_MODEL", "gpt-5.5")
DEFAULT_CALC_URL = "http://localhost:3000/calc"
MAX_TOOL_ITERATIONS = 8  # need ≥3 (calc → result → submit), buffer for alternatives


# ---------------------------------------------------------------------------
# JSON Schemas (calc tool + final response_format)
# ---------------------------------------------------------------------------

# calculate_damage tool — non-strict so optional EVs / IVs / Nature / field
# don't have to be supplied every time. Mirrors /calc PokemonInput exactly.
_BOOSTS_SCHEMA = {
    "type": "object",
    "properties": {s: {"type": "integer", "minimum": -6, "maximum": 6}
                   for s in ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")},
    "additionalProperties": False,
}

_STAT_TABLE_SCHEMA = {
    "type": "object",
    "properties": {s: {"type": "integer", "minimum": 0, "maximum": 252}
                   for s in ("hp", "atk", "def", "spa", "spd", "spe")},
    "additionalProperties": False,
}

_POKEMON_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "species": {"type": "string", "description": "Species name e.g. 'Calyrex-Shadow'"},
        "item": {"type": "string", "description": "Held item ID, '' if none/unknown"},
        "ability": {"type": "string", "description": "Ability ID, '' if unknown"},
        "level": {"type": "integer", "default": 50},
        "currentHP": {
            "description": "Current HP — string ending in '%' for percentage, number for flat HP, omit for full HP",
        },
        "status": {"type": "string", "enum": ["", "brn", "par", "psn", "tox", "slp", "frz"]},
        "teraType": {"type": "string", "description": "Pokemon's Tera type (always known via OTS)"},
        "isTera": {"type": "boolean", "description": "True if the Pokemon has already Terastallized"},
        "boosts": _BOOSTS_SCHEMA,
        "evs": _STAT_TABLE_SCHEMA,
        "ivs": _STAT_TABLE_SCHEMA,
        "nature": {"type": "string", "description": "e.g. 'Modest', 'Adamant', 'Timid'"},
    },
    "required": ["species", "item", "ability", "teraType", "boosts"],
}

_FIELD_SCHEMA = {
    "type": "object",
    "properties": {
        "gameType": {"type": "string", "enum": ["Singles", "Doubles"], "default": "Doubles"},
        "weather": {"type": "string"},
        "terrain": {"type": "string"},
        "isGravity": {"type": "boolean"},
        "isMagicRoom": {"type": "boolean"},
        "isWonderRoom": {"type": "boolean"},
        "attackerSide": {"type": "object"},
        "defenderSide": {"type": "object"},
    },
}

CALCULATE_DAMAGE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "calculate_damage",
        "description": (
            "Compute the deterministic damage range for one move from one attacker "
            "to one defender, given the field state. Returns min/max damage rolls, "
            "min/max percent of defender max HP, KO chance text, and a description. "
            "Use this to verify your most decisive damage assumptions before locking "
            "in a decision."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "attacker": _POKEMON_INPUT_SCHEMA,
                "defender": _POKEMON_INPUT_SCHEMA,
                "move": {
                    "anyOf": [
                        {"type": "string", "description": "Move name (e.g. 'Wood Hammer')"},
                        {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "isCrit": {"type": "boolean"},
                                "hits": {"type": "integer"},
                            },
                            "required": ["name"],
                        },
                    ]
                },
                "field": _FIELD_SCHEMA,
            },
            "required": ["attacker", "defender", "move"],
        },
    },
}


# Final assistant output — strict JSON schema, used via response_format.
_SLOT_ACTION_SCHEMA = {
    "type": "object",
    "description": "What this active slot does this turn. Set the unused fields to null.",
    "properties": {
        "action_type": {
            "type": "string",
            "enum": ["move", "switch", "pass"],
            "description": "'move' to use a move, 'switch' to swap to a benched mon, 'pass' if the slot is empty/fainted at start of turn",
        },
        "move": {
            "type": ["string", "null"],
            "description": "Move name — required when action_type='move', else null",
        },
        "target": {
            "type": ["string", "null"],
            "description": "PS slot id of the target ('p2a', 'p2b', 'p1a', 'p1b'), 'spread' for AoE moves, 'self' for self-targeting moves, or null when not 'move'",
        },
        "tera": {
            "type": ["boolean", "null"],
            "description": "True if Terastallizing this turn; null when not 'move'",
        },
        "switch_to": {
            "type": ["string", "null"],
            "description": "Species being switched to — required when action_type='switch', else null",
        },
    },
    "required": ["action_type", "move", "target", "tera", "switch_to"],
    "additionalProperties": False,
}

FINAL_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pre_tool_thought": {
            "type": "string",
            "description": "Brief strategic reasoning summary that leads to the chosen action.",
        },
        "action": {
            "type": "object",
            "properties": {
                "slot_1": _SLOT_ACTION_SCHEMA,
                "slot_2": _SLOT_ACTION_SCHEMA,
            },
            "required": ["slot_1", "slot_2"],
            "additionalProperties": False,
        },
    },
    "required": ["pre_tool_thought", "action"],
    "additionalProperties": False,
}


# Final-answer is a tool now (not response_format). The model has only one
# output channel — tool calls — so it can't bypass calculate_damage by
# producing a structured response directly. This is the architectural fix
# for the zero-tool-call problem we hit in the first real test run.
SUBMIT_DECISION_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_decision",
        "description": (
            "Call this exactly once per turn, after using `calculate_damage` "
            "to verify your most decisive damage assumptions, when you are "
            "ready to commit your final play. The arguments are your final "
            "structured action."
        ),
        "parameters": FINAL_OUTPUT_SCHEMA,
    },
}


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

def _species_key(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum())


# TODO(rlhf-followup): replace this prompt-driven alternative evaluation
# (rule 5 below) with proper minimax / Monte Carlo distillation. The current
# approach has the teacher cherry-picking weak alternatives because it knows
# the answer; a proper search would surface alternatives that genuinely
# competed with the chosen play.
_SHARED_RULES_TAIL = """2. The Tool Rule: You have two tools. `calculate_damage` verifies any specific damage hypothetical — switch outcomes, future-turn ranges, opposing Tera predictions, +1/+2 boosted scenarios. `submit_decision` is how you commit your final play; call it exactly once when ready. **You MUST call `calculate_damage` at least once before calling `submit_decision`.** The pre-computed threat matrix already covers current matchup damage cells — don't re-calc those; use the tool for hypotheticals only.

3. The Threat-Matrix Rule: Each line shows an Absolute damage envelope (provable from observed play). When the canonical meta spread is consistent with the inferred bounds, a Probable envelope is also shown; when it's contradicted, only Absolute is shown tagged `(off-meta)`.

4. The Spread Rule: Your team's stat spread may be presented as either exact values or as inferred per-stat ranges. When a range is given, reason from the bounds — worst case for your own survival checks, best case for your offensive checks.

5. The Alternatives Rule: Before submitting, evaluate at least one plausible alternative play (a different move on the same Pokémon, or a switch to bring in a useful matchup) using `calculate_damage`, and document why it's worse than your chosen play. The point of the calc tool is to disprove tempting alternatives, not to confirm what the threat matrix already showed.

6. The Output Rule: Commit your decision via `submit_decision` with arguments:
   - pre_tool_thought: a brief strategic reasoning summary that leads to your chosen action (mention the rejected alternative explicitly)
   - action: {{ slot_1, slot_2 }} where each slot describes the action for that active Pokémon
"""


SYSTEM_PROMPT_BO1 = """You are a top-tier competitive VGC Reg I player commanding YOUR TEAM (Player 1) in a Generation 9 doubles battle (best-of-1, **Closed Team Sheet** — only species are visible at team preview; items, abilities, moves, and Tera types are hidden until they activate or are used).

Your job each turn is to decide what each of your active Pokémon does — a move with a target (and whether to Terastallize), a switch, or pass.

YOUR TEAM (P1 — moves you haven't yet used this match are tagged as `[UNREVEALED_MOVE]`):
{p1_team_block}

CRITICAL RULES:

1. The Masking Rule: If a Pokémon on Your Side has `[UNREVEALED_MOVE]` in its moveset, treat that slot as untrusted: don't assume what the move is and don't factor it into your reasoning. (In a real match you'd know your own moves, but for this prompt we're surfacing only what's been revealed in play so far.)

""" + _SHARED_RULES_TAIL


SYSTEM_PROMPT_BO3 = """You are a top-tier competitive VGC Reg I player commanding YOUR TEAM (Player 1) in a Generation 9 doubles battle (best-of-3, **Open Team Sheet** — both players see each other's full 6-Pokémon roster, items, abilities, all 4 moves, and Tera type before turn 1; only EVs / IVs / Nature stay hidden).

YOUR TEAM (P1 — full Open Team Sheet, ★ = your selection for this game):
{p1_sheet_block}

OPPONENT'S TEAM (P2 — full Open Team Sheet; their selection of 4 of these 6 will be revealed as they switch in):
{p2_sheet_block}

CRITICAL RULES:

1. The OTS Rule: All 6 of your opponent's Pokémon, their items, abilities, moves, and Tera types are PUBLIC knowledge — reason about every one of them, including the backline. Their actual selection of 4 reveals as they switch in, and their EV / IV / Nature spreads stay hidden throughout.

""" + _SHARED_RULES_TAIL


SYNTHESIS_GROUND_TRUTH_SUFFIX = """

=== EXPERT'S DECISION (oracle truth — articulate the chain of reasoning that justifies exactly this play) ===
{ground_truth_json}
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _format_ots_block(
    sheets: list[dict[str, Any]],
    *,
    brought_keys: set[str] | None = None,
) -> str:
    """Format an OTS team sheet as a system-prompt-friendly block.

    `brought_keys` is the set of (normalized) species names actually brought
    to the current game; entries in `sheets` whose species key matches get a
    ★ marker. Pass `None` (default) for the opponent — we don't reveal which
    4 they brought.
    """
    lines: list[str] = []
    for s in sheets:
        sp = s.get("species") or "?"
        marker = "★ " if (brought_keys is not None and _species_key(sp) in brought_keys) else "  "
        item = s.get("item") or "?"
        ability = s.get("ability") or "?"
        tera = s.get("teraType") or "?"
        moves = " / ".join(m for m in (s.get("moves") or []) if m)
        lines.append(f"{marker}{sp} @ {item}, ability={ability}, tera={tera}")
        lines.append(f"      moves: {moves}")
    return "\n".join(lines)


def render_system_prompt(p1_team_block: str) -> str:
    """Bo1 (CTS) — format the system prompt with the reconstructed P1 team block."""
    return SYSTEM_PROMPT_BO1.format(p1_team_block=p1_team_block)


def render_system_prompt_bo3(
    p1_sheet: list[dict[str, Any]],
    p2_sheet: list[dict[str, Any]],
    p1_brought: set[str],
) -> str:
    """Bo3 (OTS) — both teams' full sheets in the prompt; P1's brought 4 marked with ★."""
    p1_block = _format_ots_block(p1_sheet, brought_keys=p1_brought)
    p2_block = _format_ots_block(p2_sheet, brought_keys=None)
    return SYSTEM_PROMPT_BO3.format(p1_sheet_block=p1_block, p2_sheet_block=p2_block)


async def synthesize_turn(
    system_prompt: str,
    user_prompt: str,
    human_action: dict[str, Any],
    *,
    calc_url: str = DEFAULT_CALC_URL,
    model: str = DEFAULT_MODEL,
    max_iterations: int = MAX_TOOL_ITERATIONS,
    openai_client: AsyncOpenAI | None = None,
    aiohttp_session: aiohttp.ClientSession | None = None,
) -> list[dict[str, Any]] | None:
    """Run the teacher tool-use loop. Returns the SFT-ready conversation, or None on failure.

    Architecture: tool calls are the only output channel. The model must call
    `calculate_damage` (zero or more times) and then `submit_decision` (exactly
    once) to terminate the loop. There is no `response_format` — without
    `submit_decision` the loop hits max_iterations and returns None.

    The returned messages have the ground-truth suffix stripped from the user
    prompt — they're safe to write directly to the fine-tuning JSONL.
    """
    own_openai = openai_client is None
    if own_openai:
        openai_client = AsyncOpenAI()
    own_aiohttp = aiohttp_session is None
    if own_aiohttp:
        aiohttp_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60))

    try:
        # The user message the LLM SEES has the ground-truth suffix.
        api_user_content = user_prompt + SYNTHESIS_GROUND_TRUTH_SUFFIX.format(
            ground_truth_json=json.dumps(human_action, indent=2)
        )

        api_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": api_user_content},
        ]

        tools = [CALCULATE_DAMAGE_TOOL, SUBMIT_DECISION_TOOL]
        submit_seen = False

        for iter_idx in range(max_iterations):
            # Force a tool call on every iteration. On iter 0 specifically, force
            # `calculate_damage` so the model can't shortcut straight to
            # submit_decision before doing any verification — MUST in the prompt
            # alone wasn't enough with gpt-4o.
            if iter_idx == 0:
                tool_choice: Any = {"type": "function", "function": {"name": "calculate_damage"}}
            else:
                tool_choice = "required"
            response = await openai_client.chat.completions.create(
                model=model,
                messages=api_messages,
                tools=tools,
                tool_choice=tool_choice,
                parallel_tool_calls=False,  # force sequential reasoning: calc → result → next decision
            )
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
                # Model produced content without a tool call — protocol violation
                # (tool_choice=required should prevent this, but defensive).
                return None

            for tc in msg.tool_calls:
                name = tc.function.name
                if name == "calculate_damage":
                    try:
                        args = json.loads(tc.function.arguments)
                        result = await _call_calc(aiohttp_session, calc_url, args)
                        tool_content = json.dumps(result)
                    except Exception as e:
                        tool_content = json.dumps({"error": f"{type(e).__name__}: {e}"})
                    api_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_content,
                    })
                elif name == "submit_decision":
                    # Acknowledge the commit so the saved messages are well-formed
                    # (every tool_call must have a matching tool result for OpenAI
                    # fine-tuning data validity).
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
                # Strip ground-truth from the saved user message and return.
                saved_messages = list(api_messages)
                saved_messages[1] = {"role": "user", "content": user_prompt}
                return saved_messages

        # Hit max iterations without ever seeing submit_decision.
        return None
    finally:
        if own_aiohttp:
            await aiohttp_session.close()


async def _call_calc(
    session: aiohttp.ClientSession, calc_url: str, payload: dict[str, Any]
) -> dict[str, Any]:
    async with session.post(calc_url, json=payload) as r:
        text = await r.text()
        if r.status >= 400:
            raise RuntimeError(f"/calc {r.status}: {text[:200]}")
        return json.loads(text)


# ---------------------------------------------------------------------------
# Provider abstraction (for the frontier-model bake-off)
# ---------------------------------------------------------------------------


@dataclass
class ProviderResult:
    """Per-turn metrics that bakeoff.py uses to compare providers."""
    messages: list[dict[str, Any]] | None  # SFT-ready conversation, or None on failure
    input_tokens: int = 0
    output_tokens: int = 0
    calc_calls: int = 0                    # `calculate_damage` invocations
    iterations: int = 0                    # API roundtrips
    elapsed_seconds: float = 0.0
    cost_usd: float = 0.0
    error: str | None = None


# Approximate $/1M-token pricing per (provider, model). Updated periodically;
# only used by bakeoff.py for cost-estimation. Numbers are best-guess
# placeholders for the current 2026 frontier lineup — confirm against the
# provider's pricing page before scaling.
PRICE_PER_M_TOKENS: dict[str, dict[str, tuple[float, float]]] = {
    # (input_price, output_price) in $ per 1M tokens
    "openai": {
        # Flagship line.
        "gpt-5.5":         (5.00,  20.00),
        "gpt-5.5-pro":     (15.00, 60.00),
        "gpt-5.4":         (3.00,  12.00),  # previous-gen flagship, cheaper
        # Compact tiers.
        "gpt-5.5-mini":    (0.40,  1.60),
        "gpt-5.5-nano":    (0.10,  0.40),
        # Legacy (still callable).
        "gpt-4o":          (2.50,  10.00),
        "gpt-4o-mini":     (0.15,  0.60),
        "gpt-4.1":         (2.00,  8.00),
    },
    "anthropic": {
        "claude-opus-4-7":   (15.00, 75.00),
        "claude-sonnet-4-6": (3.00,  15.00),
        "claude-haiku-4-5":  (1.00,  5.00),
        # Legacy.
        "claude-sonnet-4-5": (3.00,  15.00),
    },
    "google": {
        "gemini-3.1-pro-preview":        (1.50,  10.00),
        "gemini-3.1-flash-preview":      (0.30,  2.50),
        "gemini-3.1-flash-lite-preview": (0.10,  0.50),
        # Legacy.
        "gemini-2.5-pro":                (1.25,  10.00),
    },
}


def estimate_cost_usd(provider: str, model: str, in_tokens: int, out_tokens: int) -> float:
    table = PRICE_PER_M_TOKENS.get(provider, {})
    in_p, out_p = table.get(model, (0.0, 0.0))
    return (in_tokens * in_p + out_tokens * out_p) / 1_000_000.0


class TeacherProvider(ABC):
    """Abstract base — concrete adapters in teacher_{openai,anthropic,google}.py.

    Each implementation does its own SDK call + tool-format translation,
    but returns OpenAI-format messages so saved JSONL is comparable across
    providers.
    """

    name: str = "abstract"
    model: str = ""

    @abstractmethod
    async def synthesize_turn(
        self,
        system_prompt: str,
        user_prompt: str,
        human_action: dict[str, Any],
        *,
        calc_url: str = DEFAULT_CALC_URL,
        aiohttp_session: aiohttp.ClientSession | None = None,
    ) -> ProviderResult:
        ...


__all__ = [
    "CALCULATE_DAMAGE_TOOL",
    "DEFAULT_CALC_URL",
    "DEFAULT_MODEL",
    "FINAL_OUTPUT_SCHEMA",
    "MAX_TOOL_ITERATIONS",
    "PRICE_PER_M_TOKENS",
    "ProviderResult",
    "SUBMIT_DECISION_TOOL",
    "SYNTHESIS_GROUND_TRUTH_SUFFIX",
    "SYSTEM_PROMPT_BO1",
    "SYSTEM_PROMPT_BO3",
    "TeacherProvider",
    "_call_calc",
    "estimate_cost_usd",
    "render_system_prompt_bo3",
    "render_system_prompt",
    "synthesize_turn",
]
