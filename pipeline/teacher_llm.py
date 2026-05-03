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

Isolation contract:
    Talks only to (a) the OpenAI Chat Completions API and (b) the calc
    microservice via the `calculate_damage` tool. No replay parsing, no
    inference, no canonical-priors imports.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import aiohttp
from openai import AsyncOpenAI

DEFAULT_MODEL = os.environ.get("TEACHER_MODEL", "gpt-4o")
DEFAULT_CALC_URL = "http://localhost:3000/calc"
MAX_TOOL_ITERATIONS = 6


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


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_BASE = """You are a world-class VGC (Pokémon Video Game Championships) competitor at the Day 2 World Championships level, commanding YOUR TEAM (Player 1) in a Generation 9 VGC Reg I doubles battle.

Your job each turn is to decide what each of your active Pokémon does — a move with a target (and whether to Terastallize), a switch, or pass.

YOUR REVEALED TEAM (from this match):
{p1_team_block}

CRITICAL RULES:

1. The Masking Rule: If a Pokémon on Your Side has `[UNREVEALED_MOVE]` in its moveset, it means that move was never utilized by the human expert in this entire Bo3 series. You must assume that the unrevealed move was completely suboptimal, irrelevant, or unusable for this specific matchup. Do not attempt to guess what it is, and do not factor it into your strategic reasoning.

2. The Tool Rule: You have access to a `calculate_damage` tool that exposes the official Smogon damage calculator. Use it to verify your most decisive damage assumptions before committing — typically 1-3 calcs per turn. Do not over-query.

3. The Threat-Matrix Rule: The user message includes a pre-computed threat matrix with TWO tracks per matchup:
   - Absolute: the strict mathematical envelope from observed-damage inference (provable bounds; wide).
   - Probable (meta): the calc result assuming both Pokémon run their canonical Smogon meta spread (narrow; only as good as the prior).
   When the two tracks disagree (`[PRIOR CONTRADICTED]`), the opponent is off-meta — favor the Absolute envelope.

4. The Output Rule: After your reasoning (and any tool calls), return one final JSON object matching the response schema:
   - pre_tool_thought: a brief strategic reasoning summary that leads to your chosen action
   - action: {{ slot_1, slot_2 }} where each slot describes the action for that active Pokémon
"""


SYNTHESIS_GROUND_TRUTH_SUFFIX = """

=== EXPERT'S DECISION (oracle truth — articulate the chain of reasoning that justifies exactly this play) ===
{ground_truth_json}
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_system_prompt(p1_team_block: str) -> str:
    """Format the system prompt with the reconstructed P1 team block inserted."""
    return SYSTEM_PROMPT_BASE.format(p1_team_block=p1_team_block)


async def synthesize_turn(
    system_prompt: str,
    user_prompt: str,
    human_action: dict[str, Any],
    *,
    tools_allowed: bool = True,
    calc_url: str = DEFAULT_CALC_URL,
    model: str = DEFAULT_MODEL,
    max_iterations: int = MAX_TOOL_ITERATIONS,
    openai_client: AsyncOpenAI | None = None,
    aiohttp_session: aiohttp.ClientSession | None = None,
) -> list[dict[str, Any]] | None:
    """Run the teacher tool-use loop. Returns the SFT-ready conversation, or None on failure.

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

        tools = [CALCULATE_DAMAGE_TOOL] if tools_allowed else None
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "vgc_decision",
                "strict": True,
                "schema": FINAL_OUTPUT_SCHEMA,
            },
        }

        for _ in range(max_iterations):
            response = await openai_client.chat.completions.create(
                model=model,
                messages=api_messages,
                tools=tools,
                response_format=response_format,
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
                # Final answer — strip ground-truth from the user message before returning.
                saved_messages = list(api_messages)
                saved_messages[1] = {"role": "user", "content": user_prompt}
                return saved_messages

            # Execute every tool call this round, append tool messages.
            for tc in msg.tool_calls:
                if tc.function.name == "calculate_damage":
                    try:
                        args = json.loads(tc.function.arguments)
                        result = await _call_calc(aiohttp_session, calc_url, args)
                        tool_content = json.dumps(result)
                    except Exception as e:
                        tool_content = json.dumps({"error": f"{type(e).__name__}: {e}"})
                else:
                    tool_content = json.dumps({"error": f"unknown tool: {tc.function.name}"})

                api_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_content,
                })

        # Hit max iterations without a final answer.
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


__all__ = [
    "CALCULATE_DAMAGE_TOOL",
    "DEFAULT_CALC_URL",
    "DEFAULT_MODEL",
    "FINAL_OUTPUT_SCHEMA",
    "MAX_TOOL_ITERATIONS",
    "SYSTEM_PROMPT_BASE",
    "render_system_prompt",
    "synthesize_turn",
]
