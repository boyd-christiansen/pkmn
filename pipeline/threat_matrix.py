"""Render the per-turn damage envelope as a compact human-readable text block.

Pipeline role:
    For one BoardState (one /parse_log snapshot), enumerate every plausible
    (attacker, move, defender) triple between the two sides and ask the calc
    microservice for damage bounds.

    Each calc payload includes the volatile state from the snapshot (status,
    boosts, weather, terrain, side conditions, Tera) so the answer reflects
    the actual board, not a sterile lab calc.

    For damage that flows over the opponent boundary, two calcs are run per
    move: the *low* and *high* envelopes derived from the OpponentKnowledge
    EV bounds — see the rules below.

Inputs:
    snapshot           — one TurnSnapshot dict from /parse_log.
    p1_side            — "p1" or "p2", indicating which side is "us".
    current_knowledge  — OpponentKnowledgeState (see damage_inferencer.py).

Outputs:
    A single string formatted as a small text block, ready to be inlined
    into the SFT prompt or returned as a tool-call response.

Bounds rules (per the project spec):
    INCOMING (opp → us):
      • Low  uses opponent's min_atk / min_spa
      • High uses opponent's max_atk / max_spa
    OUTGOING (us → opp):
      • Low  uses opponent's max_hp + max_def / max_spd  (opp is bulkiest)
      • High uses opponent's min_hp + min_def / min_spd  (opp is frailest)

Isolation contract:
    HTTP-calls calc_microservice (`/calc`, `/dex/move`). No replay parsing,
    no LLM, no imports from sibling pipeline modules.
"""
from __future__ import annotations

import asyncio
from typing import Any, Mapping

import aiohttp

from damage_inferencer import (
    DEFAULT_CALC_BASE_URL,
    OpponentKnowledgeState,
    STATS,
    get_move_category,
    species_key,
)


def _build_pokemon_payload(
    pkm: dict[str, Any], evs: Mapping[str, int] | None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "species": pkm["species"],
        "item": pkm.get("item") or "",
        "ability": pkm.get("ability") or "",
        "level": 50,
        "status": pkm.get("status") or "",
        "teraType": pkm.get("teraType") or "Normal",
        "isTera": bool(pkm.get("isTerastallized")),
        "boosts": pkm.get("boosts") or {},
    }
    if evs is not None:
        payload["evs"] = dict(evs)
    return payload


def _field_payload(snapshot: dict[str, Any], attacker_side: str) -> dict[str, Any]:
    f = snapshot.get("field", {})
    is_p1_atk = attacker_side == "p1"
    return {
        "gameType": "Doubles",
        "weather": f.get("weather"),
        "terrain": f.get("terrain"),
        "attackerSide": {
            "isTailwind": bool(f.get("tailwindP1") if is_p1_atk else f.get("tailwindP2"))
        },
        "defenderSide": {
            "isTailwind": bool(f.get("tailwindP2") if is_p1_atk else f.get("tailwindP1"))
        },
    }


async def _call_calc(
    session: aiohttp.ClientSession, base_url: str, payload: dict[str, Any]
) -> dict[str, Any]:
    async with session.post(f"{base_url}/calc", json=payload) as r:
        if r.status >= 400:
            text = await r.text()
            raise RuntimeError(f"/calc {r.status}: {text[:200]}")
        return await r.json()


def _attacker_evs_for_bound(
    knowledge: OpponentKnowledgeState | None,
    attacker_species: str,
    category: str,
    bound: str,  # "low" | "high"
) -> dict[str, int] | None:
    """Pick attacker's offensive EVs for the bound. Defender side, opp attacker."""
    if knowledge is None:
        return None
    entry = knowledge.get(species_key(attacker_species))
    if entry is None:
        return None
    stat = "atk" if category == "Physical" else "spa"
    edge = "min_evs" if bound == "low" else "max_evs"
    evs = {s: 0 for s in STATS}
    evs[stat] = entry[edge][stat]
    return evs


def _defender_evs_for_bound(
    knowledge: OpponentKnowledgeState | None,
    defender_species: str,
    category: str,
    bound: str,  # "low" | "high"
) -> dict[str, int] | None:
    """Pick defender's HP+defensive EVs for the bound. Outgoing damage, opp defender."""
    if knowledge is None:
        return None
    entry = knowledge.get(species_key(defender_species))
    if entry is None:
        return None
    def_stat = "def" if category == "Physical" else "spd"
    edge = "max_evs" if bound == "low" else "min_evs"
    evs = {s: 0 for s in STATS}
    evs["hp"] = entry[edge]["hp"]
    evs[def_stat] = entry[edge][def_stat]
    return evs


def _fmt_pct(p: float) -> str:
    return f"{p:.1f}%"


def _fmt_bound(prefix: str, result: dict[str, Any]) -> str:
    return f"{prefix}: {_fmt_pct(result['minPercent'])}–{_fmt_pct(result['maxPercent'])}  ({result['koChance']})"


async def generate_threat_matrix(
    snapshot: dict[str, Any],
    p1_side: str,
    current_knowledge: OpponentKnowledgeState,
    *,
    session: aiohttp.ClientSession | None = None,
    base_url: str = DEFAULT_CALC_BASE_URL,
) -> str:
    """Build the threat-matrix text block for one turn snapshot.

    `p1_side` indicates which side is "us" — typically "p1" or "p2"; the other
    side is treated as the opponent and the opponent's EV bounds are looked
    up in `current_knowledge`.
    """
    own_session = session is None
    if own_session:
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60))
    try:
        return await _generate(snapshot, p1_side, current_knowledge, session, base_url)
    finally:
        if own_session:
            await session.close()


async def _generate(
    snapshot: dict[str, Any],
    p1_side: str,
    knowledge: OpponentKnowledgeState,
    session: aiohttp.ClientSession,
    base_url: str,
) -> str:
    p2_side = "p2" if p1_side == "p1" else "p1"
    p1 = snapshot.get(p1_side, {})
    p2 = snapshot.get(p2_side, {})
    p1_active = [m for m in p1.get("active", []) if not m.get("fainted")]
    p2_active = [m for m in p2.get("active", []) if not m.get("fainted")]

    out_lines: list[str] = [f"=== THREAT MATRIX  (turn {snapshot.get('turn', '?')}, us={p1_side}) ===", ""]

    # OUTGOING — us → opp
    out_lines.append("--- OUTGOING (us → opp) ---")
    if not p1_active or not p2_active:
        out_lines.append("(no active matchup)")
    else:
        for atk in p1_active:
            for defn in p2_active:
                lines = await _matchup(
                    session, base_url, snapshot, atk, defn,
                    attacker_side=p1_side,
                    knowledge=knowledge,
                    direction="outgoing",
                )
                out_lines.extend(lines)
    out_lines.append("")

    # INCOMING — opp → us
    out_lines.append("--- INCOMING (opp → us) ---")
    if not p1_active or not p2_active:
        out_lines.append("(no active matchup)")
    else:
        for atk in p2_active:
            for defn in p1_active:
                lines = await _matchup(
                    session, base_url, snapshot, atk, defn,
                    attacker_side=p2_side,
                    knowledge=knowledge,
                    direction="incoming",
                )
                out_lines.extend(lines)

    return "\n".join(out_lines).rstrip() + "\n"


async def _matchup(
    session: aiohttp.ClientSession,
    base_url: str,
    snapshot: dict[str, Any],
    attacker: dict[str, Any],
    defender: dict[str, Any],
    *,
    attacker_side: str,
    knowledge: OpponentKnowledgeState,
    direction: str,  # "outgoing" | "incoming"
) -> list[str]:
    moves = attacker.get("revealedMoves") or []
    if not moves:
        return []

    field_payload = _field_payload(snapshot, attacker_side)
    lines: list[str] = []

    label = (
        f"[us {attacker['species']}] vs [opp {defender['species']}]"
        if direction == "outgoing"
        else f"[opp {attacker['species']}] vs [us {defender['species']}]"
    )
    lines.append(label + (f"  (status={attacker.get('status')}, boosts={attacker.get('boosts')})" if attacker.get('status') or attacker.get('boosts') else ""))

    for move in moves:
        category = await get_move_category(session, base_url, move)
        if category == "Status":
            continue

        if direction == "outgoing":
            # opponent is the defender; vary opp's HP+defense EVs
            low_evs = _defender_evs_for_bound(knowledge, defender["species"], category, "low")
            high_evs = _defender_evs_for_bound(knowledge, defender["species"], category, "high")
            atk_payload = _build_pokemon_payload(attacker, evs=None)
            low_payload = {
                "attacker": atk_payload,
                "defender": _build_pokemon_payload(defender, evs=low_evs),
                "move": move,
                "field": field_payload,
            }
            high_payload = {
                "attacker": atk_payload,
                "defender": _build_pokemon_payload(defender, evs=high_evs),
                "move": move,
                "field": field_payload,
            }
        else:
            # opponent is the attacker; vary opp's offensive EVs
            low_evs = _attacker_evs_for_bound(knowledge, attacker["species"], category, "low")
            high_evs = _attacker_evs_for_bound(knowledge, attacker["species"], category, "high")
            def_payload = _build_pokemon_payload(defender, evs=None)
            low_payload = {
                "attacker": _build_pokemon_payload(attacker, evs=low_evs),
                "defender": def_payload,
                "move": move,
                "field": field_payload,
            }
            high_payload = {
                "attacker": _build_pokemon_payload(attacker, evs=high_evs),
                "defender": def_payload,
                "move": move,
                "field": field_payload,
            }

        try:
            low_result = await _call_calc(session, base_url, low_payload)
            high_result = await _call_calc(session, base_url, high_payload)
        except Exception as e:
            lines.append(f"  {move}  ERROR: {e}")
            continue

        lines.append(
            f"  {move:<22} "
            f"low {_fmt_pct(low_result['minPercent'])}–{_fmt_pct(low_result['maxPercent'])} "
            f"({low_result['koChance']})  |  "
            f"high {_fmt_pct(high_result['minPercent'])}–{_fmt_pct(high_result['maxPercent'])} "
            f"({high_result['koChance']})"
        )

    return lines


__all__ = ["generate_threat_matrix"]
