"""Observed-bounds damage envelope for one turn snapshot, ready for the SFT prompt.

Pipeline role:
    For one BoardState (one /parse_log snapshot), enumerate every plausible
    (attacker, move, defender) triple between the two sides and ask the calc
    microservice for the **Absolute** damage envelope of each matchup:

      - **Absolute**: the strict mathematical envelope using BOTH sides'
        current `KnowledgeState` bounds. Wide but provable.

    There is no second "Probable / canonical-meta" track. Smogon usage
    priors were removed (they were more confusing than useful in live game
    context — a meta figure next to the provable envelope invited the model
    to reason about an off-board statistic). The fallback for a stat with no
    observations is simply the fully-open `[0, 252]` envelope (rendered as
    `unknown` in the spread blocks), not a canonical spread.

Inputs:
    snapshot           — one TurnSnapshot dict from /parse_log.
    p1_side            — "p1" or "p2", whichever is "us" for the LLM.
    p1_knowledge       — KnowledgeState for side p1.
    p2_knowledge       — KnowledgeState for side p2.

Outputs:
    A single text block with one line per (attacker, move, defender),
    grouped by direction (OUTGOING us → opp, INCOMING opp → us).

Isolation contract:
    HTTP-calls calc_microservice (`/calc`, `/dex/move`). No replay parsing,
    no LLM, no canonical-priors, no imports from `damage_inferencer` beyond
    the shared types/helpers.
"""
from __future__ import annotations

from typing import Any, Mapping

import aiohttp

from damage_inferencer import (
    DEFAULT_CALC_BASE_URL,
    KnowledgeState,
    STATS,
    init_knowledge_entry,
    species_key,
)


def _build_pokemon_payload(
    pkm: dict[str, Any],
    *,
    evs: Mapping[str, int] | None = None,
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
    is_p1 = attacker_side == "p1"
    return {
        "gameType": "Doubles",
        "weather": f.get("weather"),
        "terrain": f.get("terrain"),
        "attackerSide": {"isTailwind": bool(f.get("tailwindP1") if is_p1 else f.get("tailwindP2"))},
        "defenderSide": {"isTailwind": bool(f.get("tailwindP2") if is_p1 else f.get("tailwindP1"))},
    }


async def _call_calc(
    session: aiohttp.ClientSession, base_url: str, payload: dict[str, Any]
) -> dict[str, Any]:
    async with session.post(f"{base_url}/calc", json=payload) as r:
        if r.status >= 400:
            text = await r.text()
            raise RuntimeError(f"/calc {r.status}: {text[:200]}")
        return await r.json()


def _entry_or_default(state: KnowledgeState, species: str) -> dict[str, dict[str, int]]:
    return state.get(species_key(species), init_knowledge_entry())


CHIP_DAMAGE_THRESHOLD_PCT = 15.0  # max-percent below this = "chip", rolled into a footer line
SPREAD_TARGETS = ("allAdjacentFoes", "foeSide", "allAdjacent")


_MOVE_META_CACHE: dict[str, dict[str, Any]] = {}


async def _get_move_meta(
    session: aiohttp.ClientSession, base_url: str, move_name: str
) -> dict[str, Any]:
    key = species_key(move_name)
    if key in _MOVE_META_CACHE:
        return _MOVE_META_CACHE[key]
    async with session.get(f"{base_url}/dex/move/{key}") as r:
        if r.status == 404:
            meta = {"category": "Status", "target": "self"}
        elif r.status >= 400:
            raise RuntimeError(f"/dex/move {r.status}")
        else:
            meta = await r.json()
    _MOVE_META_CACHE[key] = meta
    return meta


def _fmt_pct(p: float) -> str:
    return f"{p:.1f}%"


def _fmt_range(lo: float, hi: float) -> str:
    return f"{_fmt_pct(lo)}–{_fmt_pct(hi)}"


async def generate_threat_matrix(
    snapshot: dict[str, Any],
    p1_side: str,
    p1_knowledge: KnowledgeState,
    p2_knowledge: KnowledgeState,
    *,
    format_id: str | None = None,  # noqa: ARG001 — retained for call-site compat; unused since the meta track was removed
    session: aiohttp.ClientSession | None = None,
    base_url: str = DEFAULT_CALC_BASE_URL,
) -> str:
    """Render the Absolute-envelope threat matrix for one turn as a text block.

    Every line is the strict, provable damage range derived from both sides'
    current `KnowledgeState` bounds. No canonical-meta second track.
    """
    own_session = session is None
    if own_session:
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60))
    try:
        return await _generate(
            snapshot, p1_side, p1_knowledge, p2_knowledge, session, base_url
        )
    finally:
        if own_session:
            await session.close()


async def _generate(
    snapshot: dict[str, Any],
    p1_side: str,
    p1_knowledge: KnowledgeState,
    p2_knowledge: KnowledgeState,
    session: aiohttp.ClientSession,
    base_url: str,
) -> str:
    p2_side = "p2" if p1_side == "p1" else "p1"
    p1 = snapshot.get(p1_side, {})
    p2 = snapshot.get(p2_side, {})
    p1_active = [m for m in p1.get("active", []) if not m.get("fainted")]
    p2_active = [m for m in p2.get("active", []) if not m.get("fainted")]

    out: list[str] = [f"=== THREAT MATRIX  (turn {snapshot.get('turn', '?')}, us={p1_side}) ===", ""]

    out.append("--- OUTGOING (us → opp) ---")
    if p1_active and p2_active:
        for atk in p1_active:
            out.extend(
                await _attacker_block(
                    session, base_url, snapshot, atk, p2_active,
                    attacker_side=p1_side,
                    attacker_knowledge=p1_knowledge,
                    defender_knowledge=p2_knowledge,
                    direction="outgoing",
                )
            )
    else:
        out.append("(no active matchup)")
    out.append("")

    out.append("--- INCOMING (opp → us) ---")
    if p1_active and p2_active:
        for atk in p2_active:
            out.extend(
                await _attacker_block(
                    session, base_url, snapshot, atk, p1_active,
                    attacker_side=p2_side,
                    attacker_knowledge=p2_knowledge,
                    defender_knowledge=p1_knowledge,
                    direction="incoming",
                )
            )
    else:
        out.append("(no active matchup)")

    return "\n".join(out).rstrip() + "\n"


async def _calc_one_move_one_defender(
    session: aiohttp.ClientSession,
    base_url: str,
    move: str,
    attacker: dict[str, Any],
    defender: dict[str, Any],
    a_entry: dict[str, dict[str, int]],
    d_entry: dict[str, dict[str, int]],
    off_stat: str,
    def_stat: str,
    field_payload: dict[str, Any],
) -> dict[str, Any]:
    """Run the two boundary calcs (abs_low, abs_high) for one move on one
    defender, returning a dict ready for rendering."""
    att_low_evs = {s: 0 for s in STATS}
    att_low_evs[off_stat] = a_entry["min_evs"][off_stat]
    def_low_evs = {s: 0 for s in STATS}
    def_low_evs["hp"] = d_entry["max_evs"]["hp"]
    def_low_evs[def_stat] = d_entry["max_evs"][def_stat]

    att_high_evs = {s: 0 for s in STATS}
    att_high_evs[off_stat] = a_entry["max_evs"][off_stat]
    def_high_evs = {s: 0 for s in STATS}
    def_high_evs["hp"] = d_entry["min_evs"]["hp"]
    def_high_evs[def_stat] = d_entry["min_evs"][def_stat]

    abs_low = await _call_calc(
        session, base_url,
        {
            "attacker": _build_pokemon_payload(attacker, evs=att_low_evs),
            "defender": _build_pokemon_payload(defender, evs=def_low_evs),
            "move": move, "field": field_payload,
        },
    )
    abs_high = await _call_calc(
        session, base_url,
        {
            "attacker": _build_pokemon_payload(attacker, evs=att_high_evs),
            "defender": _build_pokemon_payload(defender, evs=def_high_evs),
            "move": move, "field": field_payload,
        },
    )

    return {"abs_low": abs_low, "abs_high": abs_high}


def _ko_chance(result: dict[str, Any]) -> str:
    """KO-chance string from the high-roll boundary calc."""
    return result["abs_high"].get("koChance") or ""


def _format_single_target_line(
    move: str, defender: dict[str, Any], result: dict[str, Any]
) -> str:
    abs_str = _fmt_range(result["abs_low"]["minPercent"], result["abs_high"]["maxPercent"])
    ko = _ko_chance(result)
    ko_part = f"  [{ko}]" if ko else ""
    return f"  {move} → {defender['species']:<14} {abs_str}{ko_part}"


def _format_spread_line(
    move: str, results_by_def: list[tuple[dict[str, Any], dict[str, Any]]]
) -> str:
    """One line for a spread move covering all listed defenders."""
    parts: list[str] = []
    ko_parts: list[str] = []
    for defender, res in results_by_def:
        abs_str = _fmt_range(res["abs_low"]["minPercent"], res["abs_high"]["maxPercent"])
        parts.append(f"{defender['species']} {abs_str}")
        ko = _ko_chance(res)
        if ko:
            ko_parts.append(f"{defender['species']}: {ko}")
    body = ", ".join(parts)
    ko_summary = f"  [{'; '.join(ko_parts)}]" if ko_parts else ""
    return f"  {move} [spread]: {body}{ko_summary}"


async def _attacker_block(
    session: aiohttp.ClientSession,
    base_url: str,
    snapshot: dict[str, Any],
    attacker: dict[str, Any],
    defenders: list[dict[str, Any]],
    *,
    attacker_side: str,
    attacker_knowledge: KnowledgeState,
    defender_knowledge: KnowledgeState,
    direction: str,
) -> list[str]:
    # Prefer the OTS-known full moveset (Bo3); fall back to chronologically
    # revealed moves (Bo1 / OTS games where the field is missing).
    known = attacker.get("knownMoves")
    moves: list[str] = (
        [m for m in known if m]
        if known
        else (attacker.get("revealedMoves") or [])
    )
    if not moves:
        return []

    field_payload = _field_payload(snapshot, attacker_side)
    a_entry = _entry_or_default(attacker_knowledge, attacker["species"])

    # Build per-defender entries once.
    def_meta = [{"pkm": d, "entry": _entry_or_default(defender_knowledge, d["species"])}
                for d in defenders]

    side_tag = "us" if direction == "outgoing" else "opp"
    label = f"[{side_tag} {attacker['species']}]"
    extras = []
    if attacker.get("status"):
        extras.append(f"status={attacker['status']}")
    if attacker.get("boosts"):
        extras.append(f"boosts={attacker['boosts']}")
    head = label + (f"  ({', '.join(extras)})" if extras else "")
    lines: list[str] = [head]
    chip_moves: list[str] = []

    for move in moves:
        meta = await _get_move_meta(session, base_url, move)
        category = meta.get("category", "Status")
        if category == "Status":
            continue
        target = meta.get("target", "normal")
        off_stat = "atk" if category == "Physical" else "spa"
        def_stat = "def" if category == "Physical" else "spd"

        # Run the calcs against every active defender on the other side.
        results_per_def: list[tuple[dict[str, Any], dict[str, Any]]] = []
        try:
            for dm in def_meta:
                res = await _calc_one_move_one_defender(
                    session, base_url, move,
                    attacker, dm["pkm"],
                    a_entry, dm["entry"],
                    off_stat, def_stat, field_payload,
                )
                results_per_def.append((dm["pkm"], res))
        except Exception as e:
            lines.append(f"  {move}  ERROR: {e}")
            continue

        # Chip filter: if the move can't break 15% on any defender, footer it.
        max_pct = max(
            res["abs_high"]["maxPercent"] for _, res in results_per_def
        ) if results_per_def else 0.0
        if max_pct < CHIP_DAMAGE_THRESHOLD_PCT:
            chip_moves.append(move)
            continue

        is_spread = target in SPREAD_TARGETS and len(results_per_def) > 1
        if is_spread:
            lines.append(_format_spread_line(move, results_per_def))
        else:
            for defender, res in results_per_def:
                lines.append(_format_single_target_line(move, defender, res))

    if chip_moves:
        lines.append(f"  …plus {len(chip_moves)} chip move(s): {', '.join(chip_moves)}")

    return lines


__all__ = ["generate_threat_matrix"]
