"""Dual-track damage envelope for one turn snapshot, ready for the SFT prompt.

Pipeline role:
    For one BoardState (one /parse_log snapshot), enumerate every plausible
    (attacker, move, defender) triple between the two sides and ask the calc
    microservice for two damage answers per matchup:

      - **Absolute**: the strict mathematical envelope using BOTH sides'
        current `KnowledgeState` bounds. Wide but provable.
      - **Probable (meta)**: the single calc result assuming both Pokémon are
        running their canonical meta spread (`canonical_priors`). Narrow,
        quick, and only as good as the prior.

    Whenever the canonical prior falls outside the absolute bounds for any
    relevant stat, the line is flagged `[PRIOR CONTRADICTED]` so the LLM
    can reason about off-meta opponents.

Inputs:
    snapshot           — one TurnSnapshot dict from /parse_log.
    p1_side            — "p1" or "p2", whichever is "us" for the LLM.
    p1_knowledge       — KnowledgeState for side p1.
    p2_knowledge       — KnowledgeState for side p2.

Outputs:
    A single text block with one line per (attacker, move, defender),
    grouped by direction (OUTGOING us → opp, INCOMING opp → us).

Isolation contract:
    HTTP-calls calc_microservice (`/calc`, `/dex/move`). Pure-data import of
    `canonical_priors`. No replay parsing, no LLM, no imports from
    `damage_inferencer` beyond the shared types/helpers.
"""
from __future__ import annotations

from typing import Any, Mapping

import aiohttp

from canonical_priors import ProbableSpread, get_probable_spread
from damage_inferencer import (
    DEFAULT_CALC_BASE_URL,
    KnowledgeState,
    STATS,
    get_move_category,
    init_knowledge_entry,
    species_key,
)


def _build_pokemon_payload(
    pkm: dict[str, Any],
    *,
    evs: Mapping[str, int] | None = None,
    nature: str | None = None,
    ivs: Mapping[str, int] | None = None,
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
    if nature is not None:
        payload["nature"] = nature
    if ivs is not None:
        payload["ivs"] = dict(ivs)
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


def _is_prior_contradicted(
    spread: ProbableSpread,
    entry: dict[str, dict[str, int]],
    relevant_stats: tuple[str, ...],
) -> bool:
    """True if the canonical EVs fall outside the proven bounds on any relevant stat."""
    for s in relevant_stats:
        ev = spread.evs.get(s, 0)
        if ev < entry["min_evs"][s] or ev > entry["max_evs"][s]:
            return True
    return False


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
    session: aiohttp.ClientSession | None = None,
    base_url: str = DEFAULT_CALC_BASE_URL,
) -> str:
    """Render the dual-track threat matrix for one turn as a text block."""
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
    for atk in p1_active:
        for defn in p2_active:
            out.extend(
                await _matchup(
                    session, base_url, snapshot, atk, defn,
                    attacker_side=p1_side,
                    attacker_knowledge=p1_knowledge,
                    defender_knowledge=p2_knowledge,
                    direction="outgoing",
                )
            )
    if not (p1_active and p2_active):
        out.append("(no active matchup)")
    out.append("")

    out.append("--- INCOMING (opp → us) ---")
    for atk in p2_active:
        for defn in p1_active:
            out.extend(
                await _matchup(
                    session, base_url, snapshot, atk, defn,
                    attacker_side=p2_side,
                    attacker_knowledge=p2_knowledge,
                    defender_knowledge=p1_knowledge,
                    direction="incoming",
                )
            )
    if not (p1_active and p2_active):
        out.append("(no active matchup)")

    return "\n".join(out).rstrip() + "\n"


async def _matchup(
    session: aiohttp.ClientSession,
    base_url: str,
    snapshot: dict[str, Any],
    attacker: dict[str, Any],
    defender: dict[str, Any],
    *,
    attacker_side: str,
    attacker_knowledge: KnowledgeState,
    defender_knowledge: KnowledgeState,
    direction: str,
) -> list[str]:
    moves = attacker.get("revealedMoves") or []
    if not moves:
        return []

    field_payload = _field_payload(snapshot, attacker_side)
    a_entry = _entry_or_default(attacker_knowledge, attacker["species"])
    d_entry = _entry_or_default(defender_knowledge, defender["species"])
    a_prior = get_probable_spread(attacker["species"])
    d_prior = get_probable_spread(defender["species"])

    label = (
        f"[us {attacker['species']}] vs [opp {defender['species']}]"
        if direction == "outgoing"
        else f"[opp {attacker['species']}] vs [us {defender['species']}]"
    )
    extras = []
    if attacker.get("status"):
        extras.append(f"atk_status={attacker['status']}")
    if attacker.get("boosts"):
        extras.append(f"atk_boosts={attacker['boosts']}")
    if defender.get("boosts"):
        extras.append(f"def_boosts={defender['boosts']}")
    head = label + (f"  ({', '.join(extras)})" if extras else "")
    lines: list[str] = [head]

    for move in moves:
        category = await get_move_category(session, base_url, move)
        if category == "Status":
            continue
        off_stat = "atk" if category == "Physical" else "spa"
        def_stat = "def" if category == "Physical" else "spd"

        # Absolute bounds: low end = bulkiest defender + weakest attacker;
        # high end = frailest defender + strongest attacker.
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

        try:
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
            prob = await _call_calc(
                session, base_url,
                {
                    "attacker": _build_pokemon_payload(
                        attacker, evs=a_prior.evs, nature=a_prior.nature, ivs=a_prior.ivs
                    ),
                    "defender": _build_pokemon_payload(
                        defender, evs=d_prior.evs, nature=d_prior.nature, ivs=d_prior.ivs
                    ),
                    "move": move, "field": field_payload,
                },
            )
        except Exception as e:
            lines.append(f"  {move}  ERROR: {e}")
            continue

        # Stats relevant to the contradiction check: the attacker's off-stat
        # for the attacker prior, the defender's HP+def-stat for the
        # defender prior. Either being outside its absolute box flags it.
        atk_contradicted = _is_prior_contradicted(a_prior, a_entry, (off_stat,))
        def_contradicted = _is_prior_contradicted(d_prior, d_entry, ("hp", def_stat))
        flag = "  [PRIOR CONTRADICTED]" if (atk_contradicted or def_contradicted) else ""

        absolute_str = _fmt_range(abs_low["minPercent"], abs_high["maxPercent"])
        probable_str = _fmt_range(prob["minPercent"], prob["maxPercent"])
        lines.append(
            f"  {move:<22} Absolute: {absolute_str}  |  Probable (meta): {probable_str}  ({prob['koChance']}){flag}"
        )

    return lines


__all__ = ["generate_threat_matrix"]
