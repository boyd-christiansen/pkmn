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

    When the canonical prior is clipped from the inferred KnowledgeState
    bounds by ≥ 40 EVs on any relevant stat (`PRIOR_CONTRADICTION_EV_GAP`),
    the Probable calc is **skipped entirely** for that line and only the
    Absolute envelope is shown, tagged `(off-meta)`. This avoids
    surfacing a meta range we've already disproven.

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


PRIOR_CONTRADICTION_EV_GAP = 40   # only flag off-meta when the canonical EV is clipped by ≥ this many EVs
CHIP_DAMAGE_THRESHOLD_PCT = 15.0  # max-percent below this = "chip", rolled into a footer line
SPREAD_TARGETS = ("allAdjacentFoes", "foeSide", "allAdjacent")


def _is_prior_contradicted(
    spread: ProbableSpread,
    entry: dict[str, dict[str, int]],
    relevant_stats: tuple[str, ...],
) -> bool:
    """True if the canonical EVs fall outside the proven bounds on any relevant
    stat by at least `PRIOR_CONTRADICTION_EV_GAP` EVs (avoids flagging on
    edge-case 1-EV clips)."""
    for s in relevant_stats:
        ev = spread.evs.get(s, 0)
        lo = entry["min_evs"][s]
        hi = entry["max_evs"][s]
        if ev < lo and (lo - ev) >= PRIOR_CONTRADICTION_EV_GAP:
            return True
        if ev > hi and (ev - hi) >= PRIOR_CONTRADICTION_EV_GAP:
            return True
    return False


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
    format_id: str | None = None,
    session: aiohttp.ClientSession | None = None,
    base_url: str = DEFAULT_CALC_BASE_URL,
) -> str:
    """Render the dual-track threat matrix for one turn as a text block.

    `format_id` is forwarded to `canonical_priors.get_probable_spread` so the
    Probable track uses real Smogon usage data when a chaos cache for that
    format is present (run `python canonical_priors.py --format-id ...` to
    populate the cache). Without a cache it falls back to the curated
    table + heuristic.
    """
    own_session = session is None
    if own_session:
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60))
    try:
        return await _generate(
            snapshot, p1_side, p1_knowledge, p2_knowledge, session, base_url, format_id
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
    format_id: str | None,
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
                    format_id=format_id,
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
                    format_id=format_id,
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
    a_prior: ProbableSpread,
    d_prior: ProbableSpread,
    off_stat: str,
    def_stat: str,
    field_payload: dict[str, Any],
) -> dict[str, Any]:
    """Run the three calcs (abs_low, abs_high, probable when not contradicted)
    for one move on one defender, returning a dict ready for rendering."""
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

    atk_contradicted = _is_prior_contradicted(a_prior, a_entry, (off_stat,))
    def_contradicted = _is_prior_contradicted(d_prior, d_entry, ("hp", def_stat))
    off_meta = atk_contradicted or def_contradicted

    prob: dict[str, Any] | None = None
    if not off_meta:
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

    return {
        "abs_low": abs_low,
        "abs_high": abs_high,
        "prob": prob,
        "off_meta": off_meta,
    }


def _ko_chance(result: dict[str, Any]) -> str:
    """Pick a representative KO-chance string. Prefer the Probable calc's
    (more decision-relevant when the meta is in scope); fall back to
    abs_high when off-meta."""
    if result.get("prob"):
        return result["prob"].get("koChance") or ""
    return result["abs_high"].get("koChance") or ""


def _format_single_target_line(
    move: str, defender: dict[str, Any], result: dict[str, Any]
) -> str:
    abs_str = _fmt_range(result["abs_low"]["minPercent"], result["abs_high"]["maxPercent"])
    ko = _ko_chance(result)
    ko_part = f"  [{ko}]" if ko else ""
    if result["off_meta"]:
        return f"  {move} → {defender['species']:<14} {abs_str}{ko_part}  (off-meta)"
    prob_str = _fmt_range(result["prob"]["minPercent"], result["prob"]["maxPercent"])
    return f"  {move} → {defender['species']:<14} {abs_str}  | meta {prob_str}{ko_part}"


def _format_spread_line(
    move: str, results_by_def: list[tuple[dict[str, Any], dict[str, Any]]]
) -> str:
    """One line for a spread move covering all listed defenders."""
    parts: list[str] = []
    any_off_meta = False
    ko_parts: list[str] = []
    for defender, res in results_by_def:
        abs_str = _fmt_range(res["abs_low"]["minPercent"], res["abs_high"]["maxPercent"])
        parts.append(f"{defender['species']} {abs_str}")
        ko = _ko_chance(res)
        if ko:
            ko_parts.append(f"{defender['species']}: {ko}")
        any_off_meta = any_off_meta or res["off_meta"]
    body = ", ".join(parts)
    ko_summary = f"  [{'; '.join(ko_parts)}]" if ko_parts else ""
    suffix = "  (off-meta)" if any_off_meta else ""
    return f"  {move} [spread]: {body}{ko_summary}{suffix}"


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
    format_id: str | None,
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
    a_prior = get_probable_spread(attacker["species"], format_id)

    # Build per-defender entries + priors once.
    def_meta = []
    for d in defenders:
        d_entry = _entry_or_default(defender_knowledge, d["species"])
        d_prior = get_probable_spread(d["species"], format_id)
        def_meta.append({"pkm": d, "entry": d_entry, "prior": d_prior})

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
                    a_entry, dm["entry"], a_prior, dm["prior"],
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
