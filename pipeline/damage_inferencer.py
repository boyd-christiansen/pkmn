"""Tighten EV bounds on opponent Pokémon by binary-searching observed damage events.

Pipeline role:
    Maintains an `OpponentKnowledgeState` per match — for each opponent species,
    a (min_evs, max_evs) box in [0, 252]^6 stat space. Each turn's observed
    damage events are fed in via `update_knowledge`, which uses the calc
    microservice to binary-search the consistent EV range and tighten the box.

    The bounds are then consumed by `threat_matrix.py` to produce low/high
    damage envelopes.

Inputs:
    snapshot_pre   — the /parse_log snapshot taken at the START of the turn
                     (provides volatile state: status, boosts, weather, ...).
    snapshot_post  — same shape, START of the next turn.
    action_log     — list[DamageEvent] of damage-dealing hits this turn.
    current_knowledge — OpponentKnowledgeState dict, mutated in place.

Outputs:
    The same `current_knowledge` dict, with `min_evs` ratcheted up and
    `max_evs` ratcheted down where new observations allow.

Isolation contract:
    HTTP-calls calc_microservice (`/calc`, `/dex/move`). No replay parsing,
    no LLM. No imports from sibling pipeline modules.

Algorithm:
    Damage % is monotonically decreasing in defender HP/Def/SpD EVs. For each
    relevant stat, two binary searches over [0, 252] find:
      - smallest EV where calc.minPercent ≤ observed.maxPercent  (new min_ev)
      - largest  EV where calc.maxPercent ≥ observed.minPercent  (new max_ev)
    Other unknowns are held at the *least restrictive* bound during each
    search (e.g. when bounding Def's lower edge, hold HP at max_evs) so that
    bounds are never over-tightened due to coupling between unknowns.

    The "fuzzy HP" tolerance widens the observation by ±0.9% to account for
    the 1% rounding the spectator sees on opponent HP bars.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping

import aiohttp

DEFAULT_CALC_BASE_URL = "http://localhost:3000"
STATS = ("hp", "atk", "def", "spa", "spd", "spe")
DEFAULT_MIN_EV = 0
DEFAULT_MAX_EV = 252
FUZZY_HP_TOLERANCE = 0.9  # percent


OpponentKnowledgeState = dict[str, dict[str, dict[str, int]]]
# Shape:
#   {
#     "calyrexshadow": {
#       "min_evs": {"hp": 0, "atk": 0, ...},
#       "max_evs": {"hp": 252, "atk": 252, ...},
#     },
#     ...
#   }


@dataclass
class DamageEvent:
    """A single damage-dealing hit observed within one turn.

    `hp_before`/`hp_after` are spectator-visible HP percentages (0-100).
    `is_ko` widens the observation: damage was at least `hp_before`, with no
    upper bound from this event alone.
    """

    attacker_species: str
    attacker_side: str  # "p1" | "p2"
    defender_species: str
    defender_side: str  # "p1" | "p2"
    move: str
    hp_before: float
    hp_after: float
    is_ko: bool = False


def species_key(species: str) -> str:
    """Normalise a species name to the @pkmn-style ID (lowercase, alnum only)."""
    return "".join(c for c in species.lower() if c.isalnum())


def init_knowledge_entry() -> dict[str, dict[str, int]]:
    return {
        "min_evs": {s: DEFAULT_MIN_EV for s in STATS},
        "max_evs": {s: DEFAULT_MAX_EV for s in STATS},
    }


def init_knowledge(species_list: list[str]) -> OpponentKnowledgeState:
    return {species_key(s): init_knowledge_entry() for s in species_list}


def _find_active_pokemon(
    snapshot: dict[str, Any], side: str, species: str
) -> dict[str, Any] | None:
    target = species_key(species)
    for p in snapshot.get(side, {}).get("active", []):
        if species_key(p["species"]) == target:
            return p
    return None


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


def observed_damage_range(
    hp_before: float, hp_after: float, is_ko: bool, fuzzy: float = FUZZY_HP_TOLERANCE
) -> tuple[float, float]:
    """Map the spectator-visible HP drop to a (min%, max%) actual-damage range."""
    if is_ko or hp_after <= 0:
        return (max(0.0, hp_before - fuzzy), float("inf"))
    nominal = hp_before - hp_after
    return (max(0.0, nominal - fuzzy), min(100.0, nominal + fuzzy))


async def _call_calc(
    session: aiohttp.ClientSession, base_url: str, payload: dict[str, Any]
) -> dict[str, Any]:
    async with session.post(f"{base_url}/calc", json=payload) as r:
        if r.status >= 400:
            text = await r.text()
            raise RuntimeError(f"/calc {r.status}: {text[:200]}")
        return await r.json()


_MOVE_CATEGORY_CACHE: dict[str, str] = {}


async def get_move_category(
    session: aiohttp.ClientSession, base_url: str, move: str
) -> str:
    """Return 'Physical' | 'Special' | 'Status'. Cached process-wide."""
    key = species_key(move)
    if key in _MOVE_CATEGORY_CACHE:
        return _MOVE_CATEGORY_CACHE[key]
    async with session.get(f"{base_url}/dex/move/{key}") as r:
        if r.status == 404:
            _MOVE_CATEGORY_CACHE[key] = "Status"
            return "Status"
        if r.status >= 400:
            raise RuntimeError(f"/dex/move {r.status}")
        cat = (await r.json()).get("category", "Status")
        _MOVE_CATEGORY_CACHE[key] = cat
        return cat


EvalFn = Callable[[int], Awaitable[tuple[float, float]]]


async def _binary_search_min_ev(eval_fn: EvalFn, target_max: float) -> int | None:
    """Smallest EV in [0, 252] where calc.minPercent ≤ target_max.

    Damage % is monotonically decreasing in defensive EVs, so a textbook
    binary search applies. Returns None if no EV in range satisfies the
    constraint (typically means our other priors are wrong or the observation
    is noisy — caller should skip the update).
    """
    lo, hi = DEFAULT_MIN_EV, DEFAULT_MAX_EV
    calc_min_lo, _ = await eval_fn(lo)
    if calc_min_lo <= target_max:
        return lo
    calc_min_hi, _ = await eval_fn(hi)
    if calc_min_hi > target_max:
        return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        calc_min, _ = await eval_fn(mid)
        if calc_min <= target_max:
            hi = mid
        else:
            lo = mid
    return hi


async def _binary_search_max_ev(eval_fn: EvalFn, target_min: float) -> int | None:
    """Largest EV in [0, 252] where calc.maxPercent ≥ target_min."""
    lo, hi = DEFAULT_MIN_EV, DEFAULT_MAX_EV
    _, calc_max_hi = await eval_fn(hi)
    if calc_max_hi >= target_min:
        return hi
    _, calc_max_lo = await eval_fn(lo)
    if calc_max_lo < target_min:
        return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        _, calc_max = await eval_fn(mid)
        if calc_max >= target_min:
            lo = mid
        else:
            hi = mid
    return lo


async def update_knowledge(
    snapshot_pre: dict[str, Any],
    snapshot_post: dict[str, Any],  # noqa: ARG001 - reserved for future use (cross-event sanity checks)
    action_log: list[DamageEvent],
    current_knowledge: OpponentKnowledgeState,
    *,
    session: aiohttp.ClientSession | None = None,
    base_url: str = DEFAULT_CALC_BASE_URL,
    fuzzy_hp_pct: float = FUZZY_HP_TOLERANCE,
) -> OpponentKnowledgeState:
    """Tighten EV bounds in `current_knowledge` from this turn's damage events.

    `snapshot_pre` provides the volatile state used in calc payloads
    (status, boosts, weather, terrain, side conditions). Mutates
    `current_knowledge` in place and returns it for chaining.
    """
    own_session = session is None
    if own_session:
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60))
    try:
        for event in action_log:
            await _process_event(
                event, snapshot_pre, current_knowledge, session, base_url, fuzzy_hp_pct
            )
    finally:
        if own_session:
            await session.close()
    return current_knowledge


async def _process_event(
    event: DamageEvent,
    snapshot_pre: dict[str, Any],
    current_knowledge: OpponentKnowledgeState,
    session: aiohttp.ClientSession,
    base_url: str,
    fuzzy: float,
) -> None:
    defender_key = species_key(event.defender_species)
    if defender_key not in current_knowledge:
        return  # we only learn about Pokémon we're tracking

    attacker_pkm = _find_active_pokemon(snapshot_pre, event.attacker_side, event.attacker_species)
    defender_pkm = _find_active_pokemon(snapshot_pre, event.defender_side, event.defender_species)
    if attacker_pkm is None or defender_pkm is None:
        return

    category = await get_move_category(session, base_url, event.move)
    if category == "Status":
        return
    def_stat = "def" if category == "Physical" else "spd"

    target_min, target_max = observed_damage_range(
        event.hp_before, event.hp_after, event.is_ko, fuzzy
    )
    field_payload = _field_payload(snapshot_pre, event.attacker_side)

    attacker_key = species_key(event.attacker_species)
    if attacker_key in current_knowledge:
        atk_entry = current_knowledge[attacker_key]
        atk_evs: Mapping[str, int] | None = {
            s: (atk_entry["min_evs"][s] + atk_entry["max_evs"][s]) // 2 for s in STATS
        }
    else:
        atk_evs = None
    attacker_payload = _build_pokemon_payload(attacker_pkm, evs=atk_evs)

    entry = current_knowledge[defender_key]

    async def eval_with(defender_evs: Mapping[str, int]) -> tuple[float, float]:
        defender_payload = _build_pokemon_payload(defender_pkm, evs=defender_evs)
        result = await _call_calc(
            session,
            base_url,
            {
                "attacker": attacker_payload,
                "defender": defender_payload,
                "move": event.move,
                "field": field_payload,
            },
        )
        return float(result["minPercent"]), float(result["maxPercent"])

    def eval_pair(hp_ev: int, def_ev: int) -> Awaitable[tuple[float, float]]:
        evs = {s: 0 for s in STATS}
        evs["hp"] = hp_ev
        evs[def_stat] = def_ev
        return eval_with(evs)

    # Tighten the relevant defensive stat (Def for physical, SpD for special).
    # Hold HP at the *least restrictive* bound so we never over-tighten via
    # cross-stat coupling: high HP makes a small min_def consistent; low HP
    # makes a large max_def consistent.
    new_min_def = await _binary_search_min_ev(
        lambda d: eval_pair(entry["max_evs"]["hp"], d), target_max
    )
    if new_min_def is not None and new_min_def > entry["min_evs"][def_stat]:
        entry["min_evs"][def_stat] = new_min_def

    if not event.is_ko:
        new_max_def = await _binary_search_max_ev(
            lambda d: eval_pair(entry["min_evs"]["hp"], d), target_min
        )
        if new_max_def is not None and new_max_def < entry["max_evs"][def_stat]:
            entry["max_evs"][def_stat] = new_max_def

    # Same shape for HP — hold defensive stat at the least restrictive end.
    new_min_hp = await _binary_search_min_ev(
        lambda h: eval_pair(h, entry["max_evs"][def_stat]), target_max
    )
    if new_min_hp is not None and new_min_hp > entry["min_evs"]["hp"]:
        entry["min_evs"]["hp"] = new_min_hp

    if not event.is_ko:
        new_max_hp = await _binary_search_max_ev(
            lambda h: eval_pair(h, entry["min_evs"][def_stat]), target_min
        )
        if new_max_hp is not None and new_max_hp < entry["max_evs"]["hp"]:
            entry["max_evs"]["hp"] = new_max_hp


__all__ = [
    "DEFAULT_CALC_BASE_URL",
    "DamageEvent",
    "FUZZY_HP_TOLERANCE",
    "OpponentKnowledgeState",
    "STATS",
    "get_move_category",
    "init_knowledge",
    "init_knowledge_entry",
    "observed_damage_range",
    "species_key",
    "update_knowledge",
]
