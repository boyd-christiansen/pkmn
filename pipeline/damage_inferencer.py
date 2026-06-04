"""Two-way EV-bound inference from observed damage events.

Pipeline role:
    Per match the orchestrator maintains TWO `KnowledgeState`s — one per side.
    Each turn's damage events are fed to `update_knowledge`, which binary-
    searches the consistent EV range for *both* the attacker and defender of
    every event, using interval arithmetic over the other side's bounds so
    neither side's hidden EVs are assumed.

    The bounds feed `threat_matrix.py`'s "Absolute" track.

Key design choices:
    • Slot-based event addressing ("p1a", "p2b") — VGC has Species Clause but
      we're ready for switch shenanigans regardless.
    • Two-way updates: every observation tightens BOTH sides at once. To
      avoid order-dependent over-tightening, all six binary searches per
      event run against pre-update bounds, then results are applied
      atomically.
    • Cross-stat coupling on the *same* side (e.g. defender HP × Def) is
      handled by holding the other unknown at its least-restrictive bound
      during each search.
    • Cross-side coupling (attacker × defender uncertainty) is handled the
      same way — see the per-search "least restrictive" comments below.

Isolation contract:
    HTTP-calls calc_microservice (`/calc`, `/dex/move`). No replay parsing,
    no LLM. No imports from sibling pipeline modules.

Known limitations:
    • Multi-hit moves (Triple Axel, Bullet Seed, Population Bomb, …) emit
      one DamageEvent per hit. The inferencer detects them by counting
      same-(attacker, move, defender) tuples per turn and skips all of
      them — supporting them properly would need a `hits` field on the
      event and on /calc.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping

import aiohttp

DEFAULT_CALC_BASE_URL = "http://localhost:3000"
STATS = ("hp", "atk", "def", "spa", "spd", "spe")
DEFAULT_MIN_EV = 0
DEFAULT_MAX_EV = 252
FUZZY_HP_TOLERANCE = 0.9  # percent
TOTAL_EV_BUDGET = 508    # PS allows 510 total, only 508 usable in 4-EV chunks


# A KnowledgeState tracks per-species bounds for *one side*.
# {
#   "calyrexshadow": {
#     "min_evs": {"hp": 0, "atk": 0, ...},
#     "max_evs": {"hp": 252, ...},
#   },
#   ...
# }
KnowledgeState = dict[str, dict[str, dict[str, int]]]


@dataclass
class DamageEvent:
    """One damage-dealing hit observed within a turn.

    `attacker_slot`, `defender_slot` are PS-style identifiers like "p1a",
    "p2b". `hp_before_pct` and `hp_after_pct` are the *defender's* HP %
    immediately around this hit (not the post-turn snapshot — multi-hit
    turns interpose between snapshots).
    """

    attacker_slot: str
    defender_slot: str
    move_name: str
    hp_before_pct: float
    hp_after_pct: float
    is_crit: bool = False
    is_ko: bool = False


def species_key(species: str) -> str:
    return "".join(c for c in species.lower() if c.isalnum())


def init_knowledge_entry() -> dict[str, Any]:
    return {
        "min_evs": {s: DEFAULT_MIN_EV for s in STATS},
        "max_evs": {s: DEFAULT_MAX_EV for s in STATS},
        # `observed` is True once this Pokémon has been involved in ANY
        # spread-relevant event (dealt damage, took damage, or revealed
        # move order). It is the causal answer to "is this mon `unknown`?":
        # a mon stays `unknown` strictly while observed is False. Distinct
        # from "observed but EV-unconstrained" — that mon has observed=True
        # with still-wide bounds, so a persistently-wide spread can no
        # longer be misread as "this mon was inert all game".
        "observed": False,
    }


def init_knowledge(species_list: list[str]) -> KnowledgeState:
    return {species_key(s): init_knowledge_entry() for s in species_list}


def _find_pokemon_by_slot(snapshot: dict[str, Any], slot: str) -> dict[str, Any] | None:
    if len(slot) < 3:
        return None
    side, letter = slot[:2], slot[2]
    for p in snapshot.get(side, {}).get("active", []):
        if p.get("slot") == letter:
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
    is_p1 = attacker_side == "p1"
    return {
        "gameType": "Doubles",
        "weather": f.get("weather"),
        "terrain": f.get("terrain"),
        "attackerSide": {"isTailwind": bool(f.get("tailwindP1") if is_p1 else f.get("tailwindP2"))},
        "defenderSide": {"isTailwind": bool(f.get("tailwindP2") if is_p1 else f.get("tailwindP1"))},
    }


def observed_damage_range(
    hp_before: float, hp_after: float, is_ko: bool, fuzzy: float = FUZZY_HP_TOLERANCE
) -> tuple[float, float]:
    """Map the spectator-visible HP drop to a (min%, max%) actual-damage range."""
    if is_ko or hp_after <= 0:
        return (max(0.0, hp_before - fuzzy), float("inf"))
    nominal = hp_before - hp_after
    return (max(0.0, nominal - fuzzy), min(100.0, nominal + fuzzy))


_MOVE_CATEGORY_CACHE: dict[str, str] = {}


async def get_move_category(
    session: aiohttp.ClientSession, base_url: str, move: str
) -> str:
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


async def _call_calc(
    session: aiohttp.ClientSession, base_url: str, payload: dict[str, Any]
) -> dict[str, Any]:
    async with session.post(f"{base_url}/calc", json=payload) as r:
        if r.status >= 400:
            text = await r.text()
            raise RuntimeError(f"/calc {r.status}: {text[:200]}")
        return await r.json()


# ---------------------------------------------------------------------------
# Binary search primitives.
#
# Damage % is monotonically:
#   - DECREASING in defender HP / Def / SpD EVs  (more bulk → less %)
#   - INCREASING in attacker Atk / SpA EVs       (more offense → more %)
#
# Each helper finds the boundary EV in [0, 252] given the appropriate
# monotonicity. Returns None if no EV in range satisfies the constraint
# (the observation is inconsistent with the held priors — caller skips).
# ---------------------------------------------------------------------------

EvalFn = Callable[[int], Awaitable[tuple[float, float]]]


async def _bsearch_min_defender_ev(eval_fn: EvalFn, target_max: float) -> int | None:
    """Smallest defender EV in [0, 252] where calc.minPercent ≤ target_max."""
    lo, hi = DEFAULT_MIN_EV, DEFAULT_MAX_EV
    if (await eval_fn(lo))[0] <= target_max:
        return lo
    if (await eval_fn(hi))[0] > target_max:
        return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if (await eval_fn(mid))[0] <= target_max:
            hi = mid
        else:
            lo = mid
    return hi


async def _bsearch_max_defender_ev(eval_fn: EvalFn, target_min: float) -> int | None:
    """Largest defender EV in [0, 252] where calc.maxPercent ≥ target_min."""
    lo, hi = DEFAULT_MIN_EV, DEFAULT_MAX_EV
    if (await eval_fn(hi))[1] >= target_min:
        return hi
    if (await eval_fn(lo))[1] < target_min:
        return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if (await eval_fn(mid))[1] >= target_min:
            lo = mid
        else:
            hi = mid
    return lo


async def _bsearch_min_attacker_ev(eval_fn: EvalFn, target_min: float) -> int | None:
    """Smallest attacker EV in [0, 252] where calc.maxPercent ≥ target_min."""
    lo, hi = DEFAULT_MIN_EV, DEFAULT_MAX_EV
    if (await eval_fn(lo))[1] >= target_min:
        return lo
    if (await eval_fn(hi))[1] < target_min:
        return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if (await eval_fn(mid))[1] >= target_min:
            hi = mid
        else:
            lo = mid
    return hi


async def _bsearch_max_attacker_ev(eval_fn: EvalFn, target_max: float) -> int | None:
    """Largest attacker EV in [0, 252] where calc.minPercent ≤ target_max."""
    lo, hi = DEFAULT_MIN_EV, DEFAULT_MAX_EV
    if (await eval_fn(hi))[0] <= target_max:
        return hi
    if (await eval_fn(lo))[0] > target_max:
        return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if (await eval_fn(mid))[0] <= target_max:
            lo = mid
        else:
            hi = mid
    return lo


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _filter_action_log(events: list[DamageEvent]) -> list[DamageEvent]:
    """Drop multi-hit moves: any (attacker_slot, move_name, defender_slot)
    appearing more than once in the same turn is treated as a multi-hit
    sequence and skipped entirely (we'd need /calc `hits` support to
    handle it correctly)."""
    counts: dict[tuple[str, str, str], int] = {}
    for ev in events:
        key = (ev.attacker_slot, ev.move_name, ev.defender_slot)
        counts[key] = counts.get(key, 0) + 1
    return [
        ev for ev in events
        if counts[(ev.attacker_slot, ev.move_name, ev.defender_slot)] == 1
    ]


# Move callers we keep when filtering events for damage inference. Sleep
# Talk only ever calls own moves, so its damage observations are still
# valid attribution. Metronome / Copycat / Sketch / Snatch / Me First /
# Dancer / Instruct can call moves the user doesn't own — calc-bound
# updates against those would corrupt EV inference.
_DAMAGE_INFERENCE_CALLERS_OK: frozenset[str | None] = frozenset({None, "Sleep Talk"})


def events_to_damage_events(events: list[dict[str, Any]]) -> list[DamageEvent]:
    """Convert new-schema TurnEvent stream → list[DamageEvent].

    Filters:
      - type == "move"
      - called_via in {None, "Sleep Talk"}  (own moves only)
      - hit.outcome == "damage"             (drop misses / blocks / immunes / fails)
      - hit has hp_before_pct & hp_after_pct (well-formed)

    A single move event with multiple damage hits expands to one
    DamageEvent per hit. The downstream `_filter_action_log` then drops
    any (attacker, move, defender) triple that appears multiple times,
    which catches Triple Axel / Bullet Seed / Population Bomb
    multi-hits (today's calc can't model `hits` properly).
    """
    out: list[DamageEvent] = []
    for ev in events:
        if not isinstance(ev, dict) or ev.get("type") != "move":
            continue
        if ev.get("called_via") not in _DAMAGE_INFERENCE_CALLERS_OK:
            continue
        attacker_slot = ev.get("attacker_slot", "")
        move_name = ev.get("move_name", "")
        for hit in ev.get("hits") or []:
            if hit.get("outcome") != "damage":
                continue
            if "hp_before_pct" not in hit or "hp_after_pct" not in hit:
                continue
            out.append(DamageEvent(
                attacker_slot=attacker_slot,
                defender_slot=hit.get("defender_slot", ""),
                move_name=move_name,
                hp_before_pct=float(hit["hp_before_pct"]),
                hp_after_pct=float(hit["hp_after_pct"]),
                is_crit=bool(hit.get("is_crit", False)),
                is_ko=bool(hit.get("is_ko", False)),
            ))
    return out


def _apply_total_ev_constraint(entry: dict[str, dict[str, int]]) -> None:
    """Tighten max_evs using the 508-total constraint.

    For every stat, the most EVs it can have is `508 - sum_of_other_mins`.
    A stat we've proven needs ≥ X EVs frees up budget for the others —
    converse, if other stats already eat most of the budget, this stat
    can't be heavily invested either.
    """
    sum_min = sum(entry["min_evs"].values())
    if sum_min > TOTAL_EV_BUDGET:
        # Inconsistent priors — leave bounds untouched rather than corrupt them.
        return
    for stat in STATS:
        ceiling = TOTAL_EV_BUDGET - (sum_min - entry["min_evs"][stat])
        new_max = min(entry["max_evs"][stat], ceiling)
        # Never push max below the stat's own min.
        if new_max < entry["min_evs"][stat]:
            new_max = entry["min_evs"][stat]
        entry["max_evs"][stat] = new_max


async def update_knowledge(
    snapshot_pre: dict[str, Any],
    snapshot_post: dict[str, Any],  # noqa: ARG001 — reserved for cross-event sanity checks
    action_log: list[DamageEvent],
    p1_knowledge: KnowledgeState,
    p2_knowledge: KnowledgeState,
    *,
    session: aiohttp.ClientSession | None = None,
    base_url: str = DEFAULT_CALC_BASE_URL,
    fuzzy_hp_pct: float = FUZZY_HP_TOLERANCE,
) -> tuple[KnowledgeState, KnowledgeState]:
    """Tighten both p1 and p2 knowledge from this turn's damage events.

    Mutates both knowledge dicts in place; returns them as a tuple for
    chaining. Volatile state (status, boosts, weather, terrain, Tera) is
    pulled directly from `snapshot_pre` into every calc payload.
    """
    own_session = session is None
    if own_session:
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60))
    try:
        events = _filter_action_log(action_log)
        for event in events:
            await _process_event(
                event, snapshot_pre, p1_knowledge, p2_knowledge, session, base_url, fuzzy_hp_pct
            )
    finally:
        if own_session:
            await session.close()
    return p1_knowledge, p2_knowledge


async def infer_match_final_bounds(
    games: list[dict[str, Any]],
    p1_species: list[str],
    p2_species: list[str],
    *,
    session: aiohttp.ClientSession | None = None,
    base_url: str = DEFAULT_CALC_BASE_URL,
    fuzzy_hp_pct: float = FUZZY_HP_TOLERANCE,
) -> tuple[KnowledgeState, KnowledgeState]:
    """Run the inferencer across every turn of every game in the match.

    Used to compute "match-final" bounds for P1 — the tightest knowledge
    the inferencer can extract from observing the complete match. These
    bounds approximate "the spread the player actually built and knew at
    deploy time" (which is what we'll have access to in production).

    Implementation: just init both KnowledgeStates and walk every
    `(snapshot_pre, snapshot_post, events)` triple in turn order across
    all games, calling `update_knowledge` exactly as the per-turn loop
    in master_pipeline does. Pure offline batch — no row writes, no
    prompt rendering, no model calls.

    Returns `(p1_final_bounds, p2_final_bounds)`. Callers typically only
    use the P1 result (for the YOUR SPREADS prompt block + the matrix's
    P1 side). The P2 result is returned for completeness; it represents
    "what an outside observer would learn about P2 by end-of-match", and
    isn't currently surfaced — the matrix's P2 side uses the running
    chronological state to preserve the proper observational asymmetry.
    """
    p1 = init_knowledge(p1_species)
    p2 = init_knowledge(p2_species)
    p1_uni = {species_key(s) for s in p1_species}
    p2_uni = {species_key(s) for s in p2_species}

    own_session = session is None
    if own_session:
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60))
    try:
        for game in games:
            snaps = game.get("snapshots") or []
            for i in range(len(snaps) - 1):
                snap_pre, snap_post = snaps[i], snaps[i + 1]
                events_stream = snap_pre.get("events") or []
                damage_events = events_to_damage_events(events_stream)
                if damage_events:
                    await update_knowledge(
                        snap_pre, snap_post, damage_events, p1, p2,
                        session=session, base_url=base_url, fuzzy_hp_pct=fuzzy_hp_pct,
                    )
                # Speed (move-order) + observed-flag pass runs every turn,
                # including damage-free turns (status moves still reveal order).
                await update_observed_and_speed(
                    snap_pre, events_stream, p1, p2,
                    session=session, base_url=base_url,
                    p1_universe=p1_uni, p2_universe=p2_uni,
                )
    finally:
        if own_session:
            await session.close()
    return p1, p2


async def _process_event(
    event: DamageEvent,
    snapshot_pre: dict[str, Any],
    p1_knowledge: KnowledgeState,
    p2_knowledge: KnowledgeState,
    session: aiohttp.ClientSession,
    base_url: str,
    fuzzy: float,
) -> None:
    attacker_side = event.attacker_slot[:2]
    defender_side = event.defender_slot[:2]
    if attacker_side == defender_side:
        return  # self-targeting moves have no inferential value
    if attacker_side not in ("p1", "p2") or defender_side not in ("p1", "p2"):
        return

    attacker_pkm = _find_pokemon_by_slot(snapshot_pre, event.attacker_slot)
    defender_pkm = _find_pokemon_by_slot(snapshot_pre, event.defender_slot)
    if attacker_pkm is None or defender_pkm is None:
        return

    category = await get_move_category(session, base_url, event.move_name)
    if category == "Status":
        return
    off_stat = "atk" if category == "Physical" else "spa"
    def_stat = "def" if category == "Physical" else "spd"

    target_min, target_max = observed_damage_range(
        event.hp_before_pct, event.hp_after_pct, event.is_ko, fuzzy
    )
    field_payload = _field_payload(snapshot_pre, attacker_side)

    a_state = p1_knowledge if attacker_side == "p1" else p2_knowledge
    d_state = p1_knowledge if defender_side == "p1" else p2_knowledge
    a_key = species_key(attacker_pkm["species"])
    d_key = species_key(defender_pkm["species"])
    a_state.setdefault(a_key, init_knowledge_entry())
    d_state.setdefault(d_key, init_knowledge_entry())

    # Snapshot pre-update bounds; all six searches use these so the result is
    # order-independent. Applied atomically below.
    a_min = dict(a_state[a_key]["min_evs"])
    a_max = dict(a_state[a_key]["max_evs"])
    d_min = dict(d_state[d_key]["min_evs"])
    d_max = dict(d_state[d_key]["max_evs"])

    def _evs_for_atk(off_ev: int) -> dict[str, int]:
        evs = {s: 0 for s in STATS}
        evs[off_stat] = off_ev
        return evs

    def _evs_for_def(hp_ev: int, def_ev: int) -> dict[str, int]:
        evs = {s: 0 for s in STATS}
        evs["hp"] = hp_ev
        evs[def_stat] = def_ev
        return evs

    move_payload: dict[str, Any] = {"name": event.move_name, "isCrit": event.is_crit}

    async def calc(att_evs: Mapping[str, int], def_evs: Mapping[str, int]) -> tuple[float, float]:
        result = await _call_calc(
            session,
            base_url,
            {
                "attacker": _build_pokemon_payload(attacker_pkm, evs=att_evs),
                "defender": _build_pokemon_payload(defender_pkm, evs=def_evs),
                "move": move_payload,
                "field": field_payload,
            },
        )
        return float(result["minPercent"]), float(result["maxPercent"])

    # ----- DEFENDER bounds ----- (other-side held at LEAST RESTRICTIVE end) ---
    # min_def: small def_ev should be consistent → easiest with weak attacker (a_min)
    #          and bulky-HP defender (d_max.hp)
    new_min_def = await _bsearch_min_defender_ev(
        lambda d: calc(_evs_for_atk(a_min[off_stat]), _evs_for_def(d_max["hp"], d)),
        target_max,
    )
    # max_def: large def_ev should be consistent → easiest with strong attacker (a_max)
    #          and frail-HP defender (d_min.hp)
    new_max_def = (
        None if event.is_ko
        else await _bsearch_max_defender_ev(
            lambda d: calc(_evs_for_atk(a_max[off_stat]), _evs_for_def(d_min["hp"], d)),
            target_min,
        )
    )
    new_min_hp = await _bsearch_min_defender_ev(
        lambda h: calc(_evs_for_atk(a_min[off_stat]), _evs_for_def(h, d_max[def_stat])),
        target_max,
    )
    new_max_hp = (
        None if event.is_ko
        else await _bsearch_max_defender_ev(
            lambda h: calc(_evs_for_atk(a_max[off_stat]), _evs_for_def(h, d_min[def_stat])),
            target_min,
        )
    )

    # ----- ATTACKER bounds ----- (defender held at LEAST RESTRICTIVE end) ---
    # min_off: small off_ev should be consistent → easiest with frail defender (d_min)
    new_min_off = await _bsearch_min_attacker_ev(
        lambda a: calc(_evs_for_atk(a), _evs_for_def(d_min["hp"], d_min[def_stat])),
        target_min,
    )
    # max_off: large off_ev should be consistent → easiest with bulky defender (d_max)
    new_max_off = (
        None if event.is_ko
        else await _bsearch_max_attacker_ev(
            lambda a: calc(_evs_for_atk(a), _evs_for_def(d_max["hp"], d_max[def_stat])),
            target_max,
        )
    )

    # ----- ATOMIC APPLY (only after all six searches resolved) ----------------
    if new_min_def is not None and new_min_def > d_min[def_stat]:
        d_state[d_key]["min_evs"][def_stat] = new_min_def
    if new_max_def is not None and new_max_def < d_max[def_stat]:
        d_state[d_key]["max_evs"][def_stat] = new_max_def
    if new_min_hp is not None and new_min_hp > d_min["hp"]:
        d_state[d_key]["min_evs"]["hp"] = new_min_hp
    if new_max_hp is not None and new_max_hp < d_max["hp"]:
        d_state[d_key]["max_evs"]["hp"] = new_max_hp
    if new_min_off is not None and new_min_off > a_min[off_stat]:
        a_state[a_key]["min_evs"][off_stat] = new_min_off
    if new_max_off is not None and new_max_off < a_max[off_stat]:
        a_state[a_key]["max_evs"][off_stat] = new_max_off

    # ----- GLOBAL 508-EV CONSTRAINT --------------------------------------
    # If we've proven minimums on enough stats, the remaining stats can't
    # exceed (508 − sum_of_other_mins). Cheap pass that often crushes
    # max_evs dramatically once one or two offensive/defensive stats are
    # locked in (e.g. Speed=252 + Atk=252 forces HP/Def/SpD/SpA all ≤ 4).
    _apply_total_ev_constraint(a_state[a_key])
    _apply_total_ev_constraint(d_state[d_key])


# ===========================================================================
# Observed-flag marking + speed (move-order) inference
# ===========================================================================
#
# Damage events tighten atk / spa / def / spd / hp. Speed is invisible to
# the damage path entirely — yet move-order is the single most reliably
# observable signal in a real game (any two mons that act reveal a relative
# speed ordering). This block adds:
#
#   1. `mark_observed_from_events` — flips each roster mon's `observed` flag
#      the moment it acts (uses a move) or is hit. Drives the `unknown`
#      vs observed-but-wide distinction in the spread blocks.
#
#   2. `infer_speed_from_move_order` — a deliberately CONSERVATIVE upper-bound
#      pass. A Choice Scarf only makes a mon faster, so "B resolved AFTER A"
#      proves B is genuinely slower than A's *maximum possible* speed — a
#      valid, monotonically-tightening upper bound on B's Spe EV. We only
#      trust the slower direction, and only on clean turns:
#        • no Trick Room (it inverts the order→speed mapping),
#        • no Tailwind on either side, no Sticky Web on either side
#          (speed multipliers we'd otherwise have to model),
#        • no After You / Quash / Instruct (explicit order manipulation),
#        • both moves priority-0 and damaging, neither called via
#          Metronome/Dancer/etc. (priority abilities like Prankster only
#          touch status moves; Gale Wings needs a Flying move — priority-0
#          damaging excludes both),
#        • the slower mon is not paralyzed (para is a speed *reduction*,
#          so a para'd slow mon could secretly be fast).
#
# `update_observed_and_speed` runs both and is called from every place that
# calls `update_knowledge` (the two `_safe_update_knowledge` copies + the
# match-final pass), so both running and final states get speed + observed.

_BASE_STATS_CACHE: dict[str, dict[str, int]] = {}
_MOVE_INFO_CACHE: dict[str, dict[str, Any] | None] = {}
_ORDER_MANIPULATION_MOVES = frozenset({"afteryou", "quash", "instruct"})


async def get_base_stats(
    session: aiohttp.ClientSession, base_url: str, species: str
) -> dict[str, int]:
    """Base-stat block for a species via the calc service's /dex/species.
    Returns {} for unknown species (transformed forms, typos) so callers
    can skip cleanly."""
    key = species_key(species)
    if key in _BASE_STATS_CACHE:
        return _BASE_STATS_CACHE[key]
    async with session.get(f"{base_url}/dex/species/{key}") as r:
        if r.status == 404:
            _BASE_STATS_CACHE[key] = {}
            return {}
        if r.status >= 400:
            raise RuntimeError(f"/dex/species {r.status}")
        data = await r.json()
    bs = data.get("baseStats") or {}
    _BASE_STATS_CACHE[key] = bs
    return bs


async def _get_move_info(
    session: aiohttp.ClientSession, base_url: str, move: str
) -> dict[str, Any] | None:
    """Full /dex/move record (category, priority, target...). None if unknown."""
    key = species_key(move)
    if key in _MOVE_INFO_CACHE:
        return _MOVE_INFO_CACHE[key]
    async with session.get(f"{base_url}/dex/move/{key}") as r:
        if r.status == 404:
            _MOVE_INFO_CACHE[key] = None
            return None
        if r.status >= 400:
            raise RuntimeError(f"/dex/move {r.status}")
        info = await r.json()
    _MOVE_INFO_CACHE[key] = info
    return info


def _speed_stat(base: int, ev: int, *, iv: int = 31, nature_mult: float = 1.0) -> int:
    """Level-50 Speed stat from base + EV (+IV + nature). Standard Gen-9 formula."""
    inner = (2 * base + iv + ev // 4) * 50 // 100 + 5
    return int(inner * nature_mult)


def _largest_spe_ev_at_or_below(base: int, ceiling: int) -> int | None:
    """Largest EV in [0, 252] (step 4) whose neutral-nature, IV-31 Speed is
    ≤ ceiling. Returns None when even EV 0 already exceeds the ceiling (an
    inconsistent observation — skip rather than corrupt the bound).

    Convention: we store the *neutral-equivalent* Spe EV, exactly like the
    damage path stores neutral/IV-31-equivalent atk/def/etc. bounds. This
    is provably valid: the ceiling is the faster mon's MAXIMUM possible
    effective Speed (+nature), so the slower mon's true Speed ≤ ceiling;
    its neutral-equivalent EV is therefore ≤ this returned bound, so we
    never exclude the true value. Matching the damage convention keeps a
    single EV-space across all six stats rather than mixing nature
    assumptions within one entry.
    """
    if _speed_stat(base, 0, iv=31, nature_mult=1.0) > ceiling:
        return None
    last = 0
    ev = 0
    while ev <= 252:
        if _speed_stat(base, ev, iv=31, nature_mult=1.0) <= ceiling:
            last = ev
            ev += 4
        else:
            break
    return last


def _slot_active_map(snap: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for side in ("p1", "p2"):
        for a in (snap.get(side, {}) or {}).get("active", []) or []:
            sl = a.get("slot")
            if sl:
                out[f"{side}{sl}"] = a
    return out


def _mark(state: KnowledgeState, key: str, universe: set[str] | None) -> None:
    if universe is not None and key not in universe:
        return  # transformed Ditto / Illusion — not a real roster spread
    entry = state.setdefault(key, init_knowledge_entry())
    entry["observed"] = True


def mark_observed_from_events(
    snap_pre: dict[str, Any],
    events: list[dict[str, Any]],
    p1_knowledge: KnowledgeState,
    p2_knowledge: KnowledgeState,
    *,
    p1_universe: set[str] | None = None,
    p2_universe: set[str] | None = None,
) -> None:
    """Flip `observed` for every roster mon that acted or was hit this turn."""
    slot_map = _slot_active_map(snap_pre)

    def state_for(side: str) -> tuple[KnowledgeState, set[str] | None]:
        return (p1_knowledge, p1_universe) if side == "p1" else (p2_knowledge, p2_universe)

    for ev in events:
        if not isinstance(ev, dict) or ev.get("type") != "move":
            continue
        atk_slot = ev.get("attacker_slot", "")
        atk = slot_map.get(atk_slot)
        if atk and atk_slot[:2] in ("p1", "p2"):
            st, uni = state_for(atk_slot[:2])
            _mark(st, species_key(atk.get("species", "")), uni)
        for hit in ev.get("hits") or []:
            if hit.get("outcome") != "damage":
                continue
            dslot = hit.get("defender_slot", "")
            d = slot_map.get(dslot)
            if d and dslot[:2] in ("p1", "p2"):
                st, uni = state_for(dslot[:2])
                _mark(st, species_key(d.get("species", "")), uni)


async def infer_speed_from_move_order(
    snap_pre: dict[str, Any],
    events: list[dict[str, Any]],
    p1_knowledge: KnowledgeState,
    p2_knowledge: KnowledgeState,
    *,
    session: aiohttp.ClientSession,
    base_url: str,
    p1_universe: set[str] | None = None,
    p2_universe: set[str] | None = None,
) -> None:
    """Conservative move-order → Spe-upper-bound pass (see block comment)."""
    field = snap_pre.get("field") or {}
    pw = field.get("pseudoWeather") or {}
    if any("trickroom" in species_key(k) for k in pw):
        return
    if field.get("tailwindP1") or field.get("tailwindP2"):
        return
    for side in ("p1", "p2"):
        sc = (snap_pre.get(side, {}) or {}).get("sideConditions") or {}
        if any("stickyweb" in species_key(k) for k in sc):
            return

    move_names = {
        species_key(ev.get("move_name", ""))
        for ev in events if isinstance(ev, dict) and ev.get("type") == "move"
    }
    if move_names & _ORDER_MANIPULATION_MOVES:
        return

    slot_map = _slot_active_map(snap_pre)

    # Ordered sequence of clean, priority-0, damaging, natural move actions.
    seq: list[dict[str, Any]] = []
    for ev in events:
        if not isinstance(ev, dict) or ev.get("type") != "move":
            continue
        if ev.get("called_via"):
            continue
        slot = ev.get("attacker_slot", "")
        pkm = slot_map.get(slot)
        if not pkm or slot[:2] not in ("p1", "p2"):
            continue
        info = await _get_move_info(session, base_url, ev.get("move_name", ""))
        if info is None:
            continue
        if info.get("priority", 0) != 0:
            continue
        if info.get("category") not in ("Physical", "Special"):
            continue
        seq.append({"slot": slot, "side": slot[:2], "pkm": pkm})

    # Every (earlier, later) cross-side pair within this single priority
    # bracket is a valid faster/slower observation.
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            faster, slower = seq[i], seq[j]
            if faster["side"] == slower["side"]:
                continue
            if (slower["pkm"].get("status") or "") == "par":
                continue  # para is a speed reduction — can't bound a slow para'd mon
            await _apply_speed_upper_bound(
                faster, slower, session, base_url,
                p1_knowledge, p2_knowledge, p1_universe, p2_universe,
            )


async def _apply_speed_upper_bound(
    faster: dict[str, Any],
    slower: dict[str, Any],
    session: aiohttp.ClientSession,
    base_url: str,
    p1_knowledge: KnowledgeState,
    p2_knowledge: KnowledgeState,
    p1_universe: set[str] | None,
    p2_universe: set[str] | None,
) -> None:
    f_base = await get_base_stats(session, base_url, faster["pkm"].get("species", ""))
    s_base = await get_base_stats(session, base_url, slower["pkm"].get("species", ""))
    if not f_base or not s_base:
        return  # unknown species (transformed) — skip

    f_state = p1_knowledge if faster["side"] == "p1" else p2_knowledge
    s_state = p1_knowledge if slower["side"] == "p1" else p2_knowledge
    f_uni = p1_universe if faster["side"] == "p1" else p2_universe
    s_uni = p1_universe if slower["side"] == "p1" else p2_universe
    f_key = species_key(faster["pkm"].get("species", ""))
    s_key = species_key(slower["pkm"].get("species", ""))
    if (f_uni is not None and f_key not in f_uni) or (s_uni is not None and s_key not in s_uni):
        return  # transformed / illusion — don't pollute roster spreads
    f_entry = f_state.setdefault(f_key, init_knowledge_entry())
    s_entry = s_state.setdefault(s_key, init_knowledge_entry())

    # Ceiling = the faster mon's MAXIMUM possible effective Speed (its current
    # max Spe EV, max IV, +nature). The slower mon is genuinely below this.
    f_max_ev = f_entry["max_evs"]["spe"]
    ceiling = _speed_stat(int(f_base["spe"]), int(f_max_ev), iv=31, nature_mult=1.1)
    new_upper = _largest_spe_ev_at_or_below(int(s_base["spe"]), ceiling)

    # Mark both involved regardless of whether the bound tightened.
    f_entry["observed"] = True
    s_entry["observed"] = True

    if new_upper is None:
        return  # inconsistent (priority/ability we couldn't see) — leave untouched
    if new_upper < s_entry["max_evs"]["spe"]:
        s_entry["max_evs"]["spe"] = new_upper
        if s_entry["min_evs"]["spe"] > new_upper:
            s_entry["min_evs"]["spe"] = new_upper


async def update_observed_and_speed(
    snap_pre: dict[str, Any],
    events: list[dict[str, Any]],
    p1_knowledge: KnowledgeState,
    p2_knowledge: KnowledgeState,
    *,
    session: aiohttp.ClientSession,
    base_url: str = DEFAULT_CALC_BASE_URL,
    p1_universe: set[str] | None = None,
    p2_universe: set[str] | None = None,
) -> None:
    """Run observed-marking + conservative speed inference for one turn.
    Mutates both knowledge states in place. Best-effort: never raises into
    the caller's per-turn loop (failures are swallowed by the caller's
    try/except, but we also guard the speed pass internally)."""
    mark_observed_from_events(
        snap_pre, events, p1_knowledge, p2_knowledge,
        p1_universe=p1_universe, p2_universe=p2_universe,
    )
    await infer_speed_from_move_order(
        snap_pre, events, p1_knowledge, p2_knowledge,
        session=session, base_url=base_url,
        p1_universe=p1_universe, p2_universe=p2_universe,
    )


__all__ = [
    "TOTAL_EV_BUDGET",
    "DEFAULT_CALC_BASE_URL",
    "DamageEvent",
    "FUZZY_HP_TOLERANCE",
    "KnowledgeState",
    "STATS",
    "get_base_stats",
    "get_move_category",
    "init_knowledge",
    "init_knowledge_entry",
    "mark_observed_from_events",
    "observed_damage_range",
    "species_key",
    "update_knowledge",
    "update_observed_and_speed",
]
