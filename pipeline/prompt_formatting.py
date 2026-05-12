"""User-prompt composition helpers.

Pipeline role:
    Composes the per-turn user prompt from a snapshot, a precomputed
    threat matrix block, the running KnowledgeState, and the full game
    history. Splits cleanly into:

      - **Current frame** — board state, actives, benches (with
        empty-slot annotation when a slot is genuinely vacant).
      - **GAME-STATE LEDGER** — faints, Tera-used, field /
        pseudo-weather / side conditions with turns-left, on-active
        volatiles, choice locks, recent item events, and a
        per-active Cumulative damage row.
      - **TURN-BY-TURN** — every prior turn's events as
        compact one-liners.
      - **SERIES STATE** *(Bo3, game ≥ 2)* — per prior game header +
        full inlined turn-by-turn rollup.
      - **YOUR SPREADS** — one-sided EV constraints per active P1 mon.

    The threat matrix block is generated separately by
    `threat_matrix.py` and concatenated at the end.

Isolation contract:
    Pure data → string transforms. Imports `_species_key` from
    `team_reconstruction` for opponent-bench filtering and Cumulative
    damage attribution; no other sibling-module imports.
"""
from __future__ import annotations

from typing import Any

from team_reconstruction import _species_key


# =============================================================================
# Per-mon line renderers
# =============================================================================


def format_p1_team_block(p1_team: dict[str, dict[str, Any]]) -> str:
    """Bullet-list of P1's reconstructed team (used in the Bo1 system prompt)."""
    lines: list[str] = []
    for entry in p1_team.values():
        moves_str = " / ".join(entry["moves"])
        item = entry["item"] or "?"
        ability = entry["ability"] or "?"
        tera = entry["teraType"] or "?"
        lines.append(
            f"  - {entry['species']} @ {item}, ability={ability}, tera={tera}\n"
            f"      moves: {moves_str}"
        )
    return "\n".join(lines)


def _summarize_active(p: dict[str, Any]) -> str:
    parts = [
        f"[{p['slot']}] {p['species']}",
        f"HP {p.get('hpPercent', '?')}%",
    ]
    if p.get("status"):
        parts.append(f"status={p['status']}")
    if p.get("item"):
        parts.append(f"item={p['item']}")
    if p.get("ability"):
        parts.append(f"ability={p['ability']}")
    tera_type = p.get("teraType")
    if tera_type:
        if p.get("isTerastallized"):
            parts.append(f"TERA-ACTIVE ({p.get('terastallizedAs') or tera_type})")
        else:
            parts.append(f"tera={tera_type}")
    boosts = p.get("boosts") or {}
    if boosts:
        parts.append("boosts=" + ",".join(f"{k}{v:+d}" for k, v in boosts.items()))
    revealed = p.get("revealedMoves") or []
    if revealed:
        parts.append("revealed=" + ",".join(revealed))
    return "  " + " | ".join(parts)


def _summarize_bench(b: dict[str, Any]) -> str:
    return f"{b['species']}{' (fainted)' if b.get('fainted') else ''}"


def _format_actives_with_empty_slots(side_snap: dict[str, Any]) -> str:
    """Render P1's active block, with explicit `[b] (empty — no Pokémon
    remaining)` lines when a slot is genuinely vacant (last mon, no
    replacement available).

    A slot is "empty" when:
      - There's no active entry for that letter, AND
      - There are 0 living bench mons (i.e. no replacement could be sent
        in next turn). If there ARE living bench mons but the slot is
        currently empty, we don't render an empty annotation — the player
        will be prompted for a replacement at end-of-turn before the next
        snapshot, so the slot isn't "permanently" empty.
    """
    actives = side_snap.get("active", []) or []
    by_slot = {a.get("slot"): a for a in actives}
    bench = side_snap.get("bench", []) or []
    living_bench = sum(1 for b in bench if not b.get("fainted"))

    lines: list[str] = []
    for letter in ("a", "b"):
        if letter in by_slot:
            lines.append(_summarize_active(by_slot[letter]))
        else:
            # Slot is unoccupied. If a living bench mon is available, the
            # parser snapshot is mid-replacement; don't annotate. If no
            # living bench mon, this is the genuine "last Pokémon" case.
            if living_bench == 0:
                lines.append(f"  [{letter}] (empty — no Pokémon remaining)")
            # else: skip; the next snapshot will fill the slot.
    return "\n".join(lines) if lines else "  (no active Pokémon)"


# =============================================================================
# YOUR SPREADS — per-stat EV info for the player's own team
# =============================================================================
#
# Semantically this block surfaces "what the player knows about their own
# Pokémon's spreads". At deploy time that's exact values from the
# team-builder JSON; during training synthesis it's the tightest bounds
# the inferencer can extract from observing the *complete* match (see
# `damage_inferencer.infer_match_final_bounds`). Either way, the model
# should treat the block as known/given — the section is deliberately NOT
# labelled "(inferred)" to avoid prompting the model to second-guess
# numbers it would in fact have at deploy.
#
# Risk acknowledged: stats with no observations (e.g. SpD of a Pokémon
# that never took special damage in this match) render as `?`. The model
# could theoretically learn "if SpD is `?`, the mon never took spdamage"
# — implicit signal leak. Mitigated by not explaining derivation in the
# prompt. TODO: substitute canonical priors for fully-open stats so the
# block looks the same whether observation was sparse or rich.
# =============================================================================


_STAT_DISPLAY = ("hp", "atk", "def", "spa", "spd", "spe")
_FULLY_OPEN_MIN = 0
_FULLY_OPEN_MAX = 252


def format_p1_known_spreads_block(
    snapshot: dict[str, Any],
    p1_knowledge: dict[str, dict[str, dict[str, int]]],
) -> str:
    """Render `=== YOUR SPREADS ===` block (no "(inferred)" tag).

    For each active P1 mon, show every stat whose bound has been tightened
    on either side beyond the fully-open `[0, 252]` defaults. Stats with
    no constraint render as `?`. If the upper bound is tightened only,
    show as `≤ N`; lower bound only, `≥ N`; both, the explicit range.
    Pinned to a single value when min == max.

    The caller should pass the **match-final** P1 KnowledgeState (from
    `damage_inferencer.infer_match_final_bounds`), NOT the running
    chronological state. This block represents "what the player knows
    about their own team" — knowledge they had at deploy time, not
    knowledge that accrues turn by turn.
    """
    actives = (snapshot.get("p1") or {}).get("active") or []
    if not actives:
        return ""
    lines: list[str] = ["=== YOUR SPREADS ==="]
    for p in actives:
        species = p.get("species") or "?"
        key = _species_key(species)
        entry = p1_knowledge.get(key)
        if not entry:
            lines.append(f"  {species}: (no observations yet)")
            continue
        mins = entry.get("min_evs", {})
        maxs = entry.get("max_evs", {})
        constrained: list[str] = []
        unconstrained: list[str] = []
        for s in _STAT_DISPLAY:
            lo = mins.get(s, _FULLY_OPEN_MIN)
            hi = maxs.get(s, _FULLY_OPEN_MAX)
            stat_label = s.capitalize()
            if lo == _FULLY_OPEN_MIN and hi == _FULLY_OPEN_MAX:
                unconstrained.append(stat_label)
            elif lo == hi:
                constrained.append(f"{stat_label} {lo}")
            elif lo == _FULLY_OPEN_MIN:
                constrained.append(f"{stat_label} ≤{hi}")
            elif hi == _FULLY_OPEN_MAX:
                constrained.append(f"{stat_label} ≥{lo}")
            else:
                constrained.append(f"{stat_label} {lo}–{hi}")
        if not constrained:
            lines.append(f"  {species}: (no observations yet)")
        else:
            tail = ", others ?" if unconstrained else ""
            lines.append(f"  {species}: " + ", ".join(constrained) + tail)
    return "\n".join(lines)


# =============================================================================
# Historical-context blocks (game-state ledger / turn-by-turn / series state)
# =============================================================================


def _slot_label(slot: str) -> str:
    """'p1a' -> 'P1[a]'."""
    if len(slot) >= 3:
        return f"{slot[:2].upper()}[{slot[2]}]"
    return slot.upper()


def _maybe_turns_left(n: int | None, total: int | None = None) -> str:
    if n is None:
        return ""
    unit = "turn" if n == 1 else "turns"
    if total is not None:
        return f" ({n}/{total} {unit} left)"
    return f" ({n} {unit} left)"


def _scan_events(snapshots: list[dict[str, Any]], current_idx: int):
    """Yield (turn_number, event) for every event in snapshots[0..current_idx-1].

    Note: snapshot at index N stores events that happened DURING turn N
    (i.e. between |turn|N and |turn|N+1 markers). For "what was the state
    at the start of turn current_idx+1", we read events from snapshots
    [0..current_idx-1] (all turns *before* the current one).
    """
    for i in range(min(current_idx, len(snapshots))):
        s = snapshots[i]
        turn_num = s.get("turn", i + 1)
        for ev in s.get("events") or []:
            yield turn_num, ev


def format_game_state_ledger(
    snapshot: dict[str, Any],
    snapshots_so_far: list[dict[str, Any]],
    current_idx: int,
) -> str:
    """=== GAME-STATE LEDGER ===  block.

    Only-when-active rows: empty rows are omitted entirely.
    """
    lines: list[str] = []
    p1, p2 = snapshot.get("p1", {}), snapshot.get("p2", {})
    field = snapshot.get("field", {})

    # Faints (always shown — useful even at 0/0 to anchor the player).
    p1_faints, p2_faints = p1.get("faints", 0), p2.get("faints", 0)
    lines.append(f"Faints:        P1 {p1_faints}/4   |   P2 {p2_faints}/4")

    # Tera used (only-when-active per side).
    tera_rows: list[str] = []
    for side, side_snap in (("P1", p1), ("P2", p2)):
        tu = side_snap.get("teraUsed")
        if tu:
            tera_rows.append(f"{side} ✓ {tu['species']} → {tu['teraType']} on T{tu['onTurn']}")
    if tera_rows:
        lines.append("Tera used:     " + "; ".join(tera_rows))

    # Field weather / terrain with turns-left if known.
    field_parts: list[str] = []
    if field.get("weather"):
        wt = field.get("weatherTurnsLeft")
        field_parts.append(f"{field['weather']}{_maybe_turns_left(wt)}")
    if field.get("terrain"):
        tt = field.get("terrainTurnsLeft")
        field_parts.append(f"{field['terrain']}{_maybe_turns_left(tt)}")
    if field_parts:
        lines.append("Field:         " + ", ".join(field_parts))

    # Pseudo-weather (Trick Room, Gravity, Magic Room, ...).
    pw = field.get("pseudoWeather") or {}
    if pw:
        pw_parts = []
        for pwid, info in pw.items():
            pw_parts.append(f"{pwid}{_maybe_turns_left((info or {}).get('turnsLeft'))}")
        lines.append("Pseudo-weather: " + ", ".join(pw_parts))

    # Side conditions (per side: tailwind / screens / spikes / safeguard).
    for side, side_snap, twin_active, tw_left in (
        ("P1", p1, field.get("tailwindP1"), field.get("tailwindP1TurnsLeft")),
        ("P2", p2, field.get("tailwindP2"), field.get("tailwindP2TurnsLeft")),
    ):
        sc = side_snap.get("sideConditions") or {}
        sc_parts: list[str] = []
        if twin_active:
            sc_parts.append(f"Tailwind{_maybe_turns_left(tw_left)}")
        for sid, info in sc.items():
            info = info or {}
            level = info.get("level")
            tl = info.get("turnsLeft")
            label = sid
            if level is not None:
                label += f" L{level}"
            sc_parts.append(f"{label}{_maybe_turns_left(tl)}")
        if sc_parts:
            lines.append(f"{side} side:       " + ", ".join(sc_parts))

    # Volatiles (only on-active mons, only when present).
    vol_parts: list[str] = []
    for side_label, side_snap in (("P1", p1), ("P2", p2)):
        for active in side_snap.get("active") or []:
            if not active:
                continue
            vols = active.get("volatiles") or {}
            slot = active.get("slot", "?")
            label = f"{side_label}[{slot}] {active.get('species', '?')}"
            for vname, vinfo in vols.items():
                vinfo = vinfo or {}
                if vname == "substitute":
                    hp = vinfo.get("hp")
                    vol_parts.append(f"{label} Substitute" + (f" ({hp} HP)" if hp is not None else ""))
                elif vname == "encoredInto":
                    vol_parts.append(f"{label} Encore-locked into {vinfo}")
                elif vname == "disabled":
                    vol_parts.append(f"{label} Disabled: {vinfo}")
                elif vname == "taunt":
                    vol_parts.append(f"{label} Taunt{_maybe_turns_left(vinfo.get('turnsLeft'))}")
                elif vname == "healBlock":
                    vol_parts.append(f"{label} Heal Block{_maybe_turns_left(vinfo.get('turnsLeft'))}")
                elif vname == "perishCount":
                    vol_parts.append(f"{label} Perish {vinfo}")
                elif vname == "confusion":
                    vol_parts.append(f"{label} confused{_maybe_turns_left(vinfo.get('turnsLeft'))}")
                elif vname == "leechSeed":
                    vol_parts.append(f"{label} Leech-Seeded")
                else:
                    vol_parts.append(f"{label} {vname}")
            # Choice lock surfaced separately — see Choice locks row.
    if vol_parts:
        lines.append("Volatiles:     " + "; ".join(vol_parts))

    # Choice locks (only-when-set, on-active).
    choice_parts: list[str] = []
    for side_label, side_snap in (("P1", p1), ("P2", p2)):
        for active in side_snap.get("active") or []:
            lock = (active or {}).get("choiceLockedInto")
            if lock:
                choice_parts.append(
                    f"{side_label}[{active.get('slot', '?')}] {active.get('species', '?')} locked into {lock}"
                )
    if choice_parts:
        lines.append("Choice locks:  " + "; ".join(choice_parts))

    # Item events (last 3 from history, only if present).
    item_history: list[str] = []
    for turn_num, ev in _scan_events(snapshots_so_far, current_idx):
        if ev.get("type") == "item_event":
            slot = ev.get("slot", "?")
            kind = ev.get("kind", "?")
            item = ev.get("item", "?")
            sl = _slot_label(slot)
            if kind == "consumed":
                item_history.append(f"{sl} {item} consumed (T{turn_num})")
            elif kind == "knocked_off":
                item_history.append(f"{sl} {item} Knocked Off (T{turn_num})")
            elif kind in ("tricked", "stolen", "flung", "incinerated", "popped", "harvested"):
                item_history.append(f"{sl} {item} {kind} (T{turn_num})")
    if item_history:
        # Only show the last 4 to keep the prompt tight.
        lines.append("Item events:   " + "; ".join(item_history[-4:]))

    # Cumulative damage taken by each currently-active mon, walking prior
    # turns' move events. Counts hits where the active was a defender +
    # turns the active spent on field. Only-when-active row.
    cumulative_lines: list[str] = []
    for side_label, side_snap in (("P1", p1), ("P2", p2)):
        for active in side_snap.get("active") or []:
            if not active:
                continue
            slot = active.get("slot", "?")
            species = active.get("species", "?")
            full_slot = f"{side_label.lower()}{slot}"  # "p1a"
            stats = _accumulate_active_stats(snapshots_so_far, current_idx, full_slot, species)
            if stats["turns_on_field"] == 0 and stats["damage_pct"] == 0:
                continue
            label = f"{side_label}[{slot}] {species}"
            if stats["damage_pct"] > 0:
                cumulative_lines.append(
                    f"{label} took {stats['damage_pct']}% across {stats['hits']} hit(s) "
                    f"over {stats['turns_on_field']} turn(s) on field"
                )
            elif stats["turns_on_field"] > 0:
                cumulative_lines.append(
                    f"{label} no damage taken over {stats['turns_on_field']} turn(s) on field"
                )
    if cumulative_lines:
        if len(cumulative_lines) == 1:
            lines.append("Cumulative:    " + cumulative_lines[0])
        else:
            lines.append("Cumulative:    " + cumulative_lines[0])
            for extra in cumulative_lines[1:]:
                lines.append("               " + extra)

    return "=== GAME-STATE LEDGER ===\n" + "\n".join(lines)


def _accumulate_active_stats(
    snapshots_so_far: list[dict[str, Any]],
    current_idx: int,
    target_slot: str,
    target_species: str,
) -> dict[str, int]:
    """Walk prior turns' events and accumulate damage / hits / turns-on-field
    for a specific (slot, species) combination.

    Stops counting backwards once the target slot's species changes (i.e.
    walking back across a switch to a different mon). Damage is summed as
    integer percent (HP before − HP after) for every damage hit on the
    target's slot during a turn where the slot's mon was the target species.
    """
    if current_idx <= 0 or not snapshots_so_far:
        return {"damage_pct": 0, "hits": 0, "turns_on_field": 0}

    target_key = _species_key(target_species)
    damage_pct = 0
    hits = 0
    turns_on_field = 0

    for i in range(min(current_idx, len(snapshots_so_far))):
        s = snapshots_so_far[i]
        side = target_slot[:2]      # "p1" or "p2"
        letter = target_slot[2]      # "a" or "b"
        active_at_t = next(
            (a for a in (s.get(side, {}) or {}).get("active", []) if a and a.get("slot") == letter),
            None,
        )
        if not active_at_t:
            continue
        if _species_key(active_at_t.get("species", "")) != target_key:
            continue
        turns_on_field += 1
        # Sum damage taken this turn from move events targeting this slot.
        for ev in s.get("events") or []:
            if ev.get("type") != "move":
                continue
            for hit in ev.get("hits") or []:
                if hit.get("defender_slot") != target_slot:
                    continue
                if hit.get("outcome") != "damage":
                    continue
                hp_b = hit.get("hp_before_pct")
                hp_a = hit.get("hp_after_pct")
                if hp_b is None or hp_a is None:
                    continue
                damage_pct += max(0, int(hp_b) - int(hp_a))
                hits += 1
    return {"damage_pct": damage_pct, "hits": hits, "turns_on_field": turns_on_field}


def _format_event_inline(ev: dict[str, Any]) -> str | None:
    """Render a single TurnEvent as a one-line string. Returns None for events
    that should be folded into another (e.g. faint events covered by a
    move's is_ko=True hit)."""
    t = ev.get("type")
    if t == "move":
        attacker_label = _slot_label(ev.get("attacker_slot", "?"))
        move_name = ev.get("move_name", "?")
        cv = ev.get("called_via")
        prefix = f"{attacker_label} "
        if cv:
            move_part = f"{cv} → {move_name}"
        else:
            move_part = move_name
        hits = ev.get("hits") or []
        if not hits:
            # Status / self-target move (Calm Mind, Protect, Rage Powder, ...).
            return f"{prefix}{move_part}"
        # Group hits by outcome for compact display.
        bits: list[str] = []
        for h in hits:
            tgt = _slot_label(h.get("defender_slot", "?"))
            outcome = h.get("outcome", "?")
            if outcome == "damage":
                hp_after = h.get("hp_after_pct")
                ko = " KO" if h.get("is_ko") else ""
                crit = " (crit)" if h.get("is_crit") else ""
                if hp_after is not None:
                    bits.append(f"{tgt} {hp_after}%{ko}{crit}")
                else:
                    bits.append(f"{tgt} damage{ko}{crit}")
            elif outcome in ("blocked", "immune", "miss", "no_effect", "fail"):
                cause = h.get("cause")
                bits.append(f"{tgt} {outcome}" + (f" by {cause}" if cause else ""))
            else:
                bits.append(f"{tgt} {outcome}")
        kind_tag = " (spread)" if len(hits) >= 2 else ""
        return f"{prefix}{move_part}{kind_tag} → " + ", ".join(bits)
    if t == "switch":
        side = ev.get("side", "?").upper()
        slot = _slot_label(ev.get("slot", "?"))
        from_sp = ev.get("from_species") or "(empty slot)"
        to_sp = ev.get("to_species", "?")
        forced = ev.get("forced_by")
        forced_part = f" (via {forced})" if forced else ""
        return f"{side} switched {slot}: {from_sp} → {to_sp}{forced_part}"
    if t == "tera":
        side = ev.get("side", "?").upper()
        return f"{side} Tera'd: {ev.get('species', '?')} → {ev.get('to_type', '?')}"
    if t == "cant_move":
        slot = _slot_label(ev.get("slot", "?"))
        reason = ev.get("reason", "?")
        attempted = ev.get("attempted_move")
        att_part = f" (tried {attempted})" if attempted else ""
        return f"{slot} couldn't move ({reason}){att_part}"
    if t == "item_event":
        slot = _slot_label(ev.get("slot", "?"))
        return f"{slot} {ev.get('kind', '?')} {ev.get('item', '?')}"
    if t == "faint":
        # Faints are usually folded into the KO'd move event. If the faint
        # is from end-of-turn residual damage (Life Orb / weather / status),
        # render it as its own line so the cause is at least visible.
        slot = _slot_label(ev.get("slot", "?"))
        return f"{slot} {ev.get('species', '?')} fainted"
    return None


def format_turn_by_turn(
    snapshots_so_far: list[dict[str, Any]],
    current_idx: int,
    *,
    game_index: int = 0,
) -> str:
    """=== TURN-BY-TURN (game N) ===  block.

    One block per prior turn in this game. Renders every event inline.
    No length cap (per design — sequence-aware reasoning is the point).
    """
    lines: list[str] = [f"=== TURN-BY-TURN (game {game_index + 1}) ==="]
    if current_idx == 0:
        lines.append("(no prior turns this game)")
        return "\n".join(lines)
    any_content = False
    for i in range(min(current_idx, len(snapshots_so_far))):
        s = snapshots_so_far[i]
        turn_num = s.get("turn", i + 1)
        events = s.get("events") or []
        if not events:
            continue
        # Drop faint events that already collapse into an `is_ko` damage hit
        # in the same turn (avoid duplicate "Rillaboom fainted" lines).
        ko_slots: set[str] = set()
        for ev in events:
            if ev.get("type") == "move":
                for h in ev.get("hits") or []:
                    if h.get("is_ko"):
                        ko_slots.add(h.get("defender_slot", ""))
        rendered_first = False
        for ev in events:
            if ev.get("type") == "faint" and _slot_label(ev.get("slot", ""))[:5].lower() in {
                "p1[a]", "p1[b]", "p2[a]", "p2[b]"
            }:
                # Suppress if the matching slot was KO'd by a damage event already.
                if ev.get("slot") in ko_slots:
                    continue
            line = _format_event_inline(ev)
            if not line:
                continue
            prefix = f"T{turn_num}: " if not rendered_first else "    "
            lines.append(prefix + line)
            rendered_first = True
        if rendered_first:
            any_content = True
    if not any_content:
        lines.append("(no actionable events recorded)")
    return "\n".join(lines)


def format_series_state(
    prior_games: list[dict[str, Any]],
    *,
    current_game_index: int,
    total_games_in_series: int,
) -> str:
    """=== SERIES STATE (Bo3, game N of M) ===  block. Bo3-only.

    For each prior game in the series: a short header (winner, turns,
    brought rosters, Tera resolutions) followed by the FULL turn-by-turn
    action log of that game.

    TODO(token-efficient-series-summary): the verbatim rollup is verbose
    and consumes a lot of attention. A learned or rule-based summarizer
    that distills "what mattered for THIS turn's decision" would be a
    big improvement. Until we have one, raw inlining keeps the model
    from missing context.
    """
    if current_game_index == 0 or not prior_games:
        return ""
    lines: list[str] = [
        f"=== SERIES STATE (Bo3, game {current_game_index + 1} of {total_games_in_series}) ==="
    ]
    for gi, g in enumerate(prior_games):
        snaps = g.get("snapshots") or []
        if not snaps:
            continue
        last = snaps[-1]
        winner = g.get("winner")
        we_won = winner == "p1"
        turns = len(snaps)
        winner_label = "we won" if we_won else ("opp won" if winner == "p2" else "tied/aborted")
        lines.append("")  # blank separator between games
        lines.append(f"--- Game {gi + 1} ({winner_label}, {turns} turns) ---")

        # Brought rosters: union of all-time-seen actives + bench across that game.
        def _brought(side_snap_seq: list[dict[str, Any]], side: str) -> list[str]:
            seen_order: list[str] = []
            seen: set[str] = set()
            for s in side_snap_seq:
                for p in (s.get(side, {}).get("active") or []):
                    sp = p.get("species")
                    if sp and sp not in seen:
                        seen.add(sp); seen_order.append(sp)
                for b in (s.get(side, {}).get("bench") or []):
                    sp = b.get("species")
                    if sp and sp not in seen:
                        seen.add(sp); seen_order.append(sp)
            return seen_order
        p1_brought = _brought(snaps, "p1")
        # For opponent's roster in series state: only include species that
        # actually saw field (chronological — the same gating logic the
        # current-frame OPP BENCH uses).
        opp_seen: set[str] = set()
        for s in snaps:
            for sp in (s.get("p2", {}) or {}).get("seenSpecies") or []:
                opp_seen.add(sp)
        p2_brought = [sp for sp in _brought(snaps, "p2") if _species_key(sp) in opp_seen]

        if p1_brought:
            lines.append(f"  We brought:  {', '.join(p1_brought)}")
        if p2_brought:
            lines.append(f"  Opp brought: {', '.join(p2_brought)}")

        # Tera resolutions (sticky teraUsed on last snapshot).
        for side_label, side in (("Our Tera:   ", "p1"), ("Opp Tera:   ", "p2")):
            tu = (last.get(side) or {}).get("teraUsed")
            if tu:
                lines.append(f"  {side_label}{tu['species']} → {tu['teraType']} on T{tu['onTurn']}")

        # Inline the full turn-by-turn action log for this prior game.
        # We pass `current_idx=len(snaps)` to render every turn.
        rollup = format_turn_by_turn(snaps, len(snaps), game_index=gi)
        # Strip the rollup's own header — we already labelled the game above.
        rollup_body = rollup.split("\n", 1)[1] if "\n" in rollup else ""
        if rollup_body:
            lines.append(rollup_body)
    return "\n".join(lines)


# =============================================================================
# Top-level user prompt
# =============================================================================


def format_user_prompt(
    snapshot: dict[str, Any],
    threat_matrix_text: str,
    *,
    p1_inferred_block: str = "",
    snapshots_so_far: list[dict[str, Any]] | None = None,
    current_idx: int = 0,
    prior_games: list[dict[str, Any]] | None = None,
    game_index: int = 0,
    total_games_in_series: int = 1,
    match_format: str = "bo1",
) -> str:
    """Compose the user prompt for one turn.

    Includes (in order): board state header, current actives + benches,
    GAME-STATE LEDGER, TURN-BY-TURN (this game), SERIES STATE (Bo3 only,
    game_index > 0), YOUR SPREADS (inferred), and the threat matrix block.
    """
    f = snapshot.get("field", {})
    field_parts = []
    if f.get("weather"):
        field_parts.append(f"weather={f['weather']}")
    if f.get("terrain"):
        field_parts.append(f"terrain={f['terrain']}")
    field_parts.append(f"P1-tailwind={'YES' if f.get('tailwindP1') else 'no'}")
    field_parts.append(f"P2-tailwind={'YES' if f.get('tailwindP2') else 'no'}")
    field_str = ", ".join(field_parts)

    p1 = snapshot.get("p1", {})
    p2 = snapshot.get("p2", {})

    p1_active_lines = _format_actives_with_empty_slots(p1)
    p2_active_lines = "\n".join(_summarize_active(p) for p in p2.get("active", []))
    # P1 bench: full brought-set (player knows their own selection from team preview).
    p1_bench = ", ".join(_summarize_bench(b) for b in p1.get("bench", [])) or "(none)"
    # P2 bench: chronologically gated — the player only learns the opponent's
    # brought selection as the opponent actually switches them in. Use
    # `seenSpecies` (emitted by the parser) as the visibility filter.
    p2_seen = {_species_key(s) for s in (p2.get("seenSpecies") or [])}
    p2_bench_visible = [
        b for b in p2.get("bench", [])
        if _species_key(b.get("species", "")) in p2_seen
    ]
    p2_bench = ", ".join(_summarize_bench(b) for b in p2_bench_visible) or "(none)"

    # New historical context blocks.
    snaps = snapshots_so_far or []
    ledger_block = format_game_state_ledger(snapshot, snaps, current_idx)
    rollup_block = format_turn_by_turn(snaps, current_idx, game_index=game_index)
    series_block = ""
    if match_format == "bo3" and prior_games:
        series_block = format_series_state(
            prior_games,
            current_game_index=game_index,
            total_games_in_series=total_games_in_series,
        )

    spreads_block = (p1_inferred_block + "\n\n") if p1_inferred_block else ""
    series_block_part = (series_block + "\n\n") if series_block else ""

    return (
        f"=== TURN {snapshot.get('turn', '?')} ===\n"
        f"Field: {field_str}\n\n"
        f"YOUR (P1) ACTIVE:\n{p1_active_lines}\n"
        f"YOUR (P1) BENCH: {p1_bench}\n\n"
        f"OPP (P2) ACTIVE:\n{p2_active_lines}\n"
        f"OPP (P2) BENCH: {p2_bench}\n\n"
        f"{ledger_block}\n\n"
        f"{rollup_block}\n\n"
        f"{series_block_part}"
        f"{spreads_block}"
        f"{threat_matrix_text}"
    )
