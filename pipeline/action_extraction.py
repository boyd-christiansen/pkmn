"""Reverse-engineer the human's per-turn decision and series-winner relabeling.

Pipeline role:
    Given a parsed match (one row from `replay_parser.py`), provide:
      - `flip_match_to_winner(match)` — relabel sides so the series
        winner is always P1 for downstream synthesis.
      - `extract_p1_actions(snap_pre, snap_post, events)` — translate the
        TurnEvent stream + pre/post snapshots into the structured action
        dict that becomes the SFT ground-truth label.

Isolation contract:
    Pure data transforms over snapshots + events. No I/O, no LLM, no
    calc service. Imports nothing from sibling pipeline modules.
"""
from __future__ import annotations

import copy
from typing import Any


# =============================================================================
# Series-winner-as-P1 flip
# =============================================================================


def _series_winner(games: list[dict]) -> str | None:
    """Return 'p1' | 'p2' | None — the player who won the majority of games.

    Each game has a `winner` field set by /parse_log. Bo1: per-game winner
    IS the series winner. Bo3: majority of game winners. None if no
    determination is possible (all ties / aborts).
    """
    p1_wins = sum(1 for g in games if g.get("winner") == "p1")
    p2_wins = sum(1 for g in games if g.get("winner") == "p2")
    if p1_wins == p2_wins:
        return None
    return "p1" if p1_wins > p2_wins else "p2"


_SLOT_FLIP = {"p1a": "p2a", "p1b": "p2b", "p1c": "p2c",
              "p2a": "p1a", "p2b": "p1b", "p2c": "p1c"}


def _flip_slot(slot: str | None) -> str | None:
    if not slot:
        return slot
    return _SLOT_FLIP.get(slot, slot)


def _flip_side(side: str | None) -> str | None:
    if side == "p1": return "p2"
    if side == "p2": return "p1"
    return side


def _flip_event_inplace(ev: dict[str, Any]) -> None:
    """Mutate a TurnEvent dict to swap p1↔p2 references."""
    t = ev.get("type")
    if t == "move":
        ev["attacker_slot"] = _flip_slot(ev.get("attacker_slot"))
        for hit in ev.get("hits") or []:
            hit["defender_slot"] = _flip_slot(hit.get("defender_slot"))
    elif t == "cant_move":
        ev["slot"] = _flip_slot(ev.get("slot"))
    elif t == "tera":
        ev["side"] = _flip_side(ev.get("side"))
        ev["slot"] = _flip_slot(ev.get("slot"))
    elif t == "switch":
        ev["side"] = _flip_side(ev.get("side"))
        ev["slot"] = _flip_slot(ev.get("slot"))
    elif t == "faint":
        ev["side"] = _flip_side(ev.get("side"))
        ev["slot"] = _flip_slot(ev.get("slot"))
    elif t == "item_event":
        ev["slot"] = _flip_slot(ev.get("slot"))


def flip_match_to_winner(match: dict) -> dict:
    """Return a copy of `match` with sides relabeled so the **series winner
    is P1**. If P1 already won (or the winner is undetermined), returns the
    match unchanged.

    Swaps:
      - top-level `players[0] ↔ players[1]`
      - per game: `teamSheets.p1 ↔ teamSheets.p2`, `winner` flipped
      - per snapshot: `p1 ↔ p2`, `field.tailwindP1 ↔ tailwindP2`,
        `field.tailwindP1TurnsLeft ↔ tailwindP2TurnsLeft`,
        every `events[i]` slot/side fields flipped per the discriminated union
    """
    games = match.get("games") or []
    winner = _series_winner(games)
    if winner != "p2":
        return match  # already P1's side, or undetermined — nothing to flip

    flipped = copy.deepcopy(match)

    flipped["players"] = [match["players"][1], match["players"][0]]

    for g in flipped["games"]:
        if g.get("teamSheets"):
            ts = g["teamSheets"]
            ts["p1"], ts["p2"] = ts["p2"], ts["p1"]
        if g.get("winner") == "p1":
            g["winner"] = "p2"
        elif g.get("winner") == "p2":
            g["winner"] = "p1"

        for snap in g.get("snapshots", []):
            snap["p1"], snap["p2"] = snap["p2"], snap["p1"]
            field = snap.get("field") or {}
            field["tailwindP1"], field["tailwindP2"] = (
                field.get("tailwindP2", False),
                field.get("tailwindP1", False),
            )
            if "tailwindP1TurnsLeft" in field or "tailwindP2TurnsLeft" in field:
                field["tailwindP1TurnsLeft"], field["tailwindP2TurnsLeft"] = (
                    field.get("tailwindP2TurnsLeft"),
                    field.get("tailwindP1TurnsLeft"),
                )
            snap["field"] = field

            for ev in snap.get("events") or []:
                _flip_event_inplace(ev)

    return flipped


# =============================================================================
# P1 action extraction
# =============================================================================


def slot_action(
    action_type: str,
    *,
    move: str | None = None,
    target: str | None = None,
    tera: bool | None = None,
    switch_to: str | None = None,
) -> dict[str, Any]:
    """Construct a single-slot action dict in the SFT label format.

    Public so other call sites (orchestrator, bakeoff) can build pass /
    placeholder actions consistently.
    """
    return {
        "action_type": action_type,
        "move": move,
        "target": target,
        "tera": tera,
        "switch_to": switch_to,
    }


def extract_p1_actions(
    snap_pre: dict[str, Any],
    snap_post: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, dict[str, Any]] | None:
    """Reverse-engineer what each P1 active slot did this turn.

    Reads the new TurnEvent stream (move / switch / cant_move / faint /
    tera / item_event). Only events that represent the human's choice
    count: move events with called_via in {None, "Sleep Talk"}, intentional
    switches (forced_by is None), and cant_move events.

    Returns a dict `{ "a": slot_action, "b": slot_action }` or `None` when
    any slot's action is ambiguous (e.g. forced out before acting).
    """
    out: dict[str, dict[str, Any]] = {}
    pre_active = {p["slot"]: p for p in snap_pre.get("p1", {}).get("active", [])}
    post_active = {p["slot"]: p for p in snap_post.get("p1", {}).get("active", [])}

    for letter in ("a", "b"):
        pre_p = pre_active.get(letter)
        post_p = post_active.get(letter)
        slot_id = f"p1{letter}"

        if pre_p is None or pre_p.get("fainted"):
            out[letter] = slot_action("pass")
            continue

        # Pull events that represent the human's CHOICE for this slot.
        # Forced switches and called-via-other-move moves are consequences,
        # not decisions, so they're excluded.
        my_moves = [
            e for e in events
            if e.get("type") == "move"
            and e.get("attacker_slot") == slot_id
            and e.get("called_via") in (None, "Sleep Talk")
        ]
        my_intentional_switches = [
            e for e in events
            if e.get("type") == "switch"
            and e.get("side") == "p1"
            and e.get("slot") == slot_id
            and e.get("forced_by") is None
        ]
        my_cant_moves = [
            e for e in events
            if e.get("type") == "cant_move" and e.get("slot") == slot_id
        ]

        # Intentional switch wins over move if both present (rare — would be
        # a parse glitch). Fall through to move otherwise.
        if my_intentional_switches:
            sw = my_intentional_switches[-1]
            out[letter] = slot_action("switch", switch_to=sw.get("to_species"))
            continue

        if my_moves:
            mv = my_moves[0]
            move_name = mv.get("move_name")
            hits = mv.get("hits") or []
            unique_targets = sorted({h["defender_slot"] for h in hits if h.get("defender_slot")})
            if not unique_targets:
                target = "self"  # status / self-target (Calm Mind, Protect, Substitute, ...)
            elif len(unique_targets) == 1:
                target = unique_targets[0]
            else:
                target = "spread"
            tera = (
                post_p is not None
                and post_p.get("species") == pre_p.get("species")
                and bool(post_p.get("isTerastallized"))
                and not bool(pre_p.get("isTerastallized"))
            )
            out[letter] = slot_action("move", move=move_name, target=target, tera=tera)
            continue

        if my_cant_moves:
            # The slot couldn't act (asleep / paralyzed / flinch / disable).
            # Recordable as a pass — the prompt rollup will explain why.
            out[letter] = slot_action("pass")
            continue

        # No move / intentional switch / cant_move from this slot. The mon
        # was likely forced out (Roar / Whirlwind / Volt Switch redirect /
        # Eject Button) before having a chance to act. We can't recover
        # the decision the human would have made — skip the turn.
        return None

    return out
