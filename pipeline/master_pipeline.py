"""Orchestrator: scraped replays in -> SFT-ready JSONL out.

Pipeline role:
    Walks through `parsed_data/{bo1,bo3}.jsonl` (produced by replay_parser),
    and for each turn of each match:
      1. reconstructs P1's team (revealed item / ability / tera / moves with
         `[UNREVEALED_MOVE]` padding for slots the human never used);
      2. extracts P1's actual two-slot decision from `snap[N].events` +
         the diff to `snap[N+1]` (move / switch / cant_move events, Tera flag);
      3. asks `threat_matrix` to render the dual-track damage envelope;
      4. drives `teacher_llm.synthesize_turn` to elicit a chain-of-thought
         that justifies that exact decision;
      5. writes the resulting OpenAI-fine-tuning conversation to
         `parsed_data/sft_training_data.jsonl`;
      6. filters the same `events` stream for damage observations and
         feeds them to `damage_inferencer.update_knowledge` to tighten
         both KnowledgeStates for the next turn.

    KnowledgeStates start at fully-open `[0, 252]` bounds — the canonical
    priors are used by `threat_matrix` for its Probable track only,
    preserving the Absolute track's strict-math guarantee.

Isolation contract:
    The only file allowed to import from every other pipeline module.
    Everything else is leaf — no cross-imports between siblings.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import aiohttp
import click
from openai import AsyncOpenAI

from teacher_llm import TeacherProvider
from tqdm.asyncio import tqdm

import canonical_priors
import damage_inferencer
import teacher_llm
import threat_matrix

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = Path(__file__).resolve().parent
DEFAULT_PARSED_DATA_DIR = PIPELINE_DIR / "parsed_data"
DEFAULT_BO3_INPUT = DEFAULT_PARSED_DATA_DIR / "bo3.jsonl"
DEFAULT_BO1_INPUT = DEFAULT_PARSED_DATA_DIR / "bo1.jsonl"
DEFAULT_OUTPUT = DEFAULT_PARSED_DATA_DIR / "sft_training_data.jsonl"
DEFAULT_CALC_BASE_URL = "http://localhost:3000"

FORMAT_ID_BY_KIND = {
    "bo1": "gen9vgc2026regi",
    "bo3": "gen9vgc2026regibo3",
}


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


def reconstruct_p1_team(games: list[dict]) -> dict[str, dict[str, Any]]:
    """Forward-scan all snapshots in a match, aggregate revealed P1 info per species.

    Pads each Pokémon's move list to exactly 4 with `"[UNREVEALED_MOVE]"`.
    """
    aggregated: dict[str, dict[str, Any]] = {}

    def _ensure(species: str) -> dict[str, Any]:
        return aggregated.setdefault(
            species,
            {
                "species": species,
                "item": None,
                "ability": None,
                "teraType": None,
                "isTerastallized": False,
                "moves": [],
            },
        )

    for game in games:
        for snap in game.get("snapshots", []):
            for p in snap.get("p1", {}).get("active", []):
                entry = _ensure(p["species"])
                if p.get("item") and not entry["item"]:
                    entry["item"] = p["item"]
                if p.get("ability") and not entry["ability"]:
                    entry["ability"] = p["ability"]
                if p.get("teraType") and not entry["teraType"]:
                    entry["teraType"] = p["teraType"]
                if p.get("isTerastallized"):
                    entry["isTerastallized"] = True
                for mv in p.get("revealedMoves") or []:
                    if mv not in entry["moves"]:
                        entry["moves"].append(mv)
            for b in snap.get("p1", {}).get("bench", []):
                _ensure(b["species"])

    for entry in aggregated.values():
        while len(entry["moves"]) < 4:
            entry["moves"].append("[UNREVEALED_MOVE]")
        entry["moves"] = entry["moves"][:4]

    return aggregated


def reconstruct_p2_species(games: list[dict]) -> list[str]:
    seen: list[str] = []
    for game in games:
        for snap in game.get("snapshots", []):
            for p in snap.get("p2", {}).get("active", []) + snap.get("p2", {}).get("bench", []):
                if p["species"] not in seen:
                    seen.append(p["species"])
    return seen


def _species_key(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum())


def _team_sheets_for_match(games: list[dict]) -> dict[str, list[dict]] | None:
    """Return the first non-null `teamSheets` from any game in the match.

    All games in a Bo3 series carry the same sheet, so we just take the
    earliest one available. None means CTS for the whole match.
    """
    for g in games:
        sheets = g.get("teamSheets")
        if sheets and sheets.get("p1") and sheets.get("p2"):
            return sheets
    return None


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

    import copy
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


def _brought_species_keys_for_game(game: dict) -> set[str]:
    """Species (normalized keys) actually brought by P1 to this single game.

    Derived from the union of P1 active + P1 bench across this game's
    snapshots. In OTS Bo3 the parser already gates P1 bench to broughtSet,
    so this naturally yields the 4 brought.
    """
    out: set[str] = set()
    for snap in game.get("snapshots", []):
        for p in snap.get("p1", {}).get("active", []):
            out.add(_species_key(p["species"]))
        for b in snap.get("p1", {}).get("bench", []):
            out.add(_species_key(b["species"]))
    return out


# ---------------------------------------------------------------------------
# Action extraction
# ---------------------------------------------------------------------------


def _slot_action(
    action_type: str,
    *,
    move: str | None = None,
    target: str | None = None,
    tera: bool | None = None,
    switch_to: str | None = None,
) -> dict[str, Any]:
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
            out[letter] = _slot_action("pass")
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
            out[letter] = _slot_action("switch", switch_to=sw.get("to_species"))
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
            out[letter] = _slot_action("move", move=move_name, target=target, tera=tera)
            continue

        if my_cant_moves:
            # The slot couldn't act (asleep / paralyzed / flinch / disable).
            # Recordable as a pass — the prompt rollup will explain why.
            out[letter] = _slot_action("pass")
            continue

        # No move / intentional switch / cant_move from this slot. The mon
        # was likely forced out (Roar / Whirlwind / Volt Switch redirect /
        # Eject Button) before having a chance to act. We can't recover
        # the decision the human would have made — skip the turn.
        return None

    return out


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def format_p1_team_block(p1_team: dict[str, dict[str, Any]]) -> str:
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


SPREAD_BOUND_WIDTH_THRESHOLD = 60  # stats whose (max - min) ≤ this get an explicit range shown
_STAT_DISPLAY = ("hp", "atk", "def", "spa", "spd", "spe")


def format_p1_inferred_spreads_block(
    snapshot: dict[str, Any],
    p1_knowledge: dict[str, dict[str, dict[str, int]]],
) -> str:
    """Render an `=== YOUR SPREADS ===` block listing per-stat EV ranges for
    each active P1 mon. Stats whose bound width exceeds the threshold are
    shown as `?` to keep the prompt tight.
    """
    actives = (snapshot.get("p1") or {}).get("active") or []
    if not actives:
        return ""
    lines: list[str] = ["=== YOUR SPREADS (inferred) ==="]
    for p in actives:
        species = p.get("species") or "?"
        key = _species_key(species)
        entry = p1_knowledge.get(key)
        if not entry:
            lines.append(f"  {species}: (no inference yet)")
            continue
        mins = entry.get("min_evs", {})
        maxs = entry.get("max_evs", {})
        parts: list[str] = []
        for s in _STAT_DISPLAY:
            lo = mins.get(s, 0)
            hi = maxs.get(s, 252)
            if hi - lo <= SPREAD_BOUND_WIDTH_THRESHOLD:
                if lo == hi:
                    parts.append(f"{s.capitalize()} {lo}")
                else:
                    parts.append(f"{s.capitalize()} {lo}–{hi}")
            else:
                parts.append(f"{s.capitalize()} ?")
        lines.append(f"  {species}: " + ", ".join(parts))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Historical-context blocks (game-state ledger / turn-by-turn / series state)
# ---------------------------------------------------------------------------


def _slot_label(slot: str) -> str:
    """'p1a' -> 'P1[a]'."""
    if len(slot) >= 3:
        return f"{slot[:2].upper()}[{slot[2]}]"
    return slot.upper()


def _maybe_turns_left(n: int | None, total: int | None = None) -> str:
    if n is None:
        return ""
    if total is not None:
        return f" ({n}/{total} turns left)"
    return f" ({n} turns left)"


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

    return "=== GAME-STATE LEDGER ===\n" + "\n".join(lines)


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
    """=== SERIES STATE (Bo3, game N of M) ===  block. Bo3-only."""
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
        lines.append(f"Game {gi + 1} ({winner_label}, {turns} turns):")

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
        p2_brought = _brought(snaps, "p2")
        if p1_brought:
            lines.append(f"  We brought:  {', '.join(p1_brought)}")
        if p2_brought:
            lines.append(f"  Opp brought: {', '.join(p2_brought)}")

        # Tera (last snapshot's stickied teraUsed has the resolution).
        for side_label, side in (("Our Tera:   ", "p1"), ("Opp Tera:   ", "p2")):
            tu = (last.get(side) or {}).get("teraUsed")
            if tu:
                lines.append(f"  {side_label}{tu['species']} → {tu['teraType']} on T{tu['onTurn']}")

        # Notable: heuristic 1-2 lines.
        notable: list[str] = []
        # Choice lock surfaced.
        for s in snaps:
            for p in (s.get("p2", {}).get("active") or []):
                lock = (p or {}).get("choiceLockedInto")
                if lock and not any("locked into" in n for n in notable):
                    notable.append(f"Opp {p.get('species', '?')} locked into {lock}")
                    break
        # Pseudo-weather setup (Trick Room).
        for s in snaps:
            pw = (s.get("field", {}) or {}).get("pseudoWeather") or {}
            for pwid in pw:
                if "trickroom" in pwid.lower() and not any("Trick Room" in n for n in notable):
                    notable.append("Opp set Trick Room")
                    break
        if notable:
            lines.append(f"  Notable:     {'; '.join(notable[:2])}")
    return "\n".join(lines)


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

    p1_active_lines = "\n".join(_summarize_active(p) for p in p1.get("active", []))
    p2_active_lines = "\n".join(_summarize_active(p) for p in p2.get("active", []))
    p1_bench = ", ".join(_summarize_bench(b) for b in p1.get("bench", [])) or "(none)"
    p2_bench = ", ".join(_summarize_bench(b) for b in p2.get("bench", [])) or "(none)"

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


# ---------------------------------------------------------------------------
# Resumability
# ---------------------------------------------------------------------------


def load_seen_keys(output_path: Path) -> set[tuple[str, int, int]]:
    seen: set[tuple[str, int, int]] = set()
    if not output_path.exists():
        return seen
    with output_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = (rec["match_id"], int(rec["game_index"]), int(rec["turn"]))
                seen.add(key)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return seen


# ---------------------------------------------------------------------------
# Match processing
# ---------------------------------------------------------------------------


def _build_teacher(provider: str, model: str | None) -> TeacherProvider:
    """Instantiate the requested provider, validating that its API key is present."""
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise click.ClickException(
                "OPENAI_API_KEY env var is required (or pass --dry-run / --provider <other>)"
            )
        from teacher_openai import OpenAIProvider
        return OpenAIProvider(model=model) if model else OpenAIProvider()
    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise click.ClickException("ANTHROPIC_API_KEY env var is required for --provider anthropic")
        from teacher_anthropic import AnthropicProvider
        return AnthropicProvider(model=model) if model else AnthropicProvider()
    if provider == "google":
        if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
            raise click.ClickException(
                "GOOGLE_API_KEY (or GEMINI_API_KEY) env var is required for --provider google"
            )
        from teacher_google import GoogleProvider
        return GoogleProvider(model=model) if model else GoogleProvider()
    raise click.ClickException(f"unknown provider: {provider}")


async def _check_calc_health(session: aiohttp.ClientSession, base_url: str) -> None:
    try:
        async with session.get(f"{base_url}/health") as r:
            if r.status != 200:
                raise RuntimeError(f"health check returned {r.status}")
    except Exception as e:
        raise click.ClickException(
            f"calc_microservice not reachable at {base_url}/health: {e}\n"
            f"  Start it with:  cd calc_microservice && npm run dev"
        )


async def process_match(
    match_record: dict[str, Any],
    *,
    output_path: Path,
    calc_base_url: str,
    teacher: TeacherProvider | None,
    aiohttp_session: aiohttp.ClientSession,
    file_lock: asyncio.Lock,
    format_id: str,
    seen_keys: set[tuple[str, int, int]],
    dry_run: bool,
    model: str,
) -> dict[str, int]:
    # Series-winner-as-P1: every SFT example is generated from the perspective
    # of the player who won the series. P2-won matches are relabeled in full.
    match_record = flip_match_to_winner(match_record)

    games = match_record.get("games") or []
    if not games:
        return {"skipped_no_games": 1}

    match_format = match_record.get("format", "bo1")
    team_sheets = _team_sheets_for_match(games) if match_format == "bo3" else None

    # Knowledge state seeding — for OTS Bo3, use the full 6-mon team sheets
    # so the threat matrix can reason about the unswitched-in backline too.
    # For CTS Bo1, fall back to whatever the snapshots reveal.
    p1_team_recon = reconstruct_p1_team(games)
    if team_sheets:
        p1_species_universe = [s["species"] for s in team_sheets["p1"]]
        p2_species_universe = [s["species"] for s in team_sheets["p2"]]
    else:
        p1_species_universe = list(p1_team_recon.keys())
        p2_species_universe = reconstruct_p2_species(games)

    p1_knowledge = damage_inferencer.init_knowledge(p1_species_universe)
    p2_knowledge = damage_inferencer.init_knowledge(p2_species_universe)

    # Bo1 system prompt is stable across all turns of the match.
    bo1_system_prompt = (
        teacher_llm.render_system_prompt(format_p1_team_block(p1_team_recon))
        if not team_sheets
        else None
    )

    stats: dict[str, int] = defaultdict(int)
    match_id = match_record.get("match_id", "unknown")

    for game_idx, game in enumerate(games):
        snapshots = game.get("snapshots") or []

        # Bo3 system prompt depends on the brought-4 of THIS game (different
        # selections per game in a series), so render per-game.
        if team_sheets:
            brought = _brought_species_keys_for_game(game)
            system_prompt = teacher_llm.render_system_prompt_bo3(
                p1_sheet=team_sheets["p1"],
                p2_sheet=team_sheets["p2"],
                p1_brought=brought,
            )
        else:
            system_prompt = bo1_system_prompt
        for i in range(len(snapshots) - 1):
            snap_pre = snapshots[i]
            snap_post = snapshots[i + 1]
            events_stream = snap_pre.get("events") or []
            turn = int(snap_pre.get("turn", 0))
            key = (match_id, game_idx, turn)
            if key in seen_keys:
                stats["already_done"] += 1
                continue

            human_action_dict = extract_p1_actions(snap_pre, snap_post, events_stream)
            if human_action_dict is None:
                stats["skipped_ambiguous"] += 1
                await _safe_update_knowledge(
                    snap_pre, snap_post, events_stream, p1_knowledge, p2_knowledge,
                    session=aiohttp_session, base_url=calc_base_url,
                )
                continue

            try:
                tm_text = await threat_matrix.generate_threat_matrix(
                    snap_pre, "p1", p1_knowledge, p2_knowledge,
                    format_id=format_id,
                    session=aiohttp_session,
                    base_url=calc_base_url,
                )
            except Exception as e:
                stats["skipped_threat_matrix_error"] += 1
                _log_error(f"[{match_id} g{game_idx} t{turn}] threat_matrix failed: {e}")
                await _safe_update_knowledge(
                    snap_pre, snap_post, events_stream, p1_knowledge, p2_knowledge,
                    session=aiohttp_session, base_url=calc_base_url,
                )
                continue

            p1_inferred = format_p1_inferred_spreads_block(snap_pre, p1_knowledge)
            user_prompt = format_user_prompt(
                snap_pre,
                tm_text,
                p1_inferred_block=p1_inferred,
                snapshots_so_far=snapshots,
                current_idx=i,
                prior_games=games[:game_idx],
                game_index=game_idx,
                total_games_in_series=len(games),
                match_format=match_format,
            )
            human_action = {
                "slot_1": human_action_dict.get("a", _slot_action("pass")),
                "slot_2": human_action_dict.get("b", _slot_action("pass")),
            }

            messages: list[dict[str, Any]] | None
            if dry_run:
                messages = _dry_run_messages(system_prompt, user_prompt, human_action)
            else:
                if teacher is None:
                    raise RuntimeError("Teacher provider missing in non-dry-run mode")
                try:
                    res = await teacher.synthesize_turn(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        human_action=human_action,
                        calc_url=f"{calc_base_url}/calc",
                        aiohttp_session=aiohttp_session,
                    )
                    messages = res.messages
                    if res.error and not messages:
                        _log_error(f"[{match_id} g{game_idx} t{turn}] teacher LLM: {res.error}")
                        stats["skipped_llm_error"] += 1
                except Exception as e:
                    stats["skipped_llm_error"] += 1
                    _log_error(f"[{match_id} g{game_idx} t{turn}] teacher LLM failed: {e}")
                    messages = None

            if messages is None:
                stats["skipped_llm_failed"] += 1
            else:
                async with file_lock:
                    with output_path.open("a") as f:
                        f.write(json.dumps({
                            "match_id": match_id,
                            "game_index": game_idx,
                            "turn": turn,
                            "format_id": format_id,
                            "messages": messages,
                        }) + "\n")
                seen_keys.add(key)
                stats["written"] += 1

            await _safe_update_knowledge(
                snap_pre, snap_post, events_stream, p1_knowledge, p2_knowledge,
                session=aiohttp_session, base_url=calc_base_url,
            )

    return dict(stats)


async def _safe_update_knowledge(
    snap_pre, snap_post, events_stream, p1_knowledge, p2_knowledge, *, session, base_url
):
    """Filter the new TurnEvent stream for damage observations and feed
    them to the binary-search inferencer. Drops Metronome / Copycat /
    Sketch / Snatch / Me First / Dancer / Instruct call-throughs (those
    can hit moves not in the user's actual kit) but keeps Sleep Talk
    (calls own moves only)."""
    damage_events = damage_inferencer.events_to_damage_events(events_stream)
    if not damage_events:
        return
    try:
        await damage_inferencer.update_knowledge(
            snap_pre, snap_post, damage_events, p1_knowledge, p2_knowledge,
            session=session, base_url=base_url,
        )
    except Exception as e:
        _log_error(f"update_knowledge failed: {e}")


def _dry_run_messages(
    system_prompt: str, user_prompt: str, human_action: dict[str, Any]
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "pre_tool_thought": "[DRY RUN — teacher LLM not invoked]",
                    "action": human_action,
                }
            ),
        },
    ]


def _log_error(msg: str) -> None:
    click.echo(msg, err=True)


# ---------------------------------------------------------------------------
# CLI / runner
# ---------------------------------------------------------------------------


def _resolve_format_id(input_path: Path, override: str | None) -> str:
    if override:
        return override
    name = input_path.stem.lower()
    if name in FORMAT_ID_BY_KIND:
        return FORMAT_ID_BY_KIND[name]
    return FORMAT_ID_BY_KIND["bo3"]


def _read_match_records(input_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with input_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


async def run(
    *,
    input_path: Path,
    output_path: Path,
    calc_base_url: str,
    format_id: str,
    limit: int | None,
    concurrency: int,
    dry_run: bool,
    model: str | None,
    provider: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timeout = aiohttp.ClientTimeout(total=180, connect=10)

    teacher: TeacherProvider | None = None
    if not dry_run:
        teacher = _build_teacher(provider, model)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        await _check_calc_health(session, calc_base_url)

        records = _read_match_records(input_path)
        click.echo(f"loaded {len(records)} match records from {input_path}")

        seen_keys = load_seen_keys(output_path)
        if seen_keys:
            click.echo(f"  resuming: {len(seen_keys)} (match, game, turn) keys already in {output_path.name}")

        if limit is not None:
            records = records[:limit]
            click.echo(f"  --limit {limit}: processing first {len(records)} matches")

        file_lock = asyncio.Lock()
        sem = asyncio.Semaphore(concurrency)

        async def worker(rec):
            async with sem:
                return await process_match(
                    rec,
                    output_path=output_path,
                    calc_base_url=calc_base_url,
                    teacher=teacher,
                    aiohttp_session=session,
                    file_lock=file_lock,
                    format_id=format_id,
                    seen_keys=seen_keys,
                    dry_run=dry_run,
                    model=model,
                )

        results = await tqdm.gather(*(worker(r) for r in records), desc="matches", unit="match")

    totals: dict[str, int] = defaultdict(int)
    for r in results:
        for k, v in r.items():
            totals[k] += v
    click.echo("\n=== summary ===")
    for k in sorted(totals):
        click.echo(f"  {k}: {totals[k]}")


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=str(DEFAULT_BO3_INPUT),
    show_default=True,
    help="Match-records JSONL from replay_parser (parsed_data/bo1.jsonl or bo3.jsonl).",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=str(DEFAULT_OUTPUT),
    show_default=True,
)
@click.option("--calc-base-url", default=DEFAULT_CALC_BASE_URL, show_default=True)
@click.option(
    "--format-id",
    default=None,
    help="Format ID for canonical-priors lookup. Auto-detected from input filename if omitted.",
)
@click.option("--limit", type=int, default=None, help="Process only the first N matches (test batch).")
@click.option("--concurrency", default=1, show_default=True,
              help="Max matches processed in parallel. Keep low (1-3) to respect OpenAI rate limits.")
@click.option("--dry-run", is_flag=True, help="Skip the LLM call; emit a placeholder assistant message.")
@click.option(
    "--provider",
    type=click.Choice(["openai", "anthropic", "google"]),
    default="openai",
    show_default=True,
    help="Which teacher LLM backend to use.",
)
@click.option(
    "--model",
    default=None,
    help="Override the default model id for the chosen provider.",
)
def cli(input_path, output_path, calc_base_url, format_id, limit, concurrency, dry_run, provider, model):
    """Generate the SFT training JSONL from parsed replay data."""
    resolved_format = _resolve_format_id(input_path, format_id)
    click.echo(
        f"using format_id={resolved_format}  dry_run={dry_run}  "
        f"provider={provider}  model={model or '(default)'}"
    )
    asyncio.run(
        run(
            input_path=input_path,
            output_path=output_path,
            calc_base_url=calc_base_url,
            format_id=resolved_format,
            limit=limit,
            concurrency=concurrency,
            dry_run=dry_run,
            model=model,
            provider=provider,
        )
    )


if __name__ == "__main__":
    cli()
