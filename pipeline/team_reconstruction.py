"""Team reconstruction helpers — extract per-side team info from parsed match games.

Pipeline role:
    Used by `master_pipeline.py` to seed knowledge states and render the
    Bo1 system-prompt team block, and by `prompt_formatting.py` for
    species-key normalization.

Isolation contract:
    Pure data transforms over snapshots. No I/O, no LLM, no calc service.
    Imports nothing from sibling pipeline modules.
"""
from __future__ import annotations

from typing import Any


def _species_key(s: str) -> str:
    """Normalize a species name to its lowercase, alphanumeric-only form.

    Used as a stable key for cross-snapshot identity (e.g.
    "Calyrex-Shadow" → "calyrexshadow"). Same convention used downstream
    by `damage_inferencer` and `threat_matrix`.
    """
    return "".join(c for c in s.lower() if c.isalnum())


def _prefer_display(new: str, old: str) -> bool:
    """True when `new` is a better display name than `old` — i.e. `old` is the
    bare id-form (equal to its own species_key) and `new` carries case/hyphen
    (e.g. prefer "Zamazenta-Crowned" over "zamazentacrowned")."""
    return old == _species_key(old) and new != _species_key(new)


def _reconstruct_team(games: list[dict], side: str) -> dict[str, dict[str, Any]]:
    """Forward-scan all snapshots, aggregate revealed `side` info per species.

    Aggregation is keyed by the normalized `_species_key`, so a forme restricted
    that appears as both its id-form ("zamazentacrowned", from a never-sent
    bench placeholder) and its display-form ("Zamazenta-Crowned", from an active
    snapshot) collapses to ONE entry that keeps the best display name. The
    returned dict is re-keyed by that display name, so callers reading `.keys()`
    (the species universe) get deduped display-form species.

    Pads each Pokémon's move list to exactly 4 with `"[UNREVEALED_MOVE]"`.
    """
    by_key: dict[str, dict[str, Any]] = {}

    def _ensure(species: str) -> dict[str, Any]:
        k = _species_key(species)
        entry = by_key.get(k)
        if entry is None:
            entry = {
                "species": species, "item": None, "ability": None,
                "teraType": None, "isTerastallized": False, "moves": [],
            }
            by_key[k] = entry
        elif _prefer_display(species, entry["species"]):
            entry["species"] = species
        return entry

    for game in games:
        for snap in game.get("snapshots", []):
            for p in snap.get(side, {}).get("active", []):
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
            for b in snap.get(side, {}).get("bench", []):
                _ensure(b["species"])

    for entry in by_key.values():
        while len(entry["moves"]) < 4:
            entry["moves"].append("[UNREVEALED_MOVE]")
        entry["moves"] = entry["moves"][:4]

    # Re-key by display species so `.keys()` is deduped display-form, not id-form.
    return {entry["species"]: entry for entry in by_key.values()}


def reconstruct_p1_team(games: list[dict]) -> dict[str, dict[str, Any]]:
    """Forward-scan P1 snapshots → revealed P1 info per species (see `_reconstruct_team`)."""
    return _reconstruct_team(games, "p1")


def reconstruct_p2_species(games: list[dict]) -> list[str]:
    """Union of every P2 species observed (active or bench) across the match."""
    seen: list[str] = []
    for game in games:
        for snap in game.get("snapshots", []):
            for p in snap.get("p2", {}).get("active", []) + snap.get("p2", {}).get("bench", []):
                if p["species"] not in seen:
                    seen.append(p["species"])
    return seen


def reconstruct_p2_team(games: list[dict]) -> dict[str, dict[str, Any]]:
    """Forward-scan all snapshots, aggregate revealed P2 info per species.

    Pure mirror of `reconstruct_p1_team` for the opponent side. Used by
    the bench renderer in Bo1 (CTS) where there's no team-sheet to fall
    back on — we only know what's been revealed during play. Each entry
    has the same `{species, item?, ability?, teraType?, isTerastallized,
    moves[]}` shape, with moves padded to 4 with `[UNREVEALED_MOVE]`.

    Callers should typically gate the output by `snapshot.p2.seenSpecies`
    before display — the player only "knows" about opponent mons they
    have observed on field, not every species the forward-scan can
    eventually identify.
    """
    return _reconstruct_team(games, "p2")


def team_sheets_for_match(games: list[dict]) -> dict[str, list[dict]] | None:
    """Return the first non-null `teamSheets` from any game in the match.

    All games in a Bo3 series carry the same sheet, so we just take the
    earliest one available. None means CTS for the whole match.
    """
    for g in games:
        sheets = g.get("teamSheets")
        if sheets and sheets.get("p1") and sheets.get("p2"):
            return sheets
    return None


def brought_species_keys_for_game(game: dict) -> set[str]:
    """Species (normalized keys) actually brought by P1 to this single game.

    Since the parser now emits P1 bench as the full pre-scanned brought-set
    on every snapshot, the union of P1 active + P1 bench at any single
    snapshot already gives the brought 4. We still take the union across
    snapshots as a safety net (handles edge cases where a brought species
    only appears in one of the two structures).
    """
    out: set[str] = set()
    for snap in game.get("snapshots", []):
        for p in snap.get("p1", {}).get("active", []):
            out.add(_species_key(p["species"]))
        for b in snap.get("p1", {}).get("bench", []):
            out.add(_species_key(b["species"]))
    return out
