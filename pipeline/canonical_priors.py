"""Canonical "meta spread" priors per species — mock implementation.

Pipeline role:
    Returns the *probable* EVs / IVs / Nature for a given species under standard
    competitive play. Used by `threat_matrix.py` to compute a Probable Range
    alongside the strict Absolute envelope from observed damage.

    This is a stopgap. Real Smogon usage data (per-species spreads from
    pokemonshowdown.com/stats/) will replace this when wired up; until then we
    use a small hand-curated table for tier-1 species plus a base-stats
    heuristic for the rest.

Isolation contract:
    Pure data lookup. No HTTP, no LLM, no imports from sibling pipeline modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

STATS = ("hp", "atk", "def", "spa", "spd", "spe")
DEFAULT_IVS: Mapping[str, int] = {s: 31 for s in STATS}


@dataclass(frozen=True)
class ProbableSpread:
    evs: Mapping[str, int]
    nature: str
    ivs: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_IVS))


def _norm(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


# Hand-curated spreads for the most common Reg I species. These are the
# answers a strong VGC player would give for "what's the standard spread on X?"
# When we wire in real Smogon usage data, this table goes away.
_CURATED: dict[str, tuple[str, dict[str, int]]] = {
    "calyrexshadow":     ("Timid",   {"hp": 4,   "atk": 0,   "def": 0,   "spa": 252, "spd": 0,   "spe": 252}),
    "calyrexice":        ("Adamant", {"hp": 252, "atk": 252, "def": 4,   "spa": 0,   "spd": 0,   "spe": 0}),
    "miraidon":          ("Modest",  {"hp": 4,   "atk": 0,   "def": 0,   "spa": 252, "spd": 0,   "spe": 252}),
    "koraidon":          ("Jolly",   {"hp": 4,   "atk": 252, "def": 0,   "spa": 0,   "spd": 0,   "spe": 252}),
    "zacian":            ("Jolly",   {"hp": 4,   "atk": 252, "def": 0,   "spa": 0,   "spd": 0,   "spe": 252}),
    "zaciancrowned":     ("Jolly",   {"hp": 4,   "atk": 252, "def": 0,   "spa": 0,   "spd": 0,   "spe": 252}),
    "zamazenta":         ("Jolly",   {"hp": 4,   "atk": 252, "def": 0,   "spa": 0,   "spd": 0,   "spe": 252}),
    "zamazentacrowned":  ("Impish",  {"hp": 252, "atk": 4,   "def": 252, "spa": 0,   "spd": 0,   "spe": 0}),
    "kyogre":            ("Modest",  {"hp": 252, "atk": 0,   "def": 0,   "spa": 252, "spd": 4,   "spe": 0}),
    "groudon":           ("Adamant", {"hp": 252, "atk": 252, "def": 4,   "spa": 0,   "spd": 0,   "spe": 0}),
    "lunala":            ("Modest",  {"hp": 252, "atk": 0,   "def": 4,   "spa": 252, "spd": 0,   "spe": 0}),
    "necrozmaduskmane":  ("Adamant", {"hp": 252, "atk": 252, "def": 4,   "spa": 0,   "spd": 0,   "spe": 0}),
    "incineroar":        ("Sassy",   {"hp": 244, "atk": 0,   "def": 4,   "spa": 0,   "spd": 252, "spe": 4}),
    "amoonguss":         ("Calm",    {"hp": 252, "atk": 0,   "def": 4,   "spa": 0,   "spd": 252, "spe": 0}),
    "rillaboom":         ("Adamant", {"hp": 252, "atk": 252, "def": 4,   "spa": 0,   "spd": 0,   "spe": 0}),
    "urshifu":           ("Adamant", {"hp": 4,   "atk": 252, "def": 0,   "spa": 0,   "spd": 0,   "spe": 252}),
    "urshifurapidstrike":("Adamant", {"hp": 4,   "atk": 252, "def": 0,   "spa": 0,   "spd": 0,   "spe": 252}),
    "fluttermane":       ("Timid",   {"hp": 4,   "atk": 0,   "def": 0,   "spa": 252, "spd": 0,   "spe": 252}),
    "ironhands":         ("Adamant", {"hp": 252, "atk": 252, "def": 4,   "spa": 0,   "spd": 0,   "spe": 0}),
    "ironbundle":        ("Timid",   {"hp": 4,   "atk": 0,   "def": 0,   "spa": 252, "spd": 0,   "spe": 252}),
    "ragingbolt":        ("Modest",  {"hp": 252, "atk": 0,   "def": 4,   "spa": 252, "spd": 0,   "spe": 0}),
    "chienpao":          ("Jolly",   {"hp": 4,   "atk": 252, "def": 0,   "spa": 0,   "spd": 0,   "spe": 252}),
    "ursalunabloodmoon": ("Modest",  {"hp": 252, "atk": 0,   "def": 4,   "spa": 252, "spd": 0,   "spe": 0}),
    "dragonite":         ("Adamant", {"hp": 4,   "atk": 252, "def": 0,   "spa": 0,   "spd": 0,   "spe": 252}),
    "grimmsnarl":        ("Sassy",   {"hp": 244, "atk": 0,   "def": 12,  "spa": 0,   "spd": 252, "spe": 0}),
    "smeargle":          ("Jolly",   {"hp": 252, "atk": 0,   "def": 4,   "spa": 0,   "spd": 0,   "spe": 252}),
    "farigiraf":         ("Sassy",   {"hp": 252, "atk": 0,   "def": 4,   "spa": 0,   "spd": 252, "spe": 0}),
    "landorustherian":   ("Adamant", {"hp": 252, "atk": 252, "def": 4,   "spa": 0,   "spd": 0,   "spe": 0}),
    "tornadustherian":   ("Timid",   {"hp": 4,   "atk": 0,   "def": 0,   "spa": 252, "spd": 0,   "spe": 252}),
    "pelipper":          ("Modest",  {"hp": 252, "atk": 0,   "def": 4,   "spa": 252, "spd": 0,   "spe": 0}),
    "whimsicott":        ("Timid",   {"hp": 252, "atk": 0,   "def": 4,   "spa": 0,   "spd": 0,   "spe": 252}),
    "indeedeefemale":    ("Calm",    {"hp": 252, "atk": 0,   "def": 4,   "spa": 0,   "spd": 252, "spe": 0}),
    "ironvaliant":       ("Timid",   {"hp": 4,   "atk": 0,   "def": 0,   "spa": 252, "spd": 0,   "spe": 252}),
    "annihilape":        ("Adamant", {"hp": 252, "atk": 252, "def": 4,   "spa": 0,   "spd": 0,   "spe": 0}),
    "tatsugiri":         ("Modest",  {"hp": 252, "atk": 0,   "def": 4,   "spa": 252, "spd": 0,   "spe": 0}),
    "garchomp":          ("Jolly",   {"hp": 4,   "atk": 252, "def": 0,   "spa": 0,   "spd": 0,   "spe": 252}),
    "ogerponhearthflame":("Adamant", {"hp": 4,   "atk": 252, "def": 0,   "spa": 0,   "spd": 0,   "spe": 252}),
    "gholdengo":         ("Modest",  {"hp": 252, "atk": 0,   "def": 4,   "spa": 252, "spd": 0,   "spe": 0}),
    "ironcrown":         ("Modest",  {"hp": 4,   "atk": 0,   "def": 0,   "spa": 252, "spd": 0,   "spe": 252}),
    "terapagosterastal": ("Modest",  {"hp": 252, "atk": 0,   "def": 4,   "spa": 252, "spd": 0,   "spe": 0}),
    "brutebonnet":       ("Adamant", {"hp": 252, "atk": 252, "def": 4,   "spa": 0,   "spd": 0,   "spe": 0}),
    "ursaluna":          ("Adamant", {"hp": 252, "atk": 252, "def": 4,   "spa": 0,   "spd": 0,   "spe": 0}),
}

# Base stats fallback table for the heuristic (for species not in _CURATED).
# Source: stats.pokemonsmogon — only used when curated lookup misses.
_BASE_STATS: dict[str, dict[str, int]] = {
    "tornadus":          {"hp": 79,  "atk": 115, "def": 70,  "spa": 125, "spd": 80,  "spe": 111},
    "rotomwash":         {"hp": 50,  "atk": 65,  "def": 107, "spa": 105, "spd": 107, "spe": 86},
    "rotomheat":         {"hp": 50,  "atk": 65,  "def": 107, "spa": 105, "spd": 107, "spe": 86},
    "tyranitar":         {"hp": 100, "atk": 134, "def": 110, "spa": 95,  "spd": 100, "spe": 61},
    "heatran":           {"hp": 91,  "atk": 90,  "def": 106, "spa": 130, "spd": 106, "spe": 77},
    "indeedee":          {"hp": 60,  "atk": 65,  "def": 55,  "spa": 105, "spd": 95,  "spe": 95},
    "landorus":          {"hp": 89,  "atk": 125, "def": 90,  "spa": 115, "spd": 80,  "spe": 101},
    "ogerpon":           {"hp": 80,  "atk": 120, "def": 84,  "spa": 60,  "spd": 96,  "spe": 110},
    "ogerponcornerstone":{"hp": 80,  "atk": 120, "def": 84,  "spa": 60,  "spd": 96,  "spe": 110},
    "ogerponwellspring": {"hp": 80,  "atk": 120, "def": 84,  "spa": 60,  "spd": 96,  "spe": 110},
    "volcarona":         {"hp": 85,  "atk": 60,  "def": 65,  "spa": 135, "spd": 105, "spe": 100},
    "ironjugulis":       {"hp": 94,  "atk": 80,  "def": 86,  "spa": 122, "spd": 80,  "spe": 108},
    "tinglu":            {"hp": 155, "atk": 110, "def": 125, "spa": 55,  "spd": 80,  "spe": 45},
    "chiyu":             {"hp": 55,  "atk": 80,  "def": 80,  "spa": 135, "spd": 120, "spe": 100},
    "ironmoth":          {"hp": 80,  "atk": 70,  "def": 60,  "spa": 140, "spd": 110, "spe": 110},
    "wochien":           {"hp": 85,  "atk": 85,  "def": 100, "spa": 95,  "spd": 135, "spe": 70},
    "scream tail":       {"hp": 115, "atk": 65,  "def": 99,  "spa": 65,  "spd": 115, "spe": 111},
    "screamtail":        {"hp": 115, "atk": 65,  "def": 99,  "spa": 65,  "spd": 115, "spe": 111},
    "porygon2":          {"hp": 85,  "atk": 80,  "def": 90,  "spa": 105, "spd": 95,  "spe": 60},
    "regigigas":         {"hp": 110, "atk": 160, "def": 110, "spa": 80,  "spd": 110, "spe": 100},
    "magearna":          {"hp": 80,  "atk": 95,  "def": 115, "spa": 130, "spd": 115, "spe": 65},
    "eternatus":         {"hp": 140, "atk": 85,  "def": 95,  "spa": 145, "spd": 95,  "spe": 130},
    "rayquaza":          {"hp": 105, "atk": 150, "def": 90,  "spa": 150, "spd": 90,  "spe": 95},
    "necrozmadawnwings": {"hp": 97,  "atk": 113, "def": 109, "spa": 157, "spd": 127, "spe": 77},
    "xerneas":           {"hp": 126, "atk": 131, "def": 95,  "spa": 131, "spd": 98,  "spe": 99},
    "yveltal":           {"hp": 126, "atk": 131, "def": 95,  "spa": 131, "spd": 98,  "spe": 99},
    "solgaleo":          {"hp": 137, "atk": 137, "def": 107, "spa": 113, "spd": 89,  "spe": 97},
    "hooh":              {"hp": 106, "atk": 130, "def": 90,  "spa": 110, "spd": 154, "spe": 90},
    "mewtwo":            {"hp": 106, "atk": 110, "def": 90,  "spa": 154, "spd": 90,  "spe": 130},
    "primarina":         {"hp": 80,  "atk": 74,  "def": 74,  "spa": 126, "spd": 116, "spe": 60},
    "rotom":             {"hp": 50,  "atk": 50,  "def": 77,  "spa": 95,  "spd": 77,  "spe": 91},
}

_GENERIC_FALLBACK_BASE = {"hp": 80, "atk": 80, "def": 80, "spa": 80, "spd": 80, "spe": 80}


def _heuristic_spread(base: Mapping[str, int]) -> tuple[str, dict[str, int]]:
    """Pick a plausible competitive spread from base stats.

    Categories (in priority order):
      1. Tier-1 offensive (max(atk, spa) ≥ 110): max attacking stat + Spe.
         Speed-positive nature if base_spe ≥ 90, otherwise neutral-+atk nature.
      2. Bulky (hp ≥ 95 AND spe ≤ 70): max HP + heavier defensive side.
      3. Default mixed: max SpA + Spe.
    """
    hp, atk, spa, spe = base["hp"], base["atk"], base["spa"], base["spe"]
    bdef, bspd = base["def"], base["spd"]

    if max(atk, spa) >= 110:
        if atk >= spa:
            nature = "Jolly" if spe >= 90 else "Adamant"
            return nature, {"hp": 4, "atk": 252, "def": 0, "spa": 0, "spd": 0, "spe": 252}
        nature = "Timid" if spe >= 90 else "Modest"
        return nature, {"hp": 4, "atk": 0, "def": 0, "spa": 252, "spd": 0, "spe": 252}

    if hp >= 95 and spe <= 70:
        if bdef > bspd:
            return "Bold", {"hp": 252, "atk": 0, "def": 252, "spa": 0, "spd": 4, "spe": 0}
        return "Calm", {"hp": 252, "atk": 0, "def": 4, "spa": 0, "spd": 252, "spe": 0}

    nature = "Timid" if spe >= 90 else "Modest"
    return nature, {"hp": 4, "atk": 0, "def": 0, "spa": 252, "spd": 0, "spe": 252}


def get_probable_spread(species: str, format_id: str | None = None) -> ProbableSpread:
    """Return the canonical meta spread for a species.

    `format_id` is reserved for when we branch on Reg I vs Reg I Bo3 vs future
    formats; ignored by the mock.
    """
    key = _norm(species)
    if key in _CURATED:
        nature, evs = _CURATED[key]
        return ProbableSpread(evs=dict(evs), nature=nature)

    base = _BASE_STATS.get(key, _GENERIC_FALLBACK_BASE)
    nature, evs = _heuristic_spread(base)
    return ProbableSpread(evs=evs, nature=nature)


__all__ = ["DEFAULT_IVS", "ProbableSpread", "STATS", "get_probable_spread"]
