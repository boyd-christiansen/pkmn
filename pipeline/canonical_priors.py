"""Canonical "meta spread" priors per species, backed by Smogon Chaos JSON.

Pipeline role:
    Returns the *probable* EVs / IVs / Nature for a given species under
    standard competitive play. Used by `threat_matrix.py` to compute a
    Probable Range alongside the strict Absolute envelope from observed
    damage.

Data source order:
    1. Real Smogon ladder usage data fetched from
       smogon.com/stats/{YYYY-MM}/chaos/{format_id}-{cutoff}.json — the
       single most-used spread per species (key with highest usage count
       under each species' "Spreads" dict).
    2. Curated table for the top ~40 Reg I species — used when chaos data
       is missing on disk OR a species genuinely has 0 usage rows.
    3. Base-stat heuristic for anything else (offensive vs bulky vs default).

Bootstrap (one-off, downloads the latest available month):
    cd pipeline
    .venv/bin/python canonical_priors.py --format-id gen9vgc2026regi
    .venv/bin/python canonical_priors.py --format-id gen9vgc2026regibo3

Isolation contract:
    Pure data lookup at runtime. The fetcher uses aiohttp but only when
    invoked via the CLI (or explicit `await fetch_chaos(...)`). Lookups
    are sync and rely on a lazily-loaded process cache.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Mapping

import aiohttp
import click

STATS = ("hp", "atk", "def", "spa", "spd", "spe")
DEFAULT_IVS: Mapping[str, int] = {s: 31 for s in STATS}

CHAOS_URL_TEMPLATE = "https://www.smogon.com/stats/{date}/chaos/{format_id}-{cutoff}.json"
DEFAULT_CHAOS_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_CHAOS_CUTOFF = 0  # 0 = all rated games (broadest), 1500 = ladder cut


def _chaos_path(format_id: str) -> Path:
    return DEFAULT_CHAOS_DIR / f"smogon_chaos_{format_id}.json"


@dataclass(frozen=True)
class ProbableSpread:
    evs: Mapping[str, int]
    nature: str
    ivs: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_IVS))
    source: str = "heuristic"  # "chaos" | "curated" | "heuristic"


def _norm(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


# ---------------------------------------------------------------------------
# Smogon chaos fetcher (CLI-invoked)
# ---------------------------------------------------------------------------


async def fetch_chaos(
    format_id: str,
    *,
    cutoff: int = DEFAULT_CHAOS_CUTOFF,
    output: Path | None = None,
    months_to_try: int = 12,
) -> tuple[str, int]:
    """Download the most recent available chaos JSON for a format.

    Walks back from this month, trying months until a 200 OK is found.
    Returns `(month, num_species)`. Raises RuntimeError if no month worked.
    """
    target = output if output is not None else _chaos_path(format_id)
    today = date.today()
    months: list[str] = []
    y, m = today.year, today.month
    for _ in range(months_to_try):
        m -= 1
        if m == 0:
            m = 12
            y -= 1
        months.append(f"{y}-{m:02d}")

    timeout = aiohttp.ClientTimeout(total=60, connect=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for month in months:
            url = CHAOS_URL_TEMPLATE.format(date=month, format_id=format_id, cutoff=cutoff)
            try:
                async with session.get(url) as r:
                    if r.status != 200:
                        continue
                    data = await r.json(content_type=None)
            except aiohttp.ClientError:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(data))
            return month, len(data.get("data", {}))
    raise RuntimeError(
        f"No chaos data for {format_id} (cutoff={cutoff}) in last {months_to_try} months"
    )


# ---------------------------------------------------------------------------
# Lazy on-disk cache lookup
# ---------------------------------------------------------------------------


_CHAOS_CACHE: dict[str, dict] = {}


def _get_chaos(format_id: str) -> dict:
    if format_id in _CHAOS_CACHE:
        return _CHAOS_CACHE[format_id]
    path = _chaos_path(format_id)
    if path.exists():
        try:
            _CHAOS_CACHE[format_id] = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            _CHAOS_CACHE[format_id] = {}
    else:
        _CHAOS_CACHE[format_id] = {}
    return _CHAOS_CACHE[format_id]


def _parse_spread_string(s: str) -> tuple[str, dict[str, int]]:
    """Parse 'Modest:0/0/0/252/4/252' into ('Modest', {hp:0, atk:0, ...})."""
    if ":" not in s:
        raise ValueError(f"missing nature separator: {s!r}")
    nature, rest = s.split(":", 1)
    parts = rest.split("/")
    if len(parts) != 6:
        raise ValueError(f"need 6 EV components: {s!r}")
    return nature, dict(zip(STATS, (int(p) for p in parts), strict=True))


def _lookup_in_chaos(species: str, chaos: dict) -> dict | None:
    data = chaos.get("data") or {}
    if not data:
        return None
    if species in data:
        return data[species]
    target = _norm(species)
    for k, v in data.items():
        if _norm(k) == target:
            return v
    return None


def _spread_from_chaos(species: str, format_id: str) -> ProbableSpread | None:
    chaos = _get_chaos(format_id)
    entry = _lookup_in_chaos(species, chaos)
    if not entry:
        return None
    spreads = entry.get("Spreads")
    if not spreads:
        return None
    top_key = max(spreads, key=lambda k: spreads[k])
    try:
        nature, evs = _parse_spread_string(top_key)
    except (ValueError, KeyError):
        return None
    return ProbableSpread(evs=evs, nature=nature, source="chaos")


# ---------------------------------------------------------------------------
# Curated + heuristic fallback (preserved from the previous version)
# ---------------------------------------------------------------------------

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


def _fallback_spread(species: str) -> ProbableSpread:
    key = _norm(species)
    if key in _CURATED:
        nature, evs = _CURATED[key]
        return ProbableSpread(evs=dict(evs), nature=nature, source="curated")
    base = _BASE_STATS.get(key, _GENERIC_FALLBACK_BASE)
    nature, evs = _heuristic_spread(base)
    return ProbableSpread(evs=evs, nature=nature, source="heuristic")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_probable_spread(species: str, format_id: str | None = None) -> ProbableSpread:
    """Return the probable competitive spread for a species.

    Lookup order:
      1. Smogon chaos data on disk (if `format_id` provided and cache present)
      2. Curated table (top ~40 Reg I species)
      3. Base-stat heuristic
    """
    if format_id:
        from_chaos = _spread_from_chaos(species, format_id)
        if from_chaos is not None:
            return from_chaos
    return _fallback_spread(species)


# ---------------------------------------------------------------------------
# CLI bootstrap
# ---------------------------------------------------------------------------


@click.command()
@click.option("--format-id", default="gen9vgc2026regi", show_default=True)
@click.option("--cutoff", default=DEFAULT_CHAOS_CUTOFF, show_default=True,
              help="Smogon usage cutoff. 0 = all rated games, 1500 = ladder cut.")
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Where to save the JSON. Default: pipeline/data/smogon_chaos_<format_id>.json",
)
@click.option("--months-to-try", default=12, show_default=True,
              help="How many months back to walk before giving up.")
def cli(
    format_id: str,
    cutoff: int,
    output: Path | None,
    months_to_try: int,
) -> None:
    """Bootstrap canonical priors by downloading Smogon Chaos JSON."""
    try:
        month, count = asyncio.run(
            fetch_chaos(format_id, cutoff=cutoff, output=output, months_to_try=months_to_try)
        )
    except RuntimeError as e:
        click.echo(f"FATAL: {e}", err=True)
        raise SystemExit(1)
    out_path = output if output is not None else _chaos_path(format_id)
    click.echo(f"Fetched {format_id} chaos for {month} → {out_path}  ({count} species)")


if __name__ == "__main__":
    cli()


__all__ = [
    "DEFAULT_CHAOS_CUTOFF",
    "DEFAULT_IVS",
    "ProbableSpread",
    "STATS",
    "fetch_chaos",
    "get_probable_spread",
]
