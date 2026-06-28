"""Top-2 Smogon "meta builds" per species, backed by Smogon Chaos JSON.

Pipeline role:
    Supplies the `=== META BUILDS ===` section of the per-turn user prompt: for
    each opponent Pokémon, the most common competitive choices (top-2 spreads +
    the shared most-common moves / item / ability / Tera). `prompt_formatting`
    renders it; the teacher reasons over it and tests it with `calculate_damage`.

    This REPLACES the Plan-v9-deleted `canonical_priors.py`, but as a different
    concern: the old module fused a single EV spread into the threat-matrix
    damage envelope (the "Probable Range" track). This one is a SEPARATE,
    clearly-labelled informational block. The threat matrix stays Absolute-only.

Honesty note — chaos data is MARGINAL:
    The chaos JSON ranks each field (Spreads, Moves, Items, Abilities, Tera)
    INDEPENDENTLY. It does NOT record which moves were run with which spread.
    So `MetaBuilds` is NOT a set of coherent joint builds — it is two common
    spreads set against the shared most-common moves/item/ability/Tera. Each
    usage% is "share of sets" for that field alone. The renderer + the
    system-prompt Meta-Builds Rule both state this so the model treats it as a
    possibility space, not a confirmed set.

Data source order (`get_meta_builds`):
    1. Smogon chaos on disk (full multi-marginal extraction)   -> source="chaos"
    2. Curated table (top ~40 Reg I mons; nature+EVs only)      -> source="curated"
    3. Base-stat heuristic (spread only)                        -> source="heuristic"
    (Chaos is confirmed available for gen9vgc2026regi + ...bo3, so 2/3 are the
    cold-start / brand-new-mon safety net.)

Bootstrap (one-off, downloads the latest available month):
    cd pipeline
    .venv/bin/python meta_builds.py --format-id gen9vgc2026regi      --chaos-cutoff 1630
    .venv/bin/python meta_builds.py --format-id gen9vgc2026regibo3   --chaos-cutoff 1630

Isolation contract:
    Pure data lookup at runtime (sync, lazily-loaded process cache). HTTP
    (`aiohttp`) is used ONLY in the bootstrap CLI / explicit `await
    fetch_chaos(...)`. No LLM, no replay parsing, no imports from sibling
    pipeline modules (a local `_norm` mirrors `damage_inferencer.species_key`
    rather than importing it, to honour the leaf-isolation rule). Imported only
    by `prompt_formatting` (renderer) and the orchestrators.
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
DEFAULT_CHAOS_CUTOFF = 1630  # skilled-ladder cut: cleaner meta than 0, less sparse than 1760

# How many of each marginal to surface.
N_SPREADS = 2
N_MOVES = 4
N_ITEMS = 2
N_ABILITIES = 2
N_TERA = 2

# Sentinels to drop from each marginal before ranking.
_DROP_MOVES = {"", "nomove", "Nothing"}
_DROP_ITEMS = {"", "nothing", "Nothing"}
_DROP_ABILITIES = {"", "(No Ability)", "No Ability"}
_DROP_TERA = {"", "Nothing", "nothing"}


def _chaos_path(format_id: str) -> Path:
    return DEFAULT_CHAOS_DIR / f"smogon_chaos_{format_id}.json"


def _norm(name: str) -> str:
    """alnum-lowercase normalizer — mirrors damage_inferencer.species_key.

    Kept local (not imported) to respect the pipeline leaf-isolation contract.
    """
    return "".join(c for c in name.lower() if c.isalnum())


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpreadOption:
    nature: str
    evs: Mapping[str, int]            # {hp:..., atk:..., ...}
    usage_pct: float                 # share of this species' sets with this spread (0 if unknown)


@dataclass(frozen=True)
class NamedShare:
    name: str
    usage_pct: float                 # share of this species' sets carrying this field value


@dataclass(frozen=True)
class MetaBuilds:
    species: str                     # display name as keyed
    spreads: tuple[SpreadOption, ...]    # up to N_SPREADS, weight-desc
    moves: tuple[NamedShare, ...]        # up to N_MOVES, weight-desc, sentinel-filtered
    items: tuple[NamedShare, ...]        # up to N_ITEMS
    abilities: tuple[NamedShare, ...]    # up to N_ABILITIES
    tera_types: tuple[NamedShare, ...]   # up to N_TERA
    source: str                      # "chaos" | "curated" | "heuristic"


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

    Walks back from this month, trying months until a 200 OK is found. Stamps
    the chosen month + cutoff into the file's `info` block for provenance.
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

    timeout = aiohttp.ClientTimeout(total=120, connect=15)
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
            # Provenance: stamp the month + cutoff into the info block (the file's
            # own `info` already carries `cutoff` + `number of battles`).
            info = data.setdefault("info", {})
            info["_fetched_month"] = month
            info["_chaos_cutoff"] = cutoff
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


def get_chaos_meta(format_id: str | None) -> dict[str, object]:
    """Provenance for the render header: month, cutoff, battle count.

    Returns `{}` when no chaos file is loaded (renderer omits the provenance)."""
    if not format_id:
        return {}
    info = (_get_chaos(format_id).get("info") or {})
    if not info:
        return {}
    return {
        "format_id": format_id,
        "month": info.get("_fetched_month"),
        "cutoff": info.get("_chaos_cutoff", info.get("cutoff")),
        "battles": info.get("number of battles"),
    }


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


def _top_n(marginal: object, n: int, *, total: float, drop: set[str]) -> tuple[NamedShare, ...]:
    """Top-n entries of a chaos marginal as NamedShares (% of `total` set mass)."""
    if not isinstance(marginal, dict) or total <= 0:
        return ()
    ranked = sorted(
        ((name, cnt) for name, cnt in marginal.items() if name not in drop and cnt > 0),
        key=lambda kv: kv[1],
        reverse=True,
    )
    return tuple(NamedShare(name=name, usage_pct=cnt / total * 100.0) for name, cnt in ranked[:n])


def _builds_from_chaos(species: str, format_id: str) -> MetaBuilds | None:
    entry = _lookup_in_chaos(species, _get_chaos(format_id))
    if not entry:
        return None
    spreads_raw = entry.get("Spreads")
    if not isinstance(spreads_raw, dict) or not spreads_raw:
        return None
    # Per-set mass: exactly one spread per set, so the Spreads total = weighted #sets.
    # Every other marginal is normalized against this so each % reads as "% of sets".
    total = float(sum(v for v in spreads_raw.values() if v > 0))
    if total <= 0:
        return None

    spreads: list[SpreadOption] = []
    for key in sorted(spreads_raw, key=lambda k: spreads_raw[k], reverse=True):
        try:
            nature, evs = _parse_spread_string(key)
        except (ValueError, KeyError):
            continue
        spreads.append(SpreadOption(nature=nature, evs=evs, usage_pct=spreads_raw[key] / total * 100.0))
        if len(spreads) >= N_SPREADS:
            break
    if not spreads:
        return None

    return MetaBuilds(
        species=species,
        spreads=tuple(spreads),
        moves=_top_n(entry.get("Moves"), N_MOVES, total=total, drop=_DROP_MOVES),
        items=_top_n(entry.get("Items"), N_ITEMS, total=total, drop=_DROP_ITEMS),
        abilities=_top_n(entry.get("Abilities"), N_ABILITIES, total=total, drop=_DROP_ABILITIES),
        tera_types=_top_n(entry.get("Tera Types"), N_TERA, total=total, drop=_DROP_TERA),
        source="chaos",
    )


# ---------------------------------------------------------------------------
# Curated + heuristic fallback (cold-start safety net)
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


def _fallback_builds(species: str) -> MetaBuilds:
    """Spread-only builds for a mon absent from chaos (cold start / brand-new mon).

    No move/item/ability/Tera data (we have no usage signal), so those render as
    'unknown'. The `source` label makes the provenance honest in the prompt."""
    key = _norm(species)
    if key in _CURATED:
        nature, evs = _CURATED[key]
        src = "curated"
    else:
        base = _BASE_STATS.get(key, _GENERIC_FALLBACK_BASE)
        nature, evs = _heuristic_spread(base)
        src = "heuristic"
    return MetaBuilds(
        species=species,
        spreads=(SpreadOption(nature=nature, evs=dict(evs), usage_pct=0.0),),
        moves=(), items=(), abilities=(), tera_types=(),
        source=src,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_meta_builds(species: str, format_id: str | None = None) -> MetaBuilds | None:
    """Top builds for one species. Lookup order: chaos -> curated -> heuristic.

    Returns None only for an empty/invalid species. Chaos is the primary source
    (confirmed available for both formats); curated + heuristic are the
    cold-start safety net and carry spread-only data (source-labelled)."""
    if not species or not species.strip():
        return None
    if format_id:
        from_chaos = _builds_from_chaos(species, format_id)
        if from_chaos is not None:
            return from_chaos
    return _fallback_builds(species)


# ---------------------------------------------------------------------------
# Future seam (NOT implemented): observation-narrowing
# ---------------------------------------------------------------------------
# Once a mon has revealed a move / item / ability, prefer builds consistent with
# the reveal (Boyd's "once we see Spore, show builds that run Spore" idea). The
# chaos marginals are independent, so this is a SOFT prior: annotate kept entries
# ("consistent with observed Intimidate") rather than silently re-rank. The
# renderer already accepts `p2_knowledge`, so this lands as a separate
# `narrow_builds(builds, revealed)` step with no signature churn. See notes/TODO.md.


# ---------------------------------------------------------------------------
# CLI bootstrap
# ---------------------------------------------------------------------------


@click.command()
@click.option("--format-id", default="gen9vgc2026regi", show_default=True)
@click.option("--chaos-cutoff", "cutoff", default=DEFAULT_CHAOS_CUTOFF, show_default=True,
              help="Smogon usage cutoff. 0=all rated, 1500/1630/1760=skill cuts.")
@click.option(
    "--output", type=click.Path(path_type=Path), default=None,
    help="Where to save the JSON. Default: pipeline/data/smogon_chaos_<format_id>.json",
)
@click.option("--months-to-try", default=12, show_default=True,
              help="How many months back to walk before giving up.")
def cli(format_id: str, cutoff: int, output: Path | None, months_to_try: int) -> None:
    """Bootstrap meta builds by downloading Smogon Chaos JSON."""
    try:
        month, count = asyncio.run(
            fetch_chaos(format_id, cutoff=cutoff, output=output, months_to_try=months_to_try)
        )
    except RuntimeError as e:
        click.echo(f"FATAL: {e}", err=True)
        raise SystemExit(1)
    out_path = output if output is not None else _chaos_path(format_id)
    click.echo(f"Fetched {format_id} chaos for {month} (cutoff={cutoff}) → {out_path}  ({count} species)")
    # Surface which marginals are present so a future month dropping a field is caught early.
    sample = json.loads(out_path.read_text()).get("data", {})
    if sample:
        any_species = next(iter(sample.values()))
        present = [k for k in ("Spreads", "Moves", "Items", "Abilities", "Tera Types") if k in any_species]
        click.echo(f"  marginals present on sample species: {', '.join(present)}")


if __name__ == "__main__":
    cli()


__all__ = [
    "DEFAULT_CHAOS_CUTOFF",
    "MetaBuilds",
    "NamedShare",
    "SpreadOption",
    "STATS",
    "fetch_chaos",
    "get_chaos_meta",
    "get_meta_builds",
]
