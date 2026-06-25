"""Deterministic, state-grounded CoT-correctness verifier.

Why this exists:
    The teacher synthesizes its chain-of-thought *knowing the answer*, so the
    reasoning is a rationalization — usually good, occasionally wrong. The
    leak filter only catches oracle references; it does NOT catch a CoT whose
    own arithmetic is false. A real example (gemini-2.5-pro, first sample):

        "Ice Beam ... deals between 31.9% and 68.6% of Koraidon's max HP.
         Since Koraidon is already at 35%, this guarantees a KO."

    The low roll (31.9%) is below 35% → NOT a guaranteed KO. A student trained
    on that learns false damage math. This module flags exactly that class.

Target-aware association (the hard part — VGC is doubles):
    A naive "find a KO-guarantee + any damage range + any HP, then check the
    arithmetic" approach false-positives constantly, because every doubles CoT
    mentions 2–4 Pokémon, each with its own HP and damage figures. The first
    version cross-paired a KO claim about one target with a range about another
    and an HP of the attacker itself. So this version requires all three to
    resolve to the SAME target species:
      • the KO-guarantee phrase's nearest species = the target,
      • the damage range's nearest species must equal that target,
      • the target's HP is taken from the ACTUAL game state (on_field_hps),
        not from whatever "N%" sits nearby in the prose.
    Only then is the arithmetic checked. Without on-field HP context we cannot
    safely verify, so we return nothing (under-flag rather than false-flag).

Checks (high precision, partial recall):
    • ko_not_guaranteed — "guaranteed KO" on a target whose state HP h sits
      inside the claimed damage band (lo < h ≤ hi): the low roll survives.
    • ko_impossible — "guaranteed KO" where the move's max roll hi < h: it
      can't KO this turn at all.
    Speed/survival claims and matrix-misstatement errors are out of scope for
    the deterministic layer — they need the LLM-judge layer (follow-up).

Lifecycle (the "meshing"):
    `find_violations(cot, on_field_hps)` is one importable function reused as a
    data-gen filter (drop/regenerate flagged rows), an eval metric (score the
    fine-tuned model's CoT), and later an RL reward term.

Isolation contract:
    Pure text + a {species: hp%} dict → findings. No calc service, no LLM.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import click

_RANGE = re.compile(r"(\d+(?:\.\d+)?)\s*%?\s*(?:to|and|[–—-])\s*(\d+(?:\.\d+)?)\s*%")
_GUARANTEE = re.compile(r"\b(guarantee\w*|secur\w*|ensur\w*)\b", re.IGNORECASE)
_KOWORD = re.compile(r"\b(KO|knockout|knock\s*out|OHKO)\b", re.IGNORECASE)
# Parse "[a] Flutter Mane | HP 1% | ..." active lines from a user prompt.
_ACTIVE_HP = re.compile(r"\[[abc]\]\s+([A-Za-z0-9'.\- ]+?)\s+\|\s+HP\s+(\d+(?:\.\d+)?)\s*%")


def _species_key(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum())


def extract_on_field_hps(user_prompt: str) -> dict[str, float]:
    """{display_species: hp_percent} for every active mon (both sides) in a
    rendered user prompt. The KO target of any 'secure the KO' claim is an
    on-field mon, so this is the authoritative HP source for the check."""
    out: dict[str, float] = {}
    for m in _ACTIVE_HP.finditer(user_prompt):
        out[m.group(1).strip()] = float(m.group(2))
    return out


def _species_positions(cot: str, species_names: list[str]) -> list[tuple[int, str]]:
    low = cot.lower()
    out: list[tuple[int, str]] = []
    for name in species_names:
        nl = name.lower()
        start = 0
        while True:
            i = low.find(nl, start)
            if i < 0:
                break
            out.append((i, name))
            start = i + len(nl)
    return out


def _nearest_species(pos: int, sp_positions: list[tuple[int, str]], max_dist: int) -> str | None:
    best, best_d = None, max_dist + 1
    for sp_pos, name in sp_positions:
        d = abs(sp_pos - pos)
        if d < best_d:
            best, best_d = name, d
    return best


def find_violations(cot: str | None, on_field_hps: dict[str, float] | None = None) -> list[dict[str, Any]]:
    """Objective KO-arithmetic errors in `cot`, grounded against `on_field_hps`
    ({species: hp%} from the turn state). Returns [] when there's no context to
    verify against (we under-flag rather than guess)."""
    if not cot or not on_field_hps:
        return []
    sp_positions = _species_positions(cot, list(on_field_hps.keys()))
    if not sp_positions:
        return []
    hp_by_key = {_species_key(k): v for k, v in on_field_hps.items()}
    ranges = [(m.start(), float(m.group(1)), float(m.group(2))) for m in _RANGE.finditer(cot)]

    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, float, float]] = set()
    for g in _GUARANTEE.finditer(cot):
        ko = next((k for k in _KOWORD.finditer(cot) if abs(k.start() - g.start()) <= 70), None)
        if not ko:
            continue
        claim_pos = (g.start() + ko.start()) // 2
        target = _nearest_species(claim_pos, sp_positions, max_dist=120)
        if not target:
            continue
        tkey = _species_key(target)
        hp = hp_by_key.get(tkey)
        if hp is None:
            continue
        for rpos, a, b in ranges:
            lo, hi = (a, b) if a <= b else (b, a)
            # The range must itself be about this same target.
            rsp = _nearest_species(rpos, sp_positions, max_dist=80)
            if not rsp or _species_key(rsp) != tkey:
                continue
            if hi < hp:
                vtype = "ko_impossible"
                detail = (f"claims a guaranteed KO on {target}, but its max roll {hi:.1f}% "
                          f"< {target}'s actual HP {hp:.0f}% — it cannot KO this turn")
            elif lo < hp <= hi:
                vtype = "ko_not_guaranteed"
                detail = (f"claims a guaranteed KO on {target}, but damage {lo:.1f}–{hi:.1f}% "
                          f"vs {target} at {hp:.0f}% HP — the low roll {lo:.1f}% < {hp:.0f}% "
                          f"leaves it alive (a likely KO, not a guaranteed one)")
            else:
                continue
            key = (vtype, tkey, lo, hi)
            if key in seen:
                continue
            seen.add(key)
            # Quote a window around the KO claim for review.
            s = max(0, claim_pos - 110)
            out.append({"type": vtype, "target": target, "detail": detail,
                        "evidence": cot[s:claim_pos + 110].strip()})
    return out


# ---------------------------------------------------------------------------
# CLI — scan a synthesized SFT JSONL and report the CoT error rate.
# ---------------------------------------------------------------------------


@click.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--examples", "n_examples", type=int, default=12, show_default=True,
              help="Max flagged CoTs to print.")
def cli(input_path: Path, n_examples: int) -> None:
    """Scan a synthesized SFT JSONL for CoTs with objective KO-arithmetic errors."""
    from teacher import extract_pre_tool_thought

    rows = flagged = shown = 0
    by_type: dict[str, int] = {}
    with input_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows += 1
            msgs = rec.get("messages") or []
            cot = extract_pre_tool_thought(msgs)
            user_prompt = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
            findings = find_violations(cot, extract_on_field_hps(user_prompt))
            if not findings:
                continue
            flagged += 1
            for fnd in findings:
                by_type[fnd["type"]] = by_type.get(fnd["type"], 0) + 1
            if shown < n_examples:
                shown += 1
                loc = f"{rec.get('match_id','?')} g{rec.get('game_index','?')} t{rec.get('turn','?')}"
                click.echo(f"\n✗ {loc}")
                for fnd in findings:
                    click.echo(f"    [{fnd['type']}] {fnd['detail']}")
                    click.echo(f"      “…{fnd['evidence']}…”")

    click.echo(f"\n=== {input_path.name}: {flagged}/{rows} CoTs flagged "
               f"({100*flagged/rows:.1f}%) ===" if rows else "no rows")
    for t, c in sorted(by_type.items(), key=lambda kv: -kv[1]):
        click.echo(f"  {t}: {c}")
    click.echo(
        "\nDeterministic layer: high precision, partial recall (target-aware KO "
        "arithmetic only). Speed/survival/matrix-misstatement errors need the "
        "LLM-judge layer (follow-up)."
    )


if __name__ == "__main__":
    cli()
