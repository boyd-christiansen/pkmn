"""Pre-training corpus audit — a battery of structural + correctness tests.

Why this exists:
    Before committing real money to a synthesis run, we want to catch
    *systematic* data problems — the kind that are invisible one row at a time
    but poison training in aggregate (e.g. the threat matrix's frail-defender
    KO label that printed "guaranteed OHKO" on a sub-100% low roll ~70% of the
    time). This is the harness to find more of them.

    It runs a registry of independent checks over a JSONL (a `--dry-run`
    preview OR a real synthesized file) and reports, per check, how many rows
    trip it plus a few examples. Each check is cheap, deterministic, and
    additive — drop a new function in `CHECKS` and it runs.

Two tiers of check:
    • PROMPT/STRUCTURE — run on every row (the user prompt is real even in a
      dry-run): section presence, matrix damage sanity, the KO-label
      regression guard, action well-formedness, tool-call validity.
    • COT — run only on real synthesized rows (skipped when the CoT is the
      `[DRY RUN ...]` placeholder): KO arithmetic vs current HP, the
      matrix-grounded over-claim check, and oracle-leak.

This is deliberately the place where "find more issues like the KO label"
work accretes; treat a clean run here as a gate before the paid synthesis.

Isolation contract:
    Read-only over a JSONL. Reuses `verify_cot_correctness`, `teacher`
    (leak + CoT extraction). No calc service, no LLM.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import click

from verify_cot_correctness import find_violations, extract_on_field_hps, _species_key

REQUIRED_SECTIONS = (
    "=== TURN ", "YOUR (P1) ACTIVE:", "OPP (P2) ACTIVE:",
    "=== GAME-STATE LEDGER ===", "=== THREAT MATRIX",
)
_DRY = "[DRY RUN"

# Matrix line parsers (OUTGOING + INCOMING share the same line grammar).
_SINGLE = re.compile(r"^\s+(.+?)\s+→\s+(.+?)\s+(\d+(?:\.\d+)?)%[–-](\d+(?:\.\d+)?)%(?:\s+\[(.+?)\])?\s*$")
_SPREAD_HEAD = re.compile(r"^\s+(.+?)\s+\[spread\]:\s+(.+?)(?:\s+\[(.+?)\])?\s*$")
_SPREAD_PAIR = re.compile(r"([A-Za-z0-9'.\- ]+?)\s+(\d+(?:\.\d+)?)%[–-](\d+(?:\.\d+)?)%")
_KO_SINGLE = re.compile(r"^O?HKO$|^\d+HKO$")  # a collapsed single KO label


def _matrix_block(user_prompt: str) -> str:
    i = user_prompt.find("=== THREAT MATRIX")
    return user_prompt[i:] if i >= 0 else ""


def _outgoing_targets(user_prompt: str) -> dict[str, list[tuple[float, float]]]:
    """{target_key: [(lo%, hi%), ...]} for our OUTGOING moves. Used by the
    matrix-grounded CoT check to ask 'does any of our moves actually guarantee
    this KO at the target's current HP?'."""
    block = _matrix_block(user_prompt)
    if "--- OUTGOING" not in block:
        return {}
    out_block = block.split("--- OUTGOING", 1)[1].split("--- INCOMING", 1)[0]
    targets: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for line in out_block.splitlines():
        sm = _SINGLE.match(line)
        if sm:
            targets[_species_key(sm.group(2))].append((float(sm.group(3)), float(sm.group(4))))
            continue
        hm = _SPREAD_HEAD.match(line)
        if hm:
            for pm in _SPREAD_PAIR.finditer(hm.group(2)):
                targets[_species_key(pm.group(1))].append((float(pm.group(2)), float(pm.group(3))))
    return dict(targets)


def _all_matrix_lines(user_prompt: str) -> list[tuple[str, float, float, str]]:
    """[(target, lo, hi, ko_label), ...] across both directions, for sanity
    + regression checks."""
    out: list[tuple[str, float, float, str]] = []
    for line in _matrix_block(user_prompt).splitlines():
        sm = _SINGLE.match(line)
        if sm:
            out.append((sm.group(2).strip(), float(sm.group(3)), float(sm.group(4)), (sm.group(5) or "").strip()))
            continue
        hm = _SPREAD_HEAD.match(line)
        if hm:
            labels = {}
            for part in (hm.group(3) or "").split(";"):
                if ":" in part:
                    k, v = part.split(":", 1)
                    labels[_species_key(k)] = v.strip()
            for pm in _SPREAD_PAIR.finditer(hm.group(2)):
                tgt = pm.group(1).strip()
                out.append((tgt, float(pm.group(2)), float(pm.group(3)), labels.get(_species_key(tgt), "")))
    return out


# ---------------------------------------------------------------------------
# Checks — each returns a list of human-readable issue strings (empty = clean)
# ---------------------------------------------------------------------------


def chk_prompt_sections(ctx: dict[str, Any]) -> list[str]:
    up = ctx["user_prompt"]
    missing = [s for s in REQUIRED_SECTIONS if s not in up]
    return [f"missing section {s!r}" for s in missing]


def chk_template_leftovers(ctx: dict[str, Any]) -> list[str]:
    up = ctx["user_prompt"]
    out = []
    if re.search(r"\{[a-z_]+\}", up):       # unrendered f-string/format slot
        out.append("unrendered {placeholder} in prompt")
    if "None%" in up or "?%" in up:
        out.append("malformed percent token (None%/?%)")
    return out


def chk_matrix_damage_sanity(ctx: dict[str, Any]) -> list[str]:
    # Note: VGC damage % can legitimately exceed 300–400% (4×-effective +
    # boosts + items), so a high upper bound is NOT a bug. Only an inverted
    # range (lo > hi), a negative roll, or a clearly-garbage value (>1000%)
    # indicates a real problem.
    out = []
    for tgt, lo, hi, _ in _all_matrix_lines(ctx["user_prompt"]):
        if lo > hi + 1e-6:
            out.append(f"{tgt}: damage low {lo}% > high {hi}% (inverted)")
        # Only a negative roll or a clearly-garbage value is a bug; frail mons
        # (Smeargle etc.) legitimately take 1000s of % from a boosted 4× hit.
        if lo < 0 or hi > 10000:
            out.append(f"{tgt}: damage value out of sane bounds {lo}–{hi}%")
    return out


def chk_ko_label_regression(ctx: dict[str, Any]) -> list[str]:
    """Guard against the KO-label optimism we fixed: a *collapsed* `OHKO`
    label must not claim a guaranteed OHKO the rolls can't back up. Honest
    ranges (`OHKO–3HKO`) are fine; a bare `OHKO` is only legitimate when the
    low roll actually KOs the defender — but the calc (and the displayed %)
    are vs MAX HP, while a guaranteed OHKO is computed vs the defender's
    CURRENT HP. So we compare the low roll to the target's current HP (from
    the ACTIVE block), with a small rounding tolerance, NOT to 100% — that
    was over-flagging correct OHKOs on chipped defenders (a 78%-of-max hit
    guaranteeing a KO on a 1%-HP mon is correct)."""
    hps = ctx["on_field_hps"]
    out = []
    for tgt, lo, hi, label in _all_matrix_lines(ctx["user_prompt"]):
        if label != "OHKO":
            continue
        cur = hps.get(tgt)
        if cur is None:
            continue  # target not on field (rare) — can't HP-check
        if lo < cur - 5.0:  # 5% tolerance for calc integer rounding
            out.append(f"{tgt}: collapsed 'OHKO' but low roll {lo}% < current HP {cur:.0f}% "
                       f"(optimism regression)")
    return out


def chk_action_structure(ctx: dict[str, Any]) -> list[str]:
    act = ctx["action"]
    if act is None:
        return ["no submit_decision action found"]
    out = []
    for slot in ("slot_1", "slot_2"):
        a = act.get(slot)
        if not isinstance(a, dict):
            out.append(f"{slot}: missing/!dict"); continue
        t = a.get("action_type")
        if t not in ("move", "switch", "pass"):
            out.append(f"{slot}: bad action_type {t!r}")
        if t == "move" and not a.get("move"):
            out.append(f"{slot}: action_type=move but no move")
        if t == "switch" and not a.get("switch_to"):
            out.append(f"{slot}: action_type=switch but no switch_to")
    return out


def chk_tool_calls_wellformed(ctx: dict[str, Any]) -> list[str]:
    msgs = ctx["messages"]
    out = []
    submits = 0
    call_ids, result_ids = set(), set()
    for m in msgs:
        for tc in (m.get("tool_calls") or []):
            call_ids.add(tc.get("id"))
            if (tc.get("function") or {}).get("name") == "submit_decision":
                submits += 1
        if m.get("role") == "tool" and m.get("tool_call_id"):
            result_ids.add(m["tool_call_id"])
    if submits != 1:
        out.append(f"submit_decision appears {submits}× (expected 1)")
    if call_ids - result_ids:
        out.append(f"{len(call_ids - result_ids)} tool_call(s) without a matching tool result")
    return out


def chk_cot_ko_arithmetic(ctx: dict[str, Any]) -> list[str]:
    if ctx["is_dry"] or not ctx["cot"]:
        return []
    return [f["detail"] for f in find_violations(ctx["cot"], ctx["on_field_hps"])]


def chk_cot_matrix_grounded(ctx: dict[str, Any]) -> list[str]:
    """v2: a 'guaranteed KO' claim where NO outgoing move can actually
    guarantee it at the target's current HP (matrix lo < HP for every move).
    Grounds the damage numbers in the matrix, not the CoT's restatement."""
    if ctx["is_dry"] or not ctx["cot"]:
        return []
    cot, hps, outgoing = ctx["cot"], ctx["on_field_hps"], ctx["outgoing"]
    if not hps or not outgoing:
        return []
    from verify_cot_correctness import _species_positions, _nearest_species, _GUARANTEE, _KOWORD
    sp_pos = _species_positions(cot, list(hps.keys()))
    out, seen = [], set()
    for g in _GUARANTEE.finditer(cot):
        ko = next((k for k in _KOWORD.finditer(cot) if abs(k.start() - g.start()) <= 70), None)
        if not ko:
            continue
        target = _nearest_species((g.start() + ko.start()) // 2, sp_pos, max_dist=120)
        if not target:
            continue
        tkey = _species_key(target)
        h = hps.get(target)
        moves = outgoing.get(tkey)
        if h is None or not moves or tkey in seen:
            continue
        best_lo = max(lo for lo, _ in moves)   # strongest (highest-low-roll) move
        if best_lo < h:
            seen.add(tkey)
            out.append(f"claims a guaranteed KO on {target}, but no outgoing move's "
                       f"low roll reaches its {h:.0f}% HP (best low roll {best_lo:.1f}%)")
    return out


def chk_oracle_leak(ctx: dict[str, Any]) -> list[str]:
    if ctx["is_dry"] or not ctx["cot"]:
        return []
    from teacher import detect_oracle_leak
    leak = detect_oracle_leak(ctx["messages"])
    return [f"oracle-leak phrase: {leak!r}"] if leak else []


CHECKS: dict[str, Callable[[dict[str, Any]], list[str]]] = {
    "prompt_sections": chk_prompt_sections,
    "template_leftovers": chk_template_leftovers,
    "matrix_damage_sanity": chk_matrix_damage_sanity,
    "ko_label_regression": chk_ko_label_regression,
    "action_structure": chk_action_structure,
    "tool_calls_wellformed": chk_tool_calls_wellformed,
    "cot_ko_arithmetic": chk_cot_ko_arithmetic,
    "cot_matrix_grounded": chk_cot_matrix_grounded,
    "oracle_leak": chk_oracle_leak,
}


def _build_ctx(rec: dict[str, Any]) -> dict[str, Any]:
    from teacher import extract_pre_tool_thought
    msgs = rec.get("messages") or []
    user_prompt = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
    cot = extract_pre_tool_thought(msgs)
    action = None
    for m in msgs:
        for tc in (m.get("tool_calls") or []):
            if (tc.get("function") or {}).get("name") == "submit_decision":
                try:
                    action = json.loads(tc["function"]["arguments"]).get("action")
                except Exception:
                    action = None
    return {
        "rec": rec, "messages": msgs, "user_prompt": user_prompt, "cot": cot,
        "action": action, "is_dry": bool(cot and _DRY in cot),
        "on_field_hps": extract_on_field_hps(user_prompt),
        "outgoing": _outgoing_targets(user_prompt),
    }


@click.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--examples", "n_examples", type=int, default=3, show_default=True)
@click.option("--limit", type=int, default=None, help="Audit only the first N rows.")
def cli(input_path: Path, n_examples: int, limit: int | None) -> None:
    """Run the full audit battery over a synthesized or dry-run JSONL."""
    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    rows = dry_rows = 0
    with input_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                counts["_unparseable_json"] += 1
                continue
            rows += 1
            ctx = _build_ctx(rec)
            if ctx["is_dry"]:
                dry_rows += 1
            for name, fn in CHECKS.items():
                try:
                    issues = fn(ctx)
                except Exception as e:
                    issues = [f"CHECK ERROR: {type(e).__name__}: {e}"]
                if issues:
                    counts[name] += 1
                    if len(examples[name]) < n_examples:
                        loc = f"{rec.get('match_id','?')} g{rec.get('game_index','?')} t{rec.get('turn','?')}"
                        examples[name].append(f"{loc}: {issues[0]}")
            if limit and rows >= limit:
                break

    click.echo(f"\n=== audit: {input_path.name} ===")
    click.echo(f"rows={rows}  (dry-run placeholders={dry_rows}; CoT checks skip those)\n")
    cot_checks = {"cot_ko_arithmetic", "cot_matrix_grounded", "oracle_leak"}
    real = rows - dry_rows
    for name in CHECKS:
        c = counts.get(name, 0)
        denom = real if name in cot_checks else rows
        rate = f"{100*c/denom:.2f}%" if denom else "n/a"
        mark = "✓" if c == 0 else "✗"
        click.echo(f"  {mark} {name}: {c} ({rate})")
        for ex in examples.get(name, []):
            click.echo(f"        {ex}")
    extra = {k: v for k, v in counts.items() if k not in CHECKS}
    for k, v in extra.items():
        click.echo(f"  ✗ {k}: {v}")
    clean = all(counts.get(n, 0) == 0 for n in CHECKS)
    click.echo("\n" + ("✓ corpus clean on all checks" if clean else
                       "✗ issues found — review above before the paid synthesis run"))


if __name__ == "__main__":
    cli()
