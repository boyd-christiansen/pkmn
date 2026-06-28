"""Distill champion VGC commentary + analysis → a tight teacher context doc.

Goal (per the project owner):
    Compress expert VGC sources into a SHORT, actionable strategy document that
    teaches the CoT teacher to JUSTIFY decisions like a champion. THREE layers
    are extracted:

      1. **Decision framework** — the ordered questions a champion works through
         each turn (win condition → threats → worst case → speed/positioning →
         commit). A reasoning SCAFFOLD.
      2. **Durable principles** — format-agnostic heuristics (positioning,
         speed control, tempo, sacrifice timing, information denial).
      3. **Strategic roles & archetypes** — the *synthesized* understanding of
         WHY roles/Pokémon-types matter (Intimidate pivots, redirection,
         Trick Room enablers, speed-control modes, win-condition types).
         Captured at the durable level, not as a pick list.

Sources (two, very different):
    • Pokémon Company live-event commentary (transcripts_txt/*VGC*.txt) — play-
      by-play; we keep the "what are they thinking" reads, drop match-specifics.
    • WolfeyVGC video transcripts (wolfey_transcripts/*.json) — a large, MIXED
      channel (~half is Nuzlockes / reactions / non-VGC). An LLM RELEVANCE GATE
      keeps only competitive-VGC strategy OR strategic analysis.

Format-staleness guard (important):
    The Wolfey corpus spans years and formats (Dynamax, old regs). Per-Pokémon
    tier claims and obsolete mechanics do NOT transfer to Reg I 2026. Every
    prompt is told to extract the DURABLE, mechanism-grounded "why" and to drop
    dated tier lists / obsolete-format specifics (Dynamax, Z-moves, etc.).

Pipeline: load → relevance-gate (flash) → MAP 3 layers (pro, concurrent) →
hierarchical REDUCE (pro) → tight vgc_context.md for HUMAN REVIEW.

Cost: small (a few $ on credits). NOT the large per-row synthesis job.

Isolation contract:
    Reads teacher/transcripts/*, calls Gemini via google-genai (Vertex when
    GOOGLE_GENAI_USE_VERTEXAI=true), writes one markdown file. No pipeline imports.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import click
from google import genai
from google.genai import types as genai_types

TEACHER_DIR = Path(__file__).resolve().parent / "teacher"
TRANSCRIPTS = TEACHER_DIR / "transcripts"
DEFAULT_OUTPUT = TEACHER_DIR / "vgc_context.md"

# Extraction (map) is a fast-tier job — pull candidate steps/principles/roles
# out of a chunk; flash is plenty and ~10× faster than pro. Synthesis (reduce)
# is where judgment matters, so that stays on pro.
MAP_MODEL = os.environ.get("MAP_MODEL_GOOGLE", "gemini-3.5-flash")
REDUCE_MODEL = os.environ.get("TEACHER_MODEL_GOOGLE", "gemini-3.1-pro-preview")
GATE_MODEL = os.environ.get("GATE_MODEL_GOOGLE", "gemini-3.5-flash")  # cheap; global-endpoint OK

CHUNK_CHARS = 30_000
MAP_CONCURRENCY = 16
GATE_CONCURRENCY = 24
# Per-call timeouts so one slow/hung call can't stall the whole asyncio.gather
# (the bug that made the first run sit at >60 min).
MAP_TIMEOUT = 90.0
GATE_TIMEOUT = 45.0
GATE_SNIPPET_CHARS = 2400      # title + this much transcript is plenty to classify
REDUCE_BATCH = 40             # chunk-extracts per intermediate reduce


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_commentary(d: Path) -> list[dict[str, Any]]:
    """Pokémon Company commentary — VGC files only (TCG/GO excluded by name)."""
    out = []
    sub = d / "transcripts_txt"
    if not sub.is_dir():
        return out
    for p in sorted(sub.glob("*.txt")):
        if "VGC" not in p.name:
            continue
        # Strip "[HH:MM:SS] Speaker N:" prefixes to flowing text.
        import re
        text = re.sub(r"\[\d{2}:\d{2}:\d{2}\]\s*Speaker\s*\d+:\s*", " ", p.read_text(errors="ignore"))
        out.append({"id": p.stem, "title": p.stem, "text": " ".join(text.split()), "source": "pokemon-company"})
    return out


def load_wolfey(d: Path) -> list[dict[str, Any]]:
    out = []
    sub = d / "wolfey_transcripts"
    if not sub.is_dir():
        return out
    for p in sorted(sub.glob("*.json")):
        try:
            j = json.loads(p.read_text(errors="ignore"))
        except json.JSONDecodeError:
            continue
        text = (j.get("transcript_text") or "").strip()
        if text:
            out.append({"id": j.get("video_id", p.stem), "title": j.get("title", ""),
                        "text": text, "source": "wolfey"})
    return out


def _chunks(text: str, size: int = CHUNK_CHARS) -> list[str]:
    if len(text) <= size:
        return [text]
    parts, i = [], 0
    while i < len(text):
        end = min(i + size, len(text))
        if end < len(text):
            dot = text.rfind(". ", i + size // 2, end)
            if dot != -1:
                end = dot + 1
        parts.append(text[i:end])
        i = end
    return parts


# ---------------------------------------------------------------------------
# Relevance gate (flash) — keep VGC strategy OR strategic analysis
# ---------------------------------------------------------------------------

_GATE_SYS = """You decide if a video is useful for distilling COMPETITIVE \
Pokémon VGC (doubles) strategic knowledge.

INCLUDE (relevant=true) if it contains any transferable competitive substance:
  • turn-by-turn decision-making / match analysis / tournament games,
  • team building, EV/spread reasoning, damage/speed benchmarks,
  • strategic principles (positioning, speed control, tempo, sacrificing),
  • the strategic ROLE or VALUE of Pokémon/archetypes — e.g. "top 10", tier
    talk, "why X is good". We want the SYNTHESIZED reasoning (why a role
    matters), even if the specific Pokémon is format-dated.

EXCLUDE (relevant=false) pure entertainment with no transferable VGC strategy:
  Nuzlockes, randomizers, ROM hacks, non-VGC games (Snap, Legends Arceus, etc.),
  franchise/character rankings, reactions to unrelated content, IRL vlogs,
  unboxings, music. When unsure, lean INCLUDE — the downstream extractor drops
  fluff per chunk."""

_GATE_SCHEMA = {"type": "object", "properties": {
    "relevant": {"type": "boolean"},
    "reason": {"type": "string", "description": "≤12 words"},
}, "required": ["relevant", "reason"]}


async def _gate_one(client: genai.Client, item: dict, sem: asyncio.Semaphore) -> bool:
    async with sem:
        prompt = f"TITLE: {item['title']}\n\nTRANSCRIPT START:\n{item['text'][:GATE_SNIPPET_CHARS]}"
        try:
            r = await asyncio.wait_for(client.aio.models.generate_content(
                model=GATE_MODEL,
                contents=[genai_types.Content(role="user", parts=[genai_types.Part(text=prompt)])],
                config=genai_types.GenerateContentConfig(
                    system_instruction=_GATE_SYS, response_mime_type="application/json",
                    response_schema=_GATE_SCHEMA, temperature=0.0),
            ), timeout=GATE_TIMEOUT)
            return bool(json.loads(r.text or "{}").get("relevant", True))
        except Exception:  # noqa: BLE001 — fail-open (keep) on gate error/timeout
            return True


# ---------------------------------------------------------------------------
# MAP — 3-layer extraction
# ---------------------------------------------------------------------------

_MAP_SCHEMA = {"type": "object", "properties": {
    "decision_steps": {"type": "array", "items": {"type": "object", "properties": {
        "step": {"type": "string"}, "questions": {"type": "string"}, "why": {"type": "string"},
    }, "required": ["step", "questions", "why"]}},
    "principles": {"type": "array", "items": {"type": "object", "properties": {
        "principle": {"type": "string"}, "why": {"type": "string"}, "applies_when": {"type": "string"},
    }, "required": ["principle", "why", "applies_when"]}},
    "roles": {"type": "array", "items": {"type": "object", "properties": {
        "role": {"type": "string", "description": "e.g. 'Intimidate pivot', 'redirector', 'Trick Room setter'"},
        "why_it_matters": {"type": "string"}, "durable": {"type": "boolean", "description": "true if format-agnostic (NOT Dynamax/old-gen-specific)"},
    }, "required": ["role", "why_it_matters", "durable"]}},
}, "required": ["decision_steps", "principles", "roles"]}

_MAP_SYS = """Extract reusable VGC (Gen 9 doubles) strategic knowledge from this \
transcript chunk, in THREE layers:

1. decision_steps — the ordered questions a champion asks on a turn (win
   condition, threats, worst-case / what loses me the game, speed & positioning,
   then commit). The THOUGHT PROCESS, not match facts.
2. principles — durable, FORMAT-AGNOSTIC heuristics (positioning, speed control,
   tempo, sacrificing, information denial). The WHY matters.
3. roles — the strategic ROLE/value of an archetype and WHY it matters
   (Intimidate pivots, redirection, Trick Room enablers, speed-control modes,
   win-condition types, disruption). Set durable=false if it depends on an
   obsolete mechanic (Dynamax, Z-moves) or a specific dated metagame.

CRITICAL: capture the timeless "WHY", not the dated "WHAT". Ignore: specific
team lists, specific games/players, banter, sponsor reads, and any claim a
damage calculator already provides. Empty arrays if nothing reusable."""


async def _map_chunk(client: genai.Client, chunk: str, sem: asyncio.Semaphore) -> dict[str, Any]:
    async with sem:
        try:
            r = await asyncio.wait_for(client.aio.models.generate_content(
                model=MAP_MODEL,
                contents=[genai_types.Content(role="user", parts=[genai_types.Part(text=chunk)])],
                config=genai_types.GenerateContentConfig(
                    system_instruction=_MAP_SYS, response_mime_type="application/json",
                    response_schema=_MAP_SCHEMA, temperature=0.2),
            ), timeout=MAP_TIMEOUT)
            return json.loads(r.text or "{}")
        except Exception as e:  # noqa: BLE001 — fail-open (drop chunk) on error/timeout
            click.echo(f"  [map] chunk failed: {type(e).__name__}", err=True)
            return {"decision_steps": [], "principles": [], "roles": []}


# ---------------------------------------------------------------------------
# REDUCE — hierarchical (batch → intermediate → final)
# ---------------------------------------------------------------------------

_REDUCE_INTERMEDIATE_SYS = """Merge and DEDUPLICATE these candidate VGC strategy \
extractions into a compact intermediate set. Keep the strongest, most general \
decision-steps, principles, and roles; drop duplicates, match-specifics, and \
anything tied to an obsolete mechanic (Dynamax/Z-moves) or a dated metagame. \
Return the same JSON shape (decision_steps, principles, roles)."""

_REDUCE_FINAL_SYS = """You are the editor producing the FINAL distilled strategy \
doc, injected into an AI's system prompt to teach it to reason like a champion \
VGC (Gen 9 doubles, Tera era) player.

Produce TIGHT markdown (target 500–900 words MAX — it rides on every training \
row) with exactly three sections:

## How a champion reasons each turn
One canonical, ordered decision sequence (~4–7 steps). Each: the question(s) to \
ask + a one-line why. A way of THINKING to internalize, NOT a checklist to recite.

## Durable principles
~6–12 strongest, most generalizable principles. One line each: principle + why.

## Strategic roles & archetypes
~6–12 archetype roles and WHY each matters (Intimidate pivot, redirector, Trick \
Room setter/attacker, speed control, win-condition, disruption). Mechanism-level \
and FORMAT-AGNOSTIC.

Hard rules: ruthless dedup; Gen-9 Tera era only — DROP anything Dynamax/Z-move/ \
old-gen-specific and any dated tier list or specific-pick claim; drop anything a \
damage calculator already gives. Crisp imperative phrasing. Output ONLY the \
markdown."""


def _flatten(maps: list[dict]) -> dict[str, list]:
    return {k: [x for m in maps for x in (m.get(k) or [])] for k in ("decision_steps", "principles", "roles")}


async def _reduce_batch(client: genai.Client, sys: str, payload: dict, schema=_MAP_SCHEMA) -> dict:
    r = await client.aio.models.generate_content(
        model=REDUCE_MODEL,
        contents=[genai_types.Content(role="user", parts=[genai_types.Part(text=json.dumps(payload))])],
        config=genai_types.GenerateContentConfig(system_instruction=sys,
                                                 response_mime_type="application/json",
                                                 response_schema=schema, temperature=0.3),
    )
    return json.loads(r.text or "{}")


async def _reduce(client: genai.Client, maps: list[dict]) -> str:
    # Stage 1: batch-reduce (CONCURRENT, timeout-guarded) to bound each call's input.
    batches = [maps[i:i + REDUCE_BATCH] for i in range(0, len(maps), REDUCE_BATCH)]
    sem = asyncio.Semaphore(8)

    async def _one(b: list[dict]) -> dict:
        async with sem:
            try:
                return await asyncio.wait_for(
                    _reduce_batch(client, _REDUCE_INTERMEDIATE_SYS, _flatten(b)), timeout=120)
            except Exception:  # noqa: BLE001 — drop a batch rather than stall
                return {"decision_steps": [], "principles": [], "roles": []}

    intermediates = await asyncio.gather(*(_one(b) for b in batches))
    merged = _flatten(list(intermediates))
    click.echo(f"  reduced to {len(merged['decision_steps'])} steps, "
               f"{len(merged['principles'])} principles, {len(merged['roles'])} roles → final pass")
    # Stage 2: final markdown (single critical call — generous timeout).
    r = await asyncio.wait_for(client.aio.models.generate_content(
        model=REDUCE_MODEL,
        contents=[genai_types.Content(role="user", parts=[genai_types.Part(text=json.dumps(merged))])],
        config=genai_types.GenerateContentConfig(system_instruction=_REDUCE_FINAL_SYS, temperature=0.3),
    ), timeout=180)
    return (r.text or "").strip()


@click.command()
@click.option("--input-dir", type=click.Path(file_okay=False, path_type=Path), default=str(TRANSCRIPTS), show_default=True)
@click.option("--output", "output_path", type=click.Path(dir_okay=False, path_type=Path), default=str(DEFAULT_OUTPUT), show_default=True)
@click.option("--gate/--no-gate", default=True, show_default=True, help="LLM relevance gate on the Wolfey corpus.")
@click.option("--gate-only", is_flag=True, help="Run only the relevance gate + report kept/dropped (no map/reduce).")
@click.option("--max-videos", type=int, default=None, help="Cap Wolfey videos (after gate) — for a cheap trial run.")
def cli(input_dir: Path, output_path: Path, gate: bool, gate_only: bool, max_videos: int | None) -> None:
    """Distill the transcripts into a tight teacher/vgc_context.md."""
    commentary = load_commentary(input_dir)
    wolfey = load_wolfey(input_dir)
    click.echo(f"loaded: {len(commentary)} VGC commentary files, {len(wolfey)} Wolfey videos "
               f"| gate_model={GATE_MODEL} map_model={MAP_MODEL}")

    async def run() -> str | None:
        client = genai.Client()
        kept_wolfey = wolfey
        if gate and wolfey:
            sem = asyncio.Semaphore(GATE_CONCURRENCY)
            flags = await asyncio.gather(*(_gate_one(client, v, sem) for v in wolfey))
            kept_wolfey = [v for v, ok in zip(wolfey, flags) if ok]
            click.echo(f"  relevance gate: kept {len(kept_wolfey)}/{len(wolfey)} Wolfey videos")
        if max_videos:
            kept_wolfey = kept_wolfey[:max_videos]
            click.echo(f"  --max-videos: trimmed to {len(kept_wolfey)} Wolfey videos")
        if gate_only:
            return None

        items = commentary + kept_wolfey
        all_chunks = [c for it in items for c in _chunks(it["text"])]
        click.echo(f"  mapping {len(all_chunks)} chunks (~{sum(len(c) for c in all_chunks)//4:,} tokens)…")
        sem = asyncio.Semaphore(MAP_CONCURRENCY)
        maps = await asyncio.gather(*(_map_chunk(client, c, sem) for c in all_chunks))
        f = _flatten(list(maps))
        click.echo(f"  mapped: {len(f['decision_steps'])} steps, {len(f['principles'])} principles, "
                   f"{len(f['roles'])} roles → reducing…")
        return await _reduce(client, list(maps))

    doc = asyncio.run(run())
    if doc is None:
        click.echo("gate-only: done (no doc written).")
        return
    output_path.write_text(doc + "\n")
    click.echo(f"\nwrote {output_path}  ({len(doc)} chars ≈ {len(doc)//4} tokens)")
    click.echo("REVIEW it before wiring into the system prompt — it rides on every training row.")


if __name__ == "__main__":
    cli()
