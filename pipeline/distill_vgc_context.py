"""Distill champion-commentary transcripts → a tight teacher VGC context doc.

Goal (per the project owner):
    Compress raw transcripts of expert VGC commentary (Pokémon Company casts,
    WolfeyVGC, etc.) into a SHORT, actionable strategic document that teaches
    the CoT teacher to JUSTIFY decisions like a champion — not to recite
    facts. Two layers are extracted:

      1. **Decision framework** — the ordered sequence of questions a champion
         works through each turn (e.g. identify win condition → assess threats
         → check worst case → commit). This becomes a reasoning SCAFFOLD.
      2. **Durable principles** — format-agnostic heuristics (positioning,
         speed control, tempo, sacrifice timing, information denial). The
         WHY matters more than the WHAT.

Pipeline (map-reduce so it scales past one context window):
    • load + clean transcripts (strip VTT/SRT timestamps + caption-roll dups),
    • MAP: chunk each transcript, extract candidate framework-steps + principles
      from each chunk as structured JSON (concurrent),
    • REDUCE: synthesize ALL candidates into one tight, deduped `vgc_context.md`.

Cost: small — a handful of Gemini calls proportional to transcript length;
effectively $0 on the GCP credits. This is NOT the large synthesis job.

Output is for HUMAN REVIEW before it's wired into the system prompt — a
curated doc rides on every one of ~93K training rows, so a wrong/generic
principle would propagate everywhere.

Isolation contract:
    Reads `teacher/transcripts/*`, calls Gemini via the google-genai SDK
    (Vertex when GOOGLE_GENAI_USE_VERTEXAI=true), writes one markdown file.
    No calc service, no pipeline-module imports.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

import click
from google import genai
from google.genai import types as genai_types

TEACHER_DIR = Path(__file__).resolve().parent / "teacher"
DEFAULT_INPUT_DIR = TEACHER_DIR / "transcripts"
DEFAULT_OUTPUT = TEACHER_DIR / "vgc_context.md"
DEFAULT_MODEL = os.environ.get("TEACHER_MODEL_GOOGLE", "gemini-3.1-pro-preview")

# ~30k chars ≈ 7.5k tokens per map chunk — small enough for fast, focused
# extraction, large enough to keep each chunk's strategic thread intact.
CHUNK_CHARS = 30_000
MAP_CONCURRENCY = 6

_VTT_TS = re.compile(r"^\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[.,]\d{3}.*$")
_SRT_IDX = re.compile(r"^\d+$")
_INLINE_TS = re.compile(r"<\d{2}:\d{2}:\d{2}[.,]\d{3}>")
_TAGS = re.compile(r"</?c[^>]*>")


def _clean_caption_text(text: str) -> str:
    """Strip VTT/SRT timestamps, cue indices, inline tags, and collapse the
    rolling-caption duplication YouTube auto-captions produce (each line
    repeats the previous line plus a few new words)."""
    lines: list[str] = []
    for raw in text.splitlines():
        ln = _TAGS.sub("", _INLINE_TS.sub("", raw)).strip()
        if not ln or ln in ("WEBVTT",) or _VTT_TS.match(ln) or _SRT_IDX.match(ln):
            continue
        if ln.startswith(("Kind:", "Language:", "NOTE ")):
            continue
        lines.append(ln)
    # Drop a line that is a prefix of the next (rolling caption), and
    # consecutive exact duplicates.
    out: list[str] = []
    for i, ln in enumerate(lines):
        if i + 1 < len(lines) and lines[i + 1].startswith(ln):
            continue
        if out and out[-1] == ln:
            continue
        out.append(ln)
    return " ".join(out)


def load_transcripts(input_dir: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for p in sorted(input_dir.iterdir()):
        if p.suffix.lower() not in (".txt", ".vtt", ".srt", ".md") or p.name == "README.txt":
            continue
        text = p.read_text(errors="ignore")
        if p.suffix.lower() in (".vtt", ".srt"):
            text = _clean_caption_text(text)
        if text.strip():
            out.append((p.name, text))
    return out


def _chunks(text: str, size: int = CHUNK_CHARS) -> list[str]:
    if len(text) <= size:
        return [text]
    parts, i = [], 0
    while i < len(text):
        end = min(i + size, len(text))
        if end < len(text):  # back up to a sentence boundary
            dot = text.rfind(". ", i + size // 2, end)
            if dot != -1:
                end = dot + 1
        parts.append(text[i:end])
        i = end
    return parts


# Structured output for the MAP step.
_MAP_SCHEMA = {
    "type": "object",
    "properties": {
        "decision_steps": {
            "type": "array",
            "description": "Ordered questions/checks a strong player works through on a turn.",
            "items": {"type": "object", "properties": {
                "step": {"type": "string", "description": "Short name, e.g. 'Identify the win condition'."},
                "questions": {"type": "string", "description": "The concrete question(s) they ask."},
                "why": {"type": "string"},
            }, "required": ["step", "questions", "why"]},
        },
        "principles": {
            "type": "array",
            "description": "Durable, format-agnostic strategic principles.",
            "items": {"type": "object", "properties": {
                "principle": {"type": "string"},
                "why": {"type": "string"},
                "applies_when": {"type": "string"},
            }, "required": ["principle", "why", "applies_when"]},
        },
    },
    "required": ["decision_steps", "principles"],
}

_MAP_SYS = """You are extracting reusable VGC (Gen 9 doubles) strategy from a \
chunk of expert commentary/coaching transcript. Pull out TWO things:

1. decision_steps — the ordered questions a champion asks themselves on a turn \
(win condition, threat assessment, worst-case / what-loses-me-the-game, speed \
& positioning, then commit). Capture the THOUGHT PROCESS, not match-specific facts.
2. principles — durable, FORMAT-AGNOSTIC heuristics (positioning, speed control, \
tempo, sacrificing, information denial, preserving win conditions). The WHY matters.

Ignore: specific team lists, specific games, banter, sponsor reads, anything \
tied to one match or one season's metagame. Only timeless decision-making. If a \
chunk has nothing reusable, return empty arrays. Be concise."""

_REDUCE_SYS = """You are the editor producing the FINAL distilled strategy doc \
that will be injected into an AI's system prompt to teach it to reason like a \
champion VGC (Gen 9 doubles) player. You are given many candidate decision-steps \
and principles extracted from expert commentary.

Produce a TIGHT markdown document (target 400–800 words MAX — it rides on every \
training example) with exactly two sections:

## How a champion reasons each turn
A single, canonical, ordered decision sequence (merge/dedupe the candidates into \
~4–7 steps). For each step: the question(s) to ask and a one-line why. This is a \
way of THINKING to internalize, not a checklist to recite.

## Durable principles
~6–12 of the strongest, most generalizable principles (merge/dedupe). One line \
each: the principle + its why.

Rules: ruthless deduplication; drop anything format-specific, anything tied to a \
single game, and any fact a damage calculator already provides. Prefer crisp, \
imperative phrasing. Output ONLY the markdown, no preamble."""


async def _map_chunk(client: genai.Client, model: str, chunk: str, sem: asyncio.Semaphore) -> dict[str, Any]:
    async with sem:
        try:
            resp = await client.aio.models.generate_content(
                model=model,
                contents=[genai_types.Content(role="user", parts=[genai_types.Part(text=chunk)])],
                config=genai_types.GenerateContentConfig(
                    system_instruction=_MAP_SYS,
                    response_mime_type="application/json",
                    response_schema=_MAP_SCHEMA,
                    temperature=0.2,
                ),
            )
            return json.loads(resp.text or "{}")
        except Exception as e:  # noqa: BLE001 — one bad chunk shouldn't sink the run
            click.echo(f"  [map] chunk failed: {type(e).__name__}: {e}", err=True)
            return {"decision_steps": [], "principles": []}


async def _reduce(client: genai.Client, model: str, mapped: list[dict[str, Any]]) -> str:
    steps = [s for m in mapped for s in (m.get("decision_steps") or [])]
    principles = [p for m in mapped for p in (m.get("principles") or [])]
    payload = json.dumps({"decision_steps": steps, "principles": principles}, indent=1)
    resp = await client.aio.models.generate_content(
        model=model,
        contents=[genai_types.Content(role="user", parts=[genai_types.Part(
            text=f"Candidate extractions ({len(steps)} steps, {len(principles)} principles):\n\n{payload}")])],
        config=genai_types.GenerateContentConfig(system_instruction=_REDUCE_SYS, temperature=0.3),
    )
    return (resp.text or "").strip()


@click.command()
@click.option("--input-dir", type=click.Path(file_okay=False, path_type=Path), default=str(DEFAULT_INPUT_DIR), show_default=True)
@click.option("--output", "output_path", type=click.Path(dir_okay=False, path_type=Path), default=str(DEFAULT_OUTPUT), show_default=True)
@click.option("--model", default=DEFAULT_MODEL, show_default=True)
def cli(input_dir: Path, output_path: Path, model: str) -> None:
    """Distill the transcripts in --input-dir into a tight vgc_context.md."""
    transcripts = load_transcripts(input_dir)
    if not transcripts:
        raise click.ClickException(
            f"no transcripts in {input_dir} (drop .txt/.vtt/.srt/.md files there)")
    all_chunks = [(name, c) for name, text in transcripts for c in _chunks(text)]
    total_chars = sum(len(c) for _, c in all_chunks)
    click.echo(f"loaded {len(transcripts)} transcript(s) → {len(all_chunks)} chunks "
               f"(~{total_chars//4:,} tokens) | model={model}")

    async def run() -> str:
        client = genai.Client()
        sem = asyncio.Semaphore(MAP_CONCURRENCY)
        mapped = await asyncio.gather(*(_map_chunk(client, model, c, sem) for _, c in all_chunks))
        ns = sum(len(m.get("decision_steps") or []) for m in mapped)
        npr = sum(len(m.get("principles") or []) for m in mapped)
        click.echo(f"  mapped: {ns} candidate steps, {npr} candidate principles → reducing…")
        return await _reduce(client, model, list(mapped))

    doc = asyncio.run(run())
    output_path.write_text(doc + "\n")
    click.echo(f"\nwrote {output_path}  ({len(doc)} chars ≈ {len(doc)//4} tokens)")
    click.echo("REVIEW it before wiring into the system prompt — it rides on every training row.")


if __name__ == "__main__":
    cli()
