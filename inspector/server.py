"""FastAPI server for the local SFT inspector.

Listens on port 8001 (so as not to collide with calc_microservice on 3000).
Reads `pipeline/parsed_data/` and serves a single-page HTML/JS frontend.

Strictly read-only: this server never writes back into the pipeline
directory. The pipeline directory is the source of truth; this is just a
viewer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import data_loader as dl
import prompt_parser as pp


HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE / "static"


app = FastAPI(title="pkmn SFT inspector", version="0.1")


# =============================================================================
# Pages
# =============================================================================


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


# Static assets (app.js, styles.css). Mounted under /static/.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# =============================================================================
# API: file index
# =============================================================================


@app.get("/api/files")
def api_files() -> dict[str, Any]:
    """Return all known JSONL files under parsed_data/, grouped by bucket."""
    files = dl.list_files()
    return {
        "current": [f.to_json() for f in files if f.bucket == "current"],
        "legacy": [f.to_json() for f in files if f.bucket == "legacy"],
    }


# =============================================================================
# API: file rows + single row
# =============================================================================


@app.get("/api/file/{path:path}/rows")
def api_file_rows(path: str) -> JSONResponse:
    """Lightweight row-list metadata for a file's sidebar."""
    try:
        rows = dl.list_rows_meta(path)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    return JSONResponse({"path": path, "rows": rows})


@app.get("/api/file/{path:path}/row/{idx}")
def api_file_row(path: str, idx: int) -> JSONResponse:
    """One row, fully parsed and annotated."""
    try:
        row = dl.read_row(path, idx)
    except (FileNotFoundError, IndexError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    if not isinstance(row, dict):
        raise HTTPException(status_code=400, detail="row is not a JSON object")

    # Branch by row kind.
    if {"match_id", "game_index", "turn", "messages"} <= set(row.keys()):
        return JSONResponse(_render_sft_row(path, idx, row))
    if {"match_id", "players", "format", "games"} <= set(row.keys()):
        return JSONResponse(_render_parsed_match_row(path, idx, row))
    return JSONResponse({
        "kind": "unknown",
        "path": path,
        "idx": idx,
        "raw": row,
    })


def _render_sft_row(path: str, idx: int, row: dict[str, Any]) -> dict[str, Any]:
    messages = row.get("messages") or []
    sys_msg = next((m for m in messages if m.get("role") == "system"), {})
    usr_msg = next((m for m in messages if m.get("role") == "user"), {})

    sys_parsed = pp.split_system_prompt(sys_msg.get("content") or "")
    usr_parsed = pp.split_user_prompt(usr_msg.get("content") or "")
    tool_loop = pp.parse_tool_loop(messages)

    # Try to resolve the source parsed match for "show source" links.
    source = dl.get_source_snapshot(
        row.get("match_id"),
        int(row.get("game_index", 0)),
        int(row.get("turn", 0)),
    )

    # We don't ship the entire source match in the response — just
    # enough to render a "go to source" jump button + a small preview.
    # Detail viewers can request the full snapshot via /api/file/{file}/row/{idx}.
    source_summary = None
    if source is not None:
        snap = source["snapshot"]
        source_summary = {
            "file": source["file"],
            "match_idx": source["match_idx"],
            "match_id": source["match_id"],
            "match_format": source["match_format"],
            "game_index": source["game_index"],
            "turn": snap.get("turn"),
            "events_count": len(snap.get("events") or []),
            "post_winner": source["post_winner"],
            "team_sheets_present": source["team_sheets"] is not None,
            "snapshot": snap,
            "snapshot_post": source["snapshot_post"],
        }

    return {
        "kind": "sft",
        "path": path,
        "idx": idx,
        "match_id": row.get("match_id"),
        "game_index": row.get("game_index"),
        "turn": row.get("turn"),
        "format_id": row.get("format_id"),
        "system": {
            "raw": sys_msg.get("content"),
            "parsed": sys_parsed,
        },
        "user": {
            "raw": usr_msg.get("content"),
            "parsed": usr_parsed,
        },
        "tool_loop": tool_loop,
        "raw_messages": messages,
        "source": source_summary,
    }


def _render_parsed_match_row(path: str, idx: int, row: dict[str, Any]) -> dict[str, Any]:
    games = row.get("games") or []
    game_summaries = []
    for gi, g in enumerate(games):
        snaps = g.get("snapshots") or []
        game_summaries.append({
            "game_index": gi,
            "replay_id": g.get("replay_id"),
            "winner": g.get("winner"),
            "turn_count": len(snaps),
            "team_sheets_present": g.get("teamSheets") is not None,
        })
    return {
        "kind": "parsed_match",
        "path": path,
        "idx": idx,
        "match_id": row.get("match_id"),
        "format": row.get("format"),
        "players": row.get("players"),
        "games": games,                        # full payload (snapshots + events)
        "game_summaries": game_summaries,
    }


# =============================================================================
# API: server health
# =============================================================================


@app.get("/api/health")
def api_health() -> dict[str, str]:
    return {"status": "ok"}
