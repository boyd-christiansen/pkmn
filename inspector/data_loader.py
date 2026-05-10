"""Read-only loaders for everything under `pipeline/parsed_data/`.

Layout assumed:
    repo_root/
    ├── pipeline/parsed_data/
    │   ├── bo1.jsonl              # parsed match records (current schema)
    │   ├── bo3.jsonl              # parsed match records (current schema)
    │   ├── sft_training_data.jsonl  (or any other "current" SFT cut)
    │   └── legacy/
    │       └── *.jsonl            # legacy SFT cuts (older prompt schema)

Categorization rule:
    - Files whose schema looks like `{match_id, players, format, games[]}`
      are PARSED MATCH files.
    - Files whose schema looks like `{match_id, game_index, turn,
      messages[]}` are SFT files.
    - Files in the `legacy/` subdir are tagged `legacy`; others are
      `current`.

Cross-reference:
    Given an SFT row's `(match_id, game_index, turn)`, the loader can
    locate the corresponding parsed-match snapshot in `bo1.jsonl` or
    `bo3.jsonl` so the UI can render "show source" links from prompt
    sections back to the underlying snapshot/events data.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
PARSED_DATA_DIR = REPO_ROOT / "pipeline" / "parsed_data"
LEGACY_DIR = PARSED_DATA_DIR / "legacy"


# =============================================================================
# File index
# =============================================================================


@dataclass
class FileEntry:
    """Metadata about one JSONL file under parsed_data/."""
    path: str               # path relative to PARSED_DATA_DIR (e.g. "bo3.jsonl" or "legacy/sft_bo3.jsonl")
    abs_path: str           # absolute filesystem path
    kind: str               # "parsed_match" | "sft" | "unknown"
    bucket: str             # "current" | "legacy"
    rows: int               # number of JSONL rows
    size_bytes: int

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "bucket": self.bucket,
            "rows": self.rows,
            "size_bytes": self.size_bytes,
        }


def _classify_first_row(row: dict[str, Any]) -> str:
    if not isinstance(row, dict):
        return "unknown"
    keys = set(row.keys())
    if {"match_id", "players", "format", "games"} <= keys:
        return "parsed_match"
    if {"match_id", "game_index", "turn", "messages"} <= keys:
        return "sft"
    return "unknown"


def _peek_file(path: Path) -> tuple[str, int]:
    """(kind, row_count) for a JSONL file. Returns ('unknown', 0) on error."""
    kind = "unknown"
    count = 0
    try:
        with path.open() as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                count += 1
                if i == 0:
                    try:
                        kind = _classify_first_row(json.loads(line))
                    except json.JSONDecodeError:
                        kind = "unknown"
    except (OSError, IOError):
        pass
    return kind, count


def list_files() -> list[FileEntry]:
    """Index every *.jsonl under parsed_data/ (including legacy/)."""
    out: list[FileEntry] = []
    if not PARSED_DATA_DIR.exists():
        return out
    for p in sorted(PARSED_DATA_DIR.rglob("*.jsonl")):
        rel = p.relative_to(PARSED_DATA_DIR).as_posix()
        bucket = "legacy" if rel.startswith("legacy/") else "current"
        kind, rows = _peek_file(p)
        try:
            size = p.stat().st_size
        except OSError:
            size = 0
        out.append(FileEntry(
            path=rel,
            abs_path=str(p),
            kind=kind,
            bucket=bucket,
            rows=rows,
            size_bytes=size,
        ))
    return out


def _resolve_file(rel_path: str) -> Path:
    """Resolve a relative file path under PARSED_DATA_DIR.

    Defends against `..` traversal — only paths that resolve under
    PARSED_DATA_DIR are allowed.
    """
    p = (PARSED_DATA_DIR / rel_path).resolve()
    if not str(p).startswith(str(PARSED_DATA_DIR.resolve())):
        raise ValueError(f"path escapes parsed_data/: {rel_path}")
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(rel_path)
    return p


# =============================================================================
# JSONL row readers
# =============================================================================


def iter_rows(rel_path: str) -> Iterator[dict[str, Any]]:
    p = _resolve_file(rel_path)
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def read_row(rel_path: str, idx: int) -> dict[str, Any]:
    for i, row in enumerate(iter_rows(rel_path)):
        if i == idx:
            return row
    raise IndexError(f"row {idx} not found in {rel_path}")


def list_rows_meta(rel_path: str) -> list[dict[str, Any]]:
    """Lightweight per-row metadata for sidebar listing.

    For SFT files: returns `[{match_id, game_index, turn, format_id}]`.
    For parsed-match files: returns `[{match_id, format, game_count, turn_count, players}]`.
    """
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(iter_rows(rel_path)):
        if not isinstance(row, dict):
            out.append({"idx": idx, "kind": "unknown"})
            continue
        keys = set(row.keys())
        if {"match_id", "game_index", "turn", "messages"} <= keys:
            out.append({
                "idx": idx,
                "kind": "sft",
                "match_id": row.get("match_id"),
                "game_index": row.get("game_index"),
                "turn": row.get("turn"),
                "format_id": row.get("format_id"),
            })
        elif {"match_id", "players", "format", "games"} <= keys:
            games = row.get("games") or []
            turn_count = sum(len(g.get("snapshots") or []) for g in games)
            out.append({
                "idx": idx,
                "kind": "parsed_match",
                "match_id": row.get("match_id"),
                "format": row.get("format"),
                "players": row.get("players"),
                "game_count": len(games),
                "turn_count": turn_count,
            })
        else:
            out.append({"idx": idx, "kind": "unknown"})
    return out


# =============================================================================
# Source-tracing: SFT row → parsed match snapshot
# =============================================================================


@lru_cache(maxsize=8)
def _index_parsed_match_file(rel_path: str) -> dict[str, int]:
    """Build a `match_id → row_idx` index for one parsed-match file."""
    out: dict[str, int] = {}
    for idx, row in enumerate(iter_rows(rel_path)):
        mid = row.get("match_id")
        if mid:
            out[mid] = idx
    return out


def find_source_match(match_id: str) -> tuple[str, int] | None:
    """Locate a parsed-match record by `match_id` across the standard files.

    Returns `(file_path, row_idx)` or None if not found. Searches `bo3.jsonl`
    then `bo1.jsonl` — the order matters because Bo3 match_ids start with
    `bo3-` and Bo1 ids look different.
    """
    for candidate in ("bo3.jsonl", "bo1.jsonl"):
        try:
            idx = _index_parsed_match_file(candidate).get(match_id)
        except FileNotFoundError:
            continue
        if idx is not None:
            return (candidate, idx)
    return None


def get_source_snapshot(match_id: str, game_index: int, turn: int) -> dict[str, Any] | None:
    """Resolve the parsed-match snapshot underlying an SFT row.

    Returns a dict like:
      {
        "file": "bo3.jsonl",
        "match_idx": 2,
        "match_id": "bo3-...",
        "game_index": 0,
        "snapshot": {turn, field, p1, p2, events},
        "snapshot_post": {...} or None,   # the snapshot that immediately follows
        "post_winner": "p1" | "p2" | None,
        "team_sheets": {p1: [...], p2: [...]} or None,
        "match_format": "bo1" | "bo3",
      }
    Returns None if the source can't be found (e.g. the parsed match isn't
    in the local sample, or game/turn indices are out of range).

    Note: SFT rows are emitted from the FLIPPED match (winner-as-P1) but the
    parsed-match files store the UNFLIPPED protocol-original. So `match_id`
    matches but the `p1`/`p2` sides may be opposite. Callers should use the
    `post_winner` field to decide if they want to flip when rendering.
    """
    src = find_source_match(match_id)
    if src is None:
        return None
    file, match_idx = src
    match = read_row(file, match_idx)
    games = match.get("games") or []
    if game_index < 0 or game_index >= len(games):
        return None
    game = games[game_index]
    snaps = game.get("snapshots") or []
    snap = next((s for s in snaps if s.get("turn") == turn), None)
    if snap is None:
        return None
    snap_idx = snaps.index(snap)
    snap_post = snaps[snap_idx + 1] if snap_idx + 1 < len(snaps) else None
    return {
        "file": file,
        "match_idx": match_idx,
        "match_id": match_id,
        "game_index": game_index,
        "snapshot": snap,
        "snapshot_post": snap_post,
        "post_winner": game.get("winner"),
        "team_sheets": game.get("teamSheets"),
        "match_format": match.get("format"),
    }
