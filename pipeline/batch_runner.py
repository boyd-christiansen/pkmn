"""Batch-mode runner for plan v4 — the OpenAI Batch API state machine.

Sibling of `master_pipeline.py` but only invoked when the user passes
`--mode batch` (or `--mode hybrid`, which fans out to this for the
non-sync portion). Reuses:
  • `damage_inferencer.{init_knowledge, infer_match_final_bounds,
     update_knowledge, events_to_damage_events}` for KnowledgeState prep.
  • `threat_matrix.generate_threat_matrix` for the per-turn matrix.
  • `prompt_formatting.{format_p1_known_spreads_block, format_user_prompt,
     format_p1_team_block}` for prompt assembly.
  • `teacher.batch_openai.BatchOpenAIProvider` for the SDK plumbing.
  • `teacher.{detect_oracle_leak, judge_match_cots, ...}` for filters.
  • `master_pipeline._run_judge_with_retries` (imported lazily inside
     `run_batch_for_matches` to avoid a circular import) for the
     post-batch judge pass. Judge re-synthesis falls back to the SYNC
     teacher — batch latency is too high to be useful for retries.

The state machine runs ONE batch cycle per tool-loop iteration. All
WorkItems at iter=K bundle into one batch submission; on completion,
the orchestrator advances each item independently (some may submit
their decision, others may need another calc round).

Shape of the state file per match (`{state_dir}/{match_id}.json`):

    {
      "match_id": "...",
      "format_id": "gen9vgc2026regibo3",
      "items": [BatchWorkItem JSON dicts — see BatchWorkItem.to_dict()],
      "active_batches": [],   # currently unused; reserved for v2 cross-cycle
                              # bookkeeping. Per-item `active_batch_id` is the
                              # actual resume breadcrumb.
      "last_updated": "2026-05-12T17:22:13Z"
    }

Resumability: on startup, `--resume` loads every state file; for each
item with `status="submitted"`, the `active_batch_id` breadcrumb points
at the batch the item was waiting on. `_resume_inflight_batches`
groups items by batch_id, re-polls each, fetches results, and applies
them BEFORE entering the normal cycle loop — so the cycle loop sees
items in known {pending, committed, failed} states only.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
import click

import damage_inferencer
import threat_matrix
from action_extraction import (
    extract_p1_actions,
    flip_match_to_winner,
    slot_action,
)
from prompt_formatting import (
    format_p1_known_spreads_block,
    format_p1_team_block,
    format_p2_inferred_spreads_block,
    format_user_prompt,
)
from team_reconstruction import (
    brought_species_keys_for_game,
    reconstruct_p1_team,
    reconstruct_p2_species,
    reconstruct_p2_team,
    team_sheets_for_match,
)
from teacher import (
    BatchOpenAIProvider,
    BatchPollStatus,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_RETRIES,
    MAX_CALC_CALLS_BEFORE_FORCE_SUBMIT,
    MAX_TOOL_ITERATIONS,
    PRODUCTION_LEAK_RETRIES,
    SYNTHESIS_GROUND_TRUTH_SUFFIX,
    TeacherProvider,
    _call_calc,
    detect_oracle_leak,
    extract_pre_tool_thought,
    judge_match_cots,
    render_system_prompt,
    render_system_prompt_bo3,
)


# ---------------------------------------------------------------------------
# Per-turn prep: shared with sync mode in master_pipeline.py
# ---------------------------------------------------------------------------


@dataclass
class TurnPrep:
    """Output of `_prepare_match_turns` for one (match, game, turn).

    Everything below is pure compute — no LLM call has happened yet. The
    sync runner calls `teacher.synthesize_turn` with these fields; the
    batch runner builds the first-cycle `api_messages` from them and
    pushes a WorkItem onto the cycle queue.

    The two prompts both exist because:
      • `user_prompt` is the SAVED user message (no ground-truth suffix).
      • `api_messages` already has the ground-truth suffix injected, so
        the first batch request can fire without further mutation.
    """
    match_id: str
    game_idx: int
    turn: int
    format_id: str
    system_prompt: str
    user_prompt: str
    human_action: dict[str, Any]
    api_messages: list[dict[str, Any]]  # system + user-with-suffix


async def _prepare_match_turns(
    match_record: dict[str, Any],
    *,
    format_id: str,
    calc_base_url: str,
    aiohttp_session: aiohttp.ClientSession,
    seen_keys: set[tuple[str, int, int]] | None = None,
) -> tuple[list[TurnPrep], dict[str, int]]:
    """Pure-prep half of `master_pipeline.process_match` — no LLM calls.

    Returns the per-turn TurnPrep list (in chronological order across all
    games of the match) plus a stats dict for skipped turns (ambiguous
    actions / threat-matrix failures / already-done resume hits).

    Threading model: this is async because `damage_inferencer` and
    `threat_matrix` are network-bound (calc microservice). Within a
    match the loop is sequential because each turn's threat matrix
    depends on `p2_running` updated by prior turns' events.
    """
    stats: dict[str, int] = {"skipped_ambiguous": 0, "skipped_threat_matrix_error": 0, "already_done": 0}
    seen_keys = seen_keys or set()
    preps: list[TurnPrep] = []

    match_record = flip_match_to_winner(match_record)
    games = match_record.get("games") or []
    if not games:
        stats["skipped_no_games"] = 1
        return preps, stats

    match_format = match_record.get("format", "bo1")
    team_sheets = team_sheets_for_match(games) if match_format == "bo3" else None

    p1_team_recon = reconstruct_p1_team(games)
    # Bo1 CTS only: forward-scan opponent species for bench metadata.
    # Bo3 OTS pulls this from team_sheets so the recon is redundant work.
    p2_team_recon = reconstruct_p2_team(games) if not team_sheets else None
    if team_sheets:
        p1_species_universe = [s["species"] for s in team_sheets["p1"]]
        p2_species_universe = [s["species"] for s in team_sheets["p2"]]
    else:
        p1_species_universe = list(p1_team_recon.keys())
        p2_species_universe = reconstruct_p2_species(games)

    p1_running = damage_inferencer.init_knowledge(p1_species_universe)
    p2_running = damage_inferencer.init_knowledge(p2_species_universe)
    match_id = match_record.get("match_id", "unknown")
    # Match-final inference can fail on malformed snapshots (one bad
    # /calc 400 from `damage_inferencer._call_calc`); fall back to open
    # P1 bounds rather than killing the entire batch run. Mirror of the
    # guard in master_pipeline.process_match.
    try:
        p1_final, _ = await damage_inferencer.infer_match_final_bounds(
            games, p1_species_universe, p2_species_universe,
            session=aiohttp_session, base_url=calc_base_url,
        )
    except Exception as e:
        click.echo(
            f"[{match_id}] match-final inference failed: {e}; "
            f"falling back to fully-open P1 bounds for this match",
            err=True,
        )
        p1_final = damage_inferencer.init_knowledge(p1_species_universe)
        stats["fallback_open_p1_bounds"] = stats.get("fallback_open_p1_bounds", 0) + 1

    bo1_system_prompt = (
        render_system_prompt(format_p1_team_block(p1_team_recon))
        if not team_sheets
        else None
    )

    for game_idx, game in enumerate(games):
        snapshots = game.get("snapshots") or []
        if team_sheets:
            brought = brought_species_keys_for_game(game)
            system_prompt = render_system_prompt_bo3(
                p1_sheet=team_sheets["p1"],
                p2_sheet=team_sheets["p2"],
                p1_brought=brought,
            )
        else:
            system_prompt = bo1_system_prompt

        for i in range(len(snapshots) - 1):
            snap_pre = snapshots[i]
            snap_post = snapshots[i + 1]
            events_stream = snap_pre.get("events") or []
            turn = int(snap_pre.get("turn", 0))
            key = (match_id, game_idx, turn)
            if key in seen_keys:
                stats["already_done"] += 1
                await _safe_update_knowledge(
                    snap_pre, snap_post, events_stream, p1_running, p2_running,
                    session=aiohttp_session, base_url=calc_base_url,
                )
                continue

            human_action_dict = extract_p1_actions(snap_pre, snap_post, events_stream)
            if human_action_dict is None:
                stats["skipped_ambiguous"] += 1
                await _safe_update_knowledge(
                    snap_pre, snap_post, events_stream, p1_running, p2_running,
                    session=aiohttp_session, base_url=calc_base_url,
                )
                continue

            try:
                tm_text = await threat_matrix.generate_threat_matrix(
                    snap_pre, "p1", p1_final, p2_running,
                    format_id=format_id,
                    session=aiohttp_session,
                    base_url=calc_base_url,
                )
            except Exception as e:
                stats["skipped_threat_matrix_error"] += 1
                click.echo(
                    f"[{match_id} g{game_idx} t{turn}] threat_matrix failed: {e}",
                    err=True,
                )
                await _safe_update_knowledge(
                    snap_pre, snap_post, events_stream, p1_running, p2_running,
                    session=aiohttp_session, base_url=calc_base_url,
                )
                continue

            p1_spreads = format_p1_known_spreads_block(snap_pre, p1_final, format_id=format_id)
            if team_sheets:
                opp_universe = [m["species"] for m in team_sheets["p2"]]
            else:
                opp_universe = None
            p2_spreads = format_p2_inferred_spreads_block(
                snap_pre, p2_running, species_universe=opp_universe,
            )
            user_prompt = format_user_prompt(
                snap_pre,
                tm_text,
                p1_inferred_block=p1_spreads,
                p2_inferred_block=p2_spreads,
                snapshots_so_far=snapshots,
                current_idx=i,
                prior_games=games[:game_idx],
                game_index=game_idx,
                total_games_in_series=len(games),
                match_format=match_format,
                team_sheets=team_sheets,
                p1_team_recon=p1_team_recon,
                p2_team_recon=p2_team_recon,
            )
            human_action = {
                "slot_1": human_action_dict.get("a", slot_action("pass")),
                "slot_2": human_action_dict.get("b", slot_action("pass")),
            }

            # Build the initial api_messages with ground-truth suffix
            # appended — same shape the sync OpenAIProvider builds.
            api_user_content = user_prompt + SYNTHESIS_GROUND_TRUTH_SUFFIX.format(
                ground_truth_json=json.dumps(human_action, indent=2)
            )
            api_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": api_user_content},
            ]

            preps.append(TurnPrep(
                match_id=match_id,
                game_idx=game_idx,
                turn=turn,
                format_id=format_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                human_action=human_action,
                api_messages=api_messages,
            ))

            await _safe_update_knowledge(
                snap_pre, snap_post, events_stream, p1_running, p2_running,
                session=aiohttp_session, base_url=calc_base_url,
            )

    return preps, stats


async def _safe_update_knowledge(
    snap_pre, snap_post, events_stream, p1_knowledge, p2_knowledge, *, session, base_url
):
    """Mirror of `master_pipeline._safe_update_knowledge` — kept duplicated
    rather than imported so this module stays an importable sibling rather
    than a circular dependency on master_pipeline."""
    damage_events = damage_inferencer.events_to_damage_events(events_stream)
    if not damage_events:
        return
    try:
        await damage_inferencer.update_knowledge(
            snap_pre, snap_post, damage_events, p1_knowledge, p2_knowledge,
            session=session, base_url=base_url,
        )
    except Exception as e:
        click.echo(f"update_knowledge failed: {e}", err=True)


# ---------------------------------------------------------------------------
# Batch WorkItem + state persistence
# ---------------------------------------------------------------------------


@dataclass
class BatchWorkItem:
    """One turn's mutable state across the batch state machine's lifetime.

    Serializes to JSON for resume. `status` transitions:
        pending      -> submitted (queued in this cycle's batch)
        submitted    -> pending (next cycle, needs more iters)
                     -> committed (model called submit_decision)
                     -> failed (API error mid-cycle)
        committed    -> leak_persistent (regex caught a leak post-hoc)
                     -> judge_flagged_persistent (judge dropped after retries)
                     -> written (row appended to output JSONL)
        failed       -> terminal; bumps `skipped_llm_error`
        leak_persistent          -> terminal; bumps `skipped_persistent_leak`
        judge_flagged_persistent -> terminal; bumps `skipped_persistent_judge_fail`
        written      -> terminal; bumps `written`
    """
    # Identity
    match_id: str
    game_idx: int
    turn: int
    format_id: str
    # Synthesis state (grows with each batch cycle)
    api_messages: list[dict[str, Any]]
    iter: int = 0
    calc_calls: int = 0
    iterations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    # Final / status
    saved_messages: list[dict[str, Any]] | None = None
    status: str = "pending"
    error: str | None = None
    # Re-synth context (kept for the sync-fallback judge retry path)
    system_prompt: str = ""
    user_prompt: str = ""
    human_action: dict[str, Any] = field(default_factory=dict)
    # Resume support: when an item is `submitted`, this holds the
    # OpenAI batch_id whose response we're waiting on. Cleared back to
    # None once the response is applied. Without this field, a crash
    # while items are submitted leaves them orphaned — the orchestrator
    # has no way on resume to find the right batch to re-poll.
    active_batch_id: str | None = None

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.match_id, self.game_idx, self.turn)

    def custom_id_for(self, cycle: int) -> str:
        """Render the OpenAI Batch custom_id for a specific cycle.

        Format: `{match}::g{game}::t{turn}::iter{cycle}`. The cycle is the
        TOOL-LOOP iteration the request represents — distinct from `iter`
        because we may re-issue the same logical request after a transient
        failure (not currently implemented but the schema supports it).
        """
        return f"{self.match_id}::g{self.game_idx}::t{self.turn}::iter{cycle}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BatchWorkItem":
        # Permissive: allow older state files lacking new fields.
        return cls(
            match_id=d["match_id"],
            game_idx=int(d["game_idx"]),
            turn=int(d["turn"]),
            format_id=d.get("format_id", ""),
            api_messages=d.get("api_messages") or [],
            iter=int(d.get("iter", 0)),
            calc_calls=int(d.get("calc_calls", 0)),
            iterations=int(d.get("iterations", 0)),
            input_tokens=int(d.get("input_tokens", 0)),
            output_tokens=int(d.get("output_tokens", 0)),
            saved_messages=d.get("saved_messages"),
            status=d.get("status", "pending"),
            error=d.get("error"),
            system_prompt=d.get("system_prompt", ""),
            user_prompt=d.get("user_prompt", ""),
            human_action=d.get("human_action") or {},
            active_batch_id=d.get("active_batch_id"),
        )


def _state_file_path(state_dir: Path, match_id: str) -> Path:
    """One match_id == one JSON file. The match_id has no path-unsafe chars
    in our corpus (format_id-replay_id), but quote to be defensive."""
    safe = match_id.replace("/", "_").replace("\\", "_")
    return state_dir / f"{safe}.json"


def _save_match_state(
    state_dir: Path,
    match_id: str,
    format_id: str,
    items: list[BatchWorkItem],
    active_batches: list[dict[str, Any]],
) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "match_id": match_id,
        "format_id": format_id,
        "items": [it.to_dict() for it in items],
        "active_batches": active_batches,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }
    path = _state_file_path(state_dir, match_id)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def _load_match_state(state_dir: Path, match_id: str) -> dict[str, Any] | None:
    path = _state_file_path(state_dir, match_id)
    if not path.exists():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Cycle processing
# ---------------------------------------------------------------------------


def _tool_choice_for(item: BatchWorkItem) -> Any:
    """Match the sync provider's two-way tool_choice logic:
       calc_calls ≥ cap → force submit_decision; otherwise → 'required'."""
    if item.calc_calls >= MAX_CALC_CALLS_BEFORE_FORCE_SUBMIT:
        return {"type": "function", "function": {"name": "submit_decision"}}
    return "required"


def _strip_ground_truth_suffix(api_messages: list[dict[str, Any]], user_prompt: str) -> list[dict[str, Any]]:
    """Replace the (system, user-with-suffix) prefix with (system,
    user-plain) so the saved messages match the sync runner's output.
    This is the same op `OpenAIProvider._do_turn` does on commit."""
    if not api_messages:
        return api_messages
    saved = list(api_messages)
    if len(saved) >= 2 and saved[1].get("role") == "user":
        saved[1] = {"role": "user", "content": user_prompt}
    return saved


async def _apply_batch_response(
    item: BatchWorkItem,
    response_record: dict[str, Any],
    *,
    aiohttp_session: aiohttp.ClientSession,
    calc_base_url: str,
) -> None:
    """Mutate `item` to reflect the result of one batch cycle.

    Equivalent of one iteration of the sync provider's tool loop:
      • append assistant message to api_messages
      • update token / iteration / calc counts
      • for each tool call: run calc (sync) or recognise submit
      • set item.status to 'pending' (continue), 'committed' (done), or 'failed'
    """
    if "error" in response_record and response_record["error"]:
        item.status = "failed"
        item.error = str(response_record["error"])
        return

    # Successful line: response.body matches a sync chat.completions response.
    resp_body = (response_record.get("response") or {}).get("body") or {}
    choices = resp_body.get("choices") or []
    usage = resp_body.get("usage") or {}
    item.input_tokens += int(usage.get("prompt_tokens", 0) or 0)
    item.output_tokens += int(usage.get("completion_tokens", 0) or 0)
    item.iterations += 1

    if not choices:
        item.status = "failed"
        item.error = "no choices in batch response"
        return

    msg = (choices[0] or {}).get("message") or {}
    tool_calls = msg.get("tool_calls") or []

    # Build the assistant message in the same shape the sync runner saves.
    assistant_msg: dict[str, Any] = {"role": "assistant"}
    if msg.get("content"):
        assistant_msg["content"] = msg["content"]
    if tool_calls:
        assistant_msg["tool_calls"] = [
            {
                "id": tc.get("id"),
                "type": "function",
                "function": {
                    "name": (tc.get("function") or {}).get("name"),
                    "arguments": (tc.get("function") or {}).get("arguments"),
                },
            }
            for tc in tool_calls
        ]
    item.api_messages.append(assistant_msg)

    if not tool_calls:
        item.status = "failed"
        item.error = "no tool_calls (protocol violation)"
        return

    submit_seen = False
    for tc in tool_calls:
        fn = tc.get("function") or {}
        name = fn.get("name")
        tc_id = tc.get("id")
        if name == "calculate_damage":
            try:
                args = json.loads(fn.get("arguments") or "{}")
                calc_result = await _call_calc(aiohttp_session, f"{calc_base_url}/calc", args)
                tool_content = json.dumps(calc_result)
            except Exception as e:
                tool_content = json.dumps({"error": f"{type(e).__name__}: {e}"})
            item.api_messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": tool_content,
            })
            item.calc_calls += 1
        elif name == "submit_decision":
            submit_seen = True
            item.api_messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": json.dumps({"status": "decision_committed"}),
            })
        else:
            item.api_messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": json.dumps({"error": f"unknown tool: {name}"}),
            })

    if submit_seen:
        item.saved_messages = _strip_ground_truth_suffix(item.api_messages, item.user_prompt)
        item.status = "committed"
    else:
        item.iter += 1
        item.status = "pending"


async def _resume_inflight_batches(
    match_items: dict[str, list[BatchWorkItem]],
    batch_provider: BatchOpenAIProvider,
    *,
    aiohttp_session: aiohttp.ClientSession,
    calc_base_url: str,
    poll_interval_seconds: float,
    max_cycle_wait_seconds: float,
    stats: dict[str, int],
) -> None:
    """Drain any batches left in-flight by a prior crash.

    Walks every match's items, groups those in `status="submitted"` by
    their `active_batch_id`, polls each batch to completion, fetches
    results, and applies to the corresponding items. Items with
    `status="submitted"` but `active_batch_id=None` (legacy state files
    without the breadcrumb) get marked failed — they can't be recovered.

    This runs BEFORE the cycle loop on resume so the loop sees consistent
    state (no items stuck waiting for batches we've forgotten about).
    """
    submitted: list[BatchWorkItem] = [
        it for items in match_items.values() for it in items
        if it.status == "submitted"
    ]
    if not submitted:
        return

    by_batch: dict[str, list[BatchWorkItem]] = {}
    orphans = 0
    for it in submitted:
        if it.active_batch_id:
            by_batch.setdefault(it.active_batch_id, []).append(it)
        else:
            it.status = "failed"
            it.error = "submitted without active_batch_id (pre-resume-fix state file?)"
            stats["skipped_llm_error"] += 1
            orphans += 1
    if orphans:
        click.echo(
            f"[batch][resume] {orphans} items with no active_batch_id — marked failed",
            err=True,
        )

    for batch_id, items_in_batch in by_batch.items():
        click.echo(
            f"[batch][resume] re-polling in-flight batch {batch_id} with "
            f"{len(items_in_batch)} items",
            err=True,
        )
        try:
            final_status = await batch_provider.poll_until_done(
                batch_id,
                poll_interval_seconds=poll_interval_seconds,
                max_wait_seconds=max_cycle_wait_seconds,
            )
        except Exception as e:
            click.echo(f"[batch][resume] poll failed for {batch_id}: {e}", err=True)
            for it in items_in_batch:
                it.status = "failed"
                it.error = f"resume_poll_failed: {type(e).__name__}: {e}"
                it.active_batch_id = None
                stats["skipped_llm_error"] += 1
            continue
        if final_status.status != "completed":
            click.echo(
                f"[batch][resume] batch {batch_id} terminal-bad status="
                f"{final_status.status}", err=True,
            )
            for it in items_in_batch:
                it.status = "failed"
                it.error = f"resume_batch_{final_status.status}"
                it.active_batch_id = None
                stats["skipped_llm_error"] += 1
            continue

        results = await batch_provider.fetch_results(batch_id)
        for it in items_in_batch:
            # When this batch was submitted, it.iter equaled the cycle.
            # Items never advanced past `submitted` before the crash, so
            # it.iter still holds the cycle index we need.
            resp = results.get(it.custom_id_for(it.iter))
            if resp is None:
                it.status = "failed"
                it.error = "missing custom_id in resume fetch"
                it.active_batch_id = None
                stats["skipped_llm_error"] += 1
                continue
            await _apply_batch_response(
                it, resp,
                aiohttp_session=aiohttp_session,
                calc_base_url=calc_base_url,
            )
            it.active_batch_id = None
            if it.status == "failed":
                stats["skipped_llm_error"] += 1


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


async def run_batch_for_matches(
    matches: list[dict[str, Any]],
    *,
    batch_provider: BatchOpenAIProvider,
    aiohttp_session: aiohttp.ClientSession,
    calc_base_url: str,
    state_dir: Path,
    output_path: Path,
    file_lock: asyncio.Lock,
    format_id: str,
    seen_keys: set[tuple[str, int, int]],
    poll_interval_seconds: float,
    max_cycle_wait_seconds: float,
    leak_retries: int = PRODUCTION_LEAK_RETRIES,
    use_judge: bool = True,
    judge_client: Any = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_retries: int = DEFAULT_JUDGE_RETRIES,
    teacher_for_judge_retry: TeacherProvider | None = None,
    resume: bool = False,
) -> dict[str, int]:
    """Drive every match's turns through the batch state machine.

    Per-cycle flow:
      1. Collect all `pending` items across all matches at the current
         cycle index.
      2. Build one batch upload (capped at MAX_REQUESTS_PER_BATCH; split
         if needed — not yet implemented in v1, asserted instead).
      3. Submit, poll, fetch.
      4. Apply each response to its item, run sync calc calls if needed.
      5. Persist all touched match states.

    After all cycles drain (or `MAX_TOOL_ITERATIONS` is hit), regex leak
    filter + per-match judge run on the committed items. Surviving rows
    are written to the output JSONL per-match atomically.
    """
    stats: dict[str, int] = {
        "matches_processed": len(matches),
        "skipped_ambiguous": 0,
        "skipped_threat_matrix_error": 0,
        "already_done": 0,
        "skipped_no_games": 0,
        "skipped_llm_error": 0,
        "skipped_persistent_leak": 0,
        "skipped_persistent_judge_fail": 0,
        "leak_retry": 0,
        "judge_pass": 0,
        "judge_flagged_total": 0,
        "judge_retried_total": 0,
        "judge_error": 0,
        "judge_cost_micro_usd": 0,
        "written": 0,
        "batches_submitted": 0,
        "batches_failed": 0,
    }

    # ---------------- Prep / WorkItem materialization ----------------

    # Per-match registry of WorkItems. Keyed by match_id for O(1) state-file IO.
    match_items: dict[str, list[BatchWorkItem]] = {}
    match_format_ids: dict[str, str] = {}
    match_user_prompts: dict[tuple[str, int, int], str] = {}

    for rec in matches:
        match_id = rec.get("match_id", "unknown")

        # Resume path: if a state file exists, restore items from disk
        # rather than re-prepping (which is idempotent but wastes calc
        # microservice calls).
        cached = _load_match_state(state_dir, match_id) if resume else None
        if cached:
            items = [BatchWorkItem.from_dict(d) for d in (cached.get("items") or [])]
            match_items[match_id] = items
            match_format_ids[match_id] = cached.get("format_id", format_id)
            for it in items:
                match_user_prompts[it.key] = it.user_prompt
            continue

        preps, prep_stats = await _prepare_match_turns(
            rec,
            format_id=format_id,
            calc_base_url=calc_base_url,
            aiohttp_session=aiohttp_session,
            seen_keys=seen_keys,
        )
        for k, v in prep_stats.items():
            stats.setdefault(k, 0)
            stats[k] += v

        items: list[BatchWorkItem] = []
        for p in preps:
            items.append(BatchWorkItem(
                match_id=p.match_id,
                game_idx=p.game_idx,
                turn=p.turn,
                format_id=p.format_id,
                api_messages=list(p.api_messages),
                system_prompt=p.system_prompt,
                user_prompt=p.user_prompt,
                human_action=p.human_action,
                status="pending",
            ))
            match_user_prompts[p.match_id, p.game_idx, p.turn] = p.user_prompt
        match_items[match_id] = items
        match_format_ids[match_id] = format_id
        _save_match_state(state_dir, match_id, format_id, items, active_batches=[])

    # Flat list view for batch building.
    def _all_items() -> list[BatchWorkItem]:
        return [it for items in match_items.values() for it in items]

    def _persist_touched(touched_match_ids: set[str]) -> None:
        for mid in touched_match_ids:
            _save_match_state(
                state_dir, mid, match_format_ids.get(mid, format_id),
                match_items[mid], active_batches=[],
            )

    # ---------------- Resume preamble: drain in-flight batches ----------------
    #
    # If we crashed mid-cycle on a prior run, some items will be on disk
    # with status="submitted" and an active_batch_id. Re-poll those
    # batches first, fetch results, apply to the items — then enter the
    # normal cycle loop with everything back in pending/committed shape.
    if resume:
        await _resume_inflight_batches(
            match_items, batch_provider,
            aiohttp_session=aiohttp_session,
            calc_base_url=calc_base_url,
            poll_interval_seconds=poll_interval_seconds,
            max_cycle_wait_seconds=max_cycle_wait_seconds,
            stats=stats,
        )
        _persist_touched(set(match_items.keys()))

    # ---------------- Cycle loop ----------------

    for cycle in range(MAX_TOOL_ITERATIONS):
        pending = [it for it in _all_items() if it.status == "pending" and it.iter == cycle]
        if not pending:
            # Every item is either done (committed / failed) or further
            # ahead in the iter counter; either way, this cycle is empty.
            break

        click.echo(f"[batch] cycle {cycle}: {len(pending)} pending items across {len({it.match_id for it in pending})} matches", err=True)

        # Build one batch.
        requests = [
            batch_provider.build_request(
                custom_id=it.custom_id_for(cycle),
                api_messages=it.api_messages,
                tool_choice=_tool_choice_for(it),
            )
            for it in pending
        ]

        # Submit.
        try:
            batch_id = await batch_provider.submit_batch(requests)
        except Exception as e:
            stats["batches_failed"] += 1
            click.echo(f"[batch] cycle {cycle} submit failed: {e}", err=True)
            for it in pending:
                it.status = "failed"
                it.error = f"submit_failed: {type(e).__name__}: {e}"
                stats["skipped_llm_error"] += 1
            _persist_touched({it.match_id for it in pending})
            break

        stats["batches_submitted"] += 1
        # Mark pending items as submitted; persist before polling so a
        # crash mid-poll can recover. `active_batch_id` is the breadcrumb
        # `--resume` follows on restart to find each item's pending batch.
        for it in pending:
            it.status = "submitted"
            it.active_batch_id = batch_id
        touched = {it.match_id for it in pending}
        _persist_touched(touched)
        click.echo(f"[batch] cycle {cycle} submitted batch_id={batch_id}", err=True)

        # Poll.
        try:
            final_status: BatchPollStatus = await batch_provider.poll_until_done(
                batch_id,
                poll_interval_seconds=poll_interval_seconds,
                max_wait_seconds=max_cycle_wait_seconds,
            )
        except Exception as e:
            stats["batches_failed"] += 1
            click.echo(f"[batch] cycle {cycle} poll failed: {e}", err=True)
            for it in pending:
                it.status = "failed"
                it.error = f"poll_failed: {type(e).__name__}: {e}"
                stats["skipped_llm_error"] += 1
            _persist_touched(touched)
            break

        if final_status.status != "completed":
            click.echo(
                f"[batch] cycle {cycle} terminal-bad status={final_status.status}",
                err=True,
            )
            stats["batches_failed"] += 1
            for it in pending:
                it.status = "failed"
                it.error = f"batch_{final_status.status}"
                stats["skipped_llm_error"] += 1
            _persist_touched(touched)
            break

        # Fetch + apply per item.
        results = await batch_provider.fetch_results(batch_id)
        for it in pending:
            resp = results.get(it.custom_id_for(cycle))
            if resp is None:
                it.status = "failed"
                it.error = "missing custom_id in batch output"
                stats["skipped_llm_error"] += 1
                it.active_batch_id = None
                continue
            await _apply_batch_response(
                it, resp,
                aiohttp_session=aiohttp_session,
                calc_base_url=calc_base_url,
            )
            # Apply finished — the item is no longer waiting on this
            # batch (either it advanced, committed, or failed). Clear
            # the breadcrumb so a future resume doesn't reprocess.
            it.active_batch_id = None
            if it.status == "failed":
                stats["skipped_llm_error"] += 1
        _persist_touched(touched)
        click.echo(
            f"[batch] cycle {cycle} processed: "
            f"committed={sum(1 for it in pending if it.status == 'committed')} "
            f"pending={sum(1 for it in pending if it.status == 'pending')} "
            f"failed={sum(1 for it in pending if it.status == 'failed')}",
            err=True,
        )

    # ---------------- Regex leak filter on committed items ----------------

    for it in _all_items():
        if it.status == "committed" and detect_oracle_leak(it.saved_messages or []):
            # v1 behavior: drop. v2 could fall back to sync re-synth here,
            # but the judge's `drop-flagged` policy already provides one
            # more chance via the post-judge sync retry. Belt-and-suspenders.
            it.status = "leak_persistent"
            stats["skipped_persistent_leak"] += 1
    _persist_touched(set(match_items.keys()))

    # ---------------- Per-match judge + write ----------------

    for match_id, items in match_items.items():
        committed = [it for it in items if it.status == "committed"]
        if not committed:
            continue

        survivors_messages: dict[tuple[int, int], list[dict[str, Any]]] = {}
        if (
            use_judge and judge_client is not None
            and teacher_for_judge_retry is not None
        ):
            # Build the same row + ctx pair the sync runner passes the judge.
            row_buffer = [
                {
                    "match_id": it.match_id,
                    "game_index": it.game_idx,
                    "turn": it.turn,
                    "format_id": it.format_id,
                    "messages": it.saved_messages or [],
                }
                for it in committed
            ]
            turn_contexts = [
                {
                    "system_prompt": it.system_prompt,
                    "user_prompt": it.user_prompt,
                    "human_action": it.human_action,
                    "game_idx": it.game_idx,
                    "turn": it.turn,
                }
                for it in committed
            ]
            # Imported lazily to avoid a circular import (master_pipeline
            # imports from batch_runner via the dispatcher).
            from master_pipeline import _run_judge_with_retries
            survivors = await _run_judge_with_retries(
                row_buffer,
                turn_contexts,
                judge_client=judge_client,
                judge_model=judge_model,
                judge_retries=judge_retries,
                teacher=teacher_for_judge_retry,
                calc_base_url=calc_base_url,
                aiohttp_session=aiohttp_session,
                leak_retries=leak_retries,
                stats=stats,
                match_id=match_id,
            )
            # Map survivors back to (game_idx, turn) for status updates.
            for row in survivors:
                survivors_messages[(int(row["game_index"]), int(row["turn"]))] = row["messages"]
            # Flag the dropped ones.
            survivor_keys = set(survivors_messages.keys())
            for it in committed:
                if (it.game_idx, it.turn) not in survivor_keys:
                    it.status = "judge_flagged_persistent"
        else:
            for it in committed:
                survivors_messages[(it.game_idx, it.turn)] = it.saved_messages or []

        # Atomic per-match write.
        async with file_lock:
            with output_path.open("a") as f:
                for it in committed:
                    msgs = survivors_messages.get((it.game_idx, it.turn))
                    if msgs is None:
                        continue
                    f.write(json.dumps({
                        "match_id": it.match_id,
                        "game_index": it.game_idx,
                        "turn": it.turn,
                        "format_id": it.format_id,
                        "messages": msgs,
                    }) + "\n")
                    seen_keys.add(it.key)
                    it.status = "written"
                    stats["written"] += 1
        _save_match_state(
            state_dir, match_id, match_format_ids.get(match_id, format_id),
            items, active_batches=[],
        )

    return stats


__all__ = [
    "BatchWorkItem",
    "TurnPrep",
    "_prepare_match_turns",
    "run_batch_for_matches",
]
