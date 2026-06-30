"""Orchestrator: scraped replays in -> SFT-ready JSONL out.

Pipeline role:
    Walks through `parsed_data/{bo1,bo3}.jsonl` (produced by replay_parser),
    and for each turn of each match:
      1. relabels sides so the series winner is P1 (`flip_match_to_winner`);
      2. extracts P1's actual two-slot decision from `snap[N].events` +
         the diff to `snap[N+1]` (move / switch / cant_move events, Tera flag);
      3. asks `threat_matrix` to render the dual-track damage envelope;
      4. drives `teacher.synthesize_turn` to elicit a chain-of-thought
         that justifies that exact decision;
      5. writes the resulting OpenAI-fine-tuning conversation to
         `parsed_data/sft_training_data.jsonl`;
      6. filters the same `events` stream for damage observations and
         feeds them to `damage_inferencer.update_knowledge` to tighten
         both KnowledgeStates for the next turn.

    KnowledgeStates start at fully-open `[0, 252]` bounds and tighten as
    observed damage + move-order events accumulate. The threat matrix
    renders the strict Absolute envelope from those bounds — there is no
    canonical-meta second track (Smogon priors were removed).

Isolation contract:
    The only file allowed to import from every other pipeline module.
    Everything else is leaf — no cross-imports between siblings.
"""
from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import aiohttp
import click
from tqdm.asyncio import tqdm

import damage_inferencer
import threat_matrix
from action_extraction import (
    DEFAULT_MIN_GAME_TURNS,
    action_consistent_with_knownset,
    extract_p1_actions,
    filter_fragment_games,
    flip_match_to_winner,
    slot_action,
)
from prompt_formatting import (
    format_p1_known_spreads_block,
    format_p1_team_block,
    format_meta_builds,
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
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_MODEL_OPENAI,
    DEFAULT_JUDGE_PROVIDER,
    DEFAULT_JUDGE_RETRIES,
    DEFAULT_MAX_CYCLE_WAIT_SECONDS,
    DEFAULT_POLL_INTERVAL_SECONDS,
    PRODUCTION_LEAK_RETRIES,
    JudgeResult,
    TeacherProvider,
    detect_oracle_leak,
    extract_pre_tool_thought,
    judge_match_cots,
    render_system_prompt,
    render_system_prompt_bo3,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = Path(__file__).resolve().parent
DEFAULT_PARSED_DATA_DIR = PIPELINE_DIR / "parsed_data"
DEFAULT_BO3_INPUT = DEFAULT_PARSED_DATA_DIR / "bo3.jsonl"
DEFAULT_BO1_INPUT = DEFAULT_PARSED_DATA_DIR / "bo1.jsonl"
DEFAULT_OUTPUT = DEFAULT_PARSED_DATA_DIR / "sft_training_data.jsonl"
DEFAULT_CALC_BASE_URL = "http://localhost:3000"
DEFAULT_BATCH_STATE_DIR = PIPELINE_DIR / "batch_state"

# Hybrid-mode quality gate thresholds. The first N matches run sync with
# the judge ON, and if either match-rate dips below MIN_MATCH_RATE or
# leak-rate exceeds MAX_LEAK_RATE we abort before submitting any batch.
DEFAULT_HYBRID_SYNC_N = 50
DEFAULT_HYBRID_MIN_MATCH_RATE = 0.95
DEFAULT_HYBRID_MAX_LEAK_RATE = 0.02

FORMAT_ID_BY_KIND = {
    "bo1": "gen9vgc2026regi",
    "bo3": "gen9vgc2026regibo3",
}


# ---------------------------------------------------------------------------
# Resumability
# ---------------------------------------------------------------------------


def load_seen_keys(output_path: Path) -> set[tuple[str, int, int]]:
    seen: set[tuple[str, int, int]] = set()
    if not output_path.exists():
        return seen
    with output_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = (rec["match_id"], int(rec["game_index"]), int(rec["turn"]))
                seen.add(key)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return seen


# ---------------------------------------------------------------------------
# Provider + calc-service plumbing
# ---------------------------------------------------------------------------


def _google_auth_configured() -> bool:
    """True when the google-genai SDK has a usable auth path. Two mutually
    exclusive modes — and the choice is a BILLING decision, not just auth:

      • AI Studio (Gemini Developer API): GOOGLE_API_KEY / GEMINI_API_KEY set.
        Bills to whatever is attached to that key — NOT a GCP credit grant.
      • Vertex AI: GOOGLE_GENAI_USE_VERTEXAI truthy + GOOGLE_CLOUD_PROJECT set
        (Application Default Credentials supply the token via
        `gcloud auth application-default login`). This is the path that draws
        down GCP committed-use / credit grants — use it to spend the credits.

    `genai.Client()` (constructed with no args in GoogleProvider + the judge)
    auto-selects: with GOOGLE_GENAI_USE_VERTEXAI=true it uses Vertex and
    ignores the API key. So switching to the credits is config-only; this
    helper just stops the pre-flight guard from rejecting the (keyless) Vertex
    path.
    """
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        return True
    use_vertex = str(os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "")).strip().lower() in (
        "1", "true", "yes",
    )
    return use_vertex and bool(os.environ.get("GOOGLE_CLOUD_PROJECT"))


_GOOGLE_AUTH_HINT = (
    "Configure Google auth one of two ways:\n"
    "  • AI Studio (personal/non-credit billing): set GEMINI_API_KEY or GOOGLE_API_KEY.\n"
    "  • Vertex AI (draws GCP credits — recommended): set GOOGLE_GENAI_USE_VERTEXAI=true, "
    "GOOGLE_CLOUD_PROJECT=<project>, GOOGLE_CLOUD_LOCATION=<region>, and run "
    "`gcloud auth application-default login`."
)


def _build_teacher(provider: str, model: str | None) -> TeacherProvider:
    """Instantiate the requested provider, validating that its API key is present."""
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise click.ClickException(
                "OPENAI_API_KEY env var is required (or pass --dry-run / --provider <other>)"
            )
        from teacher import OpenAIProvider
        return OpenAIProvider(model=model) if model else OpenAIProvider()
    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise click.ClickException("ANTHROPIC_API_KEY env var is required for --provider anthropic")
        from teacher import AnthropicProvider
        return AnthropicProvider(model=model) if model else AnthropicProvider()
    if provider == "google":
        if not _google_auth_configured():
            raise click.ClickException(
                "No Google auth configured for --provider google.\n" + _GOOGLE_AUTH_HINT
            )
        from teacher import GoogleProvider
        return GoogleProvider(model=model) if model else GoogleProvider()
    raise click.ClickException(f"unknown provider: {provider}")


async def _check_calc_health(session: aiohttp.ClientSession, base_url: str) -> None:
    try:
        async with session.get(f"{base_url}/health") as r:
            if r.status != 200:
                raise RuntimeError(f"health check returned {r.status}")
    except Exception as e:
        raise click.ClickException(
            f"calc_microservice not reachable at {base_url}/health: {e}\n"
            f"  Start it with:  cd calc_microservice && npm run dev"
        )


# ---------------------------------------------------------------------------
# Per-match orchestration
# ---------------------------------------------------------------------------


async def _synthesize_with_leak_retry(
    teacher: TeacherProvider,
    *,
    system_prompt: str,
    user_prompt: str,
    human_action: dict[str, Any],
    calc_base_url: str,
    aiohttp_session: aiohttp.ClientSession,
    leak_retries: int,
    stats: dict[str, int],
    log_prefix: str,
) -> list[dict[str, Any]] | None:
    """Single-turn synthesis with regex leak-retry. Returns clean messages or None.

    The judge (in `_run_judge_with_retries`) calls this exact helper when
    it re-synthesizes a flagged turn — that's why it's factored out. Both
    code paths share the same retry counter semantics: `leak_retry` for
    soft retries, `skipped_persistent_leak` when retries exhaust, and
    `skipped_llm_error` for API failures.
    """
    attempts = 0
    while True:
        try:
            res = await teacher.synthesize_turn(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                human_action=human_action,
                calc_url=f"{calc_base_url}/calc",
                aiohttp_session=aiohttp_session,
            )
            messages = res.messages
        except Exception as e:
            stats["skipped_llm_error"] += 1
            _log_error(f"{log_prefix} teacher LLM failed: {e}")
            return None
        if messages is None:
            if res.error:
                _log_error(f"{log_prefix} teacher LLM: {res.error}")
            stats["skipped_llm_error"] += 1
            return None
        leak = detect_oracle_leak(messages)
        if not leak:
            return messages
        if attempts >= leak_retries:
            stats["skipped_persistent_leak"] += 1
            _log_error(
                f"{log_prefix} persistent leak after {attempts+1} attempt(s) — "
                f"phrase {leak!r}; dropping row"
            )
            return None
        attempts += 1
        stats["leak_retry"] += 1
        _log_error(
            f"{log_prefix} leak retry {attempts}/{leak_retries} — phrase {leak!r}"
        )


def _build_judge_grounding(user_prompt: str, messages: list[dict[str, Any]]) -> str:
    """Ground truth the correctness judge fact-checks the CoT against: a lean
    slice of the turn (current HP + boosts from the TURN/ACTIVE header, plus
    the THREAT MATRIX) and a compact digest of every `calculate_damage` the
    teacher ran. The calc digest is what stops the judge from flagging
    legitimate boosted / Tera'd hypotheticals as if they were current-state
    errors. Sending the lean slice (not the full prompt) keeps judge input —
    and cost — down."""
    head_end = user_prompt.find("=== GAME-STATE")
    state = user_prompt[:head_end].rstrip() if head_end > 0 else ""
    mi = user_prompt.find("=== THREAT MATRIX")
    matrix = user_prompt[mi:].rstrip() if mi >= 0 else ""

    id_args: dict[str, dict[str, Any]] = {}
    calc_lines: list[str] = []
    for m in messages:
        for tc in (m.get("tool_calls") or []):
            if (tc.get("function") or {}).get("name") == "calculate_damage":
                try:
                    id_args[tc.get("id")] = json.loads(tc["function"]["arguments"])
                except (KeyError, TypeError, json.JSONDecodeError):
                    pass
        if m.get("role") == "tool" and m.get("tool_call_id") in id_args:
            a = id_args[m["tool_call_id"]]
            try:
                res = json.loads(m.get("content") or "{}")
            except (TypeError, json.JSONDecodeError):
                res = {}
            atk, dfn = (a.get("attacker") or {}), (a.get("defender") or {})
            mv = a.get("move")
            mv = mv.get("name") if isinstance(mv, dict) else mv
            tags = [t for t, on in (("atk-boosts", atk.get("boosts")),
                                    ("atk-tera", atk.get("isTera")),
                                    ("def-tera", dfn.get("isTera"))) if on]
            lo, hi, ko = res.get("minPercent"), res.get("maxPercent"), res.get("koChance")
            if lo is not None:
                tagstr = f" [{','.join(tags)}]" if tags else ""
                calc_lines.append(
                    f"  {mv} {atk.get('species','?')}→{dfn.get('species','?')}{tagstr}: "
                    f"{lo}–{hi}%" + (f"  [{ko}]" if ko else "")
                )
    grounding = (state + "\n\n" + matrix).strip()
    if calc_lines:
        grounding += ("\n\nCALC RESULTS (teacher-computed hypotheticals — verified, "
                      "treat as ground truth):\n" + "\n".join(calc_lines))
    return grounding


async def _run_judge_with_retries(
    match_buffer: list[dict[str, Any]],
    turn_contexts: list[dict[str, Any]],
    *,
    judge_client: Any,                                # AsyncOpenAI | genai.Client
    judge_provider: str,                              # "google" | "openai"
    judge_model: str,
    judge_retries: int,
    teacher: TeacherProvider,
    calc_base_url: str,
    aiohttp_session: aiohttp.ClientSession,
    leak_retries: int,
    stats: dict[str, int],
    match_id: str,
    judge_correctness: bool = True,
) -> list[dict[str, Any]]:
    """Buffered match-level judge: flag turns, re-synthesize them, repeat.

    Contract:
      - `match_buffer[i]` mutates in place when turn i is successfully
        re-synthesized; the original row dict is replaced with the new
        messages but other fields (match_id, game_index, turn) stay.
      - On any judge API error, returns `match_buffer` unchanged
        (fail-open: don't lose work because of an infra hiccup).
      - On exhausted retries, drops only the still-flagged turns
        (per plan v4 user decision: `drop-flagged`, never `drop-match`).

    `turn_contexts[i]` MUST mirror `match_buffer[i]` index-by-index: same
    length, same ordering. The judge's `turn_idx` field references this
    shared positional index — that's how we route retries back to the
    original synthesis context.
    """
    if not match_buffer:
        return match_buffer

    final_flagged: set[int] = set()
    for attempt in range(judge_retries + 1):
        # Build records for the judge. extract_pre_tool_thought returns
        # None for any turn whose submit_decision is missing or whose
        # arguments don't parse — those get an empty CoT, which the judge
        # is unlikely to flag (and which is a no-op anyway).
        turn_records = [
            {
                "turn_idx": i,
                "match_id": row["match_id"],
                "game_idx": row["game_index"],
                "turn": row["turn"],
                "pre_tool_thought": (extract_pre_tool_thought(row["messages"]) or ""),
                # Grounding (state + matrix + the teacher's calc results) lets
                # the judge fact-check quantitative claims; only built when
                # correctness-checking is on, since it ~doubles judge input.
                "grounding": (
                    _build_judge_grounding(turn_contexts[i].get("user_prompt", ""), row["messages"])
                    if judge_correctness else ""
                ),
            }
            for i, row in enumerate(match_buffer)
        ]
        jr: JudgeResult = await judge_match_cots(
            turn_records,
            client=judge_client,
            provider=judge_provider,
            model=judge_model,
            judge_correctness=judge_correctness,
        )
        stats["judge_cost_micro_usd"] += int(jr.cost_usd * 1_000_000)
        if jr.error:
            stats["judge_error"] += 1
            _log_error(
                f"[{match_id}] judge error: {jr.error}; fail-open, writing all rows"
            )
            return match_buffer
        if not jr.flagged_turn_indices:
            stats["judge_pass"] += 1
            return match_buffer

        stats["judge_flagged_total"] += len(jr.flagged_turn_indices)
        for idx in jr.flagged_turn_indices:
            reason = jr.reasons.get(idx, "")
            _log_error(
                f"[{match_id}] judge flagged turn_idx={idx} "
                f"(game={turn_records[idx]['game_idx']} turn={turn_records[idx]['turn']}): "
                f"{reason!r}"
            )

        if attempt == judge_retries:
            # Final pass — record which turns remain flagged.
            final_flagged = set(jr.flagged_turn_indices)
            break

        # Re-synthesize each flagged turn through the standard leak-retry
        # path. Successes replace the row in place; failures leave the
        # original row in place (the judge will likely flag it again next
        # pass and the retry budget will eventually drop it).
        for idx in jr.flagged_turn_indices:
            ctx = turn_contexts[idx]
            new_messages = await _synthesize_with_leak_retry(
                teacher,
                system_prompt=ctx["system_prompt"],
                user_prompt=ctx["user_prompt"],
                human_action=ctx["human_action"],
                calc_base_url=calc_base_url,
                aiohttp_session=aiohttp_session,
                leak_retries=leak_retries,
                stats=stats,
                log_prefix=(
                    f"[{match_id} g{ctx['game_idx']} t{ctx['turn']}] "
                    f"judge-retry {attempt+1}/{judge_retries}"
                ),
            )
            if new_messages is not None:
                match_buffer[idx]["messages"] = new_messages
                stats["judge_retried_total"] += 1

    # Exhausted retries — drop only still-flagged turns.
    if final_flagged:
        stats["skipped_persistent_judge_fail"] += len(final_flagged)
        return [r for i, r in enumerate(match_buffer) if i not in final_flagged]
    return match_buffer


async def process_match(
    match_record: dict[str, Any],
    *,
    output_path: Path,
    calc_base_url: str,
    teacher: TeacherProvider | None,
    aiohttp_session: aiohttp.ClientSession,
    file_lock: asyncio.Lock,
    format_id: str,
    seen_keys: set[tuple[str, int, int]],
    dry_run: bool,
    model: str,
    leak_retries: int = PRODUCTION_LEAK_RETRIES,
    use_judge: bool = True,
    judge_client: Any = None,
    judge_provider: str = DEFAULT_JUDGE_PROVIDER,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_retries: int = DEFAULT_JUDGE_RETRIES,
    judge_correctness: bool = True,
    min_game_turns: int = DEFAULT_MIN_GAME_TURNS,
) -> dict[str, int]:
    # Series-winner-as-P1: every SFT example is generated from the perspective
    # of the player who won the series. P2-won matches are relabeled in full.
    match_record = flip_match_to_winner(match_record)

    games = match_record.get("games") or []
    if not games:
        return {"skipped_no_games": 1}

    # Plan v8 — drop fragment games (too-short to carry meaningful
    # decision context). Default `min_game_turns=2` removes 0-snapshot
    # ghost games + 1-pair single-decision sweeps. Track for stats.
    games, frag_dropped = filter_fragment_games(games, min_game_turns)
    fragment_stats: dict[str, int] = {}
    if frag_dropped:
        fragment_stats["dropped_fragment_games"] = frag_dropped
    if not games:
        fragment_stats["skipped_all_games_too_short"] = 1
        return fragment_stats

    match_format = match_record.get("format", "bo1")
    team_sheets = team_sheets_for_match(games) if match_format == "bo3" else None

    # Knowledge state seeding — for OTS Bo3, use the full 6-mon team sheets
    # so the threat matrix can reason about the unswitched-in backline too.
    # For CTS Bo1, fall back to whatever the snapshots reveal.
    p1_team_recon = reconstruct_p1_team(games)
    # Bo1 CTS only: forward-scan opponent species for bench metadata
    # rendering. Bo3 OTS gets this from team_sheets directly so the
    # recon would be redundant work.
    p2_team_recon = reconstruct_p2_team(games) if not team_sheets else None
    if team_sheets:
        p1_species_universe = [s["species"] for s in team_sheets["p1"]]
        p2_species_universe = [s["species"] for s in team_sheets["p2"]]
    else:
        p1_species_universe = list(p1_team_recon.keys())
        p2_species_universe = reconstruct_p2_species(games)

    # Two parallel KnowledgeStates for P1 + one for P2:
    #   p1_running  — fed turn-by-turn; kept for diagnostics + the inspector.
    #                 NOT used in prompts (the player knows their own team).
    #   p1_final    — computed once below from the full match; surfaces as
    #                 the YOUR SPREADS block AND drives the matrix's P1 side.
    #                 Approximates "the exact spread the player built".
    #   p2_running  — fed turn-by-turn; drives the matrix's P2 side.
    #                 Models the observational asymmetry — we learn about
    #                 the opponent through play.
    stats: dict[str, int] = defaultdict(int)
    # Carry forward any pre-init bumps (e.g. fragment-game filter).
    for k, v in fragment_stats.items():
        stats[k] += v
    match_id = match_record.get("match_id", "unknown")

    p1_running = damage_inferencer.init_knowledge(p1_species_universe)
    p2_running = damage_inferencer.init_knowledge(p2_species_universe)
    # Roster-key sets gate the observed-flag + speed inference so a
    # transformed Ditto (snapshot species = the copied mon) can't pollute
    # a real roster slot's spread.
    p1_universe_keys = {damage_inferencer.species_key(s) for s in p1_species_universe}
    p2_universe_keys = {damage_inferencer.species_key(s) for s in p2_species_universe}
    # The match-final offline pass touches every damage event in every
    # game of the match. If any one event causes a /calc failure (most
    # commonly a malformed Pokémon entry on the Node side), we don't
    # want to take down the whole corpus run — fall back to fully-open
    # P1 bounds for this match and let the per-turn loop run normally.
    # The YOUR SPREADS block will render as `(no observations yet)` for
    # every P1 mon in this match, which is a sensible degradation.
    try:
        p1_final, _ = await damage_inferencer.infer_match_final_bounds(
            games, p1_species_universe, p2_species_universe,
            session=aiohttp_session, base_url=calc_base_url,
        )
    except Exception as e:
        _log_error(
            f"[{match_id}] match-final inference failed: {e}; "
            f"falling back to fully-open P1 bounds for this match"
        )
        p1_final = damage_inferencer.init_knowledge(p1_species_universe)
        stats["fallback_open_p1_bounds"] += 1

    # Bo1 system prompt is stable across all turns of the match.
    bo1_system_prompt = (
        render_system_prompt(format_p1_team_block(p1_team_recon))
        if not team_sheets
        else None
    )

    # Per-match buffers populated during the per-turn loop. We defer the
    # JSONL write until after the match-level judge runs so flagged turns
    # can be re-synthesized in place before commit.
    match_buffer: list[dict[str, Any]] = []
    turn_contexts: list[dict[str, Any]] = []

    for game_idx, game in enumerate(games):
        snapshots = game.get("snapshots") or []

        # Bo3 system prompt depends on the brought-4 of THIS game (different
        # selections per game in a series), so render per-game.
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
                continue

            human_action_dict = extract_p1_actions(snap_pre, snap_post, events_stream)
            if human_action_dict is None:
                stats["skipped_ambiguous"] += 1
                await _safe_update_knowledge(
                    snap_pre, snap_post, events_stream, p1_running, p2_running,
                    session=aiohttp_session, base_url=calc_base_url,
                    p1_universe=p1_universe_keys, p2_universe=p2_universe_keys,
                )
                continue

            # Data-integrity: an OTS move outside the mon's knownMoves is a
            # parser misparse (mirror-match slot misattribution). Drop the turn
            # rather than train a label that contradicts the sheet.
            if not action_consistent_with_knownset(snap_pre, human_action_dict):
                stats["skipped_illegal_moveset"] += 1
                await _safe_update_knowledge(
                    snap_pre, snap_post, events_stream, p1_running, p2_running,
                    session=aiohttp_session, base_url=calc_base_url,
                    p1_universe=p1_universe_keys, p2_universe=p2_universe_keys,
                )
                continue

            try:
                # Threat matrix gets the asymmetric (p1_final, p2_running) pair:
                # tight bounds for our side (we know our team), loose
                # chronological bounds for the opponent (we learn over time).
                tm_text = await threat_matrix.generate_threat_matrix(
                    snap_pre, "p1", p1_final, p2_running,
                    format_id=format_id,
                    session=aiohttp_session,
                    base_url=calc_base_url,
                )
            except Exception as e:
                stats["skipped_threat_matrix_error"] += 1
                _log_error(f"[{match_id} g{game_idx} t{turn}] threat_matrix failed: {e}")
                await _safe_update_knowledge(
                    snap_pre, snap_post, events_stream, p1_running, p2_running,
                    session=aiohttp_session, base_url=calc_base_url,
                    p1_universe=p1_universe_keys, p2_universe=p2_universe_keys,
                )
                continue

            # YOUR SPREADS surfaces match-final P1 bounds (the player's
            # own-team knowledge stand-in).
            p1_spreads = format_p1_known_spreads_block(
                snap_pre, p1_final, species_universe=p1_species_universe
            )
            # OPP SPREADS uses chronological P2 inference. Bo3 OTS lets
            # us list all 6 (player saw them at team preview); Bo1 CTS
            # gates by snapshot.p2.seenSpecies inside the renderer.
            if team_sheets:
                opp_universe = [m["species"] for m in team_sheets["p2"]]
            else:
                opp_universe = None
            p2_spreads = format_p2_inferred_spreads_block(
                snap_pre, p2_running, species_universe=opp_universe,
            )
            # META BUILDS: top-2 Smogon builds per opponent mon (reuses
            # opp_universe for the same visibility gating). Pure/sync; degrades
            # to "" on any error — a missing data file must never kill a turn.
            try:
                meta_text = format_meta_builds(
                    snap_pre, p2_running, format_id=format_id, species_universe=opp_universe,
                )
            except Exception as e:  # noqa: BLE001
                meta_text = ""
                _log_error(f"[{match_id} g{game_idx} t{turn}] format_meta_builds failed: {e}")
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
                meta_builds_text=meta_text,
            )
            human_action = {
                "slot_1": human_action_dict.get("a", slot_action("pass")),
                "slot_2": human_action_dict.get("b", slot_action("pass")),
            }

            messages: list[dict[str, Any]] | None = None
            if dry_run:
                messages = _dry_run_messages(system_prompt, user_prompt, human_action)
            else:
                if teacher is None:
                    raise RuntimeError("Teacher provider missing in non-dry-run mode")
                messages = await _synthesize_with_leak_retry(
                    teacher,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    human_action=human_action,
                    calc_base_url=calc_base_url,
                    aiohttp_session=aiohttp_session,
                    leak_retries=leak_retries,
                    stats=stats,
                    log_prefix=f"[{match_id} g{game_idx} t{turn}]",
                )

            if messages is not None:
                match_buffer.append({
                    "match_id": match_id,
                    "game_index": game_idx,
                    "turn": turn,
                    "format_id": format_id,
                    "messages": messages,
                })
                # turn_contexts must stay 1:1 with match_buffer by index
                # — the judge routes retries back via this positional
                # alignment.
                turn_contexts.append({
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "human_action": human_action,
                    "game_idx": game_idx,
                    "turn": turn,
                })

            await _safe_update_knowledge(
                snap_pre, snap_post, events_stream, p1_running, p2_running,
                session=aiohttp_session, base_url=calc_base_url,
                p1_universe=p1_universe_keys, p2_universe=p2_universe_keys,
            )

    # ------------------------------------------------------------------
    # Post-loop: match-level judge (Plan v4 workstream 1).
    # ------------------------------------------------------------------
    # Run on the buffered match. Dry-runs and judge-disabled paths skip
    # straight to write. The judge re-synthesizes flagged turns in place
    # (mutating match_buffer[idx]['messages']), and on exhausted retries
    # drops only the still-flagged turns.
    if not dry_run and use_judge and judge_client is not None and teacher is not None and match_buffer:
        survivors = await _run_judge_with_retries(
            match_buffer,
            turn_contexts,
            judge_client=judge_client,
            judge_provider=judge_provider,
            judge_model=judge_model,
            judge_retries=judge_retries,
            teacher=teacher,
            calc_base_url=calc_base_url,
            aiohttp_session=aiohttp_session,
            leak_retries=leak_retries,
            stats=stats,
            match_id=match_id,
            judge_correctness=judge_correctness,
        )
    else:
        survivors = match_buffer

    # Single atomic batch-write per match.
    if survivors:
        async with file_lock:
            with output_path.open("a") as f:
                for row in survivors:
                    f.write(json.dumps(row) + "\n")
                    seen_keys.add((row["match_id"], row["game_index"], row["turn"]))
                    stats["written"] += 1

    return dict(stats)


async def _safe_update_knowledge(
    snap_pre, snap_post, events_stream, p1_knowledge, p2_knowledge, *, session, base_url,
    p1_universe=None, p2_universe=None,
):
    """Feed the new TurnEvent stream to the inferencer. Two passes:

      1. Damage-bound inference (atk/spa/def/spd/hp). Drops Metronome /
         Copycat / Sketch / Snatch / Me First / Dancer / Instruct /
         Mirror Move / Assist / Nature Power call-throughs (those can hit
         moves not in the user's actual kit) but keeps Sleep Talk (own
         moves only).
      2. Observed-flag marking + conservative speed (move-order) inference,
         which runs EVERY turn — including damage-free turns, since status
         moves still reveal move order. `p1_universe` / `p2_universe` are
         roster-key sets gating against transformed-Ditto pollution."""
    damage_events = damage_inferencer.events_to_damage_events(events_stream)
    try:
        if damage_events:
            await damage_inferencer.update_knowledge(
                snap_pre, snap_post, damage_events, p1_knowledge, p2_knowledge,
                session=session, base_url=base_url,
            )
        await damage_inferencer.update_observed_and_speed(
            snap_pre, events_stream, p1_knowledge, p2_knowledge,
            session=session, base_url=base_url,
            p1_universe=p1_universe, p2_universe=p2_universe,
        )
    except Exception as e:
        _log_error(f"update_knowledge failed: {e}")


def _dry_run_messages(
    system_prompt: str, user_prompt: str, human_action: dict[str, Any]
) -> list[dict[str, Any]]:
    """Placeholder messages for `--dry-run` mode.

    Shaped to mirror a real teacher-LLM-produced row so the SFT inspector
    parses it identically: an assistant message with a `submit_decision`
    tool call (rather than a JSON dump in `content`), followed by the
    matching tool-response ack. The `pre_tool_thought` is a stub flagging
    that no LLM was actually called; the `action` field surfaces the
    human's actual play from the source replay as a visual cue.

    Use case: cheaply preview the full prompt corpus before paying for a
    real synthesis run. Every turn shows the exact system + user prompts
    the teacher would have seen, plus the ground-truth play it would
    have been asked to justify.
    """
    dry_call_id = "call_dry_run"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": dry_call_id,
                    "type": "function",
                    "function": {
                        "name": "submit_decision",
                        "arguments": json.dumps({
                            "pre_tool_thought": (
                                "[DRY RUN — teacher LLM was not invoked. "
                                "The `action` field below is the human "
                                "expert's actual play from the source "
                                "replay, surfaced as a placeholder so "
                                "this preview row is visually inspectable "
                                "alongside its prompt.]"
                            ),
                            "action": human_action,
                        }),
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": dry_call_id,
            "content": json.dumps({"status": "decision_committed_dry_run"}),
        },
    ]


def _log_error(msg: str) -> None:
    click.echo(msg, err=True)


# ---------------------------------------------------------------------------
# CLI / runner
# ---------------------------------------------------------------------------


def _resolve_format_id(input_path: Path, override: str | None) -> str:
    if override:
        return override
    name = input_path.stem.lower()
    if name in FORMAT_ID_BY_KIND:
        return FORMAT_ID_BY_KIND[name]
    return FORMAT_ID_BY_KIND["bo3"]


def _read_match_records(input_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with input_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _load_excluded_match_ids(path: Path | None) -> set[str]:
    """Match-ids the synthesis run must NOT touch — the eval holdout.

    Accepts the `eval_split.json` produced by `make_eval_split.py` (uses its
    `holdout` list) or a plain JSON/newline list of match-ids. Keeping the
    holdout out of training is what makes the downstream action-match eval
    honest (no train/test leakage)."""
    if path is None:
        return set()
    text = path.read_text()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return set(obj.get("holdout") or [])
        if isinstance(obj, list):
            return set(obj)
    except json.JSONDecodeError:
        pass
    return {ln.strip() for ln in text.splitlines() if ln.strip()}


def _build_judge_client(use_judge: bool, dry_run: bool, judge_provider: str) -> Any:
    """Construct the judge client. Provider dispatch (Plan v8):

      - `judge_provider="google"` → `google.genai.Client` (the production
        default; picks up `GOOGLE_API_KEY` or `GEMINI_API_KEY` from env).
      - `judge_provider="openai"` → `openai.AsyncOpenAI` (Plan v4 path;
        kept for explicit `--provider openai` runs).

    Plan v4 originally required OpenAI regardless of teacher provider so
    every judge call was cross-provider. Plan v8 dropped that — we
    standardized the teacher on Gemini and the judge with it, both for
    consistency and because the cross-provider cost / dependency was
    no longer paying for itself. Use `--judge-provider openai` to opt
    back into the cross-provider mode.
    """
    if dry_run or not use_judge:
        return None
    if judge_provider == "google":
        if not _google_auth_configured():
            raise click.ClickException(
                "No Google auth configured for --use-judge (default Gemini judge).\n"
                + _GOOGLE_AUTH_HINT
                + "\nOr pass --no-judge, or --judge-provider openai."
            )
        from google import genai
        return genai.Client()
    if judge_provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise click.ClickException(
                "OPENAI_API_KEY env var is required for --use-judge "
                "--judge-provider openai. Pass --no-judge to skip."
            )
        from openai import AsyncOpenAI
        return AsyncOpenAI()
    raise click.ClickException(f"unknown --judge-provider: {judge_provider}")


async def _run_sync(
    records: list[dict[str, Any]],
    *,
    output_path: Path,
    calc_base_url: str,
    format_id: str,
    concurrency: int,
    dry_run: bool,
    model: str | None,
    teacher: TeacherProvider | None,
    judge_client: Any,
    aiohttp_session: aiohttp.ClientSession,
    file_lock: asyncio.Lock,
    seen_keys: set[tuple[str, int, int]],
    leak_retries: int,
    use_judge: bool,
    judge_provider: str,
    judge_model: str,
    judge_retries: int,
    judge_correctness: bool,
    min_game_turns: int,
) -> dict[str, int]:
    """Sync runner: original `process_match` flow over the given records."""
    sem = asyncio.Semaphore(concurrency)

    async def worker(rec: dict[str, Any]) -> dict[str, int]:
        # Backstop: catch any unhandled exception so one bad match doesn't
        # take down the gather. process_match already handles the common
        # failure modes internally (skipped_ambiguous, skipped_threat_matrix_error,
        # fallback_open_p1_bounds, skipped_llm_error). Anything else
        # surfaces here as a `skipped_uncaught_exception` counter.
        async with sem:
            try:
                return await process_match(
                    rec,
                    output_path=output_path,
                    calc_base_url=calc_base_url,
                    teacher=teacher,
                    aiohttp_session=aiohttp_session,
                    file_lock=file_lock,
                    format_id=format_id,
                    seen_keys=seen_keys,
                    dry_run=dry_run,
                    model=model,
                    leak_retries=leak_retries,
                    use_judge=use_judge,
                    judge_client=judge_client,
                    judge_provider=judge_provider,
                    judge_model=judge_model,
                    judge_retries=judge_retries,
                    judge_correctness=judge_correctness,
                    min_game_turns=min_game_turns,
                )
            except Exception as e:
                _log_error(
                    f"[{rec.get('match_id', '?')}] UNCAUGHT exception in process_match: "
                    f"{type(e).__name__}: {e}; skipping match"
                )
                return {"skipped_uncaught_exception": 1}

    results = await tqdm.gather(
        *(worker(r) for r in records), desc="matches[sync]", unit="match",
    )
    totals: dict[str, int] = defaultdict(int)
    for r in results:
        if not isinstance(r, dict):
            continue
        for k, v in r.items():
            totals[k] += v
    return dict(totals)


async def _run_batch(
    records: list[dict[str, Any]],
    *,
    output_path: Path,
    calc_base_url: str,
    format_id: str,
    teacher: TeacherProvider | None,  # used by judge sync-fallback retries
    judge_client: Any,
    aiohttp_session: aiohttp.ClientSession,
    file_lock: asyncio.Lock,
    seen_keys: set[tuple[str, int, int]],
    leak_retries: int,
    use_judge: bool,
    judge_provider: str,
    judge_model: str,
    judge_retries: int,
    judge_correctness: bool,
    state_dir: Path,
    poll_interval_seconds: float,
    max_cycle_wait_seconds: float,
    resume: bool,
    model: str | None,
    min_game_turns: int,
) -> dict[str, int]:
    """Batch runner: drive every match's turns through one batch cycle
    per tool-loop iteration via the OpenAI Batch API.

    Per Plan v4, the batch path is OpenAI-only in v1. The state machine
    lives in `batch_runner.run_batch_for_matches`; this function builds
    the provider + threads the right arguments through.
    """
    from batch_runner import run_batch_for_matches  # local import: optional dep

    batch_provider = BatchOpenAIProvider(model=model)
    click.echo(
        f"[batch] state_dir={state_dir}  resume={resume}  "
        f"poll={poll_interval_seconds}s  max_wait={max_cycle_wait_seconds}s  "
        f"provider_model={batch_provider.model!r}"
    )

    stats = await run_batch_for_matches(
        records,
        batch_provider=batch_provider,
        aiohttp_session=aiohttp_session,
        calc_base_url=calc_base_url,
        state_dir=state_dir,
        output_path=output_path,
        file_lock=file_lock,
        format_id=format_id,
        seen_keys=seen_keys,
        poll_interval_seconds=poll_interval_seconds,
        max_cycle_wait_seconds=max_cycle_wait_seconds,
        leak_retries=leak_retries,
        use_judge=use_judge,
        judge_client=judge_client,
        judge_provider=judge_provider,
        judge_model=judge_model,
        judge_retries=judge_retries,
        judge_correctness=judge_correctness,
        teacher_for_judge_retry=teacher,
        resume=resume,
        min_game_turns=min_game_turns,
    )
    return stats


def _hybrid_gate_passed(
    sync_stats: dict[str, int],
    *,
    min_match_rate: float,
    max_leak_rate: float,
) -> tuple[bool, str]:
    """Decide whether the hybrid sync portion's quality justifies the
    batch portion. Compares match-rate and leak-rate against thresholds.

    For now we approximate match-rate using `written` over total
    attempted (excluding ambiguous / threat-matrix-error skips, which
    aren't quality signals). Leak-rate is dropped-rows / attempted.
    """
    written = sync_stats.get("written", 0)
    skipped_persistent_leak = sync_stats.get("skipped_persistent_leak", 0)
    skipped_persistent_judge = sync_stats.get("skipped_persistent_judge_fail", 0)
    attempted = written + skipped_persistent_leak + skipped_persistent_judge
    if attempted == 0:
        return False, "hybrid gate: no usable turns attempted in sync portion"
    match_rate = written / attempted
    leak_rate = (skipped_persistent_leak + skipped_persistent_judge) / attempted
    msg = (
        f"hybrid gate: written={written} drop_leak={skipped_persistent_leak} "
        f"drop_judge={skipped_persistent_judge} "
        f"→ match_rate={match_rate:.3f} (≥{min_match_rate}), "
        f"leak_rate={leak_rate:.3f} (≤{max_leak_rate})"
    )
    if match_rate < min_match_rate:
        return False, msg + " — HALT (match_rate below threshold)"
    if leak_rate > max_leak_rate:
        return False, msg + " — HALT (leak_rate above threshold)"
    return True, msg + " — OK, proceeding to batch portion"


async def _run_hybrid(
    records: list[dict[str, Any]],
    hybrid_sync_n: int,
    hybrid_min_match_rate: float,
    hybrid_max_leak_rate: float,
    *,
    output_path: Path,
    calc_base_url: str,
    format_id: str,
    concurrency: int,
    dry_run: bool,
    model: str | None,
    teacher: TeacherProvider | None,
    judge_client: Any,
    aiohttp_session: aiohttp.ClientSession,
    file_lock: asyncio.Lock,
    seen_keys: set[tuple[str, int, int]],
    leak_retries: int,
    use_judge: bool,
    judge_provider: str,
    judge_model: str,
    judge_retries: int,
    judge_correctness: bool,
    state_dir: Path,
    poll_interval_seconds: float,
    max_cycle_wait_seconds: float,
    resume: bool,
    min_game_turns: int,
) -> dict[str, int]:
    """Sync-then-batch quality-gated hybrid.

    1. Run the first `hybrid_sync_n` matches through `_run_sync`.
    2. Evaluate the gate (match-rate ≥ min, leak-rate ≤ max).
    3. If passed: run the remaining matches through `_run_batch`. If
       failed: log the failure and abort before submitting any batch
       upload — better to surface a quality regression than silently
       commit thousands of dollars to a busted run.
    """
    head = records[:hybrid_sync_n]
    tail = records[hybrid_sync_n:]
    click.echo(
        f"[hybrid] sync gate: first {len(head)} matches via sync; "
        f"remaining {len(tail)} via batch if gate passes."
    )
    sync_stats = await _run_sync(
        head,
        output_path=output_path,
        calc_base_url=calc_base_url,
        format_id=format_id,
        concurrency=concurrency,
        dry_run=dry_run,
        model=model,
        teacher=teacher,
        judge_client=judge_client,
        aiohttp_session=aiohttp_session,
        file_lock=file_lock,
        seen_keys=seen_keys,
        leak_retries=leak_retries,
        use_judge=use_judge,
        judge_provider=judge_provider,
        judge_model=judge_model,
        judge_retries=judge_retries,
        judge_correctness=judge_correctness,
        min_game_turns=min_game_turns,
    )
    passed, gate_msg = _hybrid_gate_passed(
        sync_stats,
        min_match_rate=hybrid_min_match_rate,
        max_leak_rate=hybrid_max_leak_rate,
    )
    click.echo(f"[hybrid] {gate_msg}")
    if not passed:
        sync_stats["hybrid_gate_failed"] = 1
        return sync_stats
    if not tail:
        return sync_stats

    batch_stats = await _run_batch(
        tail,
        output_path=output_path,
        calc_base_url=calc_base_url,
        format_id=format_id,
        teacher=teacher,
        judge_client=judge_client,
        aiohttp_session=aiohttp_session,
        file_lock=file_lock,
        seen_keys=seen_keys,
        leak_retries=leak_retries,
        use_judge=use_judge,
        judge_provider=judge_provider,
        judge_model=judge_model,
        judge_retries=judge_retries,
        judge_correctness=judge_correctness,
        state_dir=state_dir,
        poll_interval_seconds=poll_interval_seconds,
        max_cycle_wait_seconds=max_cycle_wait_seconds,
        resume=resume,
        model=model,
        min_game_turns=min_game_turns,
    )
    # Merge.
    combined: dict[str, int] = defaultdict(int)
    for source in (sync_stats, batch_stats):
        for k, v in source.items():
            combined[k] += v
    return dict(combined)


async def run(
    *,
    input_path: Path,
    output_path: Path,
    calc_base_url: str,
    format_id: str,
    limit: int | None,
    concurrency: int,
    dry_run: bool,
    model: str | None,
    provider: str,
    leak_retries: int = PRODUCTION_LEAK_RETRIES,
    use_judge: bool = True,
    judge_provider: str = DEFAULT_JUDGE_PROVIDER,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_retries: int = DEFAULT_JUDGE_RETRIES,
    judge_correctness: bool = True,
    mode: str = "sync",
    state_dir: Path = DEFAULT_BATCH_STATE_DIR,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    max_cycle_wait_seconds: float = DEFAULT_MAX_CYCLE_WAIT_SECONDS,
    resume: bool = False,
    hybrid_sync_n: int = DEFAULT_HYBRID_SYNC_N,
    hybrid_min_match_rate: float = DEFAULT_HYBRID_MIN_MATCH_RATE,
    hybrid_max_leak_rate: float = DEFAULT_HYBRID_MAX_LEAK_RATE,
    min_game_turns: int = DEFAULT_MIN_GAME_TURNS,
    exclude_match_ids: set[str] | None = None,
) -> None:
    """Top-level dispatcher. Picks `_run_sync` / `_run_batch` / `_run_hybrid`
    based on `mode`."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timeout = aiohttp.ClientTimeout(total=180, connect=10)

    teacher: TeacherProvider | None = None
    if not dry_run:
        teacher = _build_teacher(provider, model)

    judge_client = _build_judge_client(use_judge, dry_run, judge_provider)

    if mode == "batch" and provider != "openai":
        raise click.ClickException(
            f"--mode batch requires --provider openai in v1 (got {provider}). "
            f"Vertex Batch API adapter is a future workstream — see "
            f"notes/TODO.md for the deferred-follow-ups list. For "
            f"production with Gemini, use --mode sync; with $100K GCP "
            f"credits available the unit-cost saving from batch isn't a "
            f"hard constraint."
        )

    async with aiohttp.ClientSession(timeout=timeout) as session:
        await _check_calc_health(session, calc_base_url)

        records = _read_match_records(input_path)
        click.echo(f"loaded {len(records)} match records from {input_path}")

        if exclude_match_ids:
            before = len(records)
            records = [r for r in records if r.get("match_id") not in exclude_match_ids]
            click.echo(
                f"  --exclude-match-file: dropped {before - len(records)} holdout "
                f"matches ({len(records)} remain for training synthesis)"
            )

        seen_keys = load_seen_keys(output_path)
        if seen_keys:
            click.echo(f"  resuming: {len(seen_keys)} (match, game, turn) keys already in {output_path.name}")

        if limit is not None:
            records = records[:limit]
            click.echo(f"  --limit {limit}: processing first {len(records)} matches")

        file_lock = asyncio.Lock()

        if mode == "sync":
            totals = await _run_sync(
                records,
                output_path=output_path,
                calc_base_url=calc_base_url,
                format_id=format_id,
                concurrency=concurrency,
                dry_run=dry_run,
                model=model,
                teacher=teacher,
                judge_client=judge_client,
                aiohttp_session=session,
                file_lock=file_lock,
                seen_keys=seen_keys,
                leak_retries=leak_retries,
                use_judge=use_judge,
                judge_provider=judge_provider,
                judge_model=judge_model,
                judge_retries=judge_retries,
                judge_correctness=judge_correctness,
                min_game_turns=min_game_turns,
            )
        elif mode == "batch":
            if dry_run:
                # Batch dry-run: skip API submission, just prep + state-file.
                click.echo("[batch] --dry-run: writing state files only, no batch submission")
                from batch_runner import _prepare_match_turns, BatchWorkItem, _save_match_state
                totals = defaultdict(int)
                for rec in records:
                    preps, prep_stats = await _prepare_match_turns(
                        rec, format_id=format_id, calc_base_url=calc_base_url,
                        aiohttp_session=session, seen_keys=seen_keys,
                    )
                    for k, v in prep_stats.items():
                        totals[k] += v
                    items = [BatchWorkItem(
                        match_id=p.match_id, game_idx=p.game_idx, turn=p.turn,
                        format_id=p.format_id, api_messages=list(p.api_messages),
                        system_prompt=p.system_prompt, user_prompt=p.user_prompt,
                        human_action=p.human_action, status="pending",
                    ) for p in preps]
                    _save_match_state(state_dir, rec.get("match_id", "unknown"), format_id, items, [])
                    totals["state_files_written"] += 1
                totals = dict(totals)
            else:
                totals = await _run_batch(
                    records,
                    output_path=output_path,
                    calc_base_url=calc_base_url,
                    format_id=format_id,
                    teacher=teacher,
                    judge_client=judge_client,
                    aiohttp_session=session,
                    file_lock=file_lock,
                    seen_keys=seen_keys,
                    leak_retries=leak_retries,
                    use_judge=use_judge,
                    judge_provider=judge_provider,
                    judge_model=judge_model,
                    judge_retries=judge_retries,
                    judge_correctness=judge_correctness,
                    state_dir=state_dir,
                    poll_interval_seconds=poll_interval_seconds,
                    max_cycle_wait_seconds=max_cycle_wait_seconds,
                    resume=resume,
                    model=model,
                    min_game_turns=min_game_turns,
                )
        elif mode == "hybrid":
            totals = await _run_hybrid(
                records,
                hybrid_sync_n=hybrid_sync_n,
                hybrid_min_match_rate=hybrid_min_match_rate,
                hybrid_max_leak_rate=hybrid_max_leak_rate,
                output_path=output_path,
                calc_base_url=calc_base_url,
                format_id=format_id,
                concurrency=concurrency,
                dry_run=dry_run,
                model=model,
                teacher=teacher,
                judge_client=judge_client,
                aiohttp_session=session,
                file_lock=file_lock,
                seen_keys=seen_keys,
                leak_retries=leak_retries,
                use_judge=use_judge,
                judge_provider=judge_provider,
                judge_model=judge_model,
                judge_retries=judge_retries,
                judge_correctness=judge_correctness,
                state_dir=state_dir,
                poll_interval_seconds=poll_interval_seconds,
                max_cycle_wait_seconds=max_cycle_wait_seconds,
                resume=resume,
                min_game_turns=min_game_turns,
            )
        else:
            raise click.ClickException(f"unknown --mode: {mode}")

    click.echo("\n=== summary ===")
    for k in sorted(totals):
        if k == "judge_cost_micro_usd":
            click.echo(f"  judge_cost_usd: ${totals[k] / 1_000_000:.4f}")
            continue
        click.echo(f"  {k}: {totals[k]}")


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=str(DEFAULT_BO3_INPUT),
    show_default=True,
    help="Match-records JSONL from replay_parser (parsed_data/bo1.jsonl or bo3.jsonl).",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=str(DEFAULT_OUTPUT),
    show_default=True,
)
@click.option("--calc-base-url", default=DEFAULT_CALC_BASE_URL, show_default=True)
@click.option(
    "--format-id",
    default=None,
    help="Format ID stamped on each SFT row (Bo1 vs Bo3). Auto-detected from input filename if omitted.",
)
@click.option("--limit", type=int, default=None, help="Process only the first N matches (test batch).")
@click.option("--concurrency", default=1, show_default=True,
              help="Max matches processed in parallel. Keep low (1-3) to respect provider rate limits.")
@click.option("--dry-run", is_flag=True, help="Skip the LLM call; emit a placeholder assistant message.")
@click.option(
    "--provider",
    type=click.Choice(["openai", "anthropic", "google"]),
    default="google",
    show_default=True,
    help="Which teacher LLM backend to use. Production default is Google "
         "(Gemini) — bake-off-winning quality + GCP credit availability. "
         "OpenAI / Anthropic remain available for explicit selection.",
)
@click.option(
    "--model",
    default=None,
    help="Override the default model id for the chosen provider.",
)
@click.option(
    "--leak-retries", type=int, default=PRODUCTION_LEAK_RETRIES, show_default=True,
    help="Retries per turn when the teacher's CoT contains a ground-truth-leak phrase. "
         "0 = drop on first hit (smoke / measurement). Default in production: 3.",
)
@click.option(
    "--use-judge/--no-judge", default=True, show_default=True,
    help="Run the per-match model-judge validator (plan v4). Catches the long-tail "
         "meta-leaks that the regex filter misses. Adds ~$0.014 per match.",
)
@click.option(
    "--judge-provider",
    type=click.Choice(["openai", "google"]),
    default=DEFAULT_JUDGE_PROVIDER,
    show_default=True,
    help="Which LLM backend the judge uses. Default Google (matches the "
         "production teacher provider). Set to openai to use OpenAI for a "
         "cross-provider sanity-check.",
)
@click.option(
    "--judge-model", default=DEFAULT_JUDGE_MODEL, show_default=True,
    help="Model the judge calls. Default tracks the judge-provider's default "
         "(google → gemini-3.1-pro-preview; openai → gpt-5.5). Override with "
         "this flag or via the JUDGE_MODEL env var.",
)
@click.option(
    "--judge-retries", type=int, default=DEFAULT_JUDGE_RETRIES, show_default=True,
    help="Re-synthesis passes after the judge flags a turn. On exhaustion, drops only "
         "the flagged turns; the rest of the match commits cleanly.",
)
@click.option(
    "--judge-correctness/--no-judge-correctness", default=True, show_default=True,
    help="Also have the per-match judge fact-check each CoT's quantitative claims "
         "(damage / KO / speed) against the threat matrix + the teacher's own calc "
         "results, not just oracle-leak hygiene. Flagged turns are re-synthesized "
         "(not dropped), so good CoT isn't thrown away. Negligible added cost.",
)
@click.option(
    "--min-game-turns", type=int, default=DEFAULT_MIN_GAME_TURNS, show_default=True,
    help="Drop games whose snapshot count produces fewer than this many "
         "turn-pairs (= max possible SFT rows from the game). Default 2 "
         "removes ghost games + single-decision sweeps that lack meaningful "
         "history context. Set 0 to disable; set 4+ for richer-context-only.",
)
@click.option(
    "--exclude-match-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Hold these match-ids OUT of synthesis (the eval holdout). Accepts "
         "make_eval_split.py's eval_split.json (uses its `holdout` list) or a "
         "plain id list. Required for the pilot so the action-match eval has "
         "no train/test leakage.",
)
@click.option(
    "--mode",
    type=click.Choice(["sync", "batch", "hybrid"]),
    default="sync",
    show_default=True,
    help="Execution strategy. `sync` calls the API per-turn. `batch` uses the "
         "OpenAI Batch API (~50% cheaper, 24h SLA). `hybrid` runs the first "
         "--hybrid-sync-n matches sync as a quality gate then batches the rest.",
)
@click.option(
    "--state-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=str(DEFAULT_BATCH_STATE_DIR),
    show_default=True,
    help="Per-match resume state directory (batch / hybrid modes).",
)
@click.option(
    "--poll-interval-seconds", type=float,
    default=DEFAULT_POLL_INTERVAL_SECONDS, show_default=True,
    help="OpenAI Batch API status-poll cadence in batch / hybrid modes.",
)
@click.option(
    "--max-cycle-wait-seconds", type=float,
    default=DEFAULT_MAX_CYCLE_WAIT_SECONDS, show_default=True,
    help="Per-batch-cycle SLA. Batch API guarantees 24h (86400s); we hard-cap "
         "here to surface stuck batches.",
)
@click.option(
    "--resume", is_flag=True, default=False,
    help="Batch mode: re-use prior state files in --state-dir, picking up "
         "in-flight batches where they left off.",
)
@click.option(
    "--hybrid-sync-n", type=int,
    default=DEFAULT_HYBRID_SYNC_N, show_default=True,
    help="Hybrid mode: number of leading matches to run sync as a quality gate.",
)
@click.option(
    "--hybrid-min-match-rate", type=float,
    default=DEFAULT_HYBRID_MIN_MATCH_RATE, show_default=True,
    help="Hybrid mode halt-threshold: minimum written/attempted ratio to "
         "proceed to batch.",
)
@click.option(
    "--hybrid-max-leak-rate", type=float,
    default=DEFAULT_HYBRID_MAX_LEAK_RATE, show_default=True,
    help="Hybrid mode halt-threshold: maximum dropped/attempted ratio to "
         "proceed to batch.",
)
def cli(input_path, output_path, calc_base_url, format_id, limit, concurrency, dry_run,
        provider, model, leak_retries, use_judge, judge_provider, judge_model, judge_retries,
        judge_correctness, min_game_turns, exclude_match_file, mode, state_dir,
        poll_interval_seconds, max_cycle_wait_seconds, resume,
        hybrid_sync_n, hybrid_min_match_rate, hybrid_max_leak_rate):
    """Generate the SFT training JSONL from parsed replay data."""
    exclude_match_ids = _load_excluded_match_ids(exclude_match_file)
    resolved_format = _resolve_format_id(input_path, format_id)
    click.echo(
        f"using format_id={resolved_format}  dry_run={dry_run}  "
        f"provider={provider}  model={model or '(default)'}  leak_retries={leak_retries}  "
        f"judge={'on' if use_judge else 'off'}"
        + (f" ({judge_provider}/{judge_model}, retries={judge_retries})" if use_judge else "")
        + f"  mode={mode}"
        + (f" (sync_n={hybrid_sync_n})" if mode == "hybrid" else "")
        + f"  min_game_turns={min_game_turns}"
    )
    asyncio.run(
        run(
            input_path=input_path,
            output_path=output_path,
            calc_base_url=calc_base_url,
            format_id=resolved_format,
            limit=limit,
            concurrency=concurrency,
            dry_run=dry_run,
            model=model,
            provider=provider,
            leak_retries=leak_retries,
            use_judge=use_judge,
            judge_provider=judge_provider,
            judge_model=judge_model,
            judge_retries=judge_retries,
            judge_correctness=judge_correctness,
            mode=mode,
            state_dir=state_dir,
            poll_interval_seconds=poll_interval_seconds,
            max_cycle_wait_seconds=max_cycle_wait_seconds,
            resume=resume,
            hybrid_sync_n=hybrid_sync_n,
            hybrid_min_match_rate=hybrid_min_match_rate,
            hybrid_max_leak_rate=hybrid_max_leak_rate,
            min_game_turns=min_game_turns,
            exclude_match_ids=exclude_match_ids,
        )
    )


if __name__ == "__main__":
    cli()
