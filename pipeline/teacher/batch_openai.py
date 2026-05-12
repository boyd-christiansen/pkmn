"""OpenAI Batch API adapter for the teacher tool-loop (plan v4, W2).

This module is the SDK-side plumbing for batched synthesis. It does NOT
own the orchestration loop or the per-match state — that's master_pipeline.
The provider exposes the four operations the orchestrator needs:

  • `build_request(custom_id, api_messages, tool_choice) -> dict`
      Render one JSONL line for the batch upload.

  • `submit_batch(requests) -> batch_id`
      Upload the JSONL, create the batch, return its id.

  • `poll(batch_id) -> dict`
      Cheap status check; returns {"status", "request_counts"}.

  • `fetch_results(batch_id) -> dict[custom_id, response]`
      After completion, download and parse the output file.

Why split this from the orchestrator: the same state machine will later
swap in `BatchAnthropicProvider` / `BatchGoogleProvider` adapters with
provider-specific quirks (Anthropic Message Batches handles agentic
loops differently; Vertex has its own polling). Keeping the abstraction
narrow makes future provider work mechanical.

Architectural constraint (discovered in plan v4 exploration): OpenAI
Batch API can't span tool-loop iterations within a single request line.
So the orchestrator runs ONE batch cycle per tool-loop iteration, with
all in-flight turns at iter=K bundled into one batch upload. Calc
microservice calls happen synchronously on our side between cycles.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from .base import (
    CALCULATE_DAMAGE_TOOL,
    DEFAULT_MODEL,
    SUBMIT_DECISION_TOOL,
)

# Reasonable poll cadence: OpenAI Batch p50 latency is ~1-3h. Poll every
# minute by default so we don't spam the status endpoint but still notice
# fast completions. Callers can override via `--poll-interval-seconds`.
DEFAULT_POLL_INTERVAL_SECONDS = 60.0
# 24h SLA — OpenAI guarantees completion within this window. We hard-cap
# at this so a stuck batch doesn't hang the run indefinitely.
DEFAULT_MAX_CYCLE_WAIT_SECONDS = 24 * 60 * 60

# The Batch endpoint only accepts a single completion window value today.
_OPENAI_COMPLETION_WINDOW = "24h"
# Per-line cap in a batch upload; OpenAI's limit is 50K but we stay well
# under to give the per-cycle batches headroom and keep individual
# uploads fast.
MAX_REQUESTS_PER_BATCH = 10_000


@dataclass
class BatchPollStatus:
    """Snapshot of a batch's progress for a single poll tick.

    `status` follows the OpenAI lifecycle:
      validating → in_progress → finalizing → completed
                              ↘ failed | expired | cancelled
    The orchestrator treats anything not in {"completed"} as not-yet-ready,
    and anything in {"failed", "expired", "cancelled"} as terminal-bad.
    """
    status: str
    request_counts_total: int = 0
    request_counts_completed: int = 0
    request_counts_failed: int = 0
    output_file_id: str | None = None
    error_file_id: str | None = None
    raw: Any = None  # the SDK's full Batch object, for debugging


class BatchTeacherProvider(ABC):
    """Provider-agnostic interface the batch orchestrator drives.

    Implementations live in `teacher.batch_openai`, `teacher.batch_anthropic`
    (future), `teacher.batch_google` (future). The orchestrator never
    imports from a concrete adapter directly — it gets one passed in
    through `master_pipeline._build_batch_teacher`.
    """

    name: str = "abstract"
    model: str = ""

    @abstractmethod
    def build_request(
        self,
        *,
        custom_id: str,
        api_messages: list[dict[str, Any]],
        tool_choice: Any,
    ) -> dict[str, Any]:
        ...

    @abstractmethod
    async def submit_batch(self, requests: list[dict[str, Any]]) -> str:
        ...

    @abstractmethod
    async def poll(self, batch_id: str) -> BatchPollStatus:
        ...

    @abstractmethod
    async def fetch_results(self, batch_id: str) -> dict[str, dict[str, Any]]:
        ...

    @abstractmethod
    async def cancel(self, batch_id: str) -> None:
        ...


class BatchOpenAIProvider(BatchTeacherProvider):
    """OpenAI Batch API adapter.

    The build_request payload mirrors what `OpenAIProvider._do_turn`
    sends synchronously — same tools, same `parallel_tool_calls=False`,
    same omitted `max_tokens` / `temperature` (gpt-5.5 family rejects
    those). Differences:
      • One line per turn-iteration (no internal loop).
      • `custom_id` carries (match, game, turn, iter) so the
        orchestrator can route the response back to the right WorkItem.
      • No `response_format` — tool calls are still the only output
        channel.
    """

    name = "openai-batch"

    def __init__(
        self,
        model: str | None = None,
        *,
        client: AsyncOpenAI | None = None,
    ):
        self.model = model or os.environ.get("TEACHER_MODEL_OPENAI", DEFAULT_MODEL)
        self.client = client or AsyncOpenAI()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_request(
        self,
        *,
        custom_id: str,
        api_messages: list[dict[str, Any]],
        tool_choice: Any,
    ) -> dict[str, Any]:
        """Render one JSONL line for the OpenAI Batch upload.

        Caller is responsible for already having injected the
        SYNTHESIS_GROUND_TRUTH_SUFFIX into `api_messages[1]` (the user
        message) — the batch provider doesn't see plain user prompts,
        only the ready-to-send conversation. This mirrors how the sync
        path handles ground-truth injection: the suffix lives in
        api_messages but gets stripped from saved_messages at commit.
        """
        body: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "tools": [CALCULATE_DAMAGE_TOOL, SUBMIT_DECISION_TOOL],
            "tool_choice": tool_choice,
            "parallel_tool_calls": False,
        }
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    async def submit_batch(self, requests: list[dict[str, Any]]) -> str:
        """Upload the JSONL and create a batch. Returns the batch_id."""
        if not requests:
            raise ValueError("submit_batch called with empty requests list")
        if len(requests) > MAX_REQUESTS_PER_BATCH:
            raise ValueError(
                f"batch size {len(requests)} exceeds cap "
                f"{MAX_REQUESTS_PER_BATCH}; split before submission"
            )

        # Serialize requests to a single JSONL byte buffer. We upload
        # directly from memory rather than touching the filesystem so
        # state stays in our per-match JSON files (not scattered tmp
        # uploads).
        buf = io.BytesIO()
        for req in requests:
            buf.write(json.dumps(req).encode("utf-8"))
            buf.write(b"\n")
        buf.seek(0)

        file_obj = await self.client.files.create(
            file=("batch.jsonl", buf, "application/jsonl"),
            purpose="batch",
        )

        batch = await self.client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window=_OPENAI_COMPLETION_WINDOW,
            metadata={"source": "pkmn-pipeline-v4"},
        )
        return batch.id

    # ------------------------------------------------------------------
    # Poll
    # ------------------------------------------------------------------

    async def poll(self, batch_id: str) -> BatchPollStatus:
        """One status-check tick. Cheap; safe to call every N seconds."""
        batch = await self.client.batches.retrieve(batch_id)
        counts = batch.request_counts
        return BatchPollStatus(
            status=batch.status or "unknown",
            request_counts_total=(counts.total if counts else 0),
            request_counts_completed=(counts.completed if counts else 0),
            request_counts_failed=(counts.failed if counts else 0),
            output_file_id=batch.output_file_id,
            error_file_id=batch.error_file_id,
            raw=batch,
        )

    async def poll_until_done(
        self,
        batch_id: str,
        *,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        max_wait_seconds: float = DEFAULT_MAX_CYCLE_WAIT_SECONDS,
    ) -> BatchPollStatus:
        """Convenience wrapper: poll on a cadence until terminal status.

        Returns the final status. Terminal statuses are:
          completed  → orchestrator should fetch_results
          failed     → orchestrator marks all items in this batch failed
          expired    → same; 24h SLA breached
          cancelled  → manual cancel; orchestrator decides
        """
        t0 = time.monotonic()
        while True:
            status = await self.poll(batch_id)
            if status.status in ("completed", "failed", "expired", "cancelled"):
                return status
            if time.monotonic() - t0 > max_wait_seconds:
                # Don't return a fake "expired" — surface a clear timeout
                # so the orchestrator can log it distinctly from a
                # real OpenAI-side expiration.
                raise TimeoutError(
                    f"batch {batch_id} still {status.status} after "
                    f"{max_wait_seconds}s; client-side timeout"
                )
            await asyncio.sleep(poll_interval_seconds)

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------

    async def fetch_results(self, batch_id: str) -> dict[str, dict[str, Any]]:
        """Pull the output file and return {custom_id: response_dict}.

        Each value mirrors a sync `chat.completions.create()` response
        body — `choices[0].message`, `usage`, etc. — so the orchestrator's
        response-handling code can stay symmetric with the sync path.

        Failed lines (when `error` is present) are passed through with
        the error attached; the orchestrator marks them as failed work
        items rather than crashing.
        """
        status = await self.poll(batch_id)
        if status.status != "completed":
            raise RuntimeError(
                f"fetch_results called on batch {batch_id} with status "
                f"{status.status!r}; expected 'completed'"
            )
        if not status.output_file_id:
            raise RuntimeError(f"batch {batch_id} completed without output_file_id")

        out: dict[str, dict[str, Any]] = {}
        content = await self.client.files.content(status.output_file_id)
        # SDK gives us an httpx Response-like wrapper; iter_lines() is
        # not async, so we read the bytes and split.
        raw_text = content.text if hasattr(content, "text") else content.read().decode("utf-8")
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = rec.get("custom_id")
            if not cid:
                continue
            out[cid] = rec
        return out

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    async def cancel(self, batch_id: str) -> None:
        """Best-effort cancel — batches that have already finalized can't
        be cancelled, but we eat the SDK error rather than fail the run."""
        try:
            await self.client.batches.cancel(batch_id)
        except Exception:  # noqa: BLE001 — cancel is best-effort
            pass


__all__ = [
    "BatchPollStatus",
    "BatchTeacherProvider",
    "BatchOpenAIProvider",
    "DEFAULT_POLL_INTERVAL_SECONDS",
    "DEFAULT_MAX_CYCLE_WAIT_SECONDS",
    "MAX_REQUESTS_PER_BATCH",
]
