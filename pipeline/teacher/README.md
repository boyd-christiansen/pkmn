# teacher

Provider-agnostic teacher LLM sub-package. The orchestrator
(`master_pipeline.py`) and the batch runner (`batch_runner.py`) both
import from `teacher`; all SDK-specific code — OpenAI / Anthropic /
Google, plus the OpenAI Batch API adapter and the model-judge
validator — lives here.

This README is the deep-dive. For pipeline-level context see
[`../README.md`](../README.md); for project-wide context see
[`../../CLAUDE.md`](../../CLAUDE.md); for what's *not* yet built see
[`../../notes/TODO.md`](../../notes/TODO.md).

## Layout

```
teacher/
├── __init__.py         # public surface — re-exports everything below
├── base.py             # TeacherProvider ABC, tool schemas, system prompts,
│                       #   detect_oracle_leak + extract_pre_tool_thought,
│                       #   cost-per-million-tokens table
├── openai.py           # OpenAIProvider — production default
├── anthropic.py        # AnthropicProvider
├── google.py           # GoogleProvider
├── judge.py            # Plan v4: match-level CoT hygiene validator
└── batch_openai.py     # Plan v4: BatchTeacherProvider ABC + BatchOpenAIProvider
```

Six files. The split: `base.py` owns the contract and the prompts;
each adapter owns one SDK's tool-call format; `judge.py` and
`batch_openai.py` are the two Plan v4 additions that other sub-packages
don't need to know exist.

## Architecture principle

The orchestrator never imports a concrete adapter directly. It does:

```python
from teacher import TeacherProvider, OpenAIProvider, judge_match_cots
```

That's the entire public surface. Adapter selection happens in one
place (`master_pipeline._build_teacher`) and the rest of the pipeline
speaks only `TeacherProvider`.

## The TeacherProvider contract

`teacher/base.py` defines:

```python
class TeacherProvider(ABC):
    name: str = "abstract"
    model: str = ""

    @abstractmethod
    async def synthesize_turn(
        self,
        system_prompt: str,
        user_prompt: str,
        human_action: dict[str, Any],
        *,
        calc_url: str = DEFAULT_CALC_URL,
        aiohttp_session: aiohttp.ClientSession | None = None,
    ) -> ProviderResult: ...
```

`ProviderResult` carries everything the orchestrator needs for one turn:

| Field | Meaning |
|---|---|
| `messages: list[dict] \| None` | OpenAI fine-tuning format conversation, ground-truth suffix already stripped. `None` on error. |
| `iterations: int` | Tool-loop iterations the model ran. |
| `calc_calls: int` | How many `calculate_damage` invocations the model issued. |
| `input_tokens` / `output_tokens` | For cost accounting. |
| `cost_usd: float` | Computed via the `PRICE_PER_M_TOKENS` table in `base.py`. |
| `elapsed_seconds: float` | Wall-clock for this turn. |
| `error: str \| None` | Non-`None` if synthesis failed; orchestrator skips the row. |

### Two tools, one output channel

The model can only emit tool calls; there is no `response_format`
fallback. The two tools are defined in `base.py` as
`CALCULATE_DAMAGE_TOOL` and `SUBMIT_DECISION_TOOL`:

- `calculate_damage` — JSON-schema mirror of `/calc`'s request body.
  Used for hypotheticals the threat matrix doesn't already cover
  (switch-ins, backline matchups, future-state, Tera predictions).
  The matrix already enumerates every active-vs-active damage cell
  for the current turn, so re-calc'ing those is forbidden by the
  rewritten Tool Rule.
- `submit_decision` — exactly one call per turn, the model's commit.
  Arguments are `{pre_tool_thought: str, action: {slot_1, slot_2}}`.
  Each slot is `{action_type: "move"|"switch"|"pass", move?, target?,
  tera?, switch_to?}`. The strict JSON schema lives on this tool's
  `parameters` — that's how we keep the model from bypassing
  `calculate_damage` by emitting a direct structured response.

### Tool loop

`MAX_TOOL_ITERATIONS = 10` (soft ceiling).
`MAX_CALC_CALLS_BEFORE_FORCE_SUBMIT = 5` (upper cap; past 5 calc
calls, `tool_choice` is forced to `submit_decision`).

Two-level timeouts via `asyncio.wait_for`:
- `PER_CALL_TIMEOUT = 120s` — single API call ceiling.
- `PER_TURN_TIMEOUT = 300s` — whole tool loop ceiling, no matter
  how many iterations.

### System prompt — six rules

Identical structure across all three providers. Format-specific only
on rule 1 (Bo1 Masking Rule vs. Bo3 OTS Rule). All wording is
**present-tense** — the model is trained as if playing live, not
reviewing a recording.

1. **Masking Rule** (Bo1 CTS) or **OTS Rule** (Bo3) — what's known
   about the team sheet.
2. **Tool Rule** *(rewritten Plan v3)* — `calculate_damage` for
   hypotheticals only; no per-turn minimum. The model may
   `submit_decision` on its first iter if the matrix suffices.
3. **Threat-Matrix Rule** — Absolute (provable) and, when not
   contradicted, Probable (meta) per line. Off-meta lines tagged.
4. **Spread Rule** — your spreads may be exact (deploy time) or
   ranges (training time). Worst-case for survival, best-case for
   offense.
5. **Alternatives Rule** *(known limitation)* — briefly evaluate 1-2
   plausible alternatives before committing.
   [`TODO(rlhf-followup)`](../../notes/TODO.md#todorlhf-followup--pipelineteacherbasepy248):
   prompt-driven; should become minimax / MCTS distillation.
6. **Output Rule** — commit via `submit_decision`.

### Ground-truth handling

During the API call, the user message has the human's play appended
as a `=== TRAINING-MODE TARGET ===` suffix (rewritten Plan v3 from
`=== EXPERT'S DECISION ===` — the prior phrasing was leaking into
CoTs). The returned `messages` have that suffix **stripped** — saved
SFT examples show only board state + threat matrix. The student
model never sees the cheat.

### Leak filter

Two stages, both live in `teacher/base.py` and called by the
orchestrator on every saved row:

- `detect_oracle_leak(messages) -> str | None` — fast regex on the
  saved `pre_tool_thought`. Matches "oracle", "ground-truth", "the
  target {is,says,action,field}", "training {mode,section,target,
  example}", etc. Tightened May 2026 after the bake-off found
  Anthropic produced "the target action" / "training section" in 32%
  of saved rows.
- `extract_pre_tool_thought(messages) -> str | None` — shared
  parser; lives here so both the regex and the model judge can use
  the same extraction.

The model judge (`teacher/judge.py`) is the second-line filter; see
below.

## Adapters

| File | Default model | Provider-specific notes |
|---|---|---|
| `openai.py` | `gpt-5.5` | Chat Completions API. `tool_choice="required"` normally; `{type: function, function: {name: "submit_decision"}}` past the calc cap. **Production default — bake-off winner.** |
| `anthropic.py` | `claude-sonnet-4-6` | Anthropic Messages API. `tool_choice={type:"any"}` normally; `{type:"tool", name:"submit_decision"}` past the cap. Bake-off: 32% near-miss meta-leak rate; not used in production. |
| `google.py` | `gemini-3.1-pro-preview` | google-genai SDK. `tool_config={mode:"ANY"}` normally; restricted to `["submit_decision"]` past the cap. Adapter has its own JSON-Schema-2020-12 → OpenAPI-3.0 translator (`_translate_schema`) because Gemini's schema dialect doesn't accept JSON Schema directly. |

Per-adapter env-var overrides for the model id:
`TEACHER_MODEL_OPENAI`, `TEACHER_MODEL_ANTHROPIC`, `TEACHER_MODEL_GOOGLE`.

The cost table `PRICE_PER_M_TOKENS` in `base.py` carries
(input, output) $/M tokens for every model id we reference. Confirm
against the provider's pricing page before scaling to the full
corpus.

## Bake-off result (May 2026)

| Provider | Match% | Leak rate | $/row | Avg CoT | Status |
|---|---|---|---|---|---|
| **OpenAI gpt-5.5** | 100.0% | 0% | $0.07 | 902 chars | **Production default.** Concise, consistent. |
| Google gemini-3.1-pro | 100.0% | 0% | $0.04 | 1027 chars | Cheapest. Slower wall-clock. Viable cost-sensitive backup. |
| Anthropic claude-sonnet-4-6 | 61.3% | 32% near-miss | $0.09 | 4004 chars | Verbose; systematic meta-references in CoT. Not used in production. |

Anthropic's near-miss leak pattern — "the target action", "the
training section" — was the main motivator for Plan v4's judge
layer.

## Judge (`teacher/judge.py`, Plan v4)

Match-level CoT hygiene validator. After every match's turns are
synthesized (sync or batch), the orchestrator submits all of the
match's `pre_tool_thought` CoTs to OpenAI in **one** call and the
judge returns turn indices to retry.

### API

```python
@dataclass
class JudgeResult:
    flagged_turn_indices: list[int]
    reasons: dict[int, str]          # turn_idx → short quote from CoT
    raw_response: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    elapsed_seconds: float
    error: str | None                # fail-open: write all rows on any error

async def judge_match_cots(
    turn_records: list[dict],
    *,
    client: AsyncOpenAI,
    model: str = DEFAULT_JUDGE_MODEL,
    timeout: float = 60.0,
) -> JudgeResult: ...
```

`turn_records[i]` is `{turn_idx, match_id, game_idx, turn,
pre_tool_thought}`. The judge's response is structured JSON via
OpenAI's `response_format=json_schema` — no partial parses, no
regex on the response.

### Why per-match, not per-row

Amortizes a fixed system prompt across N turns. One call for an
8-turn match costs ~$0.014 with `gpt-5.5` (~$0.0015 with
`gpt-5.5-mini` when access opens up) versus ~$0.04 if we judged each
row separately. The judge also sees cross-turn patterns — multiple
consecutive references to the training framing are a stronger
signal than each in isolation.

### Default model

`DEFAULT_JUDGE_MODEL = "gpt-5.5"`. Plan v4 spec'd `gpt-5.5-mini` for
the ~10× cost win, but the project account doesn't currently have
access. Set `JUDGE_MODEL=gpt-5.5-mini` in env once available; no
code change needed.

### Fail-open

On any client error (network, rate limit, malformed JSON), the
function returns an empty `flagged_turn_indices` and a non-`None`
`error`. The orchestrator writes all rows as if the judge passed.
Better to ship a few possibly-leaky rows than drop a whole match
for an infra hiccup; the regex filter is still the first line of
defense.

### Truncation policy

CoTs longer than 6000ch are truncated with a visible marker before
going into the judge prompt — keeps the judge cost predictable
regardless of how chatty the teacher got.

## Batch (`teacher/batch_openai.py`, Plan v4)

OpenAI Batch API plumbing. Used by `batch_runner.run_batch_for_matches`
to issue one batch cycle per tool-loop iteration across all in-flight
matches at once.

### Why a separate module

The Batch API has a constraint that doesn't apply to sync chat:
**each request line is independent — it can't span a multi-iteration
tool loop.** So our N-iter teacher tool loop becomes N sequential
batch cycles, with each cycle bundling all in-flight turns at the
same iter index. Calc microservice calls run synchronously on our
side between cycles.

### Interface

```python
class BatchTeacherProvider(ABC):
    @abstractmethod
    def build_request(self, *, custom_id, api_messages, tool_choice) -> dict: ...
    @abstractmethod
    async def submit_batch(self, requests: list[dict]) -> str: ...
    @abstractmethod
    async def poll(self, batch_id: str) -> BatchPollStatus: ...
    @abstractmethod
    async def fetch_results(self, batch_id: str) -> dict[str, dict]: ...
    @abstractmethod
    async def cancel(self, batch_id: str) -> None: ...

class BatchOpenAIProvider(BatchTeacherProvider): ...
```

`build_request` renders one JSONL line for the upload, mirroring
exactly what `OpenAIProvider._do_turn` sends synchronously (same
tools, `parallel_tool_calls=False`, omitted `max_tokens` /
`temperature` because gpt-5.5 rejects those).

`custom_id` encodes `{match_id}::g{game}::t{turn}::iter{cycle}` so
`batch_runner.run_batch_for_matches` can route responses back to
the right `BatchWorkItem`.

`poll_until_done` is a convenience wrapper on `poll` that exits when
the batch reaches `completed | failed | expired | cancelled`.

### Anthropic / Google adapters

Not implemented in v1. Logged in
[`../../notes/TODO.md`](../../notes/TODO.md#3-anthropic--google-batch-adapters)
— the ABC abstraction is in place and siblings should slot in
mechanically.

## Imports outside this package

What the rest of the pipeline does with this package:

- `master_pipeline.py` imports `TeacherProvider`, `OpenAIProvider`,
  `AnthropicProvider`, `GoogleProvider`, `BatchOpenAIProvider`,
  `detect_oracle_leak`, `extract_pre_tool_thought`,
  `judge_match_cots`, `render_system_prompt`,
  `render_system_prompt_bo3`, several constants.
- `batch_runner.py` imports `BatchOpenAIProvider`, `BatchPollStatus`,
  `detect_oracle_leak`, `extract_pre_tool_thought`,
  `judge_match_cots`, `MAX_CALC_CALLS_BEFORE_FORCE_SUBMIT`,
  `MAX_TOOL_ITERATIONS`, etc.
- `bakeoff.py` imports `OpenAIProvider`, `AnthropicProvider`,
  `GoogleProvider`, `detect_oracle_leak`, `extract_pre_tool_thought`,
  `DEFAULT_LEAK_RETRIES`, `PRODUCTION_LEAK_RETRIES`.

Everything goes through `teacher/__init__.py`. Nothing imports
adapters directly.
