# TODO — master tracker

The single source of truth for everything not yet shipped. CLAUDE.md
and the per-component READMEs point here rather than maintain their
own (drifting) lists. **When adding a new TODO anywhere in the
project, add it here too** — including the file:line for `# TODO(...)`
markers in source.

The shipped state is documented in [CLAUDE.md](../CLAUDE.md) and the
[walkthrough](pipeline_walkthrough.md). Plan files for completed
workstreams live in `~/.claude/plans/` and aren't tracked here.

---

## Active workstreams (next up)

### 1. Real corpus run on hybrid mode

`master_pipeline.py --mode hybrid --hybrid-sync-n 50` against the full
~13K parsed-match corpus. The first 50 matches run sync as a quality
gate; if `match_rate ≥ 0.95` and `leak_rate ≤ 0.02`, the remaining
~13K go through OpenAI Batch (~50% off). Estimated cost: ~$910 vs.
~$1.5K full-sync. Estimated wall-clock: hours-to-days depending on
real Batch p50 latency.

Pre-flight checks:
- Calc microservice running (`cd calc_microservice && npm run dev`).
- Canonical priors bootstrapped for both formats.
- `parsed_data/{bo1,bo3}.jsonl` exists.
- `OPENAI_API_KEY` set (judge always uses OpenAI; teacher does too in
  batch mode).

Spot-check after the sync gate completes — inspect a few rows via the
inspector before letting the batch portion fire.

### 2. Holdout eval set

Withhold ~500 matches (random sample stratified by format) from the
SFT generation run. After the trained model lands, measure
**action-match rate** of model plays vs. the human's actual play on
the holdout. Probably want to also score CoT quality on a smaller
sub-sample.

This is the canonical sanity check. Without it we have no honest
read on whether the SFT pipeline actually teaches anything beyond
its own structure.

### 3. Anthropic / Google batch adapters

v1 of `--mode batch` is OpenAI-only. The abstraction
(`BatchTeacherProvider` ABC in `pipeline/teacher/batch_openai.py`) is
already in place; siblings `batch_anthropic.py` (Message Batches) and
`batch_google.py` (Vertex batch prediction) would slot in mechanically.

Build when those providers re-enter production rotation. Anthropic
Message Batches reportedly supports full agent loops in one batch
line (unlike OpenAI's stateless-line model); if that's confirmed it
would simplify the state machine significantly for that provider.

---

## Code-level TODOs (`# TODO(...)` markers)

Three open markers in the Python source.

### `TODO(rlhf-followup)` — pipeline/teacher/base.py:248

Replace the prompt-driven Alternatives Rule (rule 5 of the system
prompt) with minimax / Monte Carlo distillation. The current teacher
cherry-picks weak alternatives because it already knows the answer
from the `=== TRAINING-MODE TARGET ===` block; a proper search step
would surface alternatives that *genuinely competed* with the chosen
play, making the rationalization meaningful.

Architecture sketch: a separate "alternatives engine" runs a shallow
search over the action space (filtered by the threat matrix); the
teacher then articulates *why* the chosen play beat each of the
top-K alternatives. Bigger workstream — likely a separate sibling of
`master_pipeline.py` similar to `batch_runner.py`.

### `TODO(token-efficient-series-summary)` — pipeline/prompt_formatting.py:582

Learned or rule-based summarizer for `=== SERIES STATE ===`. Today
the function inlines the full prior-game turn-by-turn rollup
verbatim — high fidelity but consumes a lot of attention as Bo3
games stack up.

Goal: distill "what mattered for THIS turn's decision" rather than
"everything that happened." Could start rule-based (key events:
faints, Teras, choice-locks, weather changes); upgrade to a learned
summarizer if the heuristic falls short.

### `TODO(canonical-prior-substitution)` — pipeline/prompt_formatting.py:134

Substitute the canonical Smogon spread for fully-open stats in `=== YOUR
SPREADS ===` so the block looks the same whether observation was
sparse or rich. Masks the implicit leak Plan v3 accepted as risk:
mons that never took damage render as `(no observations yet)`, which
a student model could learn means "this mon was never hit this
match."

Implementation is small (modify `format_p1_known_spreads_block` to
fall through to `canonical_priors.get_probable_spread` when bounds
are fully open) — held back only by judgment on whether masking with
slightly-fictional spreads is worse than the implicit leak.

---

## Plan v4 follow-ups (deferred from May 2026)

Items explicitly marked "out of scope" in the plan-v4 file but worth
tracking.

### Real-time CoT-quality dashboard

Show judge pass-rate, leak-retry count, and persistent-judge-fail
rate over time. Trivial follow-up once the counters land (they
already do, via `stats["judge_*"]` in `master_pipeline.py`). Most
natural format: a small Streamlit / Plotly dashboard that reads the
final `=== summary ===` lines from previous runs' logs.

### Removing the regex leak filter

The regex + judge are belt-and-suspenders today. Once the judge has
proven itself across 1000+ matches, the regex becomes redundant.
Keep both for now; revisit after the first real corpus run lands.

### Inline calc execution inside batch

Today calc microservice calls run synchronously on our side between
batch cycles, which adds an iter-count multiplier to the batch
latency. OpenAI's Assistants / Responses API could in principle run
the tools inline and return a final result per batch line, collapsing
multi-iter turns into single-batch-line submissions.

Revisit only if batch latency becomes the dominant cost / wall-clock
factor. Today (May 2026) we've measured <2min per cycle in smoke
tests — well under the documented SLA.

### Hybrid hot-swap mid-run

`--mode hybrid` today is a static "first N sync, remainder batch"
split. A nicer model would let the orchestrator decide mid-run when
to start submitting batches (e.g. once the rolling 50-match leak rate
stabilizes). Nice-to-have; static split works.

---

## Long-horizon workstreams

### Selection-model SFT corpus

Separate dataset for the team-preview 4-of-6 pick decision. Walks
the same parsed replays, extracts P1's brought set per game,
generates one selection example per game:

```
{p1_full_6, p2_full_6, format_meta} → {brought: [4 species]}
```

Trains as its own model — no tactical state, no tool calls. Deployed
*before* the turn-play teacher in the inference pipeline (pick 4,
then play). Lives in a sibling module to `master_pipeline.py`, not in
it.

### RLHF on top of SFT

Once SFT gives us a model that can think like a pro, RLHF gives it
the chance to outclass one. The SFT corpus we're building is the
floor, not the ceiling. Out of scope until SFT empirically lands a
strong floor — but worth keeping in view because RLHF data
requirements (preference pairs, reward model design) influence
upstream pipeline choices.

---

## Inspector enhancements

Direct from `inspector/README.md`'s "What it doesn't do (intentional
v1 scope)" list. None are urgent; surface them when an operator
actually hits the limitation.

- **Live LLM invocation** (a Postman-like simulator inside the
  inspector). Would let an operator tweak a prompt and re-run it
  without dropping back to `master_pipeline.py`.
- **Raw replay browser.** Today: `jq` / a JSON viewer for the 16K-file
  `data_scraper/data/replays/` tree.
- **Annotation / notes persistence.** Saving operator scribbles
  against a row would help when reviewing flagged turns from the
  judge.
- **Search / filtering.** Find rows by `(match_id, turn)`,
  judge_flagged status, model used, etc.

---

## Data sourcing (for if/when we need more)

Originally lived in `for_the_future.md`. Verbatim below — still
accurate; the scraper covers top-500 × per-user replays. If the
corpus turns out to be too small for the model to learn from,
these are the next levers.

### Other Pokémon Showdown endpoints

#### Verified in the initial scraper build
- `pokemonshowdown.com/ladder/{format_id}.json` — top 500 ladder users, Elo desc.
- `replay.pokemonshowdown.com/search.json?user=X&format=Y&page=N` — paginated
  replay search for one user (50/page).
- `replay.pokemonshowdown.com/{replay_id}.json` / `.log` — full battle replay
  (metadata + pipe-delimited log).

#### Known but not verified — `curl` first before building on them
- **`search.json?format=Y&page=N`** — format-only search (no `user=` filter).
  Returns *all* public replays for the format, newest first. This is the path
  past the top-500 ceiling.
- **`search.json?...&before={uploadtime}`** — timestamp cursor for deep
  pagination. PS may cap page-number paging around ~25 pages and force you to
  use `before` to go further back.
- **`pokemonshowdown.com/users/{userid}.json`** — public user profile: per-
  format ratings, registration date. Lets you filter by Elo without pulling
  the whole ladder.
- **`play.pokemonshowdown.com/data/pokedex.json`, `moves.json`, `items.json`,
  `abilities.json`, `learnsets.json`, `formats-data.json`** — canonical game
  data. Exactly the knowledge base to expose to the LLM as a tool-callable
  reference for damage math, legality checks, etc.
- **`smogon.com/stats/{YYYY-MM}/{format}-{cutoff}.json`** — monthly Smogon
  usage stats: top Pokémon, move/item/teammate distributions, win rates per
  Elo cutoff. Great for evaluation and team-building context. (Already used
  by `canonical_priors.py`.)

### Can we scrape *all* battles?

Two senses:

#### 1. All publicly saved replays — yes, mostly
Via format-only `search.json` paginated until empty (using `before={uploadtime}`
once page-number paging caps out).

**Hard limit:** only games where a player clicks "save replay" land on the
replay server — probably <5% of all ladder games. PS also ages out very old
replays for low-traffic formats, but active formats like `gen9vgc2026regi`
have effectively full history.

#### 2. All games actually played, including unsaved — only via WebSocket
PS exposes `wss://sim3.psim.us/showdown/websocket`. You connect, query the
room/battle list, and join active battles as a spectator. The protocol is
documented in `pokemon-showdown/PROTOCOL.md` on GitHub.

**Caveats:**
- Forward-only — can't recover history this way.
- Continuous service, not a one-shot scrape — needs to run 24/7.
- Stateful protocol (more complex than HTTP polling).
- Be polite: one connection, reasonable join rate. PS is generally permissive
  about read-only spectating but it's not unlimited.

### Recommended order if we need more data

1. **(Already built)** Top-500 ladder × per-user replay search.
2. **Format-only `search.json` paginated to exhaustion**, filtered client-side
   by the `rating` field to keep only games above an Elo threshold (e.g.
   1500+). Likely a 5–20× data multiplier over (1). Small code change: one
   new helper in `ps_client.py`, one new stage in `scrape.py`.
3. **WebSocket spectator capture.** Only worth the operational cost once (1)
   and (2) plateau.

### Caveats to remember regardless of source

- **Saved-replay selection bias:** players save their highlight games, not
  their losses or boring games. Corpus skews toward decisive/interesting
  outcomes. Counterbalance with WebSocket capture if this becomes a problem.
- **Format ID drift:** VGC regulations rotate (`Reg I`, `Reg J`, ...). Format
  IDs change each season. The current scraper hard-codes
  `gen9vgc2026regi[bo3]`; revisit when the metagame moves on.
- **Private replays** (`private == 1` in search results) require a password.
  We skip them — they're rare in our top-500 sample.
- **Rate limits:** PS doesn't document any, doesn't send `Retry-After`/`X-
  RateLimit-*` headers, but be polite. Current scraper uses concurrency 8
  with exponential-backoff retry; that's been fine for top-500 crawls.
