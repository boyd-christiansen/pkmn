# Blake — onboarding to the pkmn project

Welcome. This repo is a generalist-LLM-plays-VGC pipeline. The full
context is in [`README.md`](README.md); this doc is the fastest path
to actually viewing what Boyd has built.

There are **three onboarding paths** below. **Path C is the fastest**
— start there if you just want to look at the prompts.

## What you'll be looking at

The pipeline turns ~16,500 high-Elo Showdown replays into ~93,900 SFT
training rows. Each row is one VGC turn: a fully-formatted prompt
(board state, threat matrix, inferred opponent spreads, etc.) +
ground-truth label (the human expert's actual play).

The **`sft_preview_dry_run_*.jsonl`** files contain those rows with
the prompt fully rendered but **no real LLM synthesis** — the
"assistant" message is a placeholder that surfaces the human's actual
play. Cheap to generate, perfect for visually inspecting the prompts.

The [`inspector/`](inspector/) subdir is a local FastAPI viewer for
browsing those rows + cross-referencing back to the parsed match
snapshots they were built from.

---

## Path C — Quickest possible (recommended for first look)

Skip the pipeline entirely. Boyd will send you a single tarball
(~250–300 MB compressed) containing the four data JSONLs you need.

1. **Clone the repo.**
   ```bash
   git clone https://github.com/boyd-christiansen/pkmn.git
   cd pkmn
   ```

2. **Get the data tarball from Boyd** (Drive / Dropbox / S3 / wherever
   you both share files; see Boyd for the link).

3. **Extract into `pipeline/parsed_data/`:**
   ```bash
   tar xzf /path/to/pkmn_preview_data.tar.gz -C pipeline/
   # → unpacks into pipeline/parsed_data/{bo1.jsonl, bo3.jsonl,
   #    sft_preview_dry_run_bo1.jsonl, sft_preview_dry_run_bo3.jsonl}
   ```

4. **Start the inspector.**
   ```bash
   cd inspector
   python3 -m venv .venv
   .venv/bin/pip install -e .
   ./run.sh
   ```
   → http://localhost:8001

5. **Browse.** Pick `sft_preview_dry_run_bo3.jsonl` from the file
   panel (use Bo3 for the richer prompts — Open Team Sheet, fuller
   metadata, longer turn-by-turn rollups in the series-state block).

   The header of every preview row shows a `DRY RUN` badge so you
   know the assistant message is a placeholder, not real synthesis.

That's it for Path C. No calc microservice, no API keys, no LLM
spend, no pipeline run. ~5 minutes if Boyd already shared the tarball.

---

## Path A — Browse prompts, regenerate locally (no API cost)

Use this if you want to regenerate the previews yourself (e.g. Boyd's
data is stale, or you want to tweak prompt code and see the diff).
Total wall-clock: ~30 min. No API keys needed.

1. **Clone the repo** (same as Path C step 1).

2. **Set up the four runtimes:**
   - `data_scraper/`: `python3 -m venv .venv && .venv/bin/pip install -e .`
   - `calc_microservice/`: `npm install`
   - `pipeline/`: `python3 -m venv .venv && .venv/bin/pip install -e .`
   - `inspector/`: `python3 -m venv .venv && .venv/bin/pip install -e .`

3. **Start the calc microservice** (must be running for everything else):
   ```bash
   cd calc_microservice && npm run dev
   # → http://localhost:3000 ; leave this running
   ```

4. **Get the raw replays.** Two options:
   - **Faster:** ask Boyd for the `replays.tar.gz` (~134 MB), unpack
     into `data_scraper/data/replays/`.
   - **From scratch:** run the scraper (~30–60 min, depends on Showdown):
     ```bash
     cd data_scraper && .venv/bin/python scrape.py
     ```

5. **Parse the replays into per-turn snapshots** (~15 sec for the full
   16K corpus):
   ```bash
   cd pipeline && .venv/bin/python replay_parser.py
   ```

6. **Generate the dry-run previews** (~6 min Bo3, ~14 min Bo1 at
   concurrency 8; can run in parallel):
   ```bash
   nohup .venv/bin/python master_pipeline.py \
       --input parsed_data/bo3.jsonl \
       --output parsed_data/sft_preview_dry_run_bo3.jsonl \
       --dry-run --no-judge --concurrency 8 > /tmp/preview_bo3.log 2>&1 &
   nohup .venv/bin/python master_pipeline.py \
       --input parsed_data/bo1.jsonl \
       --output parsed_data/sft_preview_dry_run_bo1.jsonl \
       --dry-run --no-judge --concurrency 8 > /tmp/preview_bo1.log 2>&1 &
   ```

8. **Start the inspector** (same as Path C step 4).

---

## Path B — Full real synthesis (don't, unless you're scoping cost)

Same as Path A, but drop `--dry-run --no-judge` from step 7 and set
`GOOGLE_API_KEY` (production default since Plan v8) in a top-level
`.env` file:

```bash
echo "GOOGLE_API_KEY=..." >> .env
.venv/bin/python master_pipeline.py --input parsed_data/bo3.jsonl
```

This calls the real Gemini API for every turn (~$1,800 for the full
93K-row corpus; effectively zero against the project's ~$100K GCP
credit pool). The pipeline-from-here also writes via the model judge
(`--use-judge`, on by default), which calls Gemini once per match.
For OpenAI synthesis instead, add `--provider openai
--judge-provider openai`; that path costs ~$3,300 against the project's
OpenAI account. **Don't run any real synthesis just to look at
prompts** — the dry-run path produces the same prompts at zero cost.
Path B is only relevant if you want to spot-check real model output
against the dry-run baseline.

---

## What's where

- `data_scraper/` — pulls top-500 ladder users + replays from
  Showdown. See [`data_scraper/README.md`](data_scraper/README.md).
- `calc_microservice/` — Node service wrapping `@smogon/calc` +
  `@pkmn/client` + `@pkmn/dex`. See
  [`calc_microservice/README.md`](calc_microservice/README.md).
- `pipeline/` — Python modules that turn raw replays into SFT rows.
  See [`pipeline/README.md`](pipeline/README.md) for the module map.
- `inspector/` — local FastAPI viewer for SFT rows + source data.
  See [`inspector/README.md`](inspector/README.md).
- `notes/` — design notes. The walkthrough at
  [`notes/pipeline_walkthrough.md`](notes/pipeline_walkthrough.md) is
  the deepest narrative — recommended after you've poked at a few
  rows in the inspector and want context.
- [`CLAUDE.md`](CLAUDE.md) — convention notes + gotchas while coding.
  Skim if you're going to modify the pipeline.

## Data files (not in git)

The following are gitignored — regenerable, intentionally out of
source control:

| Path | What | Size | How to get |
|---|---|---|---|
| `data_scraper/data/replays/` | Raw Showdown replays | ~134 MB | Scrape or ask Boyd for tarball |
| `pipeline/parsed_data/bo{1,3}.jsonl` | Per-turn snapshots + events | ~290 MB | `replay_parser.py` (~15 sec) |
| `pipeline/parsed_data/sft_preview_dry_run_bo{1,3}.jsonl` | Pre-rendered prompts | ~770 MB | `master_pipeline.py --dry-run` (~20 min) |

## Questions

Ping Boyd. The deepest design context is in
[`notes/pipeline_walkthrough.md`](notes/pipeline_walkthrough.md) —
walks through one real Bo3 match end-to-end and explains every design
decision the pipeline makes.
