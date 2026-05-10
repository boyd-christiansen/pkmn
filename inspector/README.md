# inspector

Local read-only web UI for inspecting SFT prompts, model responses, and the
source data those prompts were built from. Lives **outside** the pipeline
directory and never writes back into it — purely a viewer.

## What it shows

For every SFT row in `pipeline/parsed_data/` (and its `legacy/` subdir):

- **System prompt**, split into a prelude + numbered rules (1–6).
- **User prompt**, split into 8 logical sections (HEADER, ACTIVE/BENCH,
  GAME-STATE LEDGER, TURN-BY-TURN, SERIES STATE, YOUR SPREADS, THREAT
  MATRIX). Sections that aren't present in the row's schema render as
  collapsed "missing" placeholders.
- **Tool loop**: every `calculate_damage` iteration with its arguments
  and the matching tool response, then the final `submit_decision`
  with the model's chain-of-thought + structured action.
- **Source data**: when the row's parsed match is also in
  `pipeline/parsed_data/{bo1,bo3}.jsonl`, the inspector cross-references
  by `(match_id, game_index, turn)` and shows the underlying snapshot,
  events stream, and team sheets that fed the prompt.

For parsed-match files (`bo1.jsonl`, `bo3.jsonl`):

- One card per game, with full JSON snapshots collapsible per turn.

Two top-level tabs:

- **Browse** — file picker → row list → detail.
- **Compare** — pin two rows from Browse (📌 A / 📌 B buttons) and
  view them side-by-side.

## What it doesn't do (intentional v1 scope)

- No live LLM invocation. To see real model output, run
  `master_pipeline.py` or `bakeoff.py` and reload the inspector.
- No Postman-like simulator. Run the pipeline yourself if you want to
  see what would be sent for a given input.
- No raw replay browsing (the 16K-file `data_scraper/data/replays/`
  tree). Use `jq` / a JSON viewer for those.
- No annotation / notes persistence.
- No search / filtering.

## Running

```bash
cd inspector
python3 -m venv .venv
.venv/bin/pip install -e .
./run.sh
```

→ http://localhost:8001

The inspector imports nothing from the `pipeline` package, but it does
read `pipeline/parsed_data/`. If you move that directory, edit
`PARSED_DATA_DIR` in `data_loader.py`.

The default port is **8001** so it doesn't collide with
`calc_microservice` on **3000**.

## File layout

```
inspector/
├── README.md
├── pyproject.toml         (fastapi, uvicorn — no pipeline deps)
├── run.sh                 (uvicorn server:app --reload --port 8001)
├── server.py              FastAPI endpoints
├── data_loader.py         File index, JSONL row reader, source cross-reference
├── prompt_parser.py       Splits saved prompt strings back into structured sections
└── static/
    ├── index.html         Single-page app
    ├── app.js             Vanilla JS, no build step
    └── styles.css
```

## Endpoints

| Route | Description |
|---|---|
| `GET /api/files` | Index of *.jsonl files under `pipeline/parsed_data/`, grouped current/legacy |
| `GET /api/file/{path}/rows` | Lightweight row metadata for sidebar |
| `GET /api/file/{path}/row/{idx}` | One row, fully parsed + cross-referenced |
| `GET /api/health` | Server health check |
| `GET /` | Single-page HTML app |
| `GET /static/*` | Static assets |

## Schema awareness

The user-prompt section parser handles both **current** and **legacy**
prompt schemas. Files generated before the historical-context layer
landed don't have GAME-STATE LEDGER / TURN-BY-TURN / SERIES STATE
sections — those render as collapsed "missing" placeholders. The badge
in the detail header (`current` / `legacy`) tells you which schema a
row is using.
