#!/usr/bin/env bash
# Launch the inspector. Reads pipeline/parsed_data/ for browse content and
# imports pipeline/ modules for source-tracing. Stays read-only — never
# writes anything back into pipeline/.
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/uvicorn server:app --reload --port 8001 "$@"
