#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${INSERTANY3D_PYTHON:-$SCRIPT_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

CANARY_ROOT="${INSERTANY3D_CANARY_ROOT:-$(mktemp -d "${TMPDIR:-/tmp}/insertany3d-canary.XXXXXX")}"
MANIFEST="${INSERTANY3D_CANARY_MANIFEST:-$SCRIPT_ROOT/examples/batch-one-task.draft.json}"
BATCH_ID="${INSERTANY3D_CANARY_BATCH_ID:-draft_one_task}"

mkdir -p "$CANARY_ROOT"
exec "$PYTHON_BIN" "$SCRIPT_ROOT/tools/insertany3d.py" \
  --db "$CANARY_ROOT/state.sqlite3" \
  batch run-all "$BATCH_ID" \
  --manifest "$MANIFEST" \
  --root "$CANARY_ROOT/runs" \
  --fake --non-interactive --no-monitor --json \
  --max-steps "${INSERTANY3D_CANARY_MAX_STEPS:-100}"
