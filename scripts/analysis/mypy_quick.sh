#!/usr/bin/env bash
set -euo pipefail

# Quick mypy run with strict-ish flags and tee to out/last_mypy.txt

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/scripts/analysis/out"
MYPY="$ROOT/.venv_mypy/bin/mypy"
CFG="$ROOT/mypy.ini"
SRC="$ROOT/src"

mkdir -p "$OUT"

# Common repo layout: keep stubs on MYPYPATH if present; do not add SRC to avoid dual identities
if [[ -d "$ROOT/stubs" ]]; then
  export MYPYPATH="${ROOT}/stubs${MYPYPATH+:${MYPYPATH}}"
fi

# No color, show error codes to aid triage, and stable output to file
"$MYPY" "$SRC" \
  --config-file "$CFG" \
  --show-error-codes \
  --no-color-output \
  | tee "$OUT/last_mypy.txt"