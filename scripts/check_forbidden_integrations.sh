#!/usr/bin/env bash
set -euo pipefail

DEFAULT_TARGETS=(
  src
  tests/current
  tests
  scripts
  docs
  tools
)

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON:-python3}

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Unable to locate Python interpreter '$PYTHON_BIN'." >&2
  exit 1
fi

if [[ $# -gt 0 ]]; then
  TARGETS=("$@")
else
  TARGETS=("${DEFAULT_TARGETS[@]}")
fi

if [[ -z "${TARGETS[*]}" ]]; then
  echo "No targets specified for forbidden integration scan." >&2
  exit 0
fi

"$PYTHON_BIN" "$SCRIPT_DIR/check_forbidden_integrations.py" "${TARGETS[@]}"
