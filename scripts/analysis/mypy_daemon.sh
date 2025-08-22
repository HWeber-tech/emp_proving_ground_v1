#!/usr/bin/env bash
set -euo pipefail

# dmypy convenience wrapper
# Usage:
#   scripts/analysis/mypy_daemon.sh start|status|check|stop
#
# Uses local mypy.ini and ensures MYPYPATH includes src and stubs (if present).

CMD="${1:-}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MYPY_BIN="$ROOT/.venv_mypy/bin/mypy"
DMYPY_BIN="$ROOT/.venv_mypy/bin/dmypy"
CFG="$ROOT/mypy.ini"
SRC="$ROOT/src"

if [[ -d "$ROOT/stubs" ]]; then
  export MYPYPATH="${ROOT}/stubs:${SRC}${MYPYPATH+:${MYPYPATH}}"
else
  export MYPYPATH="${SRC}${MYPYPATH+:${MYPYPATH}}"
fi

if [[ ! -x "$DMYPY_BIN" ]]; then
  echo "dmypy not found at $DMYPY_BIN"
  echo "Install mypy in .venv_mypy: "
  echo "  python -m venv .venv_mypy && ./.venv_mypy/bin/pip install mypy"
  exit 127
fi

case "$CMD" in
  start)
    # Start or restart the daemon and run an initial check to warm caches.
    "$DMYPY_BIN" run -- --config-file "$CFG" --show-error-codes --no-color-output "$SRC" || true
    ;;
  status)
    "$DMYPY_BIN" status || true
    ;;
  check)
    "$DMYPY_BIN" run -- --config-file "$CFG" --show-error-codes --no-color-output "$SRC"
    ;;
  stop)
    "$DMYPY_BIN" stop || true
    ;;
  *)
    echo "Unknown or missing command. Use one of: start | status | check | stop"
    exit 2
    ;;
esac