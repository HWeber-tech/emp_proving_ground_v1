#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCANNER="$SCRIPT_DIR/check_forbidden_integrations.py"

if [ ! -f "$SCANNER" ]; then
  echo "Scanner script not found at $SCANNER" >&2
  exit 1
fi

# Build a list of candidate interpreters, preferring an explicit $PYTHON override
# while still falling back to common Python shims.
declare -a CANDIDATES=()
if [ -n "${PYTHON:-}" ]; then
  CANDIDATES+=("$PYTHON")
fi
CANDIDATES+=("python3" "python")

resolve_interpreter() {
  local candidate
  for candidate in "$@"; do
    if [[ -z "$candidate" ]]; then
      continue
    fi

    # Reject the historical codex helper path that no longer ships with the repo.
    if [[ "$candidate" == codex/assess-technical-debt-in-ci-workflows* ]]; then
      echo "Ignoring stale PYTHON override '$candidate' (helper removed)." >&2
      continue
    fi

    if [[ "$candidate" == */* ]]; then
      if [ ! -x "$candidate" ]; then
        continue
      fi
    else
      if ! command -v "$candidate" >/dev/null 2>&1; then
        continue
      fi
    fi

    if "$candidate" -V >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

PYTHON_CMD="$(resolve_interpreter "${CANDIDATES[@]}")" || {
  echo "Unable to locate a working Python interpreter. Tried: ${CANDIDATES[*]}" >&2
  echo "Install Python 3 or update the PYTHON environment variable with a valid interpreter." >&2
  exit 1
}

exec "$PYTHON_CMD" "$SCANNER" "$@"
