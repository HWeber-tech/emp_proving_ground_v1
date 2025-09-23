#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCANNER="${SCRIPT_DIR}/check_forbidden_integrations.py"

resolve_python() {
  local candidate="$1"
  if [[ -z "$candidate" ]]; then
    return 1
  fi

  local resolved=""
  if [[ "$candidate" == */* ]]; then
    if [[ ! -f "$candidate" || ! -x "$candidate" ]]; then
      return 1
    fi
    resolved="$candidate"
  else
    resolved="$(command -v "$candidate" 2>/dev/null || true)"
    if [[ -z "$resolved" ]]; then
      return 1
    fi
  fi

  if ! "$resolved" -V >/dev/null 2>&1; then
    return 1
  fi

  printf '%s\n' "$resolved"
}

PYTHON_CANDIDATES=()
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_CANDIDATES+=("${PYTHON}")
fi
PYTHON_CANDIDATES+=(python3 python python3.11 python3.10 python3.9 python3.8)

RESOLVED_PYTHON=""
for candidate in "${PYTHON_CANDIDATES[@]}"; do
  resolved="$(resolve_python "$candidate" 2>/dev/null || true)"
  if [[ -n "$resolved" ]]; then
    RESOLVED_PYTHON="$resolved"
    break
  fi
  if [[ -n "${PYTHON:-}" && "$candidate" == "${PYTHON}" ]]; then
    echo "Ignoring invalid \$PYTHON override '${PYTHON}'." >&2
  fi
done

if [[ -z "$RESOLVED_PYTHON" ]]; then
  cat <<'EOWARN' >&2
Unable to locate a working Python interpreter for the forbidden integration scan.
Install python3 or set PYTHON to a valid executable before re-running the check.
EOWARN
  exit 1
fi

exec "$RESOLVED_PYTHON" "$SCANNER" "$@"
