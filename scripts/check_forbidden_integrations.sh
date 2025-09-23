#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCANNER="${SCRIPT_DIR}/check_forbidden_integrations.py"
CODEX_TOOL="${REPO_ROOT}/codex/assess-technical-debt-in-ci-workflows"

try_codex_scan() {
  if [[ -x "${CODEX_TOOL}" ]]; then
    exec "${CODEX_TOOL}" "$@"
  fi

  # Some setups inject a random suffix into the codex helper path. Honour that
  # convention by checking for the longest matching executable before giving up.
  shopt -s nullglob
  local candidates=("${CODEX_TOOL}"*)
  shopt -u nullglob
  for candidate in "${candidates[@]}"; do
    if [[ -x "${candidate}" ]]; then
      exec "${candidate}" "$@"
    fi
  done
}

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

try_python_scan() {
  local python_candidates=()
  if [[ -n "${PYTHON:-}" ]]; then
    python_candidates+=("${PYTHON}")
  fi
  python_candidates+=(python3 python python3.11 python3.10 python3.9 python3.8)

  local resolved=""
  for candidate in "${python_candidates[@]}"; do
    resolved="$(resolve_python "$candidate" 2>/dev/null || true)"
    if [[ -n "$resolved" ]]; then
      exec "$resolved" "$SCANNER" "$@"
    fi
    if [[ -n "${PYTHON:-}" && "$candidate" == "${PYTHON}" ]]; then
      echo "Ignoring invalid \$PYTHON override '${PYTHON}'." >&2
    fi
  done

  cat <<'EOWARN' >&2
Unable to locate a working Python interpreter for the forbidden integration scan.
Install python3 or set PYTHON to a valid executable before re-running the check.
EOWARN
  exit 1
}

try_codex_scan "$@"
try_python_scan "$@"
