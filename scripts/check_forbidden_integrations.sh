#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCANNER="${SCRIPT_DIR}/check_forbidden_integrations.py"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

invoke_codex_wrapper() {
  shopt -s nullglob
  local wrappers=(
    "${REPO_ROOT}/codex/assess-technical-debt-in-ci-workflows"
    "${REPO_ROOT}/codex/assess-technical-debt-in-ci-workflows-"*
  )
  shopt -u nullglob

  local wrapper=""
  for candidate in "${wrappers[@]}"; do
    if [[ -f "$candidate" && -x "$candidate" ]]; then
      wrapper="$candidate"
      break
    fi
  done

  if [[ -n "$wrapper" ]]; then
    exec "$wrapper" "$SCANNER" "$@"
  fi
}

resolve_python() {
  local candidate="$1"
  if [[ -z "$candidate" ]]; then
    return 1
  fi

  local resolved=""
  if [[ "$candidate" == */* ]]; then
    if [[ -f "$candidate" && -x "$candidate" ]]; then
      resolved="$candidate"
    else
      return 1
    fi
  else
    resolved="$(command -v "$candidate" 2>/dev/null || true)"
    if [[ -z "$resolved" ]]; then
      return 1
    fi
  fi

  if "$resolved" -V >/dev/null 2>&1; then
    printf '%s\n' "$resolved"
    return 0
  fi

  return 1
}

run_python_scan() {
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

invoke_codex_wrapper "$@"
run_python_scan "$@"
