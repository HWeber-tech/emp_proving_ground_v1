#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bulk_type_fix.sh [--apply] [--dirs "dir1 dir2 ..."] [--include-regex REGEX] [--exclude-regex REGEX]
                       [--glob "pattern"]... [--jobs N] [--dmypy-check]

Flags:
  --apply                 Apply changes in-place (default: dry-run; prints unified diffs)
  --dirs "A B C"          Space-separated directories to scan (default: "src/core src/thinking src/trading src/ecosystem")
  --include-regex REGEX   Include filter (regex) applied to file paths
  --exclude-regex REGEX   Exclude filter (regex) applied to file paths
  --glob "pattern"        fnmatch-style glob (repeatable). Example: --glob "src/**/*.py"
  --jobs N                Parallelism for scanning (dry-run) (default: 8)
  --dmypy-check           After apply, run dmypy check to validate types
  -h, --help              Show this help

Environment:
  Prefers Python from ./.venv_mypy/bin/python, falls back to python3 if not present.

Examples:
  Dry-run defaults:
    bash scripts/cleanup/bulk_type_fix.sh --dirs "src/core src/thinking src/trading src/ecosystem" --jobs 8 --glob "src/**/*.py"

  Apply to core only and check with dmypy:
    bash scripts/cleanup/bulk_type_fix.sh --apply --dirs "src/core" --jobs 8 --dmypy-check
EOF
}

# Resolve repo root (script may be invoked from anywhere inside repo)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

# Python resolution
PY="./.venv_mypy/bin/python"
if [[ ! -x "${PY}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PY="python3"
  else
    echo "Error: No Python interpreter found (.venv_mypy/bin/python or python3)." >&2
    exit 2
  fi
fi

# Ensure chosen interpreter can import libcst; otherwise try system python3
if ! "${PY}" -c "import libcst" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1 && python3 -c "import libcst" >/dev/null 2>&1; then
    echo "[bulk_type_fix] Selected interpreter lacks libcst; switching to system python3" >&2
    PY="python3"
  else
    echo "[bulk_type_fix] libcst not available in ${PY} and python3; attempting installation via pip in current venv..." >&2
    if command -v "${REPO_ROOT}/.venv_mypy/bin/pip" >/dev/null 2>&1; then
      "${REPO_ROOT}/.venv_mypy/bin/pip" install --disable-pip-version-check --no-input libcst >/dev/null 2>&1 || true
    fi
    if ! "${PY}" -c "import libcst" >/dev/null 2>&1; then
      echo "Error: No interpreter with libcst is available." >&2
      exit 2
    fi
  fi
fi

APPLY=0
DIRS_DEFAULT=("src/core" "src/thinking" "src/trading" "src/ecosystem")
DIRS=("${DIRS_DEFAULT[@]}")
INCLUDE_REGEX=""
EXCLUDE_REGEX="stubs|tests|migrations|docs|__pycache__|site-packages|\\.venv"
JOBS=8
DMYPY_CHECK=0
GLOBS=()

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply)
      APPLY=1
      shift
      ;;
    --dirs)
      shift
      if [[ $# -eq 0 ]]; then echo "Missing value for --dirs" >&2; exit 2; fi
      # Split by spaces into array
      IFS=' ' read -r -a DIRS <<< "$1"
      shift
      ;;
    --include-regex)
      shift
      if [[ $# -eq 0 ]]; then echo "Missing value for --include-regex" >&2; exit 2; fi
      INCLUDE_REGEX="$1"
      shift
      ;;
    --exclude-regex)
      shift
      if [[ $# -eq 0 ]]; then echo "Missing value for --exclude-regex" >&2; exit 2; fi
      EXCLUDE_REGEX="$1"
      shift
      ;;
    --glob)
      shift
      if [[ $# -eq 0 ]]; then echo "Missing value for --glob" >&2; exit 2; fi
      GLOBS+=("$1")
      shift
      ;;
    --jobs)
      shift
      if [[ $# -eq 0 ]]; then echo "Missing value for --jobs" >&2; exit 2; fi
      JOBS="$1"
      shift
      ;;
    --dmypy-check)
      DMYPY_CHECK=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

# Build command
REWRITER="scripts/cleanup/explicit_any_rewriter.py"
if [[ ! -f "${REWRITER}" ]]; then
  echo "Error: Rewriter not found at ${REWRITER}" >&2
  exit 2
fi

CMD=("${PY}" "${REWRITER}")
if [[ ${APPLY} -eq 1 ]]; then
  CMD+=("--apply")
fi
# Directories
for d in "${DIRS[@]}"; do
  CMD+=("--dir" "$d")
done
# Include/Exclude
if [[ -n "${INCLUDE_REGEX}" ]]; then
  CMD+=("--include" "${INCLUDE_REGEX}")
fi
if [[ -n "${EXCLUDE_REGEX}" ]]; then
  CMD+=("--exclude" "${EXCLUDE_REGEX}")
fi
# Globs
for g in "${GLOBS[@]}"; do
  CMD+=("--glob" "${g}")
done
# Jobs
CMD+=("--jobs" "${JOBS}")

echo "[bulk_type_fix] Running: ${CMD[*]}"
set +e
"${CMD[@]}"
STATUS=$?
set -e

if [[ ${STATUS} -ne 0 ]]; then
  echo "[bulk_type_fix] Rewriter exited with status ${STATUS}" >&2
  exit ${STATUS}
fi

if [[ ${APPLY} -eq 1 && ${DMYPY_CHECK} -eq 1 ]]; then
  echo "[bulk_type_fix] Running dmypy check..."
  # Prefer dmypy from venv, fallback to mypy if dmypy missing
  if [[ -x "./.venv_mypy/bin/dmypy" ]]; then
    MYPY_ENV="MYPYPATH=stubs:src${MYPYPATH:+:${MYPYPATH}}"
    set +e
    eval "${MYPY_ENV} ./.venv_mypy/bin/dmypy run -- --config-file mypy.ini --show-error-codes --no-color-output src"
    DM_STATUS=$?
    set -e
    if [[ ${DM_STATUS} -ne 0 ]]; then
      echo "[bulk_type_fix] dmypy reported issues (exit ${DM_STATUS})." >&2
      exit ${DM_STATUS}
    fi
  else
    echo "[bulk_type_fix] dmypy not found; running mypy once..."
    set +e
    ./.venv_mypy/bin/mypy --config-file mypy.ini --show-error-codes --no-color-output src
    M_STATUS=$?
    set -e
    if [[ ${M_STATUS} -ne 0 ]]; then
      echo "[bulk_type_fix] mypy reported issues (exit ${M_STATUS})." >&2
      exit ${M_STATUS}
    fi
  fi
fi

echo "[bulk_type_fix] Completed successfully."