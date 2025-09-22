#!/usr/bin/env bash
set -euo pipefail

FORBIDDEN_REGEX='(?i)(ctrader[-_]?open[-_]?api|ctraderapi\\.com|connect\\.icmarkets\\.com|ctrader_open_api|swagger|spotware|real_ctrader_interface|from\\s+fastapi|import\\s+fastapi|import\\s+uvicorn)'

resolve_python() {
  local candidate="$1"
  local resolved

  if [ -z "$candidate" ]; then
    return 1
  fi

  # Direct path (includes slash) must exist, be executable, and not be a directory.
  if [ -x "$candidate" ] && [ ! -d "$candidate" ]; then
    if "$candidate" -V >/dev/null 2>&1; then
      printf '%s' "$candidate"
      return 0
    fi
    return 1
  fi

  # Otherwise, attempt PATH resolution.
  if command -v "$candidate" >/dev/null 2>&1; then
    resolved=$(command -v "$candidate") || return 1
    if [ -x "$resolved" ] && [ ! -d "$resolved" ] && "$resolved" -V >/dev/null 2>&1; then
      printf '%s' "$resolved"
      return 0
    fi
  fi

  return 1
}

if [ "$#" -gt 0 ]; then
  TARGETS=("$@")
else
  TARGETS=(
    "src"
    "tests"
    "scripts"
    "docs"
    "tools"
  )
fi

EXISTING_TARGETS=()
for path in "${TARGETS[@]}"; do
  if [ -e "$path" ]; then
    EXISTING_TARGETS+=("$path")
  fi
done

if [ "${#EXISTING_TARGETS[@]}" -eq 0 ]; then
  echo "No scan targets exist; skipping forbidden integration check."
  exit 0
fi

echo "Scanning ${EXISTING_TARGETS[*]} for forbidden integrations..."
PYTHON_BIN=""
PYTHON_CANDIDATE=${PYTHON:-}

if [ -n "$PYTHON_CANDIDATE" ]; then
  if resolved=$(resolve_python "$PYTHON_CANDIDATE"); then
    PYTHON_BIN="$resolved"
  else
    echo "The specified \$PYTHON ('$PYTHON_CANDIDATE') is not an executable Python interpreter; falling back to auto-detection." >&2
  fi
fi

if [ -z "$PYTHON_BIN" ]; then
  for fallback in python3 python; do
    if resolved=$(resolve_python "$fallback"); then
      PYTHON_BIN="$resolved"
      break
    fi
  done
fi

if [ -z "$PYTHON_BIN" ]; then
  echo "Unable to locate a Python interpreter. Install Python 3 (or set \$PYTHON) to run the forbidden integration check." >&2
  exit 1
fi

MATCHES=$(
"$PYTHON_BIN" - "$FORBIDDEN_REGEX" "${EXISTING_TARGETS[@]}" <<'PY'
import re
import sys
from pathlib import Path


pattern = re.compile(sys.argv[1])
targets = sys.argv[2:]

extensions = {
    ".py",
    ".pyi",
    ".ipynb",
    ".sh",
    ".txt",
    ".toml",
    ".cfg",
    ".ini",
    ".yml",
    ".yaml",
    ".md",
    ".mdx",
    ".rst",
}

root = Path.cwd()
matches = []


def iter_files(base: Path):
    if base.is_file():
        yield base
        return
    for candidate in base.rglob("*"):
        if candidate.is_file():
            yield candidate


for target in targets:
    base = Path(target)
    if not base.exists():
        continue
    for candidate in iter_files(base):
        if candidate.suffix.lower() not in extensions:
            continue
        try:
            text_stream = candidate.open("r", encoding="utf-8", errors="ignore")
        except OSError:
            continue
        with text_stream as handle:
            for lineno, line in enumerate(handle, start=1):
                if pattern.search(line):
                    try:
                        rel_path = candidate.resolve().relative_to(root)
                    except ValueError:
                        rel_path = candidate
                    matches.append(f"{rel_path.as_posix()}:{lineno}:{line.rstrip()}".rstrip())

if matches:
    sys.stdout.write("\n".join(matches))
PY
)

ALLOWLIST_PATTERNS=(
  '^scripts/check_forbidden_integrations.sh:'
  '^scripts/phase1_deduplication.py:'
)

if [ -n "$MATCHES" ]; then
  FILTERED_MATCHES="$MATCHES"
  for pattern in "${ALLOWLIST_PATTERNS[@]}"; do
    FILTERED_MATCHES=$(printf '%s\n' "$FILTERED_MATCHES" | grep -Ev "$pattern" || true)
  done

  if [ -n "$FILTERED_MATCHES" ]; then
    echo "Forbidden references detected:" >&2
    echo "$FILTERED_MATCHES" >&2
    exit 1
  fi

  echo "All detected references are allow-listed." >&2
  exit 0
fi

echo "No forbidden references found."
