#!/usr/bin/env bash
set -euo pipefail

 codex/assess-technical-debt-in-ci-workflows-q81m3f
DEFAULT_TARGETS=(src tests/current)

 codex/assess-technical-debt-in-ci-workflows
# Some CI harnesses inject a bogus $PYTHON override that points at a
# non-existent codex helper. Trying to execute the missing file produces a
# confusing "command not found" error before we have a chance to fall back to a
# real interpreter. Guard against this by clearing the override when it refers
# to a path that is not present so the resolver can progress to the usual
# detection logic.
if [[ -n "${PYTHON:-}" && "${PYTHON}" == */* && ! -e "${PYTHON}" ]]; then
  echo "Ignoring missing \$PYTHON override '${PYTHON}'." >&2
  unset PYTHON
fi


 codex/assess-technical-debt-in-ci-workflows

 codex/assess-technical-debt-in-ci-workflows
 main
 main
FORBIDDEN_REGEX='(?i)(ctrader[-_]?open[-_]?api|ctraderapi\\.com|connect\\.icmarkets\\.com|ctrader_open_api|swagger|spotware|real_ctrader_interface|from\\s+fastapi|import\\s+fastapi|import\\s+uvicorn)'

resolve_python() {
  local candidate="$1"
  local resolved

  if [ -z "$candidate" ]; then
    return 1
  fi

 codex/assess-technical-debt-in-ci-workflows
  # Direct paths (contain a slash) must point to an executable file.
  if [[ "$candidate" == */* ]]; then
    if [ -x "$candidate" ] && [ ! -d "$candidate" ]; then

 codex/assess-technical-debt-in-ci-workflows
  # Direct paths (contain a slash) must point to an executable file.
  if [[ "$candidate" == */* ]]; then
    if [ -x "$candidate" ] && [ ! -d "$candidate" ]; then

  # Direct path (includes slash) must exist, be executable, and not be a directory.
  if [ -x "$candidate" ] && [ ! -d "$candidate" ]; then
    if "$candidate" -V >/dev/null 2>&1; then
 main
 main
      printf '%s' "$candidate"
      return 0
    fi
    return 1
  fi

 codex/assess-technical-debt-in-ci-workflows
  # Otherwise resolve through PATH.
  if resolved=$(command -v "$candidate" 2>/dev/null); then
    if [ -x "$resolved" ] && [ ! -d "$resolved" ]; then

 codex/assess-technical-debt-in-ci-workflows
  # Otherwise resolve through PATH.
  if resolved=$(command -v "$candidate" 2>/dev/null); then
    if [ -x "$resolved" ] && [ ! -d "$resolved" ]; then

  # Otherwise, attempt PATH resolution.
  if command -v "$candidate" >/dev/null 2>&1; then
    resolved=$(command -v "$candidate") || return 1
    if [ -x "$resolved" ] && [ ! -d "$resolved" ] && "$resolved" -V >/dev/null 2>&1; then
 main
 main
      printf '%s' "$resolved"
      return 0
    fi
  fi

  return 1
}

 codex/assess-technical-debt-in-ci-workflows

 codex/assess-technical-debt-in-ci-workflows
 main
try_python_scan() {
  local candidate="$1"
  shift
  local resolved
  local output

  resolved=$(resolve_python "$candidate") || return 1
  RESOLVED_PYTHON="$resolved"

  if output=$("$resolved" - "$@" <<'PY' 2>&1
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
  ); then
    SCAN_RESULT="$output"
    SCAN_ERROR=""
    return 0
  fi

  SCAN_RESULT=""
  SCAN_ERROR="$output"
  return 1
}
 codex/assess-technical-debt-in-ci-workflows


 codex/assess-technical-debt-in-ci-workflows-h53fj9
FORBIDDEN_REGEX='(?i)(ctrader[-_]?open[-_]?api|ctraderapi\\.com|connect\\.icmarkets\\.com|ctrader_open_api|swagger|spotware|real_ctrader_interface|from\\s+fastapi|import\\s+fastapi|import\\s+uvicorn)'

 codex/assess-technical-debt-in-ci-workflows-34cv3v
FORBIDDEN_REGEX='(?i)(ctrader[-_]?open[-_]?api|ctraderapi\\.com|connect\\.icmarkets\\.com|ctrader_open_api|swagger|spotware|real_ctrader_interface|from\\s+fastapi|import\\s+fastapi|import\\s+uvicorn)'

FORBIDDEN_REGEX='(ctrader_open_api|swagger|spotware|real_ctrader_interface|from\\s+fastapi|import\\s+fastapi|import\\s+uvicorn)'
 main
 main
 main
 main
 main
 main

if [ "$#" -gt 0 ]; then
  TARGETS=("$@")
else
 codex/assess-technical-debt-in-ci-workflows-q81m3f
  TARGETS=("${DEFAULT_TARGETS[@]}")

  TARGETS=(
    "src"
    "tests"
    "scripts"
    "docs"
    "tools"
  )
 main
fi

EXISTING_TARGETS=()
for target in "${TARGETS[@]}"; do
  if [ -e "$target" ]; then
    EXISTING_TARGETS+=("$target")
  fi
done

if [ "${#EXISTING_TARGETS[@]}" -eq 0 ]; then
  echo "No scan targets exist; skipping forbidden integration check."
  exit 0
fi

 codex/assess-technical-debt-in-ci-workflows-q81m3f
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
else
  echo "Unable to locate python interpreter to run forbidden integration scan." >&2
  exit 127

echo "Scanning ${EXISTING_TARGETS[*]} for forbidden integrations..."
 codex/assess-technical-debt-in-ci-workflows

 codex/assess-technical-debt-in-ci-workflows
 main
PYTHON_BIN=""
PYTHON_CANDIDATE=${PYTHON:-}
RESOLVED_PYTHON=""
SCAN_RESULT=""
SCAN_ERROR=""

if [ -n "$PYTHON_CANDIDATE" ]; then
  if try_python_scan "$PYTHON_CANDIDATE" "$FORBIDDEN_REGEX" "${EXISTING_TARGETS[@]}"; then
    PYTHON_BIN="$RESOLVED_PYTHON"
    MATCHES="$SCAN_RESULT"
  else
    echo "The specified \$PYTHON ('$PYTHON_CANDIDATE') could not execute the forbidden integration scan; falling back to auto-detection." >&2
    if [ -n "$SCAN_ERROR" ]; then
      while IFS= read -r line; do
        printf '    %s\n' "$line" >&2
      done <<<"$SCAN_ERROR"
    fi
 codex/assess-technical-debt-in-ci-workflows


 codex/assess-technical-debt-in-ci-workflows
PYTHON_BIN=""
PYTHON_CANDIDATE=${PYTHON:-}

if [ -n "$PYTHON_CANDIDATE" ]; then
  if resolved=$(resolve_python "$PYTHON_CANDIDATE"); then
    PYTHON_BIN="$resolved"
  else
    echo "The specified \$PYTHON ('$PYTHON_CANDIDATE') is not an executable Python interpreter; falling back to auto-detection." >&2
 main
 main
  fi
fi

if [ -z "$PYTHON_BIN" ]; then
 codex/assess-technical-debt-in-ci-workflows

 codex/assess-technical-debt-in-ci-workflows
 main
  for fallback in python3 python3.12 python3.11 python3.10 python; do
    if try_python_scan "$fallback" "$FORBIDDEN_REGEX" "${EXISTING_TARGETS[@]}"; then
      PYTHON_BIN="$RESOLVED_PYTHON"
      MATCHES="$SCAN_RESULT"
 codex/assess-technical-debt-in-ci-workflows


  for fallback in python3 python; do
    if resolved=$(resolve_python "$fallback"); then
      PYTHON_BIN="$resolved"
 main
 main
      break
    fi
  done
fi

if [ -z "$PYTHON_BIN" ]; then
 codex/assess-technical-debt-in-ci-workflows

 codex/assess-technical-debt-in-ci-workflows
 main
  echo "Unable to locate a functional Python interpreter. Install Python 3 (or set \$PYTHON) to run the forbidden integration check." >&2
  if [ -n "$SCAN_ERROR" ]; then
    while IFS= read -r line; do
      printf '    %s\n' "$line" >&2
    done <<<"$SCAN_ERROR"
  fi
 codex/assess-technical-debt-in-ci-workflows


  echo "Unable to locate a Python interpreter. Install Python 3 (or set \$PYTHON) to run the forbidden integration check." >&2
 main
 main
  exit 1

 codex/assess-technical-debt-in-ci-workflows-h53fj9
PYTHON_BIN=""
PYTHON_CANDIDATE=${PYTHON:-}
if [ -n "$PYTHON_CANDIDATE" ]; then
  if [ -x "$PYTHON_CANDIDATE" ]; then
    PYTHON_BIN="$PYTHON_CANDIDATE"
  elif command -v "$PYTHON_CANDIDATE" >/dev/null 2>&1; then
    PYTHON_BIN=$(command -v "$PYTHON_CANDIDATE")
  else
    echo "The specified \$PYTHON ('$PYTHON_CANDIDATE') is not an executable; falling back to interpreter auto-detection." >&2
  fi
fi


 codex/assess-technical-debt-in-ci-workflows-34cv3v

 codex/assess-technical-debt-in-ci-workflows-2jce40

 codex/assess-technical-debt-in-ci-workflows-1er73t

 codex/assess-technical-debt-in-ci-workflows-7cy9fp
 main
 main
 main
PYTHON_BIN=${PYTHON:-}
 main
if [ -z "$PYTHON_BIN" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=$(command -v python3)
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=$(command -v python)
  else
    echo "Unable to locate a Python interpreter. Install Python 3 (or set \$PYTHON) to run the forbidden integration check." >&2
    exit 1
  fi
fi

MATCHES=$(
"$PYTHON_BIN" - "$FORBIDDEN_REGEX" "${EXISTING_TARGETS[@]}" <<'PY'
 codex/assess-technical-debt-in-ci-workflows-h53fj9

 codex/assess-technical-debt-in-ci-workflows-34cv3v

 codex/assess-technical-debt-in-ci-workflows-2jce40

 codex/assess-technical-debt-in-ci-workflows-1er73t


 codex/assess-technical-debt-in-ci-workflows-jurxls

 codex/assess-technical-debt-in-ci-workflows-que3tv
 main
MATCHES=$(
python - "$FORBIDDEN_REGEX" "${EXISTING_TARGETS[@]}" <<'PY'
 main
 main
 main
 main
 main
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
 codex/assess-technical-debt-in-ci-workflows-h53fj9

 codex/assess-technical-debt-in-ci-workflows-34cv3v

 codex/assess-technical-debt-in-ci-workflows-2jce40

 codex/assess-technical-debt-in-ci-workflows-1er73t
 main
 main
 main
    ".md",
    ".mdx",
    ".rst",
}

root = Path.cwd()
matches = []

 codex/assess-technical-debt-in-ci-workflows-h53fj9

 codex/assess-technical-debt-in-ci-workflows-34cv3v

 codex/assess-technical-debt-in-ci-workflows-2jce40

 codex/assess-technical-debt-in-ci-workflows-7cy9fp
    ".md",
    ".mdx",
    ".rst",

 codex/assess-technical-debt-in-ci-workflows-jurxls
    ".md",
    ".mdx",
    ".rst",

 main
 main
}

root = Path.cwd()
matches: list[str] = []
 main

 main
 main
 main

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

 codex/assess-technical-debt-in-ci-workflows-h53fj9

 codex/assess-technical-debt-in-ci-workflows-34cv3v

 codex/assess-technical-debt-in-ci-workflows-2jce40

 codex/assess-technical-debt-in-ci-workflows-1er73t

 codex/assess-technical-debt-in-ci-workflows-7cy9fp

 codex/assess-technical-debt-in-ci-workflows-jurxls


FILE_PATTERNS=(
  '--include=*.py'
  '--include=*.pyi'
  '--include=*.ipynb'
  '--include=*.sh'
  '--include=*.txt'
  '--include=*.toml'
  '--include=*.cfg'
  '--include=*.ini'
  '--include=*.yml'
  '--include=*.yaml'
)

MATCHES=$(grep -RniE "$FORBIDDEN_REGEX" --binary-files=without-match "${FILE_PATTERNS[@]}" -- "${EXISTING_TARGETS[@]}" || true)

 main
 main
 main
 main
 main
 main
 main
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
 main
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

MATCHES="$SCAN_RESULT"

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

MATCHES="$SCAN_RESULT"

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
 main
fi

"$PYTHON_BIN" - "${EXISTING_TARGETS[@]}" <<'PY'
import sys
import re
from pathlib import Path

PATTERN = re.compile(
    r"(ctrader_open_api|swagger|spotware|real_ctrader_interface|from\s+fastapi|import\s+fastapi|import\s+uvicorn)",
    re.IGNORECASE,
)


def iter_targets(paths):
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            yield path
        elif path.is_dir():
            for child in path.rglob('*'):
                if child.is_file():
                    yield child


matches = []
for file_path in iter_targets(sys.argv[1:]):
    try:
        with file_path.open('r', encoding='utf-8', errors='ignore') as handle:
            for line_number, line in enumerate(handle, start=1):
                if PATTERN.search(line):
                    matches.append(
                        f"{file_path.as_posix()}:{line_number}:{line.strip()}".rstrip(':')
                    )
    except (OSError, UnicodeDecodeError):
        continue

if matches:
    print("Forbidden references detected:", file=sys.stderr)
    for entry in matches:
        print(entry, file=sys.stderr)
    sys.exit(1)

print("No forbidden references found.")
PY
