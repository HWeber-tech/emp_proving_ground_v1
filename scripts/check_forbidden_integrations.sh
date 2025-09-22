#!/usr/bin/env bash
set -euo pipefail

FORBIDDEN_REGEX='(ctrader_open_api|swagger|spotware|real_ctrader_interface|from\\s+fastapi|import\\s+fastapi|import\\s+uvicorn)'

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
 codex/assess-technical-debt-in-ci-workflows-7cy9fp
PYTHON_BIN=${PYTHON:-}
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

 codex/assess-technical-debt-in-ci-workflows-jurxls

 codex/assess-technical-debt-in-ci-workflows-que3tv
 main
MATCHES=$(
python - "$FORBIDDEN_REGEX" "${EXISTING_TARGETS[@]}" <<'PY'
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
