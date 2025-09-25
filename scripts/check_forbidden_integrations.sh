#!/usr/bin/env bash
set -euo pipefail

DEFAULT_TARGETS=(src tests/current)

if [ "$#" -gt 0 ]; then
  TARGETS=("$@")
else
  TARGETS=("${DEFAULT_TARGETS[@]}")
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

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
else
  echo "Unable to locate python interpreter to run forbidden integration scan." >&2
  exit 127
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
