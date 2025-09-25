#!/usr/bin/env bash
set -euo pipefail

DEFAULT_TARGETS=(
  src
  tests/current
  tests
  scripts
  docs
  tools
)

 codex/rewrite-forbidden-integration-scanner-script
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON:-python3}

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Unable to locate Python interpreter '$PYTHON_BIN'." >&2
  exit 1
fi

FORBIDDEN_REGEX='(?i)(ctrader[-_]?open[-_]?api|ctraderapi\\.com|connect\\.icmarkets\\.com|swagger|spotware|real_ctrader_interface|from\\s+fastapi|import\\s+fastapi|import\\s+uvicorn)'

ALLOWLIST=(
  scripts/check_forbidden_integrations.sh
  scripts/check_forbidden_integrations.py
  scripts/phase1_deduplication.py
)
 main

if [[ $# -gt 0 ]]; then
  TARGETS=("$@")
else
  TARGETS=("${DEFAULT_TARGETS[@]}")
fi

if [[ -z "${TARGETS[*]}" ]]; then
  echo "No targets specified for forbidden integration scan." >&2
  exit 0
fi

 codex/rewrite-forbidden-integration-scanner-script
"$PYTHON_BIN" "$SCRIPT_DIR/check_forbidden_integrations.py" "${TARGETS[@]}"

PYTHON_BIN=${PYTHON:-python3}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Unable to locate Python interpreter '$PYTHON_BIN'." >&2
  exit 1
fi

ALLOWLIST_PAYLOAD=$(printf '%s\n' "${ALLOWLIST[@]}")
SCAN_OUTPUT=$(
  ALLOWLIST="$ALLOWLIST_PAYLOAD" \
    "$PYTHON_BIN" - "$FORBIDDEN_REGEX" "${TARGETS[@]}" <<'PY'
import os
import re
import sys
from pathlib import Path

pattern = re.compile(sys.argv[1])
raw_targets = sys.argv[2:]

allowlist_entries = {
    Path(entry).resolve().as_posix()
    for entry in os.environ.get("ALLOWLIST", "").splitlines()
    if entry.strip()
}

extensions = {
    ".cfg",
    ".ini",
    ".ipynb",
    ".md",
    ".mdx",
    ".py",
    ".pyi",
    ".rst",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

root = Path.cwd().resolve()

if not raw_targets:
    raw_targets = [str(root)]

matches: list[str] = []


def iter_files(base: Path):
    if base.is_file():
        yield base
        return
    for candidate in base.rglob("*"):
        if candidate.is_file():
            yield candidate


for target in raw_targets:
    base = (root / target).resolve() if not target.startswith("/") else Path(target).resolve()
    if not base.exists():
        continue
    for candidate in iter_files(base):
        if candidate.suffix.lower() not in extensions:
            continue
        resolved_path = candidate.resolve().as_posix()
        if resolved_path in allowlist_entries:
            continue
        try:
            with candidate.open("r", encoding="utf-8", errors="ignore") as handle:
                for lineno, line in enumerate(handle, start=1):
                    if pattern.search(line):
                        rel_path = candidate.resolve().relative_to(root).as_posix()
                        matches.append(f"{rel_path}:{lineno}:{line.rstrip()}".rstrip())
        except OSError:
            continue

if matches:
    print("Forbidden integration references detected:")
    print("\n".join(matches))
    sys.exit(1)

print("No forbidden integration references found.")
PY
)

SCAN_STATUS=$?

echo "$SCAN_OUTPUT"

exit $SCAN_STATUS
 main
