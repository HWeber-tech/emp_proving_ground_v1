#!/usr/bin/env bash
set -euo pipefail

# Some CI harnesses inject a bogus $PYTHON override that points at a
# non-existent helper. Trying to execute the missing file produces a confusing
# "command not found" error before we have a chance to fall back to the real
# interpreter. Guard against this by clearing the override when it refers to a
# path that is not present so the resolver can progress to the usual detection
# logic.
if [[ -n "${PYTHON:-}" && "${PYTHON}" == */* && ! -e "${PYTHON}" ]]; then
  echo "Ignoring missing \$PYTHON override '${PYTHON}'." >&2
  unset PYTHON
fi

FORBIDDEN_REGEX='(ctrader_open_api|swagger|spotware|real_ctrader_interface|from[[:space:]]+fastapi|import[[:space:]]+fastapi|import[[:space:]]+uvicorn)'

if [ "$#" -gt 0 ]; then
  TARGETS=("$@")
else
  TARGETS=(
    "src"
    "scripts"
    "tests"
    "docs"
    "config"
    "strategies"
    "examples"
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
MATCHES=$(grep -RniE "$FORBIDDEN_REGEX" --exclude "$(basename "$0")" -- "${EXISTING_TARGETS[@]}" || true)

if [ -n "$MATCHES" ]; then
  echo "Forbidden references detected:" >&2
  echo "$MATCHES" >&2
  exit 1
fi

echo "No forbidden references found."
