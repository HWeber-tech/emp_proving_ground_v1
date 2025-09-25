#!/usr/bin/env bash
set -euo pipefail

LEGACY_WRAPPER="codex/assess-technical-debt-in-ci-workflows"

if command -v "$LEGACY_WRAPPER" >/dev/null 2>&1; then
  exec "$LEGACY_WRAPPER" check-forbidden-integrations "$@"
elif [ -x "$LEGACY_WRAPPER" ]; then
  exec "$LEGACY_WRAPPER" check-forbidden-integrations "$@"
fi

FORBIDDEN_REGEX='(ctrader_open_api|swagger|spotware|real_ctrader_interface|from[[:space:]]+fastapi|import[[:space:]]+fastapi|import[[:space:]]+uvicorn)'

if [ "$#" -gt 0 ]; then
  TARGETS=("$@")
else
  TARGETS=("src" "tests/current")
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
MATCHES=$(grep -RniE "$FORBIDDEN_REGEX" -- "${EXISTING_TARGETS[@]}" || true)

if [ -n "$MATCHES" ]; then
  echo "Forbidden references detected:" >&2
  echo "$MATCHES" >&2
  exit 1
fi

echo "No forbidden references found."
