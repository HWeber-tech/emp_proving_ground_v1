#!/usr/bin/env bash
set -euo pipefail

FORBIDDEN_REGEX='(ctrader_open_api|swagger|spotware|real_ctrader_interface|from[[:space:]]+fastapi|import[[:space:]]+fastapi|import[[:space:]]+uvicorn)'

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
