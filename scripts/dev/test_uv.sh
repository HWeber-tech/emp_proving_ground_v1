#!/usr/bin/env bash
set -euo pipefail

# Ensure uvx is available; install uv to user space if missing
if ! command -v uvx >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# Defaults for local runs
export PYTHONPATH="${PYTHONPATH:-.}"
export EMP_USE_MOCK_FIX="${EMP_USE_MOCK_FIX:-1}"

# If no args provided, default to tests/current
if [ "$#" -eq 0 ]; then
  set -- tests/current
fi

# Run pytest via uvx in an isolated environment (no system changes)
uvx pytest -q "$@"