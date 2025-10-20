#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export RUN_MODE=${RUN_MODE:-mock}
export CONFIRM_LIVE=${CONFIRM_LIVE:-false}
export CONNECTION_PROTOCOL=${CONNECTION_PROTOCOL:-bootstrap}
export DATA_BACKBONE_MODE=${DATA_BACKBONE_MODE:-bootstrap}
export EMP_KILL_SWITCH=${EMP_KILL_SWITCH:-disabled}
export EMP_USE_MOCK_FIX=${EMP_USE_MOCK_FIX:-1}

python main.py "$@"
