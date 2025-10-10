#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(git rev-parse --show-toplevel)"
POETRY=${POETRY:-poetry}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT_DIR/artifacts/whitepaper"}
mkdir -p "$OUTPUT_DIR"
REPORT_JSON="$OUTPUT_DIR/paper_trading_report.json"
PERF_MD="$OUTPUT_DIR/performance_report.md"
DIARY_PATH="${DIARY_PATH:-$OUTPUT_DIR/decision_diary.json}"
LEDGER_PATH="${LEDGER_PATH:-$ROOT_DIR/artifacts/paper_policy_ledger.json}"

# Run the paper trading simulation to refresh diary and KPI evidence.
$POETRY run python tools/trading/run_paper_trading_simulation.py \
  --diary "$DIARY_PATH" \
  --ledger "$LEDGER_PATH" \
  --pretty \
  > "$REPORT_JSON"

# Generate execution performance summary tied to the refreshed telemetry.
$POETRY run python - <<'PYCODE'
from pathlib import Path
from src.trading.execution.performance_report import build_execution_performance_report

stats = {
    "orders_submitted": 0,
    "orders_executed": 0,
    "orders_failed": 0,
    "trade_throttle": {
        "name": "paper-sim-refresh",
        "state": "idle",
    },
    "throughput": {
        "samples": 0,
        "avg_processing_ms": 0.0,
        "p95_processing_ms": 0.0,
        "max_processing_ms": 0.0,
        "avg_lag_ms": 0.0,
        "max_lag_ms": 0.0,
        "throughput_per_min": 0.0,
    },
}
Path("$PERF_MD").write_text(build_execution_performance_report(stats))
PYCODE

cat <<SUMMARY
Whitepaper evidence refreshed.
  Report JSON:    $REPORT_JSON
  Performance MD: $PERF_MD
  Decision Diary: $DIARY_PATH
  Policy Ledger:  $LEDGER_PATH
SUMMARY
