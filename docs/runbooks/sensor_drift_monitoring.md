# Sensor Drift Monitoring Harness

The high-impact roadmap calls for proactive anomaly detection across the sensory
stack so trading strategies receive trustworthy features.  This runbook explains
how to run the new sensor drift harness, interpret its output, and fold the
results into operational workflows.

## Overview

* **Module:** `src/sensory/monitoring/sensor_drift.py`
* **CLI:** `scripts/check_sensor_drift.py`
* **Goal:** Compare recent sensory windows against a historical baseline using a
  pooled z-score heuristic. Any sensor whose absolute z-score exceeds the
  configured threshold is flagged for investigation.

## Running the CLI

```bash
python scripts/check_sensor_drift.py data/sensory_snapshot.csv \
  --baseline 240 \
  --evaluation 60 \
  --z-threshold 3.0 \
  --fail-on-drift \
  --output artifacts/sensory/sensor_drift_summary.json
```

* `--baseline` controls how many historical observations form the expectation
  window.
* `--evaluation` defines the most recent window used for drift detection.
* `--z-threshold` sets the absolute z-score above which sensors are marked as
  drifting.
* `--fail-on-drift` exits with a non-zero code when any sensor breaches the
  threshold — useful for CI guards.
* `--sensors` limits the analysis to a subset of columns. When omitted, all
  numeric columns are analysed.

The CLI prints a concise summary to stdout and can optionally persist the full
JSON payload for dashboards or post-mortem analysis.

## Automation Guidance

1. Schedule the CLI after nightly sensory replays so baseline windows incorporate
   the most recent clean data.
2. Store JSON summaries under `artifacts/sensory/` and surface them in the
   observability dashboard alongside latency metrics.
3. Feed flagged sensors into strategy gating logic (e.g., disable strategies
   that depend on a drifting indicator until it recovers).
4. Combine with the existing structured logging pipeline so drift events inherit
   correlation identifiers and show up in the OpenTelemetry stack.

## Live Diagnostics Helper

For deeper live-feed analysis, use the diagnostics module introduced in
`src/sensory/monitoring/live_diagnostics.py`. It replays market data through the
`RealSensoryOrgan`, captures anomaly posture, drift telemetry, and WHY sensor
quality, and returns a typed `LiveSensoryDiagnostics` payload ready for JSON
storage or Markdown evidence.

```python
from src.sensory.monitoring import build_live_sensory_diagnostics
from src.sensory.real_sensory_organ import RealSensoryOrgan, SensoryDriftConfig

organ = RealSensoryOrgan(
    drift_config=SensoryDriftConfig(baseline_window=20, evaluation_window=8),
)
diagnostics = build_live_sensory_diagnostics(live_frame, organ=organ)
report = diagnostics.as_dict()
```

When institutional credentials are available, call
`build_live_sensory_diagnostics_from_manager` with a `RealDataManager` to fetch
fresh Timescale/Redis/Kafka slices before building the report. Operators can add
the resulting JSON files to the daily perception evidence pack referenced by the
`perception_live_feeds` governance gate.

## Acceptance Criteria

* Unit tests in `tests/sensory/test_sensor_drift.py` cover drift detection,
  column selection, and guardrails for insufficient data.
* The CLI supports CSV, JSON, and Parquet inputs with helpful error messages.
* Documentation in this runbook plus the roadmap checklist is updated to reflect
  the delivered capability.
