# Sensory registry

This registry summarises the high-impact sensory organs and their
configuration surfaces. Regenerate via `python -m tools.sensory.registry`.

## HOW – sensory.how.how_sensor.HowSensor

Bridge the institutional HOW engine into the legacy sensory pipeline. The
sensor now enriches telemetry with order book analytics (imbalance, value
area, participation ratio) derived from `OrderBookAnalytics` snapshots and
ICT microstructure detections (fair value gaps, liquidity sweeps) sourced from
`ICTPatternAnalyzer`.

### Configuration

Configuration for the HOW sensor calibration thresholds.

| Field | Type | Default |
| --- | --- | --- |
| `minimum_confidence` | float | 0.2 |
| `warn_threshold` | float | 0.35 |
| `alert_threshold` | float | 0.65 |

## WHAT – sensory.what.what_sensor.WhatSensor

Pattern sensor (WHAT dimension).

*This sensor does not expose configuration parameters.*

## WHEN – sensory.when.when_sensor.WhenSensor

Temporal/context sensor (WHEN dimension).

Combines session intensity, macro event proximity, and option gamma posture to
reflect the temporal edge of acting right now versus waiting for better
conditions. Session analytics expose active/overlapping venues and countdowns
to the next opening bell for strategy scheduling.

### Configuration

Configuration for the WHEN sensor scoring function.

| Field | Type | Default |
| --- | --- | --- |
| `minimum_confidence` | float | 0.2 |
| `session_weight` | float | 0.4 |
| `news_weight` | float | 0.3 |
| `gamma_weight` | float | 0.3 |
| `news_decay_minutes` | int | 120 |
| `gamma_near_fraction` | float | 0.01 |
| `gamma_pressure_normalizer` | float | 50000.0 |

## WHY – sensory.why.why_sensor.WhySensor

Macro proxy sensor (WHY dimension) with yield-curve awareness plus narrative
hooks that aggregate economic calendar sentiment and macro regime flags.

*This sensor does not expose configuration parameters.*

## ANOMALY – sensory.anomaly.anomaly_sensor.AnomalySensor

Detect displacements in price/volume behaviour for the ANOMALY dimension.

### Configuration

Configuration for anomaly detection thresholds.

| Field | Type | Default |
| --- | --- | --- |
| `window` | int | 32 |
| `warn_threshold` | float | 0.4 |
| `alert_threshold` | float | 0.7 |
