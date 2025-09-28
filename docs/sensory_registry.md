# Sensory registry

This registry summarises the high-impact sensory organs and their
configuration surfaces. Regenerate via `python -m tools.sensory.registry`.

## HOW – sensory.how.how_sensor.HowSensor

Bridge the institutional HOW engine into the legacy sensory pipeline.

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
conditions.

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

Macro proxy sensor (WHY dimension) with yield-curve awareness.

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
