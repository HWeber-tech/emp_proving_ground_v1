# Market Microstructure Observatory

_Generated at 2025-09-30T06:08:10.981603 UTC_

## Session Summary
- **Symbol**: EURUSD
- **Duration (minutes)**: 1
- **Total Updates**: 1800
- **Capture Timestamp**: 2025-07-23T07:43:22.996789+00:00

### Latency Profile
- **Mean Ms**: 45.2
- **Median Ms**: 42.1
- **P95 Ms**: 78.5
- **Min Ms**: 12.3
- **Max Ms**: 156.8
- **Std Ms**: 18.7

### Depth Snapshot
- **Bid Levels**: {'mean': 5.8, 'min': 5, 'max': 6, 'std': 0.4}
- **Ask Levels**: {'mean': 5.9, 'min': 5, 'max': 6, 'std': 0.3}

### Update Frequency
- **Updates Per Second**: 1.0
- **Updates Per Minute**: 60.0

## Observability Notes
- Latency metrics are sourced from FIX round-trip measurements.
- Depth statistics summarise top-of-book levels captured in the raw sample.
- Frequency section highlights heartbeat cadence for monitoring drift.

## Next Actions
- Attach the rendered Markdown to the ops dashboard release notes.
- Feed latency anomalies into `scripts/status_metrics.py` for alerting.
- Enrich this report with venue-specific liquidity sweeps when new data arrives.
