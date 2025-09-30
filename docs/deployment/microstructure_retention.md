# Microstructure Archive Retention Guide

The high-impact roadmap mandates archiving market microstructure datasets in
tiered storage so the freshest order-flow analytics remain close to the runtime
while historical studies and compliance reviews leverage low-cost object
storage.  The `MicrostructureArchive` utilities implement this policy by
splitting retention across a "hot" and "cold" tier with defaults sourced from the
EMP Encyclopedia cost matrix.

## Default Retention Targets

| Tier | Storage class | Retention | Notes |
| --- | --- | --- | --- |
| Hot | Tier-0 SSD / DuckDB cache | 14 days | Keeps the latest intraday order-flow snapshots available for trading model calibration and anomaly investigations without incurring additional infrastructure spend. |
| Cold | Tier-1 object storage (OCI/B2/S3) | 365 days | Archives monthly roll-ups for compliance, research replay, and encyclopedia-aligned cost efficiency. |

Use the helper in `src/data_foundation/microstructure/archive.py` to embed these
expectations inside automation:

```python
from pathlib import Path
from src.data_foundation.microstructure import (
    MicrostructureArchive,
    MicrostructureArchiveConfig,
    build_retention_guidance,
)

config = MicrostructureArchiveConfig(
    hot_path=Path("data/microstructure/hot"),
    cold_path=Path("data/microstructure/cold"),
)
archive = MicrostructureArchive(config)

# Persist a snapshot emitted by MarketMicrostructureAnalyzer
archive.archive("EUR/USD", footprint_records)

# Promote hot files to cold storage and prune expired data
archive.enforce_retention()

# Render Markdown guidance for ops runbooks or dashboards
hot_guidance, cold_guidance = build_retention_guidance(config)
```

Operators should schedule `archive.enforce_retention()` daily.  The helper
promotes files older than three days into the cold tier and automatically prunes
snapshots beyond each tier’s retention window.  Adjust the durations to match
desk-specific policies, but keep the hot tier small enough to satisfy the Tier-0
budget envelope documented in the encyclopedia.

## Integration Tips

- Store the archive paths inside `config/data_foundation/` so deployments across
dev/staging/prod remain reproducible.
- Mirror the retention summary inside the professional runtime by converting the
output of `build_retention_guidance` into a Markdown block alongside existing
data backbone telemetry.
- When exporting microstructure studies to research notebooks, read from the
cold tier to avoid disturbing the latest operational snapshots.
