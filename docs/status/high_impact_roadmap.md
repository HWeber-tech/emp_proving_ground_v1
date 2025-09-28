# High-impact roadmap status

The high-impact roadmap tracks the three document-driven streams called out in
[`docs/roadmap.md`](../roadmap.md). Run the helper to refresh the status table
before demos or reviews:

```bash
python -m tools.roadmap.high_impact --format markdown
```

| Stream | Status | Summary | Next checkpoint |
| --- | --- | --- | --- |
| Stream A – Institutional data backbone | Ready | Timescale ingest, Redis caching, Kafka streaming, and Spark exports ship with readiness telemetry and failover tooling. | Exercise cross-region failover and automated scheduler cutover using the readiness feeds. |
| Stream B – Sensory cortex & evolution uplift | Ready | All five sensory organs operate with drift telemetry and catalogue-backed evolution lineage exports. | Extend live-paper experiments and automated tuning loops using evolution telemetry. |
| Stream C – Execution, risk, compliance, ops readiness | Ready | FIX pilots, risk/compliance workflows, ROI telemetry, and operational readiness publish evidence for operators. | Expand broker connectivity with drop-copy reconciliation and extend regulatory telemetry coverage. |

For narrative reports or dashboards, export the detailed format to a companion
file:

```bash
python -m tools.roadmap.high_impact --format detail --output docs/status/high_impact_roadmap_detail.md
```

The command produces a stream-by-stream summary that includes the captured
evidence list for each initiative.
