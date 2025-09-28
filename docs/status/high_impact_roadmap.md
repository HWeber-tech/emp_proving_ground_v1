# High-impact roadmap status

The high-impact roadmap tracks the three document-driven streams called out in
[`docs/roadmap.md`](../roadmap.md). Run the helper to refresh the status table
before demos or reviews:

```bash
python -m tools.roadmap.high_impact --format markdown
```

To focus on specific streams, provide one or more ``--stream`` flags:

```bash
python -m tools.roadmap.high_impact --stream "Stream A – Institutional data backbone" --format detail
```

For an at-a-glance rollup of the portfolio, render the summary view:

```bash
python -m tools.roadmap.high_impact --format summary
```

For dashboards that only need aggregate counts, export the portfolio JSON view:

```bash
python -m tools.roadmap.high_impact --format portfolio-json
```

When triaging gaps, render the attention report to list any missing
requirements:

```bash
python -m tools.roadmap.high_impact --format attention
```

To feed dashboards that only need the missing requirements, emit the JSON
attention view:

```bash
python -m tools.roadmap.high_impact --format attention-json
```

To update both this summary and the detailed evidence companion file in one
shot, use the refresh flag:

```bash
python -m tools.roadmap.high_impact --refresh-docs
```

When writing to alternate locations (for example in CI workspaces), provide
explicit paths for the summary and detail files:

```bash
python -m tools.roadmap.high_impact --refresh-docs \
  --summary-path /tmp/high_impact_summary.md \
  --detail-path /tmp/high_impact_detail.md
```

<!-- HIGH_IMPACT_SUMMARY:START -->
| Stream | Status | Summary | Next checkpoint |
| --- | --- | --- | --- |
| Stream A – Institutional data backbone | Ready | Timescale ingest, Redis caching, Kafka streaming, and Spark exports ship with readiness telemetry and failover tooling. | Exercise cross-region failover and automated scheduler cutover using the readiness feeds. |
| Stream B – Sensory cortex & evolution uplift | Ready | All five sensory organs operate with drift telemetry and catalogue-backed evolution lineage exports. | Extend live-paper experiments and automated tuning loops using evolution telemetry. |
| Stream C – Execution, risk, compliance, ops readiness | Ready | FIX pilots, risk/compliance workflows, ROI telemetry, and operational readiness publish evidence for operators. | Expand broker connectivity with drop-copy reconciliation and extend regulatory telemetry coverage. |
<!-- HIGH_IMPACT_SUMMARY:END -->

For narrative reports or dashboards, export the detailed format to a companion
file:

```bash
python -m tools.roadmap.high_impact --format detail --output docs/status/high_impact_roadmap_detail.md
```

The command produces a stream-by-stream summary that includes the captured
evidence list for each initiative.

To persist the attention view alongside these reports, provide an explicit
destination (or rely on the default location under `docs/status/`):

```bash
python -m tools.roadmap.high_impact --refresh-docs \
  --attention-path docs/status/high_impact_roadmap_attention.md
```
