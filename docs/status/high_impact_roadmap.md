# High-impact roadmap status

The high-impact tracker distills the encyclopedia alignment work into three
streams so release reviews and planning meetings can see gaps immediately.
Refresh the report before demos or milestone sign-off:

```bash
python -m tools.roadmap.high_impact --refresh-docs
```

Render a single format on demand when dashboards or briefs need a specific
payload:

```bash
python -m tools.roadmap.high_impact --format summary
python -m tools.roadmap.high_impact --format detail
python -m tools.roadmap.high_impact --format attention
```

JSON companions are emitted with the same flags (`--format …-json`). When
writing to alternate destinations, provide explicit paths:

```bash
python -m tools.roadmap.high_impact --refresh-docs \
  --summary-path /tmp/high_impact_summary.md \
  --detail-path /tmp/high_impact_detail.md \
  --attention-path /tmp/high_impact_attention.md
```

> **Reminder:** `--refresh-docs` always evaluates the full portfolio; combine it
> with explicit paths rather than stream filters so published status pages never
> omit a stream.

<!-- HIGH_IMPACT_PORTFOLIO:START -->
# High-impact roadmap summary

- Total streams: 3
- Ready: 0
- Attention needed: 3

The core abstractions mirror the encyclopedia narrative, but every stream still
relies on scaffolding or deprecated paths rather than production-grade
implementations.【F:docs/DEVELOPMENT_STATUS.md†L7-L35】【F:docs/technical_debt_assessment.md†L31-L112】

## Streams

### Stream A – Institutional data backbone

*Status:* Attention
*Summary:* The layered architecture and canonical `SystemConfig` exist, yet the
institutional ingest/caching/streaming stack remains mock-driven with no
Timescale, Redis, or Kafka services in operation.【F:docs/architecture/overview.md†L9-L48】【F:docs/DEVELOPMENT_STATUS.md†L19-L35】
*Next checkpoint:* Stand up a real ingest slice with parameterised SQL, managed
caches, supervised async tasks, and regression tests that exercise institutional
telemetry before declaring parity.【F:docs/development/remediation_plan.md†L34-L141】【F:docs/technical_debt_assessment.md†L33-L101】
*Actionable checklist:*
  - [ ] Provision production-grade Timescale/Redis/Kafka instances and connect them through supervised builders.
  - [ ] Parameterise all ingest SQL and remove `eval` usage flagged in the remediation plan.【F:docs/development/remediation_plan.md†L34-L72】
  - [ ] Add ingest telemetry regression tests and CI coverage beyond the current 76% baseline.【F:docs/ci_baseline_report.md†L8-L27】

### Stream B – Sensory cortex & evolution uplift

*Status:* Attention
*Summary:* Concept chapters are mapped to code stubs, but HOW/ANOMALY organs,
evolution pipelines, and catalogue integrations still ship as placeholders with
`NotImplementedError` paths and thin heuristics; the new integrated sensory organ
fuses WHY/WHAT/WHEN/HOW/ANOMALY signals with lineage and telemetry yet still
relies on synthetic feeds until institutional ingest lands. Fresh regression
coverage now proves the enhanced HOW organ discriminates bullish versus bearish
flows through the institutional engine wiring, but market validation remains
blocked on live ingest.【F:docs/DEVELOPMENT_STATUS.md†L19-L35】【F:src/sensory/real_sensory_organ.py†L20-L336】【F:tests/sensory/test_real_sensory_organ.py†L1-L107】【F:tests/sensory/test_dimension_organs.py†L130-L186】
*Next checkpoint:* Replace scaffolding with executable organs, wire lineage
telemetry, and complete the evolution engine so strategies can mutate against
real data feeds.【F:docs/development/remediation_plan.md†L92-L167】
*Actionable checklist:*
  - [ ] Implement executable HOW/ANOMALY sensory organs with documented drift metrics.
  - [ ] Seed catalogue-backed genomes and capture lineage telemetry for governance sign-off.
  - [ ] Extend regression suites to cover sensory drift and adaptive loops beyond FIX mocks.【F:docs/DEVELOPMENT_STATUS.md†L19-L35】

### Stream C – Execution, risk, compliance, ops readiness

*Status:* Attention
*Summary:* The order lifecycle mirrors the encyclopedia chapters, yet risk and
compliance enforcement are hollow, async entrypoints remain partially migrated,
and deprecated exports still leak into runtime consumers. Event bus shutdown now
runs under the task supervisor with explicit metadata and the final dry run
harness rotates structured/raw logs, emits typed progress snapshots, and ships a
watch CLI so operations can monitor multi-day rehearsals live, but deterministic
risk controls are still required before expanding pilots.【F:docs/technical_debt_assessment.md†L33-L121】【F:src/core/__init__.py†L14-L33】【F:docs/DEVELOPMENT_STATUS.md†L19-L35】【F:src/core/_event_bus_impl.py†L501-L590】【F:src/operations/final_dry_run.py†L307-L405】【F:src/operations/final_dry_run_progress.py†L15-L210】【F:tools/operations/final_dry_run_watch.py†L21-L176】【F:tests/operations/test_final_dry_run_progress.py†L12-L120】
*Next checkpoint:* Finish the runtime builder rollout, adopt supervised tasks,
enforce risk policies, and retire deprecated facades before expanding broker
coverage.【F:docs/technical_debt_assessment.md†L33-L101】【F:docs/development/remediation_plan.md†L34-L167】
*Actionable checklist:*
  - [ ] Complete the runtime builder migration and introduce a task supervision layer.
  - [ ] Enforce deterministic risk APIs and surface policy breaches through telemetry dashboards.
  - [x] Remove deprecated config shims and undefined exports (`get_risk_manager`) from public modules, locking the canonical facade behind regression coverage and audit evidence.【F:src/core/__init__.py†L17-L56】【F:tests/risk/test_risk_manager_impl_additional.py†L267-L277】【F:docs/reports/governance_risk_phase2_followup_audit.md†L1-L24】
<!-- HIGH_IMPACT_PORTFOLIO:END -->

<!-- HIGH_IMPACT_SUMMARY:START -->
| Stream | Status | Summary | Next checkpoint |
| --- | --- | --- | --- |
| Stream A – Institutional data backbone | Attention | Canonical layering exists, but institutional ingest/caching/streaming still run on mocks without Timescale/Redis/Kafka services. | Deploy real services, secure SQL paths, and instrument supervised telemetry across ingest tasks. |
| Stream B – Sensory cortex & evolution uplift | Attention | Encyclopedia dimensions map to stubs; HOW/ANOMALY organs and evolution loops remain incomplete. | Implement executable organs, connect lineage telemetry, and exercise evolution against live data feeds. |
| Stream C – Execution, risk, compliance, ops readiness | Attention | Lifecycle scaffolding is present, yet risk/compliance enforcement and runtime entrypoints remain hollow. | Complete the builder migration, enforce risk policies, and retire deprecated facades before expanding pilots. |
<!-- HIGH_IMPACT_SUMMARY:END -->

To export the detailed evidence companion file, render the detail view:

```bash
python -m tools.roadmap.high_impact --format detail \
  --output docs/status/high_impact_roadmap_detail.md
```

Persist the attention view alongside these reports when dashboards track gap
lists explicitly:

```bash
python -m tools.roadmap.high_impact --refresh-docs \
  --attention-path docs/status/high_impact_roadmap_attention.md
```
