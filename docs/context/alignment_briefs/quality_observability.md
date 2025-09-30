# Alignment brief – Quality & observability guardrails

## Concept promise

- The encyclopedia frames the data foundation as the bedrock that must deliver
  reliable, high-quality, real-time data to every upper layer, reinforcing that
  perception, execution, and governance rely on disciplined validation and
  monitoring.【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L1710-L1779】
- Architecture guidance elevates orchestration and operational domains as
  first-class surfaces responsible for policy enforcement and observability
  endpoints that keep institutional operators informed.【F:docs/architecture/overview.md†L11-L60】

## Reality snapshot

- CI passes with 76% coverage, yet operational metrics, trading positions, data
  foundation loaders, and sensory signals retain large untested regions, leaving
  regression gaps across critical pathways.【F:docs/ci_baseline_report.md†L8-L27】
- Debt assessments flag testing and observability as high-risk: coverage remains
  fragile at 76%, flaky-test telemetry lacks downstream alerting, and the alert
  channel drill plus Slack/webhook mirrors are still pending.【F:docs/technical_debt_assessment.md†L19-L112】
- The CI health dashboard captures numerous telemetry feeds (risk, ingest,
  sensory, evolution, operational readiness), but several entries call out
  follow-up work such as extending regression suites, wiring Slack relays, and
  surfacing sensory fixtures, underscoring that observability remains
  incomplete.【F:docs/status/ci_health.md†L10-L108】

## Gap themes

1. **Regression depth** – Expand deterministic coverage across ingest, sensory,
   risk, and runtime orchestration, closing the hotspots highlighted in the CI
   baseline and technical debt assessment.
2. **Telemetry fidelity** – Ensure every roadmap surface (ingest, risk,
   compliance, evolution, operations) exposes actionable metrics with
   provenance, drill guides, and slack/webhook relays.
3. **Status hygiene** – Keep CI dashboards, regression backlog, and context
   packs synchronised with delivery so reviewers inherit accurate narrative and
   validation pointers.

## Delivery plan

### Now (0–30 days)

- Add regression tickets/tests for uncovered modules (`operational.metrics`,
  `trading.models.position`, sensory WHY organ) and capture ownership in the
  regression backlog so coverage gains can be tracked.【F:docs/ci_baseline_report.md†L18-L27】【F:docs/technical_debt_assessment.md†L133-L154】
  - Progress: Added a pytest regression for the SQLite-backed portfolio monitor
    covering position lifecycle management and metrics generation so trading
    telemetry now participates in CI coverage.【F:tests/trading/test_real_portfolio_monitor.py†L1-L77】
- Wire Slack/webhook mirrors for CI alerts, rehearse the forced-failure drill,
  and record MTTA/MTTR in the health dashboard per the operational telemetry
  stream roadmap.【F:docs/technical_debt_assessment.md†L156-L174】【F:docs/status/ci_health.md†L74-L76】
- Refresh CI dashboard rows as telemetry lands, noting validation hooks and
  outstanding actions so stakeholders see live gaps (e.g., sensory fixture
  rollout, ingest metrics coverage).【F:docs/status/ci_health.md†L21-L76】

### Next (30–90 days)

- Re-enable quarantined pytest suites behind feature flags, extending coverage
  to legacy/integration flows while monitoring flake telemetry for drift.
  【F:docs/technical_debt_assessment.md†L84-L112】
- Instrument CI summaries with ingest freshness, Kafka lag, risk policy
  violations, and sensory drift deltas, exporting machine-readable snapshots for
  dashboards and compliance audits.【F:docs/status/ci_health.md†L21-L73】【F:docs/status/ci_health.md†L95-L108】
- Publish regression progress and telemetry deltas in weekly status updates,
  linking roadmap checkmarks to the refreshed briefs so discovery stays aligned.

### Later (90+ days)

- Establish continuous validation pipelines that gate deployments on coverage,
  telemetry thresholds, and alert-drill freshness, mirroring the long-horizon
  testing & observability mitigation plan.【F:docs/technical_debt_assessment.md†L95-L112】
- Automate documentation freshness checks (roadmap, briefs, CI health) to flag
  stale sections and open follow-up issues when telemetry or coverage regresses.
- Fold quality guardrails into professional readiness reporting so operators
  see coverage drift, flake spikes, and alert posture alongside ingest health.

## Validation & telemetry

- Track coverage deltas via `tests/.telemetry/ci_metrics.json` and surface them
  in dashboards; add assertions in regression suites to prevent silent drops.
  【F:docs/status/ci_health.md†L13-L15】
- Maintain CI dashboard entries for ingest, risk, sensory, evolution, and
  operational telemetry, updating notes/tests as suites land so reviewers can
  trace validation hooks directly from the brief.【F:docs/status/ci_health.md†L21-L73】
- Ensure Slack/webhook alert mirrors remain configured and documented so CI
  failures reach responders promptly, with drill results recorded in the
  dashboard maintenance checklist.【F:docs/status/ci_health.md†L74-L108】

## Dependencies & coordination

- Regression expansion depends on ingest/risk/sensory/evolution squads landing
  executable organs and telemetry as documented in their respective briefs.
- Operational readiness teams must keep runbooks and alert channels aligned so
  new telemetry (e.g., Kafka lag, compliance readiness) feeds the same incident
  response workflow.【F:docs/technical_debt_assessment.md†L156-L174】【F:docs/status/ci_health.md†L31-L76】
