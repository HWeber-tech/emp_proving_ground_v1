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
  - Progress: Added guardrail regressions for the trading position model to
    assert timestamp updates, profit recalculations, and close flows so the
    lightweight execution telemetry remains deterministic under CI coverage.【F:tests/trading/test_position_model_guardrails.py†L1-L105】
  - Progress: System validation evaluator now normalises structured reports into
    readiness snapshots, annotates failing checks, derives alert events, and
    publishes via the failover helper under pytest coverage so dashboards retain
    degradation evidence even when the runtime bus misbehaves.【F:src/operations/system_validation.py†L1-L312】【F:tests/operations/test_system_validation.py†L1-L195】
  - Progress: Event bus failover helper now powers security, system validation,
    compliance readiness, incident response, and evolution experiment
    publishers, replacing ad-hoc blanket handlers with typed errors and
    structured logging so transport regressions escalate consistently across
    modules.【F:src/operations/event_bus_failover.py†L1-L174】【F:src/operations/incident_response.py†L675-L715】【F:src/operations/evolution_experiments.py†L297-L342】【F:tests/operations/test_event_bus_failover.py†L1-L164】【F:tests/operations/test_incident_response.py†L135-L179】【F:tests/operations/test_evolution_experiments.py†L135-L191】
  - Progress: Execution readiness telemetry now rides the shared failover
    helper, logging runtime publish failures, escalating unexpected exceptions,
    and falling back to the global bus under pytest coverage so dashboards keep
    receiving readiness snapshots even when the runtime transport degrades.【F:src/operations/execution.py†L611-L648】【F:tests/operations/test_execution.py†L100-L134】
  - Progress: Incident response readiness now evaluates policy/state mappings,
    emits Markdown snapshots, derives roster/backlog alerts, and publishes via
    the failover helper under pytest coverage so operators inherit actionable
    escalations instead of silent outages.【F:src/operations/incident_response.py†L1-L715】【F:tests/operations/test_incident_response.py†L1-L200】
  - Progress: Guardrail manifest tests pin the ingest orchestration, ingest
    scheduler, risk policy, and observability suites to the CI guardrail marker
    so coverage drops or marker drift block merges before the broader regression
    run, with pytest verifying target existence and guardrail tagging across the
    matrix.【F:tests/runtime/test_guardrail_suite_manifest.py†L18-L40】【F:tests/data_foundation/test_ingest_scheduler.py†L1-L28】
  - Progress: Event bus health tests now assert queue backlog escalation,
    dropped-event surfacing, and the shared failover helper’s runtime/global bus
    fallbacks so operational telemetry keeps raising alarms when both transports
    degrade.【F:src/operations/event_bus_health.py†L143-L259】【F:tests/operations/test_event_bus_health.py†L22-L235】
  - Progress: Strategy performance telemetry now normalises execution events,
    captures ROI/net PnL metadata, renders Markdown summaries, and publishes via
    the shared failover helper under pytest coverage so dashboards inherit the
    same hardened transport as other operational feeds.【F:src/operations/strategy_performance.py†L200-L531】【F:tests/operations/test_strategy_performance.py†L68-L193】
  - Progress: Health monitor probes guard optional psutil imports, log resource
    sampling failures, persist bounded histories, and surface event-bus
    snapshots, with asyncio regressions ensuring the loop logs unexpected errors
    instead of hanging silently.【F:src/operational/health_monitor.py†L61-L200】【F:tests/operational/test_health_monitor.py†L74-L176】
  - Progress: Risk telemetry panels now attach limit values, ratios, and
    violation states to observability dashboard entries while preserving the
    serialised payloads, with pytest coverage asserting limit-status escalation
    so operators inherit actionable risk summaries instead of opaque aggregates.【F:src/operations/observability_dashboard.py†L254-L309】【F:tests/operations/test_observability_dashboard.py†L201-L241】
  - Progress: Observability dashboard now exposes a remediation summary capsule
    that counts failing/warning/healthy panels and lists affected slices under
    regression coverage so CI exporters can consume a canonical operational
    readiness signal without recomputing severities.【F:src/operations/observability_dashboard.py†L60-L109】【F:tests/operations/test_observability_dashboard.py†L60-L116】
  - Progress: Operational readiness telemetry now enriches snapshots with
    per-status breakdowns and component maps so dashboards can render severity
    chips without reimplementing escalation logic, with pytest and docs locking
    the contract alongside the runtime exposure.【F:src/operations/operational_readiness.py†L200-L256】【F:tests/operations/test_operational_readiness.py†L1-L86】【F:docs/status/operational_readiness.md†L1-L34】【F:tests/runtime/test_professional_app_timescale.py†L722-L799】
  - Progress: Coverage matrix CLI now exposes lagging domains via the new
    `identify_laggards` helper, supports a `--fail-below-threshold` guardrail,
    and carries pytest coverage so CI can fail fast when coverage drops instead
    of relying on manual dashboards.【F:tools/telemetry/coverage_matrix.py†L184-L304】【F:tests/tools/test_coverage_matrix.py†L1-L182】
  - Progress: Observability dashboard surfaces operational readiness as a
    first-class panel, counting component severities, embedding metadata, and
    feeding remediation summaries under pytest coverage so responders inherit a
    consolidated operational view without bespoke wiring.【F:src/operations/observability_dashboard.py†L443-L493】【F:tests/operations/test_observability_dashboard.py†L135-L236】
  - Progress: Configuration audit telemetry now evaluates `SystemConfig` diffs,
    annotates tracked toggles and extras, renders Markdown summaries, and
    publishes via the shared failover helper so configuration changes generate a
    durable audit trail for dashboards and governance reviews under pytest
    coverage.【F:src/operations/configuration_audit.py†L1-L235】【F:tests/operations/test_configuration_audit.py†L1-L164】
  - Progress: Ingest trend telemetry logging now records runtime publish
    fallbacks, raises on unexpected errors, and escalates global bus outages with
    pytest coverage so data backbone dashboards expose genuine gaps instead of
    silently skipping degraded snapshots.【F:src/operations/ingest_trends.py†L303-L336】【F:tests/operations/test_ingest_trends.py†L90-L148】
  - Progress: Timescale ingest regressions now cover migrator bootstrap,
    idempotent upserts for empty/changed plans, macro event ingestion, and the
    backbone orchestrator’s lifecycle/metadata guardrails so coverage catches
    silent failures across institutional ingest windows.【F:tests/data_foundation/test_timescale_ingest.py†L1-L359】【F:tests/data_foundation/test_timescale_backbone_orchestrator.py†L1-L200】
  - Progress: Cache health publishing now logs primary bus errors, only falls back
    when runtime failures occur, and raises on unexpected or global-bus outages
    under pytest guardrails so readiness telemetry surfaces real cache incidents
    instead of quietly failing to publish.【F:src/operations/cache_health.py†L143-L245】【F:tests/operations/test_cache_health.py†L15-L138】
- Progress: Operational metrics instrumentation now has targeted regressions for
  logging escalation, lazy gauge fallbacks, Prometheus exporter idempotence, and
  registry sink adapters so CI surfaces metric failures deterministically and
  remediation plans inherit documented evidence. Latest coverage exercises the
  failure fallback hook, sanitised FIX wrappers, and latency bounds so telemetry
  captures degraded instrumentation instead of silently dropping metrics.【F:src/operational/metrics.py†L1-L200】【F:tests/operational/test_metrics.py†L200-L328】
- Progress: Bootstrap stack now logs sensory listener, liquidity prober, and
  control-centre callback failures with structured metadata so optional hooks
  surface errors without disrupting bootstrap decisions, under pytest coverage
  that captures the emitted diagnostics.【F:src/orchestration/bootstrap_stack.py†L81-L258】【F:tests/current/test_bootstrap_stack.py†L164-L213】
- Wire Slack/webhook mirrors for CI alerts, rehearse the forced-failure drill,
  and record MTTA/MTTR in the health dashboard per the operational telemetry
  stream roadmap.【F:docs/technical_debt_assessment.md†L156-L174】【F:docs/status/ci_health.md†L74-L76】
- Refresh CI dashboard rows as telemetry lands, noting validation hooks and
  outstanding actions so stakeholders see live gaps (e.g., sensory fixture
  rollout, ingest metrics coverage).【F:docs/status/ci_health.md†L21-L76】
- Progress: CI now runs a dedicated guardrail marker job ahead of the coverage
  sweep so ingest, risk, and observability guardrails run in isolation and fail
  fast when regressions surface, with the workflow and pytest marker contract
  documenting the enforced scope.【F:.github/workflows/ci.yml†L79-L123】【F:pytest.ini†L1-L25】【F:tests/data_foundation/test_timescale_backbone_orchestrator.py†L1-L28】

- Progress: Sensory drift telemetry publisher now routes through the shared
  event-bus failover helper, logging runtime and global-bus degradations while
  tests assert the fallback contract so operators keep receiving deterministic
  drift alerts when the primary transport misbehaves.【F:src/operations/sensory_drift.py†L247-L276】【F:tests/operations/test_sensory_drift.py†L17-L163】
- Progress: System validation snapshots now attach failing-check names and
  messages to metadata and Markdown while reusing the shared failover helper so
  operational dashboards display the exact broken checks even during runtime bus
  degradation, with pytest covering metadata capture and failover paths.【F:src/operations/system_validation.py†L127-L321】【F:tests/operations/test_system_validation.py†L77-L160】

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
- Domain snapshots are now captured alongside overall coverage in
  `tests/.telemetry/ci_metrics.json`, flagging lagging domains directly in CI
  telemetry so remediation progress is visible without ad-hoc parsing.
  【F:tools/telemetry/ci_metrics.py†L1-L212】【F:tests/tools/test_ci_metrics.py†L1-L236】
- Coverage telemetry now records lagging-domain counts, worst-performing
  slices, and observability dashboard remediation snapshots via the CLI so
  dashboards inherit actionable readiness deltas alongside coverage trendlines.
  【F:tools/telemetry/ci_metrics.py†L112-L337】【F:tools/telemetry/update_ci_metrics.py†L1-L169】【F:tests/tools/test_ci_metrics.py†L180-L309】
- Remediation progress snapshots now live alongside coverage/formatter trendlines
  in `tests/.telemetry/ci_metrics.json` thanks to the `--remediation-status`
  CLI, capturing roadmap evidence (label, statuses, source, notes) for dashboards
  and audits with pytest guarding the JSON contract.【F:tools/telemetry/update_ci_metrics.py†L1-L184】【F:tools/telemetry/ci_metrics.py†L1-L210】【F:tests/tools/test_ci_metrics.py†L1-L340】【F:tests/.telemetry/ci_metrics.json†L1-L5】
- Remediation summary exporter reads the metrics feed, renders Markdown tables
  with delta call-outs, honours slice limits, omits deltas for non-numeric
  statuses, and ships with a CLI/pytest contract so status updates and briefs
  stay in sync without manual collation.【F:tools/telemetry/remediation_summary.py†L1-L220】【F:tests/tools/test_remediation_summary.py†L22-L125】
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
