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

1. **Decision loop narration** – Expand deterministic coverage across ingest,
   sensory, risk, and runtime orchestration while attaching narrated decision
   trails, policy ledger provenance, and sigma stability checkpoints so the new
   fast-weight loop exposes auditable reasoning alongside coverage gains.
2. **Drift detection & throttling** – Elevate sensory drift detectors,
   Page–Hinkley monitors, and throttle-state transitions into first-class
   telemetry streams with SLO instrumentation, Prometheus exports, and replay
   determinism checks that prove throttles activate before sigma instability
   breaches.
3. **Observability proof surface** – Keep CI dashboards, regression backlog,
   diary schema artefacts, and validation packets synchronised with delivery so
   reviewers inherit decision narration, policy provenance, and validation hooks
   for the drift/throttle surfaces without spelunking ad-hoc logs.

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
    degradation evidence even when the runtime bus misbehaves.【F:src/operations/system_validation.py†L233-L889】【F:tests/operations/test_system_validation.py†L1-L195】
  - Progress: Event bus failover helper now powers security, system validation,
    compliance readiness, incident response, evolution experiment, and evolution
    tuning publishers, replacing ad-hoc blanket handlers with typed errors and
    structured logging so transport regressions escalate consistently across
    modules.【F:src/operations/event_bus_failover.py†L1-L174】【F:src/operations/incident_response.py†L675-L715】【F:src/operations/evolution_experiments.py†L297-L342】【F:src/operations/evolution_tuning.py†L410-L433】【F:tests/operations/test_event_bus_failover.py†L1-L164】【F:tests/operations/test_incident_response.py†L135-L179】【F:tests/operations/test_evolution_experiments.py†L135-L191】【F:tests/operations/test_evolution_tuning.py†L226-L281】
  - Progress: Dry-run audit ingestion now streams gz/bz2/xz JSONL archives and trims trailing `Z` suffixes from diary timestamps so compressed evidence packets drop straight into observability dashboards, with regression coverage over gzip/bz2 fixtures and the diary normaliser.【F:src/operations/dry_run_audit.py†L31-L671】【F:src/understanding/decision_diary.py†L41-L54】【F:tests/operations/test_dry_run_audit.py†L1-L47】
  - Progress: Async event-bus diagnostics now expose `task_snapshots()` (with TopicBus delegation) and the professional runtime records the resulting metadata under `event_bus_tasks`, giving observers a live roster of supervised worker and handler tasks while keeping the probe best-effort; regression coverage asserts the metadata contract and summary wiring.【F:src/core/_event_bus_impl.py†L291-L708】【F:src/runtime/predator_app.py†L1491-L1512】【F:tests/current/test_event_bus_task_supervision.py†L103-L156】【F:tests/current/test_runtime_professional_app.py†L433-L461】
  - Progress: Execution readiness telemetry now rides the shared failover
    helper, logging runtime publish failures, escalating unexpected exceptions,
    and falling back to the global bus under pytest coverage so dashboards keep
    receiving readiness snapshots even when the runtime transport degrades.【F:src/operations/execution.py†L611-L648】【F:tests/operations/test_execution.py†L100-L134】
  - Progress: Thinking-model normalisers now guard against exploding `.dict()`
    payloads and attribute errors, with dedicated regression tests proving the
    defensive paths so AlphaTrade narration surfaces keep ingesting predictive and
    adversarial outputs without flake-prone casts.【F:src/thinking/models/normalizers.py†L26-L182】【F:tests/thinking/test_normalizers.py†L1-L85】
  - Progress: Red-team and competitive intelligence mappers now harden their
    `_to_mapping` helpers with defensive attribute access, structured warnings on
    dict failures, and regression tests that exercise Phase-0 security scenarios so
    adversarial outputs keep flowing to telemetry without silent drops.【F:src/thinking/adversarial/red_team_ai.py†L22-L91】【F:src/thinking/competitive/competitive_intelligence_system.py†L33-L117】【F:tests/thinking/test_security_hardening_phase0.py†L30-L48】
  - Progress: Competitive understanding storage now writes to `emp:competitive_understanding:*` and emits `UnderstandingReportTD` payloads with only the canonical `understanding_id`, removing the legacy alias so telemetry dashboards and downstream consumers adopt the updated schema with guardrail coverage for straggling imports.【F:src/thinking/competitive/competitive_intelligence_system.py†L868-L885】【F:src/thinking/models/types.py†L61-L69】【F:tests/current/test_understanding_validator.py†L8-L16】
  - Progress: Phase 3 intelligence orchestrator mirrors the canonical `competitive_understanding` payload onto the legacy `competitive_intelligence` key while async guardrails assert both paths resolve the same telemetry, letting dashboards migrate safely without breaking older consumers.【F:src/intelligence/__init__.py†L212-L284】【F:tests/intelligence/test_phase3_intelligence_orchestrator.py†L1-L83】
  - Progress: Incident response readiness now evaluates policy/state mappings,
    emits Markdown snapshots, derives roster/backlog alerts, and publishes via
    the failover helper under pytest coverage so operators inherit actionable
    escalations instead of silent outages.【F:src/operations/incident_response.py†L1-L715】【F:tests/operations/test_incident_response.py†L1-L200】
  - Progress: Guardrail manifest tests pin the ingest orchestration, ingest
    scheduler, risk policy, and observability suites to the CI guardrail marker,
    and assert the workflow runs `pytest -m guardrail` plus enumerates guardrail
    domains in the coverage sweep so marker drift, missing files, or workflow
    regressions block merges before the broader regression run.【F:tests/runtime/test_guardrail_suite_manifest.py†L18-L91】【F:tests/data_foundation/test_ingest_scheduler.py†L1-L28】
  - Progress: Event bus health tests now assert queue backlog escalation,
    dropped-event surfacing, and the shared failover helper’s runtime/global bus
    fallbacks so operational telemetry keeps raising alarms when both transports
    degrade.【F:src/operations/event_bus_health.py†L143-L259】【F:tests/operations/test_event_bus_health.py†L22-L235】
  - Progress: Strategy performance telemetry now normalises execution events,
    captures ROI/net PnL metadata, renders Markdown summaries, and publishes via
    the shared failover helper under pytest coverage so dashboards inherit the
    same hardened transport as other operational feeds.【F:src/operations/strategy_performance.py†L200-L531】【F:tests/operations/test_strategy_performance.py†L68-L193】
  - Progress: Legacy operational health monitor shim now raises a guided
    `ModuleNotFoundError`, and regression tests ensure imports fail so observability
    tooling adopts the canonical operations telemetry surface instead of the retired
    implementation.【F:src/operational/health_monitor.py:1】【F:tests/operational/test_health_monitor.py:1】
  - Progress: Risk telemetry panels now attach limit values, ratios, and
    violation states to observability dashboard entries while preserving the
    serialised payloads, with pytest coverage asserting limit-status escalation
    so operators inherit actionable risk summaries instead of opaque aggregates.【F:src/operations/observability_dashboard.py†L627-L694】【F:tests/operations/test_observability_dashboard.py†L220-L320】
  - Progress: Observability dashboard composer now fuses ROI, risk, latency,
    backbone, operational readiness, and quality panels into a single snapshot,
    escalating severities from ROI status, risk-limit breaches, event-bus/SLO
    lag, and coverage posture while retaining structured metadata for each panel
    so dashboards and exporters inherit a complete readiness view.【F:src/operations/observability_dashboard.py†L566-L905】【F:tests/operations/test_observability_dashboard.py†L220-L320】
  - Progress: Observability dashboard metadata now auto-fills panel status counts
    and severity maps next to the remediation capsule, and exports Markdown
    summaries so exporters and runbooks can ingest a machine-readable readiness
    snapshot without recomputing severities, with pytest locking the
    contract.【F:src/operations/observability_dashboard.py†L147-L177】【F:src/operations/observability_dashboard.py†L907-L924】【F:tests/operations/test_observability_dashboard.py†L294-L316】
  - Progress: Observability dashboard now exposes a remediation summary capsule
    that counts failing/warning/healthy panels and lists affected slices under
    regression coverage so CI exporters can consume a canonical operational
    readiness signal without recomputing severities.【F:src/operations/observability_dashboard.py†L110-L145】【F:tests/operations/test_observability_dashboard.py†L309-L316】
  - Progress: Operational readiness telemetry now enriches snapshots with
    per-status breakdowns, component status maps, rolled-up issue counts, and
    gate metadata, emitting optional gate alerts while routing through the
    failover helper so dashboards and responders inherit deterministic
    remediation context under pytest coverage documenting evaluation, alert
    derivation, and publish fallbacks.【F:src/operations/operational_readiness.py†L113-L744】【F:tests/operations/test_operational_readiness.py†L86-L389】【F:docs/status/operational_readiness.md†L1-L140】【F:tests/runtime/test_professional_app_timescale.py†L722-L799】
  - Progress: The whitepaper now records the roadmap Definition-of-Done coverage
    and a replay/evidence checklist covering data backbone hydration, governed
    simulation runs, diary validation, and KPI regeneration so observability
    reviewers can reproduce the paper-trading packet directly from versioned
    documentation instead of ad-hoc notes.【F:docs/AlphaTrade_Whitepaper.md†L30-L36】【F:docs/AlphaTrade_Whitepaper.md†L341-L362】
  - Progress: Dry-run audit evaluator now applies minimum runtime and uptime
    thresholds, emitting structured incidents when evidence windows are too
    short or availability slips, with regression coverage confirming the new
    guardrails surface through summaries and full evaluations.【F:src/operations/dry_run_audit.py†L542-L726】【F:tests/operations/test_dry_run_audit.py†L71-L253】
  - Progress: Runtime builder now consumes the sensory organ status feed,
    publishes the hardened summary/metrics telemetry, and stores the latest
    snapshots on the professional app so the summary surface emits Markdown and
    JSON blocks for responders under regression coverage.【F:src/runtime/runtime_builder.py†L322-L368】【F:src/runtime/predator_app.py†L600-L1139】【F:tests/runtime/test_runtime_builder.py†L121-L207】【F:tests/runtime/test_professional_app_timescale.py†L1328-L1404】
  - Progress: ProfessionalPredatorApp now binds the constructed `RuntimeApplication`
    into its summary payload so observability exports include ingestion and
    trading workload metadata by default, with runtime builder and test coverage
    guaranteeing the workloads surface verbatim for operators.【F:src/runtime/predator_app.py†L328-L391】【F:src/runtime/predator_app.py†L1498-L1509】【F:src/runtime/runtime_builder.py†L3940-L3949】【F:tests/runtime/test_runtime_builder.py†L309-L348】
  - Progress: RuntimeApplication now offers `create_background_task` and `register_background_task` helpers so supervised runs can spawn or adopt ad-hoc coroutines without leaking outside the TaskSupervisor, with regression proving tracked snapshots report the custom tasks and metadata counts.【F:src/runtime/runtime_builder.py†L298-L326】【F:src/runtime/runtime_builder.py†L896-L918】【F:tests/runtime/test_runtime_builder.py†L1952-L2009】
- Progress: Coverage matrix CLI now exposes lagging domains via the
  `identify_laggards` helper, exports the list of covered source files, and
  enforces required regression suites through `--require-file`, failing the
  build and logging missing paths under pytest coverage when critical files
  fall out of reports.【F:tools/telemetry/coverage_matrix.py†L83-L357】【F:tests/tools/test_coverage_matrix.py†L136-L225】
- Progress: Coverage matrix domain classifier now aliases legacy
  `src/intelligence` modules into the canonical understanding namespace so
  reports no longer split decision-loop coverage across divergent labels, with
  CLI regressions asserting the normalised file list includes the re-mapped
  paths.【F:tools/telemetry/coverage_matrix.py†L12-L60】【F:tests/tools/test_coverage_matrix.py†L120-L168】
- Progress: Coverage matrix reports now emit a `deprecated_files` list when
  Cobertura payloads reference retired namespaces (e.g. `src/market_intelligence`),
  ensuring drift surfaces in guardrail Markdown/JSON outputs instead of being
  silently aliased, with regression coverage locking the behaviour.【F:tools/telemetry/coverage_matrix.py†L52-L219】【F:tests/tools/test_coverage_matrix.py†L120-L172】
- Progress: CI workflow now runs the coverage matrix and minimum coverage
  guardrail steps after the guarded pytest job, enforcing ingest/risk targets,
  writing Markdown/summary outputs, and failing builds when thresholds slip,
  with guardrail tests asserting the steps remain in place.【F:.github/workflows/ci.yml†L90-L135】【F:tests/runtime/test_guardrail_suite_manifest.py†L98-L135】
- Progress: Coverage guardrail evaluator now parses Cobertura XML, checks
  ingest/risk targets against configurable thresholds, highlights missing
  modules, and surfaces JSON/text reports with failure exit codes so CI hooks and
  local audits can block on coverage regressions deterministically.【F:tools/telemetry/coverage_guardrails.py†L1-L268】【F:tests/tools/test_coverage_guardrails.py†L1-L83】
  - Progress: Coverage guardrails now require the ingest configuration loader, with
    CI enforcing the `--require-file` flag, the manifest enumerating the new
    target, the evaluator treating it as a default requirement, and the guarded
    Timescale config test marked so configuration coverage gaps fail fast under
    regression coverage.【F:.github/workflows/ci.yml†L118-L123】【F:tests/runtime/test_guardrail_suite_manifest.py†L23-L118】【F:tools/telemetry/coverage_guardrails.py†L24-L36】【F:tests/tools/test_coverage_guardrails.py†L40-L101】【F:tests/data_foundation/test_timescale_config.py†L1-L44】
  - Progress: Guardrail workflow now requires the ingest scheduler suite in the
    pytest guardrail step and adds the scheduler module to the coverage matrix so
    supervised ingest loops stay under explicit enforcement, with manifest tests
    asserting the workflow keeps both requirements.【F:.github/workflows/ci.yml†L108-L142】【F:tests/runtime/test_guardrail_suite_manifest.py†L103-L151】
  - Progress: Quality telemetry snapshot builder now normalises coverage,
    staleness, and remediation trends into a typed `QualityTelemetrySnapshot`,
    escalating WARN/FAIL severities, retaining lagging-domain metadata, and
    capturing remediation notes so CI exports feed dashboards with deterministic
    coverage posture evidence.【F:src/operations/quality_telemetry.py†L1-L168】【F:tests/operations/test_quality_telemetry.py†L9-L53】
  - Progress: Observability dashboard surfaces operational readiness as a
    first-class panel, counting component severities, embedding metadata, and
    feeding remediation summaries under pytest coverage so responders inherit a
    consolidated operational view without bespoke wiring.【F:src/operations/observability_dashboard.py†L754-L815】【F:tests/operations/test_observability_dashboard.py†L220-L293】
  - Progress: Observability dashboard now renders a policy reflection panel that
    summarises analysed decisions, highlights top tactics/experiments/tags, embeds
    reviewer insights, and preserves exported Markdown/metadata so compliance and
    readiness dashboards inherit adaptive strategy evidence under guardrail
    coverage.【F:src/operations/observability_dashboard.py†L285-L390】【F:tests/operations/test_observability_dashboard.py†L627-L706】
  - Progress: Observability dashboard introduces a compliance & governance panel
    that aggregates loop compliance events, maps severity to panel status, and
    exports Markdown/JSON payloads with breach counts plus remediation details so
    responders see enforcement posture alongside ROI and readiness metrics under
    refreshed regression coverage.【F:src/operations/observability_dashboard.py†L200-L540】【F:tests/operations/test_observability_dashboard.py†L297-L420】
  - Progress: Observability dashboard guard CLI grades snapshot freshness,
    required panels, failing slices, and normalised overall status strings while
    emitting JSON or human-readable summaries with status-driven exit codes so
    CI hooks and drills can block on stale, failing, or WARN observability
    evidence under pytest coverage.【F:tools/telemetry/dashboard_guard.py†L1-L220】【F:tests/tools/test_dashboard_guard.py†L16-L140】
  - Progress: Configuration audit telemetry now evaluates `SystemConfig` diffs,
    annotates tracked toggles and extras, renders Markdown summaries with
    severity breakdowns, and publishes via the shared failover helper so
    configuration changes generate a durable audit trail with explicit severity
    counts and highest-risk fields for dashboards and governance reviews under
    pytest coverage.【F:src/operations/configuration_audit.py†L90-L210】【F:tests/operations/test_configuration_audit.py†L24-L86】
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
- Progress: Operational metrics instrumentation now wraps Prometheus access in
  lazy proxies, records first-failure warnings, hardens exporter startup with
  typed port parsing plus telemetry-sink import logging, exports understanding
  throttle gauges, and exposes compliance/governance counters for breaches,
  warnings, and promotions so dashboards and SLOs can track enforcement posture;
  guardrail suites cover gauge fallbacks, throttle snapshots, and dashboard
  wiring.【F:src/operational/metrics.py†L160-L360】【F:src/understanding/metrics.py†L1-L65】【F:tests/operational/test_metrics.py†L310-L360】【F:tests/understanding/test_understanding_metrics.py†L62-L125】【F:tests/operations/test_observability_dashboard.py†L297-L420】
- Progress: Trade throttle instrumentation now streams rolling throughput,
  latency, and backlog metrics via the shared `ThroughputMonitor`, exposes the
  snapshot through `TradingManager.get_execution_stats()`, and pairs it with
  `TradingManager.assess_throughput_health()` plus refreshed docs so operations
  teams can assert latency budgets from a single verdict; the throttle summary
  now surfaces `max_trades`, remaining credits, utilisation, retry/reset timers,
  and the resolved scope/context dictionaries (with `scope_key`) under guardrail
  coverage.【F:src/trading/execution/performance_monitor.py†L1-L127】【F:src/trading/trading_manager.py†L1991-L2031】【F:docs/performance/trading_throughput_monitoring.md†L60-L74】【F:tests/trading/execution/test_performance_monitor.py†L17-L156】【F:tests/trading/test_trading_manager_execution.py†L524-L550】
- Progress: Performance health reporting now renders the same assessment as a
  Markdown evidence pack via `build_performance_health_report`, with
  `TradingManager.generate_performance_health_report()` exposing the helper for
  runbooks; the renderer calls out the new backlog analytics (`latest_lag_ms`,
  `p95_lag_ms`, `breach_rate`, `max_breach_streak`), and docs/tests capture
  healthy vs degraded paths so operators can export supervisory snapshots
  without bespoke scripts.【F:src/trading/execution/performance_report.py†L104-L193】【F:src/trading/trading_manager.py†L1696-L1757】【F:docs/performance/trading_throughput_monitoring.md†L86-L133】【F:tests/trading/execution/test_performance_report.py†L39-L131】
- Progress: `collect_performance_baseline()` now aggregates execution stats, throughput verdicts, backlog/resource posture, and Markdown evidence while the companion CLI scripts sample throttled bursts; the baseline payload and docs emphasise the new backlog analytics (`latest_lag_ms`, `p95_lag_ms`, `breach_rate`, `max_breach_streak`) so observability packs archive reproducible performance baselines under regression coverage.【F:src/trading/execution/performance_baseline.py†L21-L72】【F:tools/performance_baseline.py†L1-L108】【F:docs/performance/performance_baseline.md†L20-L83】【F:tests/trading/execution/test_performance_report.py†L39-L131】 Latest snapshot adds the per-scope throttle roster as `throttle_scopes` and exposes the same payload through `TradingManager.get_trade_throttle_scope_snapshots()`, with guardrail suites asserting the execution stats, baseline helper, and docs all surface the per-strategy guardrail state and clear it when throttling is disabled.【F:src/trading/trading_manager.py†L621-L725】【F:docs/performance/performance_baseline.md†L10-L74】【F:tests/trading/test_trading_manager_execution.py†L2092-L2138】【F:tests/trading/test_trading_manager_execution.py†L2648-L2666】
  - Progress: Bootstrap stack now logs sensory listener, liquidity prober, and
  control-centre callback failures with structured metadata so optional hooks
  surface errors without disrupting bootstrap decisions, under pytest coverage
  that captures the emitted diagnostics.【F:src/orchestration/bootstrap_stack.py†L81-L258】【F:tests/current/test_bootstrap_stack.py†L164-L213】
- Progress: AlphaTrade loop orchestrator now coordinates the understanding router,
  DriftSentry gate, policy ledger stages, and decision diary so tactic experiments
  emit stage-aware drift metadata and governance evidence in one pass, with guardrail
  coverage locking forced-paper postures and recorded artefacts.【F:src/orchestration/alpha_trade_loop.py†L48-L200】【F:tests/orchestration/test_alpha_trade_loop.py†L1-L161】
- ✅ Slack/webhook mirrors for CI alerts ship via the CI failure alerts workflow,
  while the alert drill and metrics tooling record MTTA/MTTR timelines under
  regression coverage so dashboards mirror forced-failure rehearsals without
  manual collation.【F:.github/workflows/ci-failure-alerts.yml†L1-L188】【F:tools/telemetry/alert_drill.py†L29-L172】【F:tools/telemetry/update_ci_metrics.py†L134-L279】【F:tests/tools/test_alert_drill.py†L9-L58】【F:tests/tools/test_ci_metrics.py†L340-L618】
- ✅ CI dashboard rows and the weekly status digest now capture telemetry deltas
  through the shared status digest tooling, keeping roadmap evidence aligned
  with the latest coverage, formatter, remediation, and freshness exports.【F:docs/status/ci_health.md†L10-L108】【F:docs/status/quality_weekly_status.md†L18-L35】【F:tools/telemetry/status_digest.py†L1-L347】
- Progress: Decision narration capsule builder/publisher now normalises policy
  ledger diffs, sigma stability telemetry, and throttle states before emitting
  Markdown/JSON payloads through the shared failover helper so AlphaTrade
  reviewers inherit a resilient, single-trail diary feed aligned with the
  observability schema under pytest coverage.【F:src/operations/observability_diary.py†L3-L392】【F:tests/operations/test_observability_diary.py†L1-L190】
- Progress: Understanding diagnostics builder and CLI emit sensory→belief→router→policy snapshots, add a dedicated `understanding_acceptance` pytest marker, and guard the export contract with tests so graph diagnostics stay aligned with observability deliverables; the latest iteration synthesises fast-weight health metrics (utilisation, sparsity, dominance) and renders them in the Markdown/JSON outputs under regression coverage.【F:src/understanding/diagnostics.py†L389-L958】【F:tools/understanding/graph_diagnostics.py†L1-L82】【F:tests/understanding/test_understanding_diagnostics.py†L15-L39】【F:tests/tools/test_understanding_graph_cli.py†L8-L19】【F:pytest.ini†L2-L27】
- ✅ Sensory drift regressions now bundle deterministic Page–Hinkley replays,
  throttle metadata checks, and Prometheus export fixtures so CI reproduces the
  alert catalogue and telemetry expectations deterministically.【F:tests/operations/fixtures/page_hinkley_replay.json†L1-L128】【F:tests/operations/test_sensory_drift.py†L157-L218】【F:src/understanding/metrics.py†L1-L65】
- ✅ Understanding-loop SLO probes grade latency, drift freshness, and replay
  determinism while exporting Prometheus gauges with regression coverage to
  guard the observability contract.【F:src/operations/slo.py†L300-L417】【F:tests/operations/test_slo.py†L101-L226】
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
  degradation, with pytest covering metadata capture and failover paths.【F:src/operations/system_validation.py†L724-L889】【F:tests/operations/test_system_validation.py†L77-L160】

### Next (30–90 days)

- Re-enable quarantined pytest suites behind feature flags, extending coverage
  to legacy/integration flows while monitoring flake telemetry for drift.
  【F:docs/technical_debt_assessment.md†L84-L112】
- Instrument CI summaries with ingest freshness, Kafka lag, risk policy
  violations, sensory drift deltas, throttle-state occupancy, and sigma
  stability SLOs, exporting machine-readable snapshots for dashboards and
  compliance audits.【F:docs/status/ci_health.md†L21-L73】【F:docs/status/ci_health.md†L95-L108】
- Land fast-weight replay determinism harnesses in staging, expanding
  Prometheus exporters with per-loop drift and throttle gauges, and align diary
  schema updates with AlphaTrade probe coverage milestones so reviewers can
  crosswalk delivery against the execution plan.
- Publish regression progress and telemetry deltas in weekly status updates,
  linking roadmap checkmarks to the refreshed briefs so discovery stays aligned. 【F:docs/status/quality_weekly_status.md†L1-L26】【F:tools/telemetry/ci_digest.py†L1-L337】

### Later (90+ days)

- Establish continuous validation pipelines that gate deployments on coverage,
  telemetry thresholds, and alert-drill freshness, mirroring the long-horizon
  testing & observability mitigation plan.【F:docs/technical_debt_assessment.md†L95-L112】
- Automate documentation freshness checks (roadmap, briefs, CI health) to flag
  stale sections and open follow-up issues when telemetry or coverage regresses.
- Fold quality guardrails into professional readiness reporting so operators
  see coverage drift, flake spikes, alert posture, throttle health, and
  sigma-stability posture alongside ingest health.
- Deliver the AlphaTrade observability workbook that maps diary schema entries
  to probe coverage, policy ledger provenance, replay determinism checkpoints,
  and theory packet links so future loop upgrades inherit a reusable validation
  blueprint.

## Validation & telemetry

- Stand up a deterministic replay harness for the AlphaTrade fast-weight loop,
  exposing diary-aligned probes, throttle state transitions, and Page–Hinkley
  detector outputs as fixtures that reviewers can execute locally or in CI to
  verify drift controls and decision narration remain stable across releases.
- Ship a theory packet template that documents sigma stability metrics, policy
  ledger provenance, replay scenarios, and validation hooks so observability
  reviewers inherit a consistent proof kit for new drift/throttle surfaces.
- Add validation hooks to the Prometheus exporters and policy ledger pipeline
  that publish provenance hashes, throttle occupancy, and replay checksums,
  ensuring SLO instrumentation and observability packets share identical audit
  trails.
- Track coverage deltas via `tests/.telemetry/ci_metrics.json` and surface them
  in dashboards; add assertions in regression suites to prevent silent drops.
  【F:docs/status/ci_health.md†L13-L15】
- Flake telemetry feed now captures the adaptive release thresholds regression
  signature (node id, diff, duration) so automation reviews can prioritise
  gating fixes with deterministic evidence.【F:tests/.telemetry/flake_runs.json†L1-L20】
- Domain snapshots are now captured alongside overall coverage in
  `tests/.telemetry/ci_metrics.json`, flagging lagging domains directly in CI
  telemetry so remediation progress is visible without ad-hoc parsing.
  【F:tools/telemetry/ci_metrics.py†L1-L260】【F:tests/tools/test_ci_metrics.py†L1-L300】
- Coverage telemetry now records lagging-domain counts, formatted notes, worst-performing
  slices, and optional remediation snapshots via the CLI so dashboards inherit
  actionable readiness deltas alongside coverage trendlines.【F:tools/telemetry/ci_metrics.py†L120-L260】【F:tools/telemetry/update_ci_metrics.py†L1-L240】【F:tests/tools/test_ci_metrics.py†L200-L360】
- CI metrics staleness summary now inspects coverage, formatter, domain, and
  remediation feeds to flag stale telemetry windows with timestamps and age
  calculations so roadmap checkpoints surface expired evidence automatically
  under pytest coverage.【F:tools/telemetry/ci_metrics.py†L214-L320】【F:tests/tools/test_ci_metrics.py†L210-L360】
- Remediation progress snapshots now live alongside coverage/formatter trendlines
  in `tests/.telemetry/ci_metrics.json` thanks to the `--remediation-status`
  CLI and the new `--coverage-remediation` mode, capturing roadmap evidence
  (label, statuses, source, notes) and coverage laggard summaries for dashboards
  and audits with pytest guarding the JSON contract.【F:tools/telemetry/update_ci_metrics.py†L1-L240】【F:tools/telemetry/ci_metrics.py†L1-L260】【F:tests/tools/test_ci_metrics.py†L1-L420】【F:tests/.telemetry/ci_metrics.json†L1-L6】
- CI metrics staleness guard exposes a CLI that summarises coverage, formatter,
  domain, and remediation trend freshness, failing builds when telemetry goes
  stale or evidence is missing so roadmap reviews inherit up-to-date snapshots
  under pytest coverage documenting stale detection and JSON output flows.【F:tools/telemetry/ci_metrics_guard.py†L1-L142】【F:tests/tools/test_ci_metrics_guard.py†L1-L99】
- Remediation summary exporter reads the metrics feed, renders Markdown tables
  with delta call-outs, honours slice limits, omits deltas for non-numeric
  statuses, and ships with a CLI/pytest contract so status updates and briefs
  stay in sync without manual collation.【F:tools/telemetry/remediation_summary.py†L1-L220】【F:tests/tools/test_remediation_summary.py†L22-L125】
- CI dashboard rows and weekly status updates now flow from
  `python -m tools.telemetry.status_digest`, which fuses coverage, formatter,
  remediation, freshness, and observability telemetry into Markdown with pytest
  backing so stakeholders can paste evidence-backed updates directly into
  briefs or sprint notes.【F:tools/telemetry/status_digest.py†L1-L667】【F:tests/tools/test_status_digest.py†L1-L217】
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
