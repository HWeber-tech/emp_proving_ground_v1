# Modernisation roadmap – Season reset

This reset distils the latest audit, technical debt, and status reports into a
fresh execution plan. It assumes the conceptual architecture mirrors the EMP
Encyclopedia while acknowledging that most subsystems remain scaffolding.

AlphaTrade's Perception → Adaptation → Reflection loop now anchors every near-term
investment. See `docs/High-Impact Development Roadmap.md` for the live-shadow
Understanding Loop v1 pilot, follow-on promotion gates, and the telemetry proof
kit that the roadmap calls back to in each checklist.

## Current parity snapshot

| Signal | Reality | Evidence |
| --- | --- | --- |
| Architecture | Layered domains and canonical `SystemConfig` definitions are in place, enforcing the core → sensory → thinking → trading → orchestration stack described in the encyclopedia. | 【F:docs/architecture/overview.md†L9-L48】 |
| Delivery state | The codebase is still a development framework: evolution, intelligence, execution, and strategy layers run on mocks; there is no production ingest, risk sizing, or portfolio management. | 【F:docs/DEVELOPMENT_STATUS.md†L7-L35】 |
| Quality posture | CI passes with 76% coverage, but hotspots include operational metrics, position models, and configuration loaders; runtime validation checks still fail. | 【F:docs/ci_baseline_report.md†L8-L27】【F:docs/technical_debt_assessment.md†L31-L112】 |
| Debt hotspots | Hollow risk management, unsupervised async tasks, namespace drift, and deprecated exports continue to surface in audits. | 【F:docs/technical_debt_assessment.md†L33-L80】【F:src/core/__init__.py†L11-L36】 |
| Legacy footprint | Canonical risk, evolution, and analytics modules now resolve through their source packages: the deprecated core risk manager shim is gone, stress/VaR helpers route through `src/risk/analytics`, and integration guides still trail reality. | 【F:src/config/risk/risk_config.py†L1-L161】【F:src/core/__init__.py†L11-L36】【F:docs/reports/CLEANUP_REPORT.md†L71-L104】【F:src/risk/analytics/var.py†L19-L121】 |

## Gaps to close

- [ ] **Operational data backbone** – Deliver real Timescale/Redis/Kafka services,
  parameterise SQL, and supervise ingest tasks instead of relying on mocks.
  - *Progress*: Timescale retention telemetry now validates schema/table/timestamp
    identifiers, parameterises retention queries, and documents the contract via
    regression tests so institutional slices cannot inject raw SQL through policy
    definitions. The snapshot elevates WARN/FAIL states deterministically,
    attaches coverage/span metadata for every dimension, records the evaluated
    policy catalogue, and publishes through the shared failover helper so
    dashboards inherit severity-aware retention evidence even when the runtime
    bus degrades.【F:src/operations/retention.py†L1-L334】【F:tests/operations/test_data_retention.py†L1-L220】
  - *Progress*: Timescale reader now builds SQLAlchemy Core selects with sanitised
    identifiers and bound filters, eliminating hand-written SQL while the security
    regression keeps fuzzed identifier guards in place.【F:src/data_foundation/persist/timescale_reader.py†L352】【F:tests/data_foundation/test_timescale_reader_security.py†L1】
  - *Progress*: Execution readiness journal reflects tables via SQLAlchemy, inserts
    snapshots with bound parameters, and summarises status/service counts so
    Timescale auditing no longer shells out raw text queries under the regression
    suite.【F:src/data_foundation/persist/timescale.py†L1956-L2064】【F:tests/data_foundation/test_timescale_execution_journal.py†L104】
  - *Progress*: Data backbone readiness telemetry now exposes failover triggers,
    optional trigger metadata, and recovery plan payloads under pytest coverage so
    institutional dashboards surface degraded optional slices alongside the
    Timescale recovery blueprint instead of masking the drill-down details.【F:src/operations/data_backbone.py†L488-L515】【F:tests/operations/test_data_backbone.py†L289-L347】
  - *Progress*: Timescale ingest scheduler can now register its background loop with
    the runtime task supervisor, tagging interval/jitter metadata and exposing
    live snapshots so operators see supervised ingest jobs instead of orphaned
    `create_task` handles, with pytest covering steady-state execution, failure
    cut-offs, jitter bounds, supervisor telemetry, and snapshot publishing so
    the guardrail remains deterministic. The guardrail manifest now pins the
    ingest scheduler regression to the `guardrail` matrix, ensuring CI fails
    fast if the scheduler suite drifts or loses its guardrail marker.【F:src/data_foundation/ingest/scheduler.py†L1-L138】【F:tests/data_foundation/test_ingest_scheduler.py†L1-L200】【F:tests/runtime/test_guardrail_suite_manifest.py†L18-L90】
  - *Progress*: Bootstrap runtime integration tests now stub the `_execute_timescale_ingest`
    hook and export list, locking the runtime builder’s ingest entrypoint so
    supervised launches keep wiring institutional ingest toggles under guardrail
    coverage.【F:tests/current/test_bootstrap_runtime_integration.py†L96-L111】【F:tests/runtime/test_bootstrap_runtime_sensory.py†L96-L110】
  - *Progress*: Pricing cache now hashes ingest parameters with `blake2b` for
    deterministic dataset artefacts, writes metadata and issues manifests,
    enforces retention policies, and logs cleanup failures under regression
    coverage so bootstrap slices leave auditable evidence without leaking stale
    files.【F:src/data_foundation/cache/pricing_cache.py†L68】【F:tests/data_foundation/cache/test_pricing_cache.py†L1】
  - *Progress*: Cobertura coverage guardrail tooling now parses XML reports,
    asserts coverage for ingest, risk, and observability hotspots, flags missing
    targets, and exits non-zero for WARN/FAIL thresholds so CI pipelines and
    local audits block on regression drift; pytest locks success/failure paths
    and the guardrail manifest keeps the guardrail suites wired into the
    dedicated CI marker.【F:tools/telemetry/coverage_guardrails.py†L1-L268】【F:tests/tools/test_coverage_guardrails.py†L1-L83】【F:tests/runtime/test_guardrail_suite_manifest.py†L18-L90】
  - *Progress*: CI workflow now runs the coverage matrix and minimum coverage
    guardrail steps after the guarded pytest job, enforcing ingest/risk targets,
    appending Markdown summaries, and failing the build when thresholds slip,
    with guardrail tests asserting the steps remain in place.【F:.github/workflows/ci.yml†L90-L135】【F:tests/runtime/test_guardrail_suite_manifest.py†L98-L135】
  - *Progress*: Production ingest slice now orchestrates Timescale runs and
    supervised services from a single entrypoint, wiring the institutional
    provisioner, Kafka bridge, and Redis cache through the shared
    `TaskSupervisor` while exposing deterministic summaries for dashboards and
    start/stop lifecycles under pytest coverage so operators inherit a managed
    ingest surface instead of bespoke wiring.【F:src/data_foundation/ingest/production_slice.py†L1-L170】【F:tests/data_foundation/test_production_ingest_slice.py†L1-L176】
  - *Progress*: Timescale backbone orchestrator now enriches every daily,
    intraday, and macro slice with requested symbol/event counts, fetched row
    totals, ingest result metadata, and macro window provenance so ingest
    telemetry exposes what was fetched, normalised, or skipped; guardrail tests
    cover macro window fallbacks, empty payloads, publisher failure logging, and
    metadata emission to prevent regressions in institutional ingest
    coverage.【F:src/data_foundation/ingest/timescale_pipeline.py†L70-L213】【F:tests/data_foundation/test_timescale_backbone_orchestrator.py†L1-L426】
  - *Progress*: Macro ingest now falls back to an internal no-op fetcher when no
    provider is wired, letting backbone drills drop the legacy FRED scaffold while
    keeping optional macro windows explicit in the cleanup report.【F:src/data_foundation/ingest/timescale_pipeline.py†L21】【F:docs/reports/CLEANUP_REPORT.md†L120】
  - *Progress*: Institutional ingest provisioner now spins up supervised
    Timescale schedules alongside Redis caches and Kafka consumers, wiring the
    bridge into the task supervisor with redacted metadata, publishing a managed
    manifest that lists configured topics, and exposing async/sync connectivity
    probes so dashboards can surface recovery requirements and live health checks
    without bespoke wiring. It now auto-configures Redis via the shared client
    helper when no factory is supplied, warns when a client cannot be created,
    and surfaces the active Redis backing class in service summaries and
    manifests so operators know which cache implementation is live under pytest
    coverage of the configure path.【F:src/data_foundation/ingest/institutional_vertical.py†L96-L399】【F:tests/runtime/test_institutional_ingest_vertical.py†L140-L185】【F:docs/operations/timescale_failover_drills.md†L1-L27】
  - *Progress*: Professional runtime builder now invokes the institutional
    provisioner automatically, reuses managed Redis clients when present,
    defers the next scheduled run after manual ingest, records
    managed-connector manifests plus scheduler snapshots, and propagates the
    managed manifest into data-backbone readiness telemetry so Tier‑1 launches
    inherit supervised ingest connectors with explicit connector listings in
    readiness dashboards.【F:src/runtime/runtime_builder.py†L1124-L1895】【F:src/runtime/runtime_builder.py†L2453-L2568】【F:docs/operations/timescale_failover_drills.md†L7-L33】
  - *Progress*: Tier-0 Yahoo ingest now sanitises symbols/intervals, enforces
    mutually exclusive period versus window arguments, normalises timestamps,
    and writes through a DuckDB helper that escapes table identifiers and binds
    parameters with pytest coverage, while the new market data gateway logs
    rejected requests and reuses the hardened fetcher so entry-level datasets
    inherit safe defaults.【F:src/data_foundation/ingest/yahoo_ingest.py†L82-L305】【F:src/data_foundation/ingest/yahoo_gateway.py†L1-L53】【F:tests/data_foundation/test_yahoo_ingest_security.py†L1-L132】【F:tests/data_foundation/test_yahoo_gateway.py†L1-L69】
  - *Progress*: Timescale ingest helpers now validate schema/table identifiers
    before emitting SQL, raising deterministic errors on unsafe names and
    pinning the contract via regression tests so institutional slices cannot
    smuggle raw SQL through policy or schedule configuration.【F:src/data_foundation/persist/timescale.py†L1-L120】【F:tests/data_foundation/test_timescale_ingest.py†L1-L83】
  - *Progress*: Timescale ingestor now reflects tables through SQLAlchemy,
    performs PostgreSQL upserts via `pg_insert`, binds SQLite fallbacks, and
    chunks writes so ingest runs avoid hand-written SQL while retaining
    deterministic freshness metrics under regression coverage.【F:src/data_foundation/persist/timescale.py†L2337-L2489】【F:tests/data_foundation/test_timescale_ingest.py†L165-L220】
  - *Progress*: JSONL persistence now raises typed errors for unserialisable payloads,
    logs filesystem failures, and cleans up partial files so ingest tooling surfaces
    genuine persistence faults instead of emitting empty paths under silent
    fallbacks.【F:src/data_foundation/persist/jsonl_writer.py†L1-L69】【F:tests/data_foundation/test_jsonl_writer.py†L1-L37】
  - *Progress*: Legacy parquet-writer scaffolding has been removed so ingest
    persistence now routes through the pricing cache helpers that hash
    parameters, write manifests, and enforce retention under regression
    coverage, shrinking the operational dead-code surface.【F:docs/reports/CLEANUP_REPORT.md†L110-L170】【F:src/data_foundation/cache/pricing_cache.py†L1-L194】
  - *Progress*: Ingest telemetry publisher now logs recoverable local bus
    failures, escalates unexpected exceptions, and falls back to the global bus
    under pytest coverage so ingest snapshots are not silently dropped when the
    runtime transport degrades.【F:src/data_foundation/ingest/telemetry.py†L33-L99】【F:tests/data_foundation/test_ingest_publishers.py†L1-L164】
  - *Progress*: Operational readiness aggregation fuses system validation,
    incident response, alerts, and dashboard telemetry into enriched snapshots
    that expose per-status counts, component issue catalogs, and structured
    alert contexts; publishing now rides the shared failover helper with pytest
    coverage for alert routing and bus failover so responders inherit actionable
    metadata when WARN/FAIL posture shifts.【F:src/operations/operational_readiness.py†L113-L373】【F:src/operations/incident_response.py†L242-L715】【F:src/operations/system_validation.py†L470-L889】【F:tests/operations/test_operational_readiness.py†L86-L221】【F:docs/status/operational_readiness.md†L1-L73】
- [ ] **Sensory + evolution execution** – Replace HOW/ANOMALY stubs, wire lineage
  telemetry, and prove adaptive strategies against recorded data.
  - *Progress*: HOW and ANOMALY sensors now clamp minimum confidence, sanitise
    sequence payloads, surface dropped-sample counts, attach order-book
    analytics, and persist lineage metadata with shared threshold assessments so
    downstream telemetry inherits auditable context under pytest coverage.【F:src/sensory/anomaly/anomaly_sensor.py†L21-L277】【F:src/sensory/how/how_sensor.py†L21-L210】【F:tests/sensory/test_how_anomaly_sensors.py†L187-L302】
  - *Progress*: Ecosystem optimizer now defends against unsafe genomes and
    malformed regime metadata by normalising canonical models, skipping
    non-numeric parameters, and logging adapter failures with pytest coverage on
    each guardrail so evolution runs cannot silently corrupt state.【F:src/ecosystem/optimization/ecosystem_optimizer.py†L59-L230】【F:tests/ecosystem/test_ecosystem_optimizer_hardening.py†L1-L70】
  - *Progress*: Default evolution seeding now cycles through catalogue-inspired
    genome templates, ingests recorded experiment manifests into additional
    templates, derives jitter/metrics/tags from those artifacts, and injects
    lineage metadata so baseline populations mirror the institutional strategy
    library plus recent experiments. The sampler now writes parent IDs,
    mutation histories, and performance fingerprints onto each genome, doubles
    as the default bootstrap path inside the population manager, and surfaces
    parent/mutation counts through lineage telemetry so orchestrator dashboards
    expose richer provenance, with pytest guarding sampler rotation, metadata
    propagation, and seeded genome context.【F:src/core/evolution/seeding.py†L82-L140】【F:src/core/population_manager.py†L62-L383】【F:src/evolution/lineage_telemetry.py†L200-L228】【F:tests/evolution/test_realistic_seeding.py†L48-L88】【F:tests/current/test_population_manager_with_genome.py†L86-L108】【F:tests/current/test_evolution_orchestrator.py†L112-L310】
  - *Progress*: Portfolio evolution falls back gracefully when optional
    scikit-learn dependencies are missing by logging the degraded path, returning
    deterministic cluster bucketing, and exercising the guards under
    regression tests so adaptive runs keep producing actionable recommendations
    even in minimal environments.【F:src/intelligence/portfolio_evolution.py†L47-L142】【F:tests/intelligence/test_portfolio_evolution_security.py†L1-L169】
  - *Progress*: Evolution orchestrator now honours an `EVOLUTION_ENABLE_ADAPTIVE_RUNS`
    feature flag, exposing helpers for tests and gating champion registration,
    catalogue updates, and telemetry so governance can disable adaptive loops
    until approvals land; the latest uplift records a structured
    `AdaptiveRunDecision` snapshot (source, raw flag, reason) alongside the
    boolean gate so dashboards and reviewers inherit auditable evidence of how
    adaptive runs were resolved.【F:src/evolution/feature_flags.py†L1-L91】【F:src/orchestration/evolution_cycle.py†L150-L325】【F:tests/current/test_evolution_orchestrator.py†L64-L330】【F:tests/evolution/test_feature_flags.py†L1-L46】
  - *Progress*: Evolution engine now records seed provenance on every
    initialization and generation, summarising catalogue template counts,
    species tags, and seeded totals for the population manager while lineage
    telemetry emits the enriched payload under pytest coverage so dashboards and
    governance reviews inherit deterministic seed metadata instead of opaque
    populations.【F:src/core/evolution/engine.py†L65-L336】【F:src/core/population_manager.py†L115-L183】【F:src/evolution/lineage_telemetry.py†L1-L200】【F:tests/current/test_evolution_orchestrator.py†L83-L120】【F:tests/evolution/test_lineage_snapshot.py†L8-L66】
  - *Progress*: Evolution experiment telemetry hardens publishing with explicit
    exception capture, markdown fallback logging, and pytest scenarios that
    simulate transport failures so dashboards and runbooks inherit reliable
    snapshots of paper-trading ROI and backlog posture.【F:src/operations/evolution_experiments.py†L40-L196】【F:tests/operations/test_evolution_experiments.py†L1-L126】
  - *Progress*: Evolution readiness evaluator now fuses the adaptive-run feature
    flag, seed provenance statistics, and lineage telemetry into a governance
    snapshot, rendering Markdown/JSON summaries, capturing issues, and exposing
    champion metadata so reviewers can gate adaptive runs deterministically
    under pytest coverage.【F:src/operations/evolution_readiness.py†L1-L206】【F:tests/operations/test_evolution_readiness.py†L1-L118】
  - *Progress*: Recorded sensory replay evaluator converts archived sensory
    snapshots into deterministic fitness metrics, now emitting a structured
    trade ledger with confidence/strength metadata, tracking maximum loss
    streaks, and reporting average trade durations so adaptive runs surface
    auditable replay evidence with richer drawdown diagnostics under pytest
    coverage. A companion telemetry builder promotes those metrics into
    lineage-backed Markdown/JSON snapshots, flags drawdown/return severities,
    and captures best/worst trade diagnostics so governance reviewers inherit a
    ready-to-publish replay dossier with deterministic thresholds.【F:src/evolution/evaluation/recorded_replay.py†L266-L425】【F:src/evolution/evaluation/telemetry.py†L1-L203】【F:tests/evolution/test_recorded_replay_evaluator.py†L66-L102】【F:tests/evolution/test_recorded_replay_telemetry.py†L1-L88】
  - *Progress*: HOW and ANOMALY sensors now embed sanitised lineage records,
    compute shared threshold posture assessments, and surface state/breach
    metadata on every signal so downstream consumers can audit provenance and
    escalation context, with pytest coverage locking the helper and sensory
    flows.【F:src/sensory/how/how_sensor.py†L67-L194】【F:src/sensory/anomaly/anomaly_sensor.py†L121-L220】【F:src/sensory/thresholds.py†L1-L76】【F:tests/sensory/test_how_anomaly_sensors.py†L87-L175】【F:tests/sensory/test_thresholds.py†L1-L57】
  - *Progress*: Integrated sensory organ fuses WHY/WHAT/WHEN/HOW/ANOMALY signals,
    records lineage and audit trails, instruments sensor-drift windows with a
    configurable baseline/evaluation policy, publishes telemetry snapshots, and
    now serialises per-dimension metadata plus harvested numeric telemetry so
    runtime status/metrics surfaces inherit audit-ready values under pytest
    coverage.【F:src/sensory/real_sensory_organ.py†L23-L233】【F:src/sensory/real_sensory_organ.py†L392-L489】【F:tests/sensory/test_real_sensory_organ.py†L96-L183】
  - *Progress*: Component integrator now instantiates canonical sensory, trading,
    evolution, risk, and governance subsystems, registers legacy aliases for the
    HOW/WHAT/WHEN organs, captures the enforced `RiskConfig` summary, and surfaces
    the shared risk API runbook so integration checks and governance reviews
    observe the true wiring under pytest coverage.【F:src/integration/component_integrator.py†L1-L170】【F:src/integration/component_integrator_impl.py†L1-L139】【F:tests/integration/test_component_integrator_impl.py†L1-L44】
  - *Progress*: Sensory metrics telemetry now converts the organ status feed into
    dimension-level metrics, extracts numeric audit/order-book telemetry for each
    dimension, captures drift-alert provenance, and publishes via the event-bus
    failover helper so dashboards receive strength/confidence/threshold and raw
    telemetry snapshots even when the runtime bus fails, with pytest locking the
    contract and failover path.【F:src/operations/sensory_metrics.py†L1-L200】【F:tests/operations/test_sensory_metrics.py†L1-L130】
  - *Progress*: Sensory summary publisher now normalises integrated sensor
    payloads into ranked Markdown/JSON snapshots, captures drift metadata, and
    emits telemetry via the event-bus failover helper so dashboards inherit
    resilient sensory rollups backed by pytest coverage of runtime and failover
    paths.【F:src/operations/sensory_summary.py†L1-L215】【F:tests/operations/test_sensory_summary.py†L1-L155】
  - *Progress*: Sensory lineage publisher now normalises HOW/ANOMALY lineage
    records, keeps a bounded inspection history, and publishes via either
    runtime or global event-bus bridges while the real sensory organ pipes its
    dimension metadata through the publisher so responders inherit auditable
    provenance snapshots under pytest coverage of publish/fallback paths.【F:src/sensory/lineage_publisher.py†L1-L193】【F:src/sensory/real_sensory_organ.py†L41-L489】【F:tests/sensory/test_lineage.py†L11-L145】【F:tests/sensory/test_real_sensory_organ.py†L96-L183】
  - *Progress*: Professional runtime now captures the integrated sensory status
    feed, publishes the hardened summary/metrics telemetry, and caches the last
    snapshots so the Predator app summary exposes Markdown/JSON payloads for
    responders under regression coverage of the builder and app surfaces.【F:src/runtime/runtime_builder.py†L322-L368】【F:src/runtime/predator_app.py†L600-L1139】【F:tests/runtime/test_runtime_builder.py†L121-L207】【F:tests/runtime/test_professional_app_timescale.py†L1328-L1404】
  - *Progress*: Core module now re-exports the canonical sensory organ
    implementation, normalises drift-config inputs, and drops the legacy stub
    fallback so runtime consumers always receive the real organ with regression
    coverage verifying alias stability and drift configuration coercion.【F:src/core/__init__.py†L11-L36】【F:src/core/sensory_organ.py†L1-L36】【F:tests/core/test_core_sensory_exports.py†L1-L22】
  - *Progress*: Market data recorder/replayer now serialises lightweight order
    books to JSONL, logs feature-writer failures, skips malformed payloads, and
    guards file-handle shutdown so sensory backtests and feature pipelines can
    rely on deterministic capture/replay under pytest coverage.【F:src/operational/md_capture.py†L1-L87】【F:tests/operational/test_md_capture.py†L1-L53】
- [ ] **Risk and runtime safety** – Enforce `RiskConfig`, finish the builder rollout,
  adopt supervised async lifecycles, and purge deprecated facades.
  - *Progress*: Safety manager normalises kill-switch paths (including env and relative inputs), logs unreadable sentinels, and keeps system config exports in sync so live-mode guardrails stay enforceable with regression coverage for the configuration shim.【F:src/governance/safety_manager.py†L21-L74】【F:src/governance/system_config.py†L139-L413】【F:tests/governance/test_security_phase0.py†L47-L85】
  - *Progress*: The trading risk gateway now drives portfolio checks through the
    real risk manager, records liquidity and policy telemetry, rejects intents
    that breach drawdown/exposure/liquidity limits, and publishes a
    runbook-backed `risk_config_summary` snapshot so downstream tooling inherits
    the enforced configuration and escalation URL with every limit payload.【F:src/trading/risk/risk_gateway.py†L170-L390】【F:src/trading/trading_manager.py†L105-L210】【F:tests/current/test_risk_gateway_validation.py†L326-L350】
  - *Progress*: Trading manager initialises its portfolio risk manager via the
    canonical `get_risk_manager` facade, exposes the core engine’s snapshot and
    assessment APIs, and keeps execution telemetry aligned with the deterministic
    risk manager surfaced by the runtime builder.【F:src/trading/trading_manager.py†L105-L147】【F:src/risk/risk_manager_impl.py†L533-L573】
  - *Progress*: Deterministic trading risk API now attaches structured metadata
    and a contract runbook to every `RiskApiError`, exposes a shared
    `RISK_API_RUNBOOK` alias for supervisors, and renders sector exposure maps,
    combined budget totals, volatility targets, leverage windows,
    sector-instrument counts, and latest policy snapshots in its summaries so
    downstream telemetry inherits the full allocation context plus escalation
    guidance; runtime builder and trading manager summarise the enforced risk
    config, with `get_risk_status()` exporting the canonical summary plus the
    shared runbook pointer for deterministic triage.【F:docs/api/risk.md†L1-L24】【F:docs/operations/runbooks/risk_api_contract.md†L1-L31】【F:src/trading/risk/risk_api.py†L1-L158】【F:src/runtime/runtime_builder.py†L323-L353】【F:src/trading/trading_manager.py†L672-L714】【F:tests/runtime/test_runtime_builder.py†L200-L234】【F:tests/trading/test_risk_api.py†L90-L152】【F:tests/trading/test_trading_manager_execution.py†L670-L684】
  - *Progress*: Trading manager now emits dedicated risk interface telemetry via
    snapshot/error helpers that render Markdown summaries, publish structured
    payloads on the event bus, and persist the latest posture for discovery,
    with pytest asserting snapshot and alert propagation so supervisors inherit
    actionable evidence when enforcement fails.【F:src/trading/risk/risk_interface_telemetry.py†L1-L156】【F:src/trading/trading_manager.py†L741-L759】【F:tests/trading/test_trading_manager_execution.py†L651-L667】
  - *Progress*: FIX integration pilot now exports supervised runtime metadata,
    exposes a `run_forever` trading workload, and ships a builder helper that
    wraps the pilot into a runtime application so brokers inherit the risk API
    runbook, task-supervisor stats, trading-manager risk summary, and graceful
    shutdown semantics under pytest coverage of the runtime harness.【F:src/runtime/fix_pilot.py†L112-L165】【F:src/runtime/fix_pilot.py†L225-L236】【F:src/runtime/fix_pilot.py†L496-L517】【F:src/runtime/__init__.py†L15-L107】【F:tests/runtime/test_fix_pilot.py†L166-L260】
  - *Progress*: Bootstrap control center, bootstrap runtime status, and FIX pilot
    snapshots now resolve the trading manager’s risk interface payload, cache the
    shared runbook metadata, and surface it in operator telemetry so control
    rooms, status CLIs, and pilot dashboards expose the same escalation guidance
    under regression coverage.【F:src/operations/bootstrap_control_center.py†L99-L350】【F:src/runtime/bootstrap_runtime.py†L210-L334】【F:src/runtime/fix_pilot.py†L22-L318】【F:tests/current/test_bootstrap_control_center.py†L151-L180】【F:tests/runtime/test_fix_pilot.py†L115-L178】
  - *Progress*: `RiskConfig` now normalises sector/instrument mappings, rejects
    duplicate or missing sector limits, enforces that individual and combined
    sector budgets never exceed the global exposure cap, and continues to
    enforce position sizing plus research-mode overrides so governance reviews
    inherit deterministic, de-duplicated risk inputs under pytest
    coverage.【F:src/config/risk/risk_config.py†L10-L213】【F:tests/risk/test_risk_config_validation.py†L39-L90】
  - *Progress*: Risk policy guardrail suite now exercises approvals, exposure
    breaches, leverage warnings, research-mode overrides, closing-position
    allowances, price fallbacks, ratio metadata, and derived-equity scenarios
    under the `guardrail` marker so CI fails fast when institutional limit
    enforcement or telemetry contracts regress.【F:tests/trading/test_risk_policy.py†L1-L511】
  - *Progress*: Runtime builder now resolves the canonical `RiskConfig` from the
    trading manager, validates mandatory thresholds, wraps invalid payloads in a
    deterministic runtime error, and logs the enforced posture under regression
    coverage so supervised launches cannot proceed with missing or malformed
    limits. The builder now refuses to launch when mandatory stop-loss controls
    are disabled outside research mode, emitting the shared risk API runbook
    alias so supervisors inherit a consistent escalation path under pytest
    coverage.【F:src/runtime/runtime_builder.py†L323-L353】【F:src/trading/risk/risk_api.py†L23-L44】【F:tests/runtime/test_runtime_builder.py†L200-L234】
  - *Progress*: A supervised runtime runner now wraps professional workloads in a
    shared `TaskSupervisor`, wiring graceful signal handling, optional timeouts,
    and deterministic shutdown callbacks so runtime launches inherit the same
    lifecycle guarantees as the builder, with pytest exercising normal and
    timeout-driven exits.【F:src/runtime/runtime_runner.py†L1-L120】【F:main.py†L71-L125】【F:tests/runtime/test_runtime_runner.py†L1-L58】
  - *Progress*: Risk policy evaluation now derives equity from cash and open
    positions when balances are missing, normalises string portfolio payloads,
    skips malformed position entries, and continues to enforce mandatory stop
    losses plus resolved price fallbacks so CI catches guardrail drift before it
    reaches execution flows.【F:src/trading/risk/risk_policy.py†L29-L238】【F:tests/trading/test_risk_policy.py†L178-L511】
  - *Progress*: Policy telemetry helpers now serialise deterministic decision
    snapshots, render Markdown summaries, and publish violation alerts with
    embedded runbook metadata while the trading manager escalates breached
    guardrails and regression tests lock the payload contract, giving operators
    an actionable feed plus an escalation playbook whenever policy violations
    surface.【F:src/trading/risk/policy_telemetry.py†L1-L285】【F:src/trading/trading_manager.py†L920-L991】【F:docs/operations/runbooks/risk_policy_violation.md†L1-L51】【F:tests/trading/test_risk_policy_telemetry.py†L1-L199】
  - *Progress*: Parity checker telemetry now resolves the metrics sink once,
    logs failures to access or publish gauges, and emits guarded order/position
    mismatch counts so institutional monitors see parity outages instead of
    silently dropping telemetry when instrumentation breaks.【F:src/trading/monitoring/parity_checker.py†L53-L156】
  - *Progress*: FIX broker interface risk telemetry now falls back to the
    deterministic risk API contract when provider lookups fail, merges gateway
    limit snapshots with interface summaries, and always includes the shared
    risk API runbook so manual pilot alerts retain actionable escalation
    metadata under pytest coverage.【F:src/trading/integration/fix_broker_interface.py†L211-L330】【F:tests/trading/test_fix_broker_interface_events.py†L170-L239】
  - *Progress*: FIX broker interface now routes every manual intent through the real
    risk gateway, publishes structured rejection telemetry with policy snapshots,
    deterministic severity flags, and runbook links, and records the gateway
    decision/portfolio metadata on approved orders so FIX pilots inherit the same
    deterministic guardrails plus a manual risk block playbook under pytest
    coverage.【F:src/trading/integration/fix_broker_interface.py†L211-L604】【F:tests/trading/test_fix_broker_interface_events.py†L14-L239】【F:docs/operations/runbooks/manual_fix_order_risk_block.md†L1-L38】
- [x] **Quality and observability** – Expand regression coverage, close the
  documentation gap, and track remediation progress through CI snapshots.
  - [x] Publish decision narration capsules that link policy-ledger diffs, sigma
    stability metrics, and throttle states into the observability diary schema
    so AlphaTrade reviewers inherit a single provenance trail.【F:docs/context/alignment_briefs/quality_observability.md†L178-L182】
  - *Progress*: Decision narration capsule helpers now normalise ledger diffs,
    sigma stability, throttle states, and publish Markdown/JSON payloads through
    the event-bus failover helper so observability diaries stay resilient under
    runtime outages with pytest guarding the contract.【F:src/operations/observability_diary.py†L3-L392】【F:tests/operations/test_observability_diary.py†L1-L190】
  - [x] Extend sensory drift regressions with Page–Hinkley sentries, replay
    determinism fixtures, and Prometheus exports that document throttle
    behaviour for the understanding loop.【F:docs/context/alignment_briefs/quality_observability.md†L184-L186】
  - [x] Instrument SLO probes for loop latency, drift alert freshness, and replay
    determinism across Prometheus exporters and guardrail suites.【F:docs/context/alignment_briefs/quality_observability.md†L187-L189】
  - [x] Wire Slack/webhook alert mirrors, rehearse forced-failure drills, and log
    MTTA/MTTR in CI dashboards so responders stay aligned with telemetry
    changes.【F:docs/context/alignment_briefs/quality_observability.md†L171-L174】
  - [x] Refresh CI dashboard rows and weekly status updates with telemetry
    deltas so roadmap evidence remains synchronised with delivery.【F:docs/context/alignment_briefs/quality_observability.md†L175-L177】
  - *Progress*: CI health snapshot and weekly status log now capture coverage and
    remediation deltas with evidence pointers, keeping roadmap artefacts aligned
    with the latest telemetry exports.【F:docs/status/ci_health.md†L10-L21】【F:docs/status/quality_weekly_status.md†L18-L35】
  - *Progress*: Event bus health publishing now routes through the shared
    failover helper, logging runtime publish failures, propagating metadata to
    the global bus, and raising typed errors when both transports degrade so
    operators see deterministic alerts instead of silent drops. Guardrail tests
    capture primary fallbacks, global outages, and backlog escalation.【F:src/operations/event_bus_health.py†L143-L259】【F:tests/operations/test_event_bus_health.py†L22-L235】
  - *Progress*: Evolution tuning telemetry publisher now reuses the shared
    failover helper, warning on runtime bus outages, escalating unexpected
    errors, and exercising fallback/global-error coverage so tuning snapshots
    stay observable when transports degrade.【F:src/operations/evolution_tuning.py†L410-L433】【F:tests/operations/test_evolution_tuning.py†L226-L281】
  - *Progress*: Execution readiness telemetry now rides the shared failover
    helper, logging runtime publish failures, escalating unexpected runtime and
    global bus errors, and falling back deterministically to the global topic
    so dashboards keep receiving readiness snapshots, with pytest coverage
    documenting the fallback contract.【F:src/operations/execution.py†L611-L648】【F:tests/operations/test_execution.py†L100-L134】
  - *Progress*: Regulatory telemetry publisher now reuses the failover helper,
    logging runtime publish failures, escalating unexpected errors, and
    documenting global-bus fallbacks so compliance dashboards retain snapshots
    even during runtime outages.【F:src/operations/regulatory_telemetry.py†L11-L388】【F:tests/operations/test_regulatory_telemetry.py†L18-L160】
  - *Progress*: Strategy performance telemetry aggregates execution/rejection
    ratios, ROI snapshots, and rejection reasons into Markdown summaries, then
    publishes the payload via the shared failover helper so dashboards and
    runtime reports inherit the same hardened transport guarantees under pytest
    coverage.【F:src/operations/strategy_performance.py†L200-L531】【F:tests/operations/test_strategy_performance.py†L68-L193】
  - *Progress*: CI metrics tooling now summarises trend staleness across
    coverage, formatter, domain, and remediation feeds, flagging stale
    telemetry windows with timestamps and age calculations so roadmap evidence
    highlights expired observability snapshots under pytest coverage.【F:tools/telemetry/ci_metrics.py†L214-L320】【F:tests/tools/test_ci_metrics.py†L210-L360】
  - *Progress*: Alert drill CLI and metrics updater now generate timeline JSON
    payloads for forced-failure rehearsals, parse MTTA/MTTR data, and append
    alert-response entries to the CI metrics feed so dashboards surface
    acknowledgement and recovery cadence alongside coverage trends under pytest
    coverage of the CLI/aggregator path.【F:tools/telemetry/alert_drill.py†L29-L172】【F:tools/telemetry/update_ci_metrics.py†L134-L279】【F:tools/telemetry/ci_metrics.py†L597-L658】【F:tests/tools/test_alert_drill.py†L9-L58】【F:tests/tools/test_ci_metrics.py†L340-L618】
  - *Progress*: Incident response readiness now parses policy/state mappings into
    a severity snapshot, derives targeted alert events, and publishes telemetry
    via the shared failover helper so operators get actionable runbook, roster,
    and backlog evidence under pytest coverage covering escalation, dispatch,
    and publish failure paths.【F:src/operations/incident_response.py†L1-L715】【F:tests/operations/test_incident_response.py†L1-L200】
  - *Progress*: System validation evaluator ingests JSON/structured reports,
    normalises timestamps and success rates, logs malformed history payloads at
    debug, renders Markdown, derives alert events, and routes/publishes snapshots
    through the failover helper so readiness dashboards retain failing-check
    context even when the runtime bus degrades, with pytest guarding evaluation,
    alerting, and failover flows.【F:src/operations/system_validation.py†L233-L889】【F:tests/operations/test_system_validation.py†L1-L195】
  - *Progress*: Coverage matrix CLI now surfaces lagging domains, exports the
    full set of covered source files, and enforces required guardrail suites via
    `--require-file`, failing CI when critical reports disappear and logging
    missing paths under pytest coverage.【F:tools/telemetry/coverage_matrix.py†L83-L357】【F:tests/tools/test_coverage_matrix.py†L136-L225】
  - *Progress*: Observability dashboard integrates operational readiness
    snapshots as a first-class panel, summarising component severities and
    surfacing degraded services alongside risk, latency, and backbone telemetry
    under regression coverage so responders inherit a consolidated operational
    view.【F:src/operations/observability_dashboard.py†L443-L493】【F:tests/operations/test_observability_dashboard.py†L135-L236】
  - *Progress*: Observability dashboard guard CLI now grades snapshot freshness,
    required panels, failing slices, and normalised overall status strings with
    machine-readable output plus severity-driven exit codes so CI pipelines and
    drills can block on stale, failing, or operator-reported WARN/FAIL snapshots
    under pytest coverage.【F:tools/telemetry/dashboard_guard.py†L1-L220】【F:tests/tools/test_dashboard_guard.py†L16-L140】
  - *Progress*: Configuration audit telemetry now normalises `SystemConfig`
    diffs, grades tracked toggles plus extras, renders Markdown summaries with a
    severity breakdown, and publishes via the shared failover helper so
    configuration changes leave a durable, event-bus-backed audit trail with
    explicit severity counts and highest-risk fields for operators and
    governance reviewers.【F:src/operations/configuration_audit.py†L90-L210】【F:tests/operations/test_configuration_audit.py†L24-L86】
  - *Progress*: CI metrics staleness guard CLI summarises coverage, formatter,
    domain, and remediation telemetry ages, supports human/JSON output, and
    fails builds when trends go stale or evidence is missing under pytest
    coverage so roadmap reviews inherit fresh observability evidence without
    manual checks.【F:tools/telemetry/ci_metrics_guard.py†L1-L142】【F:tests/tools/test_ci_metrics_guard.py†L1-L99】
  - *Progress*: Health monitor resource probes now normalise psutil import
    failures, log probe errors, retain bounded history, and surface event-bus
    diagnostics so operational responders get actionable state even when optional
    dependencies are missing, with asyncio-loop regressions covering each guardrail.【F:src/operational/health_monitor.py†L61-L200】【F:tests/operational/test_health_monitor.py†L74-L176】
  - *Progress*: Bootstrap control centre helpers now log champion payload,
    trading-manager method, and formatter failures, keeping operational
    diagnostics visible during bootstrap runs and documenting the logging
    behaviour under pytest.【F:src/operations/bootstrap_control_center.py†L31-L115】【F:tests/current/test_bootstrap_control_center.py†L178-L199】
  - *Progress*: Bootstrap orchestration now wraps sensory listeners,
    liquidity probers, and control-centre callbacks with structured error
    logging so optional observability hooks surface failures without breaking
    the decision loop, with pytest capturing the emitted diagnostics.【F:src/orchestration/bootstrap_stack.py†L81-L258】【F:tests/current/test_bootstrap_stack.py†L164-L213】
  - *Progress*: Observability dashboard risk telemetry now annotates each metric
    with limit values, ratios, and violation statuses while preserving serialised
    payloads, backed by regression coverage so operators inherit actionable risk
    summaries instead of opaque aggregates.【F:src/operations/observability_dashboard.py†L254-L309】【F:tests/operations/test_observability_dashboard.py†L201-L241】
  - *Progress*: Observability dashboard composer now fuses ROI, risk, latency,
    backbone, operational readiness, and quality panels into a single snapshot,
    escalating severities from ROI status, risk-limit breaches, event-bus/SLO
    lag, and coverage posture while retaining structured metadata for each panel
    so dashboards and exporters inherit a complete readiness view.【F:src/operations/observability_dashboard.py†L250-L420】【F:tests/operations/test_observability_dashboard.py†L198-L266】
  - *Progress*: Observability dashboard metadata now auto-populates panel status
    counts and per-panel severity maps alongside the remediation capsule so CI
    exporters and runbooks can ingest a machine-readable readiness snapshot
    without recomputing counts, under pytest coverage that locks the contract.【F:src/operations/observability_dashboard.py†L486-L508】【F:tests/operations/test_observability_dashboard.py†L189-L237】
  - *Progress*: Observability dashboard now emits a remediation summary capsule
    that aggregates panel severities, highlights failing/warning slices, and is
    regression-tested so CI status exporters can consume a canonical
    institutional readiness snapshot without re-deriving counts.【F:src/operations/observability_dashboard.py†L60-L109】【F:tests/operations/test_observability_dashboard.py†L60-L116】
  - *Progress*: Operational readiness aggregation now fuses system validation,
    incident response, and ingest SLO snapshots into a single severity grade,
    emits Markdown/JSON for dashboards, derives routed alert events, hardens
    publish failover via the shared helper, and enriches metadata with status
    breakdowns so dashboards can render severity chips without recomputing the
    logic, under pytest coverage and documented contract updates.【F:src/operations/operational_readiness.py†L1-L373】【F:tests/operations/test_operational_readiness.py†L1-L221】【F:docs/status/operational_readiness.md†L1-L60】【F:tests/runtime/test_professional_app_timescale.py†L722-L799】
  - *Progress*: FIX pilot telemetry now evaluates compliance, risk, drop-copy, and
    queue posture against configurable policies, publishes snapshots through the
    failover helper, and documents the publishing contract under pytest so FIX
    deployments surface actionable readiness evidence even when the runtime bus
    degrades.【F:src/operations/fix_pilot.py†L62-L373】【F:tests/operations/test_fix_pilot_ops.py†L1-L164】
  - *Progress*: Compliance readiness snapshots now include workflow portfolio
    status alongside trade and KYC telemetry, escalating blocked checklists,
    surfacing active task counts, and retaining the hardened publish failover so
    governance teams see actionable workflow posture even during runtime bus
    outages.【F:src/operations/compliance_readiness.py†L262-L420】【F:tests/operations/test_compliance_readiness.py†L58-L213】
  - *Progress*: Governance reporting cadence now publishes through the shared
    failover helper with typed escalation messages, preserving cadence payloads
    when the runtime bus degrades and documenting fallback behaviour under
    regression coverage so compliance reviewers always receive the compiled
    KYC/AML, regulatory, and audit telemetry bundle.【F:src/operations/governance_reporting.py†L437-L519】【F:tests/operations/test_governance_reporting.py†L1-L226】
  - *Progress*: System validation telemetry now attaches failing check names and
    messages to snapshot metadata and Markdown output while continuing to route
    through the shared failover helper, so readiness dashboards surface the
    precise failing checks even when the runtime bus degrades, with pytest
    verifying metadata capture and failover escalation.【F:src/operations/system_validation.py†L724-L889】【F:tests/operations/test_system_validation.py†L77-L160】
  - *Progress*: Professional readiness publisher now reuses the hardened
    failover helper, logging runtime fallbacks, refusing unexpected errors, and
    supporting injected global bus factories under pytest coverage so
    operational readiness telemetry is preserved when the primary transport
    fails instead of silently dropping snapshots.【F:src/operations/professional_readiness.py†L268-L305】【F:tests/operations/test_professional_readiness.py†L164-L239】
  - *Progress*: Sensory drift telemetry publisher now routes through the shared
    failover helper, logging runtime and global-bus degradation while retaining
    deterministic payload metadata so dashboards receive alerts even when the
    primary bus misbehaves.【F:src/operations/sensory_drift.py†L247-L276】【F:tests/operations/test_sensory_drift.py†L17-L163】
  - *Progress*: Operational metrics regression suite now exercises fallback
    execution, FIX wrapper sanitisation, and bounded latency histograms so guardrail
    telemetry captures instrumentation failures instead of dropping them silently.
    【F:src/operational/metrics.py†L1-L200】【F:tests/operational/test_metrics.py†L200-L328】
  - *Progress*: Prometheus exporter hardening now narrows port parsing failures
    to typed errors, logs the parsing context, and records telemetry sink import
    issues so metrics startup surfaces actionable warnings instead of swallowing
    unexpected exceptions.【F:src/operational/metrics.py†L545-L608】
  - *Progress*: Guardrail manifest tests now enforce the presence and pytest marker
    coverage of ingest orchestration, risk policy, and observability suites, and
    assert that the CI workflow runs `pytest -m guardrail` plus enumerates the
    guardrail-critical domains in the coverage sweep so the pipeline fails fast
    if files, markers, or workflow hooks drift.【F:tests/runtime/test_guardrail_suite_manifest.py†L18-L91】
  - *Progress*: CI telemetry tooling now records remediation status snapshots via
    the `--remediation-status` CLI flag and validates the JSON contract under
    pytest so roadmap evidence, dashboard feeds, and audits inherit structured
    remediation progress without manual spreadsheets.【F:tools/telemetry/update_ci_metrics.py†L10-L176】【F:tests/tools/test_ci_metrics.py†L180-L332】【F:tests/.telemetry/ci_metrics.json†L1-L6】
  - *Progress*: Coverage telemetry recorder now flags lagging domains, captures
    the worst-performing slice, and tags threshold breaches in the CI metrics
    feed while the CLI ingests observability dashboard snapshots into the
    remediation trend so status exports inherit actionable coverage and
    operational readiness deltas.【F:tools/telemetry/ci_metrics.py†L112-L337】【F:tools/telemetry/update_ci_metrics.py†L1-L169】【F:tests/tools/test_ci_metrics.py†L180-L309】
  - *Progress*: Flake telemetry feed now records the adaptive release thresholds regression (node id, diff, duration) so automation reviews surface the gating failure alongside deterministic metadata for triage.【F:tests/.telemetry/flake_runs.json†L1-L20】
  - *Progress*: CI digest CLI now renders dashboard rows and weekly digests from
    the metrics JSON, calculating coverage/domain/remediation deltas with pytest
    coverage and wiring straight into the status log so teams can paste evidence
    into the backlog and weekly reports without manual collation.【F:tools/telemetry/ci_digest.py†L1-L240】【F:tests/tools/test_ci_digest.py†L1-L152】【F:docs/status/ci_health.md†L13-L19】【F:docs/status/quality_weekly_status.md†L1-L26】
  - *Progress*: Quality telemetry snapshot builder now normalises coverage,
    staleness, and remediation trends into a typed `QualityTelemetrySnapshot`,
    escalating WARN/FAIL severities, retaining lagging-domain metadata, and
    capturing remediation notes so CI exports feed dashboards with deterministic
    coverage posture evidence.【F:src/operations/quality_telemetry.py†L1-L168】【F:tests/operations/test_quality_telemetry.py†L9-L53】
  - *Progress*: Remediation summary exporter renders telemetry snapshots into
    Markdown tables with delta call-outs, honours slice limits, omits deltas for
    non-numeric statuses, and ships with a CLI/pytest contract so status reports
    can ingest `tests/.telemetry/ci_metrics.json` without hand-curated decks.【F:tools/telemetry/remediation_summary.py†L1-L220】【F:tests/tools/test_remediation_summary.py†L22-L125】
  - *Progress*: Status digest CLI fuses coverage, formatter, remediation,
    freshness, and observability dashboard telemetry into Markdown for CI table
    rows or weekly updates, with pytest locking the CLI contract so briefs and
    sprint notes stay evidence-backed.【F:tools/telemetry/status_digest.py†L1-L347】【F:tests/tools/test_status_digest.py†L1-L217】【F:docs/context/alignment_briefs/quality_observability.md†L262-L271】
- [ ] **Dead code and duplication** – Triage the 168-file dead-code backlog and
  eliminate shim exports so operators see a single canonical API surface.【F:docs/reports/CLEANUP_REPORT.md†L71-L188】
  - *Progress*: Removed the deprecated risk and evolution configuration shims,
    deleted the core risk manager/stress/VaR stand-ins, and pointed callers at
    the canonical implementations so consumers converge on the real modules when
    new services arrive.【F:docs/reports/CLEANUP_REPORT.md†L71-L104】【F:src/core/__init__.py†L16-L46】【F:src/risk/analytics/var.py†L19-L121】
  - *Progress*: Retired the legacy strategy template package and rewrote the
    canonical mean reversion regression to exercise the modern trading
    strategies API, shrinking the dead-code backlog and aligning tests with the
    production surface.【F:docs/reports/CLEANUP_REPORT.md†L87-L106】【F:tests/current/test_mean_reversion_strategy.py†L1-L80】
  - *Progress*: Core configuration module now proxies every legacy accessor to
    the canonical `SystemConfig`, preserving environment overrides, YAML import
    compatibility, and debug coercion so downstream consumers can migrate
    without duplicating parsing logic while the shim stops drifting from the
    source of truth.【F:src/core/configuration.py†L1-L188】
  - *Progress*: Operational package import now aliases `src.operational.event_bus`
    to the canonical core implementation, keeping legacy paths alive while tests
    assert the wiring so cleanup work can retire the shim safely.【F:src/operational/__init__.py†L1-L38】【F:tests/operational/test_event_bus_alias.py†L162-L178】

## Roadmap cadence

### Now (0–30 days)

- [x] **Stabilise runtime entrypoints** – Move all application starts through
  `RuntimeApplication` and register background jobs under a task supervisor to
  eliminate unsupervised `create_task` usage. Runtime CLI invocations and the
  bootstrap sensory loop now run under `TaskSupervisor`, ensuring graceful
  signal/time-based shutdown paths.【F:docs/technical_debt_assessment.md†L33-L56】【F:src/runtime/cli.py†L206-L249】【F:src/runtime/bootstrap_runtime.py†L227-L268】
  - *Progress*: Phase 3 orchestrator now spawns continuous analysis and performance
    monitors via the shared task supervisor, drains background tasks on stop, and
    ships a guardrail smoke test so thinking pipelines inherit the same supervised
    lifecycle contract as runtime entrypoints.【F:src/thinking/phase3_orchestrator.py†L103-L276】【F:tests/current/test_orchestration_runtime_smoke.py†L19-L102】
- [ ] **Security hardening sprint** – Execute the remediation plan’s Phase 0:
  parameterise SQL, remove `eval`, and address blanket exception handlers in
  operational modules.【F:docs/development/remediation_plan.md†L34-L72】
    - *Progress*: Hardened the SQLite-backed real portfolio monitor with managed
      connections, parameterised statements, and narrowed exception handling to
      surface operational failures instead of masking them.【F:src/trading/portfolio/real_portfolio_monitor.py†L1-L572】
    - *Progress*: Strategy registry now opens per-operation SQLite connections,
      raises typed errors, and uses parameterised statements so governance writes
      are supervised instead of silently swallowed.【F:src/governance/strategy_registry.py†L1-L347】
    - *Progress*: Data retention telemetry guards schema/table/timestamp
      identifiers and uses SQLAlchemy parameter binding so operators cannot
      inject raw SQL through policy configuration, with tests covering the
      hardened contract.【F:src/operations/retention.py†L42-L195】【F:tests/operations/test_data_retention.py†L1-L118】
    - *Progress*: Yahoo ingest persistence sanitises DuckDB table names, uses
      parameterised deletes, and asserts gateway error handling so bootstrap
      persistence cannot be hijacked by crafted identifiers or silent
      failures.【F:src/data_foundation/ingest/yahoo_ingest.py†L82-L151】【F:tests/data_foundation/test_yahoo_ingest_security.py†L32-L80】【F:tests/data_integration/test_yfinance_gateway_security.py†L12-L56】
    - *Progress*: Retired the legacy `icmarkets_robust_application` scaffolding so
      the security sweep no longer claims coverage for a deleted stub; the
      cleanup report now records the removal while the typed FIX connection
      manager remains the canonical broker entrypoint.【F:docs/reports/CLEANUP_REPORT.md†L110-L170】【F:src/operational/fix_connection_manager.py†L21-L320】
    - *Progress*: Portfolio tracker now falls back to an atomic JSON state store,
      logs corrupted or missing snapshots, and persists fills by symbol with
      regression coverage so parity checks and compliance telemetry inherit a
      deterministic portfolio view even without Redis.【F:src/trading/monitoring/portfolio_tracker.py†L1-L139】【F:tests/trading/test_portfolio_tracker_security.py†L1-L30】
    - *Progress*: Security posture publishing now warns and falls back to the
      global bus when runtime publishing fails, raises on unexpected errors, and
      documents the error-handling paths under pytest so telemetry outages cannot
      disappear silently.【F:src/operations/security.py†L536-L579】【F:tests/operations/test_security.py†L148-L263】
    - *Progress*: Event bus failover helper now powers security, system
      validation, compliance readiness, incident response, and evolution
      experiment publishing, replacing ad-hoc blanket handlers with typed errors
      and structured logging so transport regressions escalate deterministically
      across feeds.【F:src/operations/event_bus_failover.py†L1-L174】【F:src/operations/incident_response.py†L350-L375】【F:src/operations/evolution_experiments.py†L297-L342】【F:tests/operations/test_event_bus_failover.py†L1-L164】【F:tests/operations/test_incident_response.py†L123-L167】【F:tests/operations/test_evolution_experiments.py†L135-L191】
- [x] **Context pack refresh** – Replace legacy briefs with the updated context in
  `docs/context/alignment_briefs` so discovery and reviews inherit the same
  narrative reset (this change set).
- [ ] **Coverage guardrails** – Extend the CI baseline to include ingest orchestration
  and risk policy regression tests, lifting coverage beyond the fragile 76% line.
  - *Progress*: Added an end-to-end regression for the real portfolio monitor to
    exercise data writes, analytics, and reporting flows under pytest, closing a
    previously untested gap in the trading surface.【F:tests/trading/test_real_portfolio_monitor.py†L1-L77】
  - *Progress*: Added ingest observability and risk policy telemetry regression tests
    so CI surfaces regressions in data backbone snapshots and policy evaluation
    markdown output.【F:tests/data_foundation/test_ingest_observability.py†L1-L190】【F:tests/trading/test_risk_policy_telemetry.py†L1-L124】
  - *Progress*: Data backbone readiness coverage now asserts failover trigger
    metadata and Timescale recovery plan serialisation, while risk policy
    regression tests lock mandatory stop-loss and equity budget enforcement so
    coverage extensions land with actionable guardrails instead of brittle
    placeholders.【F:tests/operations/test_data_backbone.py†L289-L347】【F:tests/trading/test_risk_policy.py†L178-L222】
  - *Progress*: Coverage telemetry now emits per-domain matrices from the
    coverage XML, with CLI tooling and pytest coverage documenting the JSON/markdown
    contract so dashboards can flag lagging domains without scraping CI logs.【F:tools/telemetry/coverage_matrix.py†L1-L199】【F:tests/tools/test_coverage_matrix.py†L1-L123】【F:docs/status/ci_health.md†L13-L31】
  - *Progress*: CI now renders an ingest coverage matrix after every guarded
    pytest run, enforcing `coverage.xml` generation and minimum coverage for the
    institutional Timescale pipeline while appending the Markdown summary to the
    GitHub Actions run; the guardrail manifest locks the workflow step so
    coverage gating cannot be removed silently.【F:.github/workflows/ci.yml†L95-L120】【F:tools/telemetry/coverage_matrix.py†L1-L199】【F:tests/runtime/test_guardrail_suite_manifest.py†L98-L114】
  - *Progress*: CI workflow now fails fast if ingest, operations, trading, and
    governance suites regress by pinning pytest entrypoints and coverage include
    lists to those domains, preventing partial runs from passing unnoticed.【F:.github/workflows/ci.yml†L79-L120】【F:pytest.ini†L1-L14】【F:pyproject.toml†L45-L85】
  - *Progress*: Ingest trend and Kafka readiness publishers now log event bus
    failures, raise on unexpected exceptions, fall back to the global bus with
    structured warnings, and surface offline-global cases under pytest coverage so
    telemetry gaps raise alerts instead of disappearing silently.【F:src/operations/ingest_trends.py†L303-L336】【F:tests/operations/test_ingest_trends.py†L90-L148】【F:src/operations/kafka_readiness.py†L305-L328】【F:tests/operations/test_kafka_readiness.py†L219-L345】
  - *Progress*: Kafka readiness suite now asserts required-topic and consumer
    coverage, tolerates epoch millisecond lag timestamps, and renders Markdown
    summaries so operations can embed readiness tables directly in incident
    updates.【F:tests/operations/test_kafka_readiness.py†L116-L207】
  - *Progress*: Security telemetry regression suite now exercises runtime-bus
    fallbacks, global-bus escalation, and unexpected-error handling so security
    posture publishing surfaces outages deterministically instead of silently
    discarding events.【F:tests/operations/test_security.py†L101-L211】
  - *Progress*: Cache health telemetry now logs primary bus failures, only falls
    back once runtime errors are captured, and raises on unexpected or global-bus
    errors with pytest guardrails so readiness dashboards record real outages
    instead of silent drops.【F:src/operations/cache_health.py†L143-L245】【F:tests/operations/test_cache_health.py†L15-L138】
  - *Progress*: Trading position model guardrails now run under pytest,
    asserting timestamp updates, profit recalculations, and close flows so the
    lightweight execution telemetry remains deterministic under CI coverage.【F:tests/trading/test_position_model_guardrails.py†L1-L105】
  - *Progress*: Timescale ingest coverage now exercises migrator setup, idempotent
    daily/intraday upserts, and macro event pathways so empty plans and windowed
    flows keep writing deterministic telemetry under CI guardrails.【F:tests/data_foundation/test_timescale_ingest.py†L1-L213】
  - *Progress*: Timescale ingest orchestrator regression suite now validates engine
    lifecycle hooks, publisher metadata, empty-plan short-circuits, and guardrails
    for missing intraday fetchers so institutional ingest cannot regress silently.【F:tests/data_foundation/test_timescale_backbone_orchestrator.py†L1-L200】
  - *Progress*: CI now runs a dedicated `pytest -m guardrail` job ahead of the
    coverage sweep, ensuring ingest, risk, and observability guardrail tests are
    executed in isolation with deterministic markers and failing fast when
    regressions surface.【F:.github/workflows/ci.yml†L79-L123】【F:pytest.ini†L1-L25】【F:tests/data_foundation/test_timescale_backbone_orchestrator.py†L1-L28】【F:tests/operations/test_event_bus_health.py†L1-L155】
  - *Progress*: Coverage guardrail CLI now parses Cobertura reports, enforces
    minimum percentages across ingest/risk targets, flags missing files, and
    exposes JSON/text summaries with failure exit codes so CI and local triage
    can block on regression hotspots deterministically.【F:tools/telemetry/coverage_guardrails.py†L1-L268】【F:tests/tools/test_coverage_guardrails.py†L1-L83】
  - *Progress*: Runtime builder coverage now snapshots ingest plan dimensions,
    trading metadata, and enforced risk summaries, while risk policy regressions
    assert portfolio price fallbacks so ingest orchestration and risk sizing
    guardrails stay under deterministic pytest coverage.【F:tests/runtime/test_runtime_builder.py†L1-L196】【F:tests/trading/test_risk_policy.py†L1-L511】
  - *Progress*: Risk policy regression enforces minimum position sizing while the
    observability dashboard tests assert limit-status escalation so CI catches
    governance and telemetry drift before it hits production surfaces.【F:tests/trading/test_risk_policy.py†L311-L333】【F:tests/operations/test_observability_dashboard.py†L222-L241】
  - *Progress*: Observability logging and dashboard suites now carry the
    `guardrail` marker so CI can gatekeep their execution ahead of the broader
    coverage sweep.【F:tests/observability/test_logging.py†L18-L24】【F:tests/operations/test_observability_dashboard.py†L24-L31】
  - *Progress*: Ingest scheduler guardrails now exercise run-loop shutdown,
    failure cut-offs, jitter windows, supervisor telemetry, snapshot builders,
    and event publishing so Timescale scheduling instrumentation surfaces issues
    immediately instead of stalling silently.【F:tests/data_foundation/test_ingest_scheduler.py†L1-L200】
  - *Progress*: Risk policy warn-threshold coverage asserts that leverage and
    exposure checks flip to warning states before violating limits, capturing
    ratios, thresholds, and metadata so compliance reviewers can trust the
    policy telemetry feed when positions approach guardrails.【F:tests/trading/test_risk_policy.py†L125-L170】

- [x] **AlphaTrade understanding loop sprint (Days 0–14)** – Stand up the live-shadow
  Perception → Adaptation → Reflection loop so AlphaTrade parity work can ship
  without capital risk.【F:docs/High-Impact Development Roadmap.md†L5-L21】【F:docs/High-Impact Development Roadmap.md†L73-L76】
  - *Progress*: Belief/regime scaffolding now ships `BeliefState` buffers,
    Hebbian updates, and regime FSM emitters that publish event-bus payloads with
    PSD guardrails, golden fixtures, and guardrail pytest coverage so live-shadow
    inputs bind to stable schemas.【F:src/understanding/belief.py†L39-L347】【F:tests/intelligence/test_belief_updates.py†L129-L200】【F:tests/intelligence/golden/belief_snapshot.json†L1-L120】
  - [x] Implement `UnderstandingRouter` fast-weight adapters with feature gating,
    configuration schema, and guardrail tests so strategy routing stays
    auditable.【F:src/understanding/router.py†L70-L240】【F:src/understanding/router_config.py†L1-L320】【F:tests/understanding/test_understanding_router_config.py†L1-L88】【F:docs/context/examples/understanding_router.md†L1-L64】
  - [x] Automate decision diaries and the probe registry with CLI exports and
    governance hooks so reviewers inherit narrated decisions and probe
    ownership.【F:docs/context/sprint_briefs/understanding_loop_v1.md†L63-L76】
  - [x] Stand up drift sentry detectors, alert policies, and runbook updates that
    tie Page–Hinkley/variance thresholds into readiness dashboards.【F:docs/context/sprint_briefs/understanding_loop_v1.md†L78-L91】【F:docs/High-Impact Development Roadmap.md†L52-L53】
  - *Progress*: Understanding drift sentry now evaluates belief/regime metrics, publishes failover-aware telemetry, derives alert payloads, and pipes runbook metadata into operational readiness so incident responders inherit a single drift component across dashboards and alert policies under regression coverage.【F:src/operations/drift_sentry.py†L1-L399】【F:tests/intelligence/test_drift_sentry.py†L43-L135】【F:tests/operations/test_operational_readiness.py†L200-L283】【F:docs/operations/runbooks/drift_sentry_response.md†L1-L69】
  - *Progress*: DriftSentry gate now ingests sensory drift snapshots, applies confidence/notional guardrails, and surfaces gating telemetry through runtime bootstrap and Predator app summaries under dedicated trading manager regressions so drift incidents halt paper promotions with documented evidence.【F:src/trading/gating/drift_sentry_gate.py†L1-L200】【F:src/runtime/bootstrap_runtime.py†L161-L177】【F:src/runtime/predator_app.py†L1012-L1024】【F:tests/trading/test_trading_manager_execution.py†L187-L260】【F:tests/trading/test_drift_sentry_gate.py†L61-L153】
  - *Progress*: Sensory drift regression suite now ships a deterministic Page–Hinkley
    replay fixture and metadata assertions so escalations reproduce the alert
    catalog, runbook link, and detector stats with evidence bundles backed by
    pytest coverage.【F:tests/operations/fixtures/page_hinkley_replay.json†L1-L128】【F:tests/operations/test_sensory_drift.py†L157-L218】
  - [x] Deliver the policy ledger store, rebuild CLI, and governance checklist so
    promotions trace back to DecisionDiary evidence.【F:docs/context/sprint_briefs/understanding_loop_v1.md†L93-L107】【F:docs/High-Impact Development Roadmap.md†L53-L54】
  - *Progress*: Policy ledger store now persists promotion history, approvals, threshold overrides, and diary evidence, with a rebuild CLI that regenerates enforceable risk configs and router guardrails while exporting governance workflows under pytest coverage so AlphaTrade promotions stay auditable.【F:src/governance/policy_ledger.py†L1-L200】【F:src/governance/policy_rebuilder.py†L1-L141】【F:tools/governance/rebuild_policy.py†L1-L112】【F:tests/governance/test_policy_ledger.py†L33-L181】【F:tests/tools/test_rebuild_policy_cli.py†L11-L41】
  - [x] Provide graph diagnostics CLI, guardrailed acceptance workflow, and
    operational dashboard tile so AlphaTrade deltas remain observable.【F:docs/context/sprint_briefs/understanding_loop_v1.md†L108-L128】
  - *Progress*: Understanding diagnostics builder now emits sensory→belief→router→policy graphs with snapshot exports, wrapped by a CLI that renders JSON/DOT/Markdown and guarded by the `understanding_acceptance` marker plus dedicated pytest suite.【F:src/understanding/diagnostics.py†L395-L542】【F:src/understanding/__init__.py†L3-L22】【F:tools/understanding/graph_diagnostics.py†L1-L82】【F:tests/understanding/test_understanding_diagnostics.py†L15-L29】【F:pytest.ini†L2-L27】
  - *Progress*: Observability dashboard now renders an understanding-loop panel summarising regime confidence, drift exceedances, experiments, and ledger approvals when diagnostics land, and escalates to WARN with CLI guidance whenever snapshots are missing so operators rebuild artifacts deterministically under guardrail tests.【F:src/operations/observability_dashboard.py†L536-L565】【F:tests/operations/test_observability_dashboard.py†L389-L413】
  - *Progress*: Understanding metrics exporter now normalises throttle states into Prometheus gauges and hooks the observability dashboard so every loop snapshot publishes throttle posture, with replay fixtures and guardrail tests locking the gauge contract.【F:src/operational/metrics.py†L43-L428】【F:src/understanding/metrics.py†L1-L65】【F:tests/operational/test_metrics.py†L310-L360】【F:tests/understanding/test_understanding_metrics.py†L62-L125】【F:tests/operations/test_observability_dashboard.py†L394-L436】
  - *Progress*: Bootstrap runtime now instantiates the real sensory organ with a
    drift-tuned history buffer, streams observations into cortex metrics,
    publishes summary/metrics/drift telemetry via the event-bus failover helper,
    and exposes samples/audits via `status()` so supervisors inherit sensory
    posture and live telemetry under dedicated runtime coverage.【F:src/runtime/bootstrap_runtime.py†L214-L492】【F:tests/runtime/test_bootstrap_runtime_sensory.py†L120-L196】

### Next (30–90 days)

- [ ] **Institutional ingest vertical** – Provision managed Timescale/Redis/Kafka
  environments, implement supervised connectors, and document failover drills.
- [ ] **Sensory cortex uplift** – Deliver executable HOW/ANOMALY organs, instrument
  drift telemetry, and expose metrics through runtime summaries and the event
  bus.
  - *Progress*: Real sensory organ now attaches metrics payloads to every
    snapshot broadcast, wrapping dimension strength/confidence telemetry and the
    integrated signal alongside drift summaries so downstream dashboards receive
    a single event with metrics, lineage, and posture metadata under pytest
    coverage.【F:src/sensory/real_sensory_organ.py†L198-L205】【F:tests/sensory/test_real_sensory_organ.py†L132-L158】
- [ ] **Evolution engine foundation** – Seed realistic genomes, wire lineage
  snapshots, and gate adaptive runs behind feature flags until governance reviews
  complete.【F:docs/development/remediation_plan.md†L92-L167】
  - *Progress*: Realistic genome seeding now materialises catalogue templates with jitter bounds, attaches lineage and performance metadata to spawned genomes, and refreshes orchestrator lineage snapshots so population statistics expose provenance under guardrail tests.【F:src/core/evolution/seeding.py†L1-L200】【F:src/orchestration/evolution_cycle.py†L125-L220】【F:tests/current/test_evolution_orchestrator.py†L60-L133】【F:tests/current/test_population_manager_with_genome.py†L91-L127】
  - *Progress*: Recorded dataset helpers now persist real sensory observations to JSONL, keep lineage/drift metadata intact, and reload them into replay evaluators with strict/append guards so adaptive fitness runs can hydrate live evidence instead of mocks under pytest coverage.【F:src/evolution/evaluation/datasets.py†L1-L171】【F:src/evolution/__init__.py†L21-L71】【F:tests/evolution/test_recorded_dataset.py†L1-L108】
- [ ] **Risk API enforcement** – Align trading modules with deterministic risk
  interfaces, surface policy violations via telemetry, and add escalation runbooks.
  - *Progress*: Risk gateway wiring now normalises intents, enforces
    drawdown/exposure/liquidity guardrails, and publishes policy decisions so
    trading managers consume the same deterministic risk manager path as the
    runtime builder.【F:src/trading/trading_manager.py†L1-L320】【F:src/trading/risk/risk_gateway.py†L161-L379】【F:tests/current/test_risk_gateway_validation.py†L74-L206】
  - *Progress*: Risk gateway decisions now attach deterministic `risk_reference`
    payloads with runbook links, limit snapshots, and risk-config summaries,
    caching the metadata for approved and rejected intents while broker events
    surface the same context under regression coverage so responders inherit a
    single audit trail across telemetry surfaces.【F:src/trading/risk/risk_gateway.py†L224-L519】【F:tests/current/test_risk_gateway_validation.py†L93-L407】【F:tests/trading/test_fix_broker_interface_events.py†L15-L152】
  - *Progress*: Trading manager now hydrates governance surfaces with the
    gateway’s limits snapshot, merges runtime-derived risk metadata, and
    normalises `risk_reference` payloads while surfacing shared runbooks so
    operations dashboards, status calls, and interface inspectors expose the
    same audited configuration under pytest coverage.【F:src/trading/trading_manager.py†L786-L939】【F:src/trading/risk/risk_gateway.py†L396-L485】【F:tests/trading/test_trading_manager_execution.py†L1125-L1171】【F:tests/current/test_risk_gateway_validation.py†L391-L424】
  - *Progress*: Professional runtime summaries now pin the shared risk API
    runbook, attach runtime metadata, merge resolved interface details, and
    surface structured `RiskApiError` payloads so operators inherit actionable
    posture even when integrations degrade under pytest coverage.【F:src/runtime/predator_app.py†L995-L1063】【F:tests/current/test_runtime_professional_app.py†L304-L364】
  - *Progress*: Liquidity prober tasks now run under the shared supervisor,
    capture deterministic risk metadata or runbook-tagged failures, and expose
    regression coverage so execution telemetry inherits auditable risk context
    for every probe burst.【F:src/trading/execution/liquidity_prober.py†L38-L334】【F:tests/trading/test_execution_liquidity_prober.py†L64-L123】
- [ ] **AlphaTrade loop expansion (Days 15–90)** – Graduate the live-shadow pilot
  into tactic experimentation, paper trading, and limited live promotions once V1
  stabilises.【F:docs/High-Impact Development Roadmap.md†L74-L76】
  - [ ] Expand PolicyRouter tactics and fast-weight experimentation while
    automating reflection summaries so reviewers see emerging strategies without
    spelunking telemetry dumps.【F:docs/High-Impact Development Roadmap.md†L74-L74】
  - *Progress*: PolicyRouter now tracks tactic objectives/tags, bulk-registers and
    updates tactics, exposes experiment registries, and ships a reflection digest
    that summarises streaks, regime mix, and experiment share while pruning
    expired experiments and exporting reviewer-ready reflection reports so
    reviewers spot emerging strategies without spelunking telemetry dumps under
    expanded pytest coverage.【F:src/thinking/adaptation/policy_router.py†L175-L525】【F:tests/thinking/test_policy_router.py†L248-L308】
  - *Progress*: AdversarialTrainer now logs generator signature mismatches,
    captures unexpected training failures with stack traces, and preserves
    heuristic fallbacks so migration bugs surface during experimentation without
    stalling adaptive runs.【F:src/thinking/adversarial/adversarial_trainer.py†L14-L140】
  - *Progress*: Prediction, survival, and red-team normalisers now swallow
    exploding `.dict()` calls and attribute errors while regression tests lock the
    defensive paths so AlphaTrade analysis surfaces stay resilient to integration
    payload drift.【F:src/thinking/models/normalizers.py†L26-L182】【F:tests/thinking/test_normalizers.py†L1-L85】
  - [ ] Enable selective paper-trade execution with DriftSentry gating
    promotions and PolicyLedger enforcing audit coverage ahead of live capital
    exposure.【F:docs/High-Impact Development Roadmap.md†L75-L75】
  - *Progress*: Trading manager now wires in DriftSentry gating, recording decisions, experiment events, and risk summaries whenever drift blocks or warns on paper trades so selective execution honours governance guardrails under pytest coverage.【F:src/trading/trading_manager.py†L183-L367】【F:src/trading/trading_manager.py†L584-L612】【F:tests/trading/test_trading_manager_execution.py†L187-L260】
  - *Progress*: Adaptive release thresholds now derive ledger stages,
    tighten confidence/notional guardrails based on sensory drift severity, and
    feed TradingManager gating plus release posture telemetry so promotions
    honour governance approvals under guardrail tests.【F:src/trading/gating/adaptive_release.py†L47-L211】【F:src/trading/trading_manager.py†L183-L659】【F:tests/trading/test_adaptive_release_thresholds.py†L57-L138】【F:tests/trading/test_trading_manager_execution.py†L423-L472】

### Later (90+ days)

- [ ] **Operational readiness** – Expand incident response, alert routing, and system
  validation so professional deployments can demonstrate reliability.
- [ ] **Dead-code eradication** – Batch-delete unused modules flagged by the cleanup
  report and tighten import guards to prevent shims from resurfacing.【F:docs/reports/CLEANUP_REPORT.md†L71-L188】
  - *Progress*: Dead-code tracker CLI now parses the cleanup report, highlights
    live shim exports, flags missing candidates, and emits text/JSON summaries
    so platform hygiene reviews can prioritise deletions and wire guardrails
    without manually scraping Markdown lists.【F:tools/cleanup/dead_code_tracker.py†L1-L145】【F:tests/tools/test_dead_code_tracker.py†L1-L42】
  - *Progress*: Removed the legacy `src.intelligence.adversarial_training` shim
    and retargeted the intelligence facade to load the canonical
    `thinking.adversarial` implementations directly, shrinking the cleanup
    backlog while preserving lazy public imports for phase-three orchestrators.【F:src/intelligence/__init__.py†L40-L105】
  - *Progress*: Retired the placeholder sensory config and macro ingest helpers,
    noting their removal in the cleanup report so hygiene reviews reflect the new
    Timescale fallback wiring instead of dead scaffolding.【F:docs/reports/CLEANUP_REPORT.md†L88】【F:src/data_foundation/ingest/timescale_pipeline.py†L21】
  - *Progress*: Cleanup automation now flags the retired sensory dimension shims directly in the report and drops them from the phase-two consolidation script so dead organs stay archived instead of being rehydrated by tooling.【F:docs/reports/CLEANUP_REPORT.md†L168-L179】【F:scripts/phase2_sensory_consolidation.py†L45-L144】
- [ ] **Governance and compliance** – Build the reporting cadence for KYC/AML,
  regulatory telemetry, and audit storage prior to live-broker pilots.【F:docs/technical_debt_assessment.md†L58-L112】
  - *Progress*: Governance reporting cadence now assembles compliance readiness,
    regulatory telemetry, and Timescale audit evidence into a single artefact,
    escalates overall status, publishes via the event-bus failover helper, and
    trims persisted histories so audits inherit deterministic evidence with
    pytest covering scheduling, publishing, and storage flows.【F:src/operations/governance_reporting.py†L1-L520】【F:tests/operations/test_governance_reporting.py†L1-L226】
  - *Progress*: Compliance readiness snapshots now normalise trade-surveillance and
    KYC components, escalate severities deterministically, and render markdown
    evidence with regression coverage so governance cadences inherit reliable
    compliance posture telemetry.【F:src/operations/compliance_readiness.py†L1-L220】【F:tests/operations/test_compliance_readiness.py†L1-L173】
  - *Progress*: Governance cadence runner now honours forced executions,
    metadata overrides, and context-pack lookups while orchestrating interval
    gating, audit evidence collection, report persistence, and event-bus
    publishing so institutional deployments can schedule or manually trigger the
    cadence behind injectable providers under pytest coverage.【F:src/operations/governance_cadence.py†L1-L166】【F:src/operations/governance_reporting.py†L604-L635】【F:tests/operations/test_governance_cadence.py†L1-L120】
  - *Progress*: Governance cadence CLI resolves SystemConfig extras into JSON
    context packs, layers optional snapshot overrides, supports forced runs, and
    emits Markdown/JSON outputs so operators can run the cadence without the
    runtime while preserving persisted history and metadata provenance under
    pytest coverage.【F:tools/governance/run_cadence.py†L1-L368】【F:tests/tools/test_run_governance_cadence.py†L47-L138】
  - *Progress*: Governance report export CLI now loads compliance/regulatory/audit
    snapshots, persists history with metadata, emits Markdown alongside JSON, and
    records regression coverage so operators can script cadence exports without
    bespoke tooling.【F:tools/telemetry/export_governance_report.py†L1-L260】【F:tests/tools/test_export_governance_report.py†L1-L139】
  - *Progress*: Policy ledger now records tactic promotions, approvals, and
    threshold overrides, builds governance workflow checklists, and the trading
    manager/runtime builder publish the combined compliance snapshots so KYC and
    trade surveillance inherit staged release thresholds under regression
    coverage.【F:src/governance/policy_ledger.py†L1-L405】【F:src/compliance/workflow.py†L1-L419】【F:src/trading/trading_manager.py†L640-L764】【F:src/runtime/runtime_builder.py†L2920-L2987】【F:tests/compliance/test_compliance_workflow.py†L1-L182】【F:tests/trading/test_trading_manager_execution.py†L430-L512】
  - *Progress*: Runtime builder now publishes the enforced risk configuration as
    telemetry and the professional runtime records the broadcast payload so risk
    summaries mirror the exact configuration emitted to operations dashboards
    under pytest coverage of the event flow and summary surface.【F:src/runtime/runtime_builder.py†L633-L734】【F:src/runtime/predator_app.py†L472-L1009】【F:tests/runtime/test_runtime_builder.py†L340-L420】

## Actionable to-do tracker

| Status | Task | Owner hint | Linkage |
| --- | --- | --- | --- |
| [ ] | Stand up production-grade ingest slice with parameterised SQL and supervised tasks | Data backbone squad | Now → Operational data backbone |
| [ ] | Deliver executable HOW/ANOMALY organs with lineage telemetry and regression coverage | Sensory cortex squad | Now/Next → Sensory + evolution execution |
| [ ] | Roll out deterministic risk API and supervised runtime builder across execution modules | Execution & risk squad | Now/Next → Risk and runtime safety |
| [x] | Expand CI to cover ingest orchestration, risk policies, and observability guardrails | Quality guild | Now → Quality and observability |
| [ ] | Purge deprecated shims and close dead-code backlog | Platform hygiene crew | Later → Dead code and duplication |

- *Progress*: Risk and evolution configuration now source directly from their
  canonical modules with the legacy shims removed, shrinking the cleanup queue
  and preventing namespace drift.【F:docs/reports/CLEANUP_REPORT.md†L74-L84】【F:src/config/risk/risk_config.py†L1-L72】【F:src/core/evolution/engine.py†L13-L52】

## Execution guardrails

- Keep policy, lint, types, and pytest checks green on every PR; treat CI failures
  as blockers.
- Update context packs and roadmap status pages alongside significant feature
  work; stale documentation is considered a regression.【F:docs/technical_debt_assessment.md†L90-L112】
- Maintain the truth-first status culture: mock implementations must remain
  labelled and roadmapped until replaced by production-grade systems.【F:docs/DEVELOPMENT_STATUS.md†L7-L35】

## Automation updates — 2025-10-07T19:26:47Z

### Last 4 commits
- 98fd31f variant-4 (2025-10-07)
- 2aa59b0 variant-3 (2025-10-07)
- f9c0243 docs(docs): tune 4 files (2025-10-07)
- a0152f4 docs(docs): tune 2 files (2025-10-07)

## Automation updates — 2025-10-07T15:30:42Z

### Last 4 commits
- eb9f8db Auto squash docs (2025-10-07)
- 85a2db9 Auto squash watch-docs (2025-10-07)
- 2fced69 Auto squash watch-variant (2025-10-07)
- bc3da75 Auto squash watch-docs (2025-10-07)
## Automation updates — 2025-10-07T16:13:29Z

### Last 4 commits
- ed2d3f8 Auto cherry-pick variant 4 (2025-10-07)
- ba15189 Auto cherry-pick variant 3 (2025-10-07)
- b42e927 Auto cherry-pick variant 2 (2025-10-07)
- 5561e6a Auto cherry-pick variant 1 (2025-10-07)

## Automation updates — 2025-10-07T21:11:11Z

### Last 4 commits
- 40007e8 refactor(trading): tune 4 files (2025-10-07)
- bddd656 refactor(sensory): tune 2 files (2025-10-07)
- eb32f08 docs(docs): tune 3 files (2025-10-07)
- 5622148 refactor(docs): tune 3 files (2025-10-07)
