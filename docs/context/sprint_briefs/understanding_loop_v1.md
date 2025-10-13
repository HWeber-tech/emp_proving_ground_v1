# Sprint brief – Understanding loop uplift (v1)

**Sprint window:** 14 days to turn the perception → belief → decision loop backlog into executable tickets that harden sensory routing, adaptive memory, and governance controls ahead of institutional data coming online.

## Concept anchors

- The roadmap "Next" outcomes call for a sensory cortex uplift, evolution engine foundation, and risk API enforcement, tying regime-aware routing and governance gates directly to near-term delivery goals.【F:docs/roadmap.md†L523-L537】
- The sensory cortex alignment brief highlights the need for executable organs, drift telemetry, and real data wiring so downstream decisions inherit trustworthy signals.【F:docs/context/alignment_briefs/sensory_cortex.md†L5-L74】
- The evolution engine alignment brief stresses lifecycle integration, lineage telemetry, and governance gates for adaptive intelligence, which the understanding loop must expose.【F:docs/context/alignment_briefs/evolution_engine.md†L5-L81】
- Risk & compliance alignment work requires deterministic policy enforcement, telemetry, and supervised runtime entrypoints, anchoring the policy ledger/gate deliverables in this sprint.【F:docs/context/alignment_briefs/institutional_risk_compliance.md†L5-L140】
- Operational readiness guidance reinforces supervised loops, drift monitoring, and observability hygiene, ensuring the drift sentry and diagnostic surfaces land with production guardrails.【F:docs/context/alignment_briefs/operational_readiness.md†L5-L88】

## Reality signals

- Sensory organs still ride synthetic inputs and only recently gained lineage-aware telemetry, so the belief/regime track must bridge mocks to executable routing with coverage.【F:docs/context/alignment_briefs/sensory_cortex.md†L13-L64】
- Evolution subsystems remain skeletal despite seeded telemetry scaffolding, underscoring the need for decision diaries and fast-weight adapters that surface lineage and governance metadata.【F:docs/context/alignment_briefs/evolution_engine.md†L13-L80】
- Risk enforcement has canonicalised configs and telemetry but still relies on manual coordination; policy-ledger work must extend these guardrails into the understanding loop artifacts.【F:docs/context/alignment_briefs/institutional_risk_compliance.md†L31-L140】
- Operational readiness tracks highlight ongoing supervision and alert gaps, meaning drift sentry and graph diagnostics have to plug into existing readiness dashboards instead of bespoke tooling.【F:docs/context/alignment_briefs/operational_readiness.md†L11-L88】

## Day-by-day orbit

| Days | Theme | Focus |
| --- | --- | --- |
| 1–2 | Belief & Regime foundations | Wire posterior belief buffers to live sensory payloads, align regime detectors, and document state diagrams. |
| 3–5 | Router + fast-weights | Implement task router, fast-weight adapters, and gating heuristics to channel beliefs into strategy intents. |
| 6–7 | Decision diary + probes | Capture decision justifications, lineage probes, and export governance-ready transcripts. |
| 8–9 | Drift sentry | Build Page–Hinkley/variance sentries that monitor belief drift and trigger supervised remediations. |
| 10–11 | Policy ledger & gate | Persist policy deltas, expose rebuild CLI, and gate adaptive updates behind ledger approvals. |
| 12–14 | Graph diagnostics & acceptance | Visualise understanding loop DAGs, add acceptance tests, and promote diagnostics into readiness dashboards. |

## Workstreams and ticket scaffolding

### Days 1–2 – Belief & Regime scaffolding (≈3 tickets)

- **Deliverables**
  - `BeliefState` data model covering priors, posterior snapshots, and decay logic, fed by existing sensory organ summaries and regime classifiers.【F:docs/context/alignment_briefs/sensory_cortex.md†L20-L74】
  - Regime adapter consolidating volatility/market-state detectors into a single `RegimeSignal` published on the event bus with provenance metadata.【F:docs/roadmap.md†L523-L529】
  - Belief state diagram + glossary drop in the context pack to unlock ticket creation for subsequent layers.【F:docs/context/alignment_briefs/sensory_cortex.md†L36-L74】
  - *Progress*: Belief buffer and emitter publish schema-versioned `BeliefState`
    snapshots with Hebbian PSD guardrails and regime FSM events, backed by
    guardrail pytest coverage and golden fixtures to prevent contract drift.【F:src/understanding/belief.py†L39-L347】【F:tests/understanding/test_belief_updates.py†L111-L239】【F:tests/understanding/golden/belief_snapshot.json†L1-L939】
  - *Progress*: Regime FSM now emits structured transition events with latency/volatility metadata, retains a bounded history, and surfaces health metrics so routing audits can trace regime flips under regression coverage.【F:src/understanding/belief.py†L695-L878】【F:tests/understanding/test_belief_updates.py†L327-L394】
  - *Progress*: Dynamic sensory payloads now backfill missing dimensions,
    extend the canonical feature order, and align prior covariance shapes so
    live ingest that adds or drops features continues emitting stable
    posteriors; the new regression covers expansion and contraction scenarios
    to lock in the behaviour.【F:src/understanding/belief.py†L310-L449】【F:tests/understanding/test_belief_updates.py†L264-L309】
  - *Progress*: Calibration utilities now extract Hebbian learning rates, decay
    floors, and calm/storm thresholds from historical EURUSD slices, then rebuild
    belief/regime components that survive calm-versus-erratic replay suites under
    guardrail coverage so tickets land with production-ready defaults.【F:src/understanding/belief_regime_calibrator.py†L1-L175】【F:tests/operations/test_belief_regime_calibrator.py†L34-L136】
  - *Progress*: Live belief manager now clamps regime threshold scaling with a
    hysteresis window, emits cached-ingest belief/regime telemetry, and carries
    guardrail coverage that replays volatility spikes to prove monotonic scaling
    and healthy regime classification for the sprint’s acceptance artifacts.【F:src/understanding/live_belief_manager.py†L18-L168】【F:tests/understanding/test_live_belief_manager.py†L210-L296】【F:tests/integration/test_operational_backbone_pipeline.py†L198-L295】
- **Validation artifacts**
  - Pytest suite around `hebbian_step` updater verifying low-rank covariance (`sigma`) remains PSD after updates across calm/normal/storm fixtures.
  - Golden JSON beliefs/regime snapshots used by doctest-like regression to guard schema drift.
  - Structured docstrings (Markdown tables) enumerating priors/posterior keys for documentation reuse.
- **Guardrails**
  - Add guardrail marker for `tests/understanding/test_belief_updates.py::test_hebbian_low_rank_sigma` so CI blocks rank inflation.
  - Event-bus contract test asserting `RegimeSignal` includes timestamp, confidence, and lineage fields before publishing.
  - Static schema check verifying belief buffers refuse symbols missing sensory lineage metadata.

### Days 3–5 – Router + fast-weights (≈4 tickets)

- **Deliverables**
  - `UnderstandingRouter` service that ingests belief snapshots and routes to downstream strategy slots with configurable weighting.
  - Fast-weight adapter (`fast_weights.py`) implementing Hebbian-inspired updates with decay, toggled by feature flag until governance sign-off.【F:docs/context/alignment_briefs/evolution_engine.md†L20-L81】
  - Router configuration schema appended to the sprint brief for ticket templating, including default policies for bootstrap vs. institutional tiers.【F:docs/context/alignment_briefs/institutional_risk_compliance.md†L70-L140】
- **Validation artifacts**
  - Batch simulation harness replaying recorded sensory snapshots to assert deterministic routing outcomes (baseline vs. fast-weight).
  - Benchmark notebook logging latency/variance impact of fast-weight updates for review.
  - Type-checked config examples in `docs/context/examples/understanding_router.md` for reference.
- **Guardrails**
  - Add CI guard ensuring fast-weight updates revert when `fast_weights_enabled` flag is false.
  - Register router contract tests under `pytest -m guardrail` that assert only approved strategy IDs receive high-confidence intents.
  - Static analysis hook verifying router configs declare policy references found in the policy ledger.
- **Progress**
  - PolicyRouter now enforces regime-aware topology switches with latency deadlines,
    tracks tactic objectives/tags, aggregates fast-weight metrics, and emits
    reviewer-ready reflection summaries/digests so emerging strategies surface without
    spelunking raw telemetry under expanded pytest coverage.【F:src/thinking/adaptation/policy_router.py†L201-L720】【F:tests/thinking/test_policy_router.py†L1-L210】
  - Operational backbone pipeline now boots attached `LiveBeliefManager` and
    `UnderstandingRouter` components during ingest rehearsals, returning
    belief/regime snapshots, understanding decisions, and ingest-failure
    telemetry so the router workstream can exercise end-to-end evidence under
    guardrail coverage even when Timescale falls back to cached frames.【F:src/data_foundation/pipelines/operational_backbone.py†L82-L366】【F:tests/integration/test_operational_backbone_pipeline.py†L198-L295】【F:tools/data_ingest/run_operational_backbone.py†L327-L378】
  - Recalibration workflow now reapplies belief-buffer hyperparameters and regime thresholds from real-market frames, clearing historical state, publishing telemetry, and validating PSD covariance alongside bus events under guardrail coverage.【F:src/understanding/live_belief_manager.py†L200-L257】【F:src/understanding/belief.py†L314-L363】【F:tests/understanding/test_live_belief_manager.py†L204-L260】【F:tests/understanding/test_belief_updates.py†L312-L354】
  - Reflection builder now surfaces emerging tactics and experiments with first/last
    seen timestamps, decision counts, share, and gating metadata, rendering reviewer
    insights that call out regime filters and minimum confidence so governance sees
    new strategies without replaying raw telemetry under refreshed pytest coverage.【F:src/thinking/adaptation/policy_reflection.py†L153-L213】【F:src/thinking/adaptation/policy_router.py†L977-L1033】【F:tests/thinking/test_policy_reflection_builder.py†L80-L113】【F:tests/thinking/test_policy_router.py†L243-L277】
  - Understanding router now supports Hebbian fast-weight adapters with
    deterministic decay, persists multiplier history, and serialises gate thresholds,
    feature values, required flags, and expiry metadata in guardrail coverage so
    governance reviews inherit auditable weight provenance even as experiments adapt
    over time.【F:src/understanding/router.py†L70-L240】【F:tests/understanding/test_understanding_router.py†L1-L185】
  - Router configuration schema ships as a typed loader with tier defaults and
    guardrailed examples so bootstrap vs. institutional deployments stay in sync with
    governance policy; context-pack examples document canonical YAML snippets for ticket
    templating.【F:src/understanding/router_config.py†L1-L320】【F:tests/understanding/test_understanding_router_config.py†L1-L88】【F:docs/context/examples/understanding_router.md†L1-L64】
  - Understanding metrics exporter now publishes throttle posture to Prometheus and wires the observability dashboard to emit those gauges whenever loop snapshots land, with replay fixtures and guardrail tests documenting the contract so fast-weight throttles stay observable.【F:src/operational/metrics.py†L43-L428】【F:src/understanding/metrics.py†L1-L65】【F:tests/understanding/test_understanding_metrics.py†L62-L125】【F:tests/operations/test_observability_dashboard.py†L609-L624】

### Days 6–7 – Decision diary & probes (≈3 tickets)

- **Deliverables**
  - Decision diary pipeline that captures pre/post routing state, policy contexts, and rationale text blobs ready for governance export.【F:docs/context/alignment_briefs/evolution_engine.md†L24-L81】【F:docs/context/alignment_briefs/institutional_risk_compliance.md†L100-L140】
  - Probe registry describing each telemetry probe (belief drift, router divergence, policy overrides) with owners and escalation paths.【F:docs/context/alignment_briefs/operational_readiness.md†L40-L88】
  - Markdown template for decision diary entries linked from the alignment briefs/context pack.
- **Validation artifacts**
  - Snapshot tests for diary Markdown/JSON exports ensuring ordering, timestamps, and policy IDs align.
  - CLI smoke test (`understanding_diary export --since ...`) verifying probes and diary entries share correlation IDs.
  - Event bus integration test confirming diary snapshots publish to the governance topic with fallbacks.
- **Guardrails**
  - Governance gate requiring diary entry presence before `fast_weights_enabled` toggles persist.
  - Lint rule or pre-commit check verifying diaries reference existing probe IDs.
  - Observability dashboard row automatically failing if diary export lag exceeds SLA (hooked via coverage matrix helper).【F:docs/context/alignment_briefs/quality_observability.md†L10-L188】
- **Progress**
  - Decision narration capsule builder/publisher now normalises ledger diffs, sigma stability, and throttle states, then publishes Markdown/JSON payloads via the shared failover helper so diary deliverables can plug directly into governance and observability dashboards.【F:src/operations/observability_diary.py†L3-L392】【F:tests/operations/test_observability_diary.py†L1-L190】
  - Policy router can ingest recorded reflection summaries and the decision diary CLI renders reflection digests with window, tactic, and experiment limits so reviewers rebuild adaptive insights from stored diaries without rerunning the loop, under regression coverage for ingestion and CLI flows.【F:src/thinking/adaptation/policy_router.py†L480-L640】【F:tests/thinking/test_policy_router.py†L200-L256】【F:tools/understanding/decision_diary_cli.py†L150-L260】【F:tests/tools/test_decision_diary_cli.py†L1-L200】
  - Observability dashboard now exposes a policy reflection panel summarising analysed decisions, leading tactics/experiments/tags/objectives, reviewer insights, and exported Markdown/JSON payloads so compliance reviewers see diary posture alongside readiness snapshots under guardrail coverage.【F:src/operations/observability_dashboard.py†L285-L390】【F:tests/operations/test_observability_dashboard.py†L627-L706】
  - WHAT sensor exports trend-strength telemetry into diary lineage so probes capture directional market context, with pytest fixtures asserting bullish vs. bearish discrimination and metadata parity across value, lineage, and default signals.【F:src/sensory/what/what_sensor.py†L83-L200】【F:tests/sensory/test_primary_dimension_sensors.py†L35-L174】
  - AlphaTrade loop now injects belief/probe attribution into trade intents, trade metadata, and diary annotations, and the trading manager records coverage metrics with warnings when attribution drops below the 90% goal so the roadmap's observability DoD has executable telemetry.【F:src/orchestration/alpha_trade_runner.py†L168-L234】【F:src/trading/trading_manager.py†L3587-L3637】【F:tests/trading/test_trading_manager_execution.py†L760-L821】

### Days 8–9 – Drift sentry (≈3 tickets)

- **Deliverables**
  - Drift sentry module combining Page–Hinkley and rolling variance detectors for belief/regime metrics with configurable severity thresholds.【F:docs/context/alignment_briefs/sensory_cortex.md†L66-L82】【F:docs/context/alignment_briefs/operational_readiness.md†L40-L88】
  - Alert policy mapping drift severities to operational readiness dashboards and incident response runbooks.【F:docs/context/alignment_briefs/operational_readiness.md†L40-L88】
  - Runbook updates describing drift remediation procedures and expected telemetry artifacts.
- **Validation artifacts**
  - Page–Hinkley trigger regression in `tests/operations/test_drift_sentry.py` verifying detection latency and false-positive bounds.
  - Synthetic drift scenario suite capturing incremental vs. abrupt change cases with golden alerts.
  - Grafana mock screenshot demonstrating drift metrics for context pack inclusion.
- **Guardrails**
  - CI guard verifying drift sentry publishes WARN within N ticks when synthetic drift fixtures exceed threshold.
  - Alert-router integration test ensuring incident response receives drift alerts through failover helper.【F:docs/context/alignment_briefs/operational_readiness.md†L40-L88】
  - Static config validator requiring drift thresholds reference documented policy IDs.
- **Progress**
  - Drift sentry detectors now evaluate belief/regime metrics, emit
    Page–Hinkley/variance telemetry, and feed operational readiness with
    alert-ready payloads plus runbook metadata so incident response inherits
    AlphaTrade drift posture under regression coverage.【F:src/operations/drift_sentry.py†L1-L279】【F:tests/operations/test_drift_sentry.py†L1-L103】【F:src/operations/operational_readiness.py†L209-L347】【F:docs/operations/runbooks/drift_sentry_response.md†L1-L69】
  - DriftSentry gate now consumes sensory drift snapshots, enforces confidence/notional guardrails before trade execution, and publishes structured gating telemetry—event-bus payloads plus runtime summaries—across bootstrap and Predator surfaces. Release routing now mirrors that telemetry by emitting forced-route events with audit metadata so dashboards inherit the full decision trail, under pytest coverage spanning the gate helper, telemetry module, and trading manager integration.【F:src/trading/gating/drift_sentry_gate.py†L1-L200】【F:src/trading/gating/telemetry.py†L1-L199】【F:src/trading/trading_manager.py†L418-L525】【F:src/trading/trading_manager.py†L829-L900】【F:src/runtime/bootstrap_runtime.py†L161-L177】【F:src/runtime/predator_app.py†L1012-L1024】【F:tests/trading/test_trading_manager_execution.py†L1079-L1098】【F:tests/trading/test_drift_sentry_gate.py†L61-L153】【F:tests/trading/test_drift_gate_telemetry.py†L10-L159】
  - Sensory drift regression now includes a deterministic Page–Hinkley replay fixture and metadata assertions so diagnostic runs reproduce the detector catalog, severity counts, and runbook link that readiness and incident response surfaces expect, under pytest coverage.【F:tests/operations/fixtures/page_hinkley_replay.json†L1-L128】【F:tests/operations/test_sensory_drift.py†L157-L218】

### Days 10–11 – Policy ledger & gate (≈3 tickets)

- **Deliverables**
  - Policy ledger store tracking regime-aware policy deltas, approvals, and linked diary entries.【F:docs/context/alignment_briefs/institutional_risk_compliance.md†L55-L140】
  - `rebuild_policy` CLI that replays ledger entries to regenerate enforceable policies for bootstrap/institutional tiers.【F:docs/roadmap.md†L533-L537】
  - Governance checklist linking ledger states to compliance readiness snapshots for alignment brief updates.【F:docs/context/alignment_briefs/institutional_risk_compliance.md†L115-L140】
- **Validation artifacts**
  - CLI integration test verifying `rebuild_policy` reproduces canonical `RiskConfig` + router gating settings.
  - Ledger migration test ensuring historical entries replay deterministically after schema changes.
  - Markdown changelog summarising ledger updates with runbook references.
- **Guardrails**
  - Feature flag preventing ledger promotions without matching decision diary evidence IDs.
  - Guardrail test asserting ledger rejects unsigned policy deltas (missing reviewer metadata).
  - CI step diffing ledger snapshots to block silent policy drift.
- **Progress**
  - Policy ledger store now records staged promotions with diary evidence, threshold overrides, and policy deltas, exposes release helpers, and powers a rebuild CLI that regenerates enforceable risk configs plus router guardrails while exporting governance workflows under pytest coverage so AlphaTrade promotions stay auditable end to end.【F:src/governance/policy_ledger.py†L1-L200】【F:src/governance/policy_rebuilder.py†L1-L141】【F:tools/governance/rebuild_policy.py†L1-L112】【F:tests/governance/test_policy_ledger.py†L33-L181】【F:tests/tools/test_rebuild_policy_cli.py†L11-L41】

### Days 12–14 – Graph diagnostics & acceptance (≈4 tickets)

- **Deliverables**
  - Understanding-loop graph diagnostic CLI rendering DAGs of sensory → belief → router → policy flows with lineage overlays.【F:docs/context/alignment_briefs/sensory_cortex.md†L25-L82】【F:docs/context/alignment_briefs/evolution_engine.md†L60-L81】
  - Acceptance test suite running end-to-end synthetic cycles validating diaries, ledger rebuilds, and drift sentry behaviour.【F:docs/roadmap.md†L523-L537】
  - Dashboard tile for operational readiness showing loop health, drift status, and policy gate posture.【F:docs/context/alignment_briefs/operational_readiness.md†L40-L88】
  - Context-pack appendix summarising diagnostic commands and export locations.【F:docs/context/alignment_briefs/quality_observability.md†L10-L188】
- **Validation artifacts**
  - Snapshot tests of generated DAG images/JSON verifying node metadata and lineage tags.
  - Acceptance workflow in CI (e.g., `pytest -m understanding_acceptance`) replaying recorded scenarios and asserting diary + ledger synchronisation.
  - Telemetry contract docs ensuring dashboard tile queries align with readiness schema.
- **Guardrails**
  - Guardrail marker for acceptance suite to fail fast on drift between telemetry schemas and diagnostics.
  - Static doc check verifying context-pack appendix stays in sync with CLI help output.
  - Observability dashboard guard ensuring missing diagnostic payloads escalate to WARN.【F:docs/context/alignment_briefs/quality_observability.md†L10-L188】
- **Progress**
  - Understanding diagnostics builder exports sensory→belief→router→policy graphs as structured snapshots, surfaced via the `tools/understanding/graph_diagnostics.py` CLI with JSON/DOT/Markdown renderers under dedicated pytest coverage and the `understanding_acceptance` marker.【F:src/understanding/diagnostics.py†L395-L542】【F:tools/understanding/graph_diagnostics.py†L1-L82】【F:tests/understanding/test_understanding_diagnostics.py†L15-L29】【F:pytest.ini†L2-L27】
  - Observability dashboard now renders an understanding-loop panel summarising regime confidence, drift exceedances, ledger approvals, and experiment mix whenever diagnostics snapshots are supplied, and escalates to WARN with CLI guidance when artifacts are missing so operators can rebuild acceptance payloads deterministically under guardrail coverage.【F:src/operations/observability_dashboard.py†L822-L875】【F:tests/operations/test_observability_dashboard.py†L582-L624】

## Definition of Done checkpoints

- Belief/regime models publish documented schemas with guardrail tests covering Hebbian updates, regime routing, and drift triggers.
- Bootstrap runtime publishes sensory summary/metrics/drift telemetry via the
  failover helper and exposes samples/audits/metrics through `status()` so
  live-shadow reviewers can observe the loop without bespoke scripts.【F:src/runtime/bootstrap_runtime.py†L214-L492】【F:tests/runtime/test_bootstrap_runtime_sensory.py†L120-L196】
- Router, diary, and ledger artifacts integrate with governance/operational readiness dashboards without bespoke tooling, updating context packs alongside code. The loop runner now persists structured trade outcomes (including throttle snapshots) and normalised performance health back into diary metadata so the same execution evidence powers dashboards, audits, and paper-sim reports under regression coverage, and the bootstrap stack mirrors the contract while annotating throttled runs with readable diary notes and persisted throttle payloads.【F:docs/context/alignment_briefs/quality_observability.md†L10-L188】【F:docs/context/alignment_briefs/operational_readiness.md†L40-L88】【F:src/orchestration/alpha_trade_runner.py†L184-L432】【F:src/orchestration/bootstrap_stack.py†L241-L446】【F:src/trading/trading_manager.py†L1413-L1501】【F:tests/orchestration/test_alpha_trade_runner.py†L167-L239】【F:tests/current/test_bootstrap_stack.py†L220-L273】
- Acceptance workflow exercises the full understanding loop and is marked as a guardrail job in CI, with reproducible fixtures for ticket derivation.【F:docs/roadmap.md†L523-L537】

## Instrumentation & documentation plan

- Extend coverage matrix CLI inputs with understanding-loop modules so lagging domains surface in telemetry reports.【F:docs/context/alignment_briefs/quality_observability.md†L10-L188】
- Publish new docs pages (belief glossary, router schema, drift sentry runbook) and reference them from the sensory/evolution alignment briefs to keep the refreshed context pack coherent.【F:docs/context/alignment_briefs/sensory_cortex.md†L36-L82】【F:docs/context/alignment_briefs/evolution_engine.md†L20-L81】
- Capture sprint-close metrics (belief drift, routing latency, ledger rebuild time) in CI health snapshot updates for cross-stream visibility.【F:docs/context/alignment_briefs/quality_observability.md†L10-L188】

## Dependencies & coordination

- Coordinate with ingest and sensory cortex delivery to ensure real data feeds land before fast-weight experiments graduate from feature-flagged mode.【F:docs/context/alignment_briefs/sensory_cortex.md†L66-L82】
- Align with evolution governance to reuse diary/ledger artifacts for adaptive-run approvals and lineage exports.【F:docs/context/alignment_briefs/evolution_engine.md†L63-L81】
- Sync with compliance/operational readiness teams so drift sentry alerts, policy ledger checkpoints, and dashboard tiles feed existing runbooks instead of creating shadow workflows.【F:docs/context/alignment_briefs/institutional_risk_compliance.md†L100-L140】【F:docs/context/alignment_briefs/operational_readiness.md†L40-L88】
