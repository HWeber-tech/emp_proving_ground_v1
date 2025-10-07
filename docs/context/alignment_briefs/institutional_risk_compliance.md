# Alignment brief – Institutional risk, execution & compliance telemetry

## Concept promise

- The encyclopedia mandates advanced risk management with regulatory compliance
  frameworks (MiFID II, Dodd-Frank) and portfolio controls layered onto the
  execution stack.【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L293-L436】【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L4520-L4528】
- Risk mitigation chapters emphasise real-time monitoring, VaR/ES analytics, and
  alerting pipelines that must accompany institutional trading.【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L110-L3161】

## Reality snapshot

- Development status confirms that execution, risk, and strategy subsystems are
  mock frameworks without real broker connectivity or portfolio management.【F:docs/DEVELOPMENT_STATUS.md†L19-L35】
- Technical debt audits highlight hollow risk enforcement and unsupervised async
  entrypoints; the legacy `get_risk_manager` shim has now been removed so the
  canonical export reflects the real implementation.【F:docs/technical_debt_assessment.md†L33-L80】【F:src/core/__init__.py†L11-L36】【F:docs/reports/CLEANUP_REPORT.md†L71-L104】
- Canonical risk configuration now resides in `src/config/risk/risk_config.py`,
  yet dead-code sweeps still list additional risk and compliance files as unused,
  complicating canonicalisation.【F:src/config/risk/risk_config.py†L1-L72】【F:docs/reports/CLEANUP_REPORT.md†L71-L175】

## Gap themes

1. **Deterministic risk enforcement** – Implement policy checks for every
   `RiskConfig` field, with telemetry for violations and governance overrides.
2. **Runtime safety** – Finish the builder migration, introduce task supervision,
   and document shutdown/restart drills.
3. **Compliance telemetry** – Provide KYC/AML workflows, trade surveillance, and
   audit journaling aligned with encyclopedia claims.

## Delivery plan

### Now (0–30 days)

- Inventory risk and compliance modules, delete unused shims, and mark remaining
  placeholders as deprecated to maintain truth-first status reporting.【F:docs/reports/CLEANUP_REPORT.md†L71-L175】
- Harden SQL and exception handling in execution/risk modules per remediation
  plan Phase 0.【F:docs/development/remediation_plan.md†L34-L72】
- Extend CI to include baseline risk regression tests covering exposure, leverage,
  and drawdown limits; capture findings in `docs/status`.
  - Progress: Risk policy regression now derives equity from cash and open
    positions when balances are missing, records consistent risk/exposure ratio
    metadata, and enforces mandatory stop losses so CI flags policy guardrail
    drift before it reaches execution.【F:src/trading/risk/risk_policy.py†L29-L238】【F:tests/trading/test_risk_policy.py†L178-L465】
  - Progress: Risk policy warn-threshold coverage asserts leverage and exposure
    checks escalate to warnings before breaching limits, capturing ratios,
    thresholds, and projected exposure metadata so compliance teams can monitor
    approaching guardrails without waiting for outright violations.【F:tests/trading/test_risk_policy.py†L125-L170】
  - Progress: Guardrail-marked risk policy suite now covers approvals,
    research-mode overrides, minimum size enforcement, closing trades, derived
    equity, and market price fallbacks so institutional limit enforcement remains
    pinned to the `guardrail` CI job.【F:tests/trading/test_risk_policy.py†L1-L465】
- Progress: Policy telemetry builders serialise decision snapshots, emit Markdown
  summaries, and publish violation alerts with embedded escalation metadata while
  the trading manager mirrors the feed and the new runbook documents the response,
  giving governance a deterministic alert surface when violations occur.【F:src/trading/risk/policy_telemetry.py†L1-L285】【F:src/trading/trading_manager.py†L920-L991】【F:docs/operations/runbooks/risk_policy_violation.md†L1-L51】【F:tests/trading/test_risk_policy_telemetry.py†L1-L199】
- Progress: Risk gateway decisions now attach cached `risk_reference` payloads
  with the risk API runbook, enforced limits, and policy/config summaries, and
  expose the same metadata through broker events so responders inherit a single
  audit context across telemetry surfaces under regression coverage.【F:src/trading/risk/risk_gateway.py†L224-L519】【F:tests/current/test_risk_gateway_validation.py†L93-L407】【F:tests/trading/test_fix_broker_interface_events.py†L15-L152】
- Progress: Parity checker telemetry now wraps gauge publication in defensive
  logging, recording order and position mismatches even when the metrics sink
  misbehaves so compliance dashboards surface reconciliation issues instead of
  silently masking telemetry drops.【F:src/trading/monitoring/parity_checker.py†L53-L156】
- Progress: Canonical `RiskConfig` now normalises instrument/sector inputs,
  rejects duplicate sector limits, enforces sector budgets against the global
  exposure cap, and retains the research-mode and position-sizing guards so
  compliance reviews inherit deterministic, de-duplicated limits under pytest
  coverage.【F:src/config/risk/risk_config.py†L10-L205】【F:tests/risk/test_risk_config_validation.py†L39-L70】
- Progress: Runtime builder now resolves the canonical `RiskConfig`, validates
  thresholds, wraps invalid payloads in runtime errors, and records enforced
  metadata under regression coverage so supervised launches cannot proceed with
  missing or malformed limits, aligning runtime posture with compliance
  expectations. Launches now fail fast when mandatory stop-loss enforcement is
  disabled outside research mode, surfacing the shared risk API runbook alias so
  supervisors inherit a consistent escalation path.【F:src/runtime/runtime_builder.py†L323-L353】【F:tests/runtime/test_runtime_builder.py†L200-L234】
- Progress: Deterministic trading risk API centralises config/status resolution,
  exposes runtime-ready metadata snapshots, advertises the shared
  `RISK_API_RUNBOOK` alias, and drives the runtime builder’s enforcement path so
  supervisors and docs consume a single hardened contract under pytest coverage,
  exporting volatility targets, leverage windows, sector-instrument counts, and
  latest policy metadata with runbook links so downstream telemetry and runtime
  errors point to the same remediation guide.【F:src/trading/risk/risk_api.py†L1-L158】【F:src/runtime/runtime_builder.py†L323-L353】【F:tests/trading/test_risk_api.py†L90-L152】【F:tests/runtime/test_runtime_builder.py†L200-L234】
- Progress: Trading risk interface telemetry helpers now publish structured
  snapshots and contract-violation alerts with Markdown summaries, updating the
  trading manager’s cached posture, while bootstrap control center, bootstrap
  runtime status, and FIX pilot snapshots resolve the interface payload to embed
  the shared runbook in operator telemetry under pytest coverage so governance
  receives actionable enforcement evidence when the interface degrades.【F:src/trading/risk/risk_interface_telemetry.py†L1-L156】【F:src/trading/trading_manager.py†L741-L759】【F:src/operations/bootstrap_control_center.py†L99-L350】【F:src/runtime/bootstrap_runtime.py†L210-L334】【F:src/runtime/fix_pilot.py†L22-L318】【F:tests/trading/test_trading_manager_execution.py†L651-L667】【F:tests/current/test_bootstrap_control_center.py†L151-L180】【F:tests/runtime/test_fix_pilot.py†L115-L178】
- Progress: FIX broker interface risk rejections now merge gateway policy
  snapshots, provider summaries, and deterministic risk API fallbacks while
  always attaching the shared runbook so manual pilots inherit actionable
  escalation metadata even when provider lookups fail, under pytest coverage.【F:src/trading/integration/fix_broker_interface.py†L211-L330】【F:tests/trading/test_fix_broker_interface_events.py†L170-L239】
- Progress: FIX integration pilot now exposes supervised runtime metadata,
  publishes a `run_forever` workload wrapper, and wires a runtime builder helper
  so pilots surface the risk runbook, task-supervisor posture, and trading
  manager risk summary while guaranteeing graceful shutdown, under pytest
  coverage of the runtime harness and public exports.【F:src/runtime/fix_pilot.py†L112-L165】【F:src/runtime/fix_pilot.py†L225-L236】【F:src/runtime/fix_pilot.py†L496-L517】【F:src/runtime/__init__.py†L15-L107】【F:tests/runtime/test_fix_pilot.py†L166-L260】
- Progress: Compliance readiness snapshots now consolidate trade surveillance,
  KYC telemetry, and workflow checklist status, escalating blocked items,
  surfacing active task counts, and exposing markdown evidence with pytest
  guardrails so governance cadences inherit truthful compliance posture
  summaries.【F:src/operations/compliance_readiness.py†L262-L420】【F:tests/operations/test_compliance_readiness.py†L58-L213】
- Progress: Governance reporting cadence now uses the shared failover helper to
  publish compiled KYC/AML, regulatory, and audit evidence bundles with typed
  escalation logs so runtime outages still deliver governance telemetry, with
  pytest scenarios covering fallback behaviour.【F:src/operations/governance_reporting.py†L437-L519】【F:tests/operations/test_governance_reporting.py†L1-L200】
- Progress: Policy ledger store now enforces staged promotions with diary
  evidence, approval metadata, and threshold overrides, and the rebuild CLI
  regenerates enforceable risk configs plus router guardrails while exporting
  governance workflow snapshots under pytest coverage so compliance reviews can
  trace AlphaTrade tactics from ledger to runtime enforcement without manual
  reconciliation.【F:src/governance/policy_ledger.py†L1-L200】【F:src/governance/policy_rebuilder.py†L1-L141】【F:tools/governance/rebuild_policy.py†L1-L112】【F:tests/governance/test_policy_ledger.py†L33-L181】【F:tests/tools/test_rebuild_policy_cli.py†L11-L41】

### Next (30–90 days)

- Implement deterministic risk API enforcement with telemetry topics (policy,
  exposure, breaches) and runtime summaries.
  - Progress: Trading manager now sources its portfolio risk manager through the
    canonical deterministic facade and exposes the core engine’s snapshot and
    assessment hooks so execution flows share the same enforcement path as the
    runtime builder.【F:src/trading/trading_manager.py†L105-L147】【F:src/risk/risk_manager_impl.py†L533-L573】
  - Progress: FIX broker interface now validates manual intents via the real risk
    gateway, publishes structured rejection telemetry, and records approved
    decisions on order state so FIX pilots inherit the same guardrails and
    evidence trail under pytest coverage as orchestrated runtimes.【F:src/trading/integration/fix_broker_interface.py†L38-L524】【F:tests/trading/test_fix_broker_interface_events.py†L13-L202】
- Progress: Governance cadence runner orchestrates interval gating, audit
  evidence collection, report persistence, and event-bus publishing so the
  compliance squad has a single supervised entrypoint for regulatory reporting
  under pytest coverage.【F:src/operations/governance_cadence.py†L1-L167】【F:tests/operations/test_governance_cadence.py†L1-L206】
- Wire compliance workflows (KYC, trade surveillance) with markdown exports and
  optional Timescale journaling to satisfy audit requirements.
- Complete runtime builder adoption so FIX pilots, simulators, and eventual live
  bridges share the same supervised entrypoint.【F:docs/technical_debt_assessment.md†L33-L56】

### Later (90+ days)

- Integrate drop-copy reconciliation, regulatory reporting feeds, and operator
  runbooks for incident response.
- Establish governance reviews and escalation procedures for policy overrides.
- Remove legacy risk/compliance templates once canonical implementations are
  proven in regression suites.【F:docs/reports/CLEANUP_REPORT.md†L71-L175】

## Dependencies & coordination

- Data backbone and sensory upgrades must land to supply accurate telemetry for
  risk decisions and compliance reporting.
- Operational readiness (alert routing, incident response) should evolve in sync
  so violations raise actionable signals.【F:docs/technical_debt_assessment.md†L90-L112】
