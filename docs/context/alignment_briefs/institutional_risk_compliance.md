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
- Technical debt audits highlight hollow risk enforcement, unsupervised async
  entrypoints, and namespace drift (`get_risk_manager` export) that mislead API
  consumers.【F:docs/technical_debt_assessment.md†L33-L80】【F:src/core/__init__.py†L11-L51】
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
  - Progress: Risk policy regression enforces mandatory stop losses, positive
    equity budgets, and violation telemetry so CI fails fast when policy guardrails
    drift, keeping execution blockers visible to compliance reviewers.【F:src/trading/risk/risk_policy.py†L120-L246】【F:tests/trading/test_risk_policy.py†L117-L205】
  - Progress: Risk policy warn-threshold coverage now asserts leverage and
    exposure checks escalate to warnings before breaching limits, capturing
    ratios, thresholds, and projected exposure metadata so compliance teams can
    monitor approaching guardrails without waiting for outright violations.【F:tests/trading/test_risk_policy.py†L69-L142】
- Progress: Policy telemetry builders serialise decision snapshots, emit Markdown
  summaries, and publish violation alerts with embedded escalation metadata while
  the trading manager mirrors the feed and the new runbook documents the response,
  giving governance a deterministic alert surface when violations occur.【F:src/trading/risk/policy_telemetry.py†L1-L285】【F:src/trading/trading_manager.py†L642-L686】【F:docs/operations/runbooks/risk_policy_violation.md†L1-L51】【F:tests/trading/test_risk_policy_telemetry.py†L1-L199】
- Progress: Canonical `RiskConfig` enforces positive position sizing, cross-field
  exposure relationships, and research-mode overrides, emitting warnings when
  mandatory stop losses are disabled outside research and blocking inconsistent
  payloads under pytest coverage so compliance reviews inherit deterministic
  guardrails.【F:src/config/risk/risk_config.py†L1-L161】【F:tests/risk/test_risk_config_validation.py†L1-L36】
- Progress: Runtime builder now resolves the canonical `RiskConfig`, validates
  thresholds, wraps invalid payloads in runtime errors, and records enforced
  metadata under regression coverage so supervised launches cannot proceed with
  missing or malformed limits, aligning runtime posture with compliance
  expectations.【F:src/runtime/runtime_builder.py†L298-L337】【F:tests/runtime/test_runtime_builder.py†L158-L200】
- Progress: Compliance readiness snapshots now consolidate trade surveillance and
  KYC telemetry, escalate severities deterministically, and expose markdown
  evidence with pytest guardrails so governance cadences inherit truthful
  compliance posture summaries.【F:src/operations/compliance_readiness.py†L1-L220】【F:tests/operations/test_compliance_readiness.py†L1-L173】

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
