# Alignment brief – Institutional risk, execution & compliance telemetry

**Why this brief exists:** The concept blueprint promises institutional-grade execution with strict risk, compliance, and
regulatory workflows, yet our context packs previously focused on data backbone and evolution slices. This brief ties the
risk/compliance stream to the shipped telemetry surfaces (risk policy, execution readiness, KYC/trade monitors, governance
checklists) so future tickets inherit the same story the roadmap and encyclopedia articulate.【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L332-L338】【F:docs/roadmap.md†L29-L37】

## Concept promise

- Layer 4 of the concept architecture highlights order management, real-time risk monitoring, and regulatory reporting as core
  institutional capabilities the execution stack must deliver.【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L332-L338】
- Enterprise outcomes explicitly call for MiFID II / Dodd-Frank compliance, audit readiness, and professional support services,
  setting expectations for telemetry and governance evidence.【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L639-L648】
- The roadmap gap table captures the need for FIX/EMS connectivity, expanded risk controls, and executable compliance workflows
  with audit persistence, anchoring this stream’s deliverables.【F:docs/roadmap.md†L29-L37】

## Reality snapshot (September 2025)

- `RiskPolicy` and `RiskManagerImpl` enforce min/max sizing, leverage, drawdown throttles, and mandatory stop-loss rules while
  exposing policy metadata for telemetry consumers.【F:src/trading/risk/risk_policy.py†L62-L200】【F:src/risk/risk_manager_impl.py†L1-L200】
- `evaluate_execution_readiness` grades fill rates, rejection ratios, latency, connection health, and drop-copy posture into
  Markdown-friendly snapshots, with runtime builder wiring publishing and persistence hooks.【F:src/operations/execution.py†L400-L520】【F:src/runtime/runtime_builder.py†L2320-L2520】
- `TradeComplianceMonitor` and `KycAmlMonitor` emit policy-backed telemetry, journal snapshots to Timescale, and maintain history
  used by the professional runtime summary and readiness evaluators.【F:src/compliance/trade_compliance.py†L200-L320】【F:src/compliance/kyc.py†L165-L276】【F:src/data_foundation/persist/timescale.py†L900-L1265】
- Aggregators `evaluate_compliance_readiness` and `evaluate_compliance_workflows` fuse trade/KYC status with governance tasks,
  publish runtime events, and persist Markdown for operators with regression coverage in place.【F:src/operations/compliance_readiness.py†L230-L320】【F:src/compliance/workflow.py†L800-L893】【F:tests/operations/test_compliance_readiness.py†L1-L90】【F:tests/compliance/test_compliance_workflow.py†L1-L150】
- The professional runtime records risk, execution, and compliance snapshots, exposing them via `summary()` and storing
  Timescale journals for audits under dedicated tests.【F:src/runtime/predator_app.py†L520-L1084】【F:tests/runtime/test_professional_app_timescale.py†L200-L460】

## Gap map

| Concept excerpt | Observable gap | Impact |
| --- | --- | --- |
| Layer 4 mandates regulatory reporting and audit trails.【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L332-L338】 | Compliance journals live in Timescale but there is no automated export to evidence packs or the portfolio snapshot. | Stakeholders must query the database manually to validate compliance claims, slowing reviews. |
| Roadmap calls for FIX/EMS pilot with expanded risk gates and live reconciliation.【F:docs/roadmap.md†L29-L37】 | FIX pilot telemetry exists, yet risk/compliance briefs do not summarise its posture or remaining gaps. | Teams lack a context pack summarising FIX governance expectations, risking duplicated discovery. |
| Enterprise outcomes require regulatory readiness and professional support evidence.【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L639-L648】 | CI health snapshot references risk policy and compliance telemetry but omits the latest audit/journal counts. | Operational reviewers cannot easily confirm the audit trail coverage promised to clients. |

## Delivery plan

### Now (30-day outlook)

1. **Surface audit evidence** – Extend the CI health snapshot and roadmap portfolio table with compliance journal counts and
   execution readiness status so documentation mirrors the runtime evidence.【F:docs/status/ci_health.md†L150-L240】【F:tools/roadmap/snapshot.py†L78-L150】
2. **Context-pack the FIX pilot** – Publish a companion brief summarising `FixIntegrationPilot`, drop-copy reconciliation, and
   execution telemetry hooks so future tickets inherit the compliance context.【F:src/runtime/fix_pilot.py†L1-L210】【F:src/runtime/fix_dropcopy.py†L1-L220】
3. **Runtime summary enhancements** – Add a composite “risk & compliance posture” block to `ProfessionalPredatorApp.summary()`
   that links risk policy decisions, execution readiness, and compliance readiness into one operator view under pytest coverage.【F:src/runtime/predator_app.py†L520-L1084】【F:tests/runtime/test_professional_app_timescale.py†L200-L460】

### Next (60-day outlook)

1. **Evidence export tooling** – `tools.telemetry.export_risk_compliance_snapshots` now bundles risk policy, execution readiness,
   compliance readiness/workflow blocks, and Timescale journal statistics into a JSON payload for governance packs, backed by
   the new journal aggregation helpers and CLI regression coverage.【F:tools/telemetry/export_risk_compliance_snapshots.py†L1-L308】【F:src/data_foundation/persist/timescale.py†L1211-L1703】【F:tests/tools/test_risk_compliance_export.py†L1-L113】【F:tests/data_foundation/test_timescale_compliance_journal.py†L57-L206】【F:tests/data_foundation/test_timescale_execution_journal.py†L86-L123】
2. **Policy versioning** – Store risk/compliance policy hashes alongside telemetry and expose diffs in runtime metadata so
   auditors can track configuration drift between runs.【F:src/trading/risk/policy_telemetry.py†L1-L208】【F:src/runtime/runtime_builder.py†L2190-L2320】
3. **FIX pilot coverage** – Expand FIX telemetry tests to include failure-path reconciliation and compliance workflow hooks,
   ensuring the pilot remains under regression as new brokers land.【F:tests/runtime/test_fix_pilot.py†L1-L180】【F:tests/operations/test_fix_pilot_ops.py†L1-L90】

### Later (90-day+ considerations)

- Integrate ROI and strategy-performance telemetry with compliance readiness to evidence how governance gates impact
  profitability claims.【F:src/operations/roi.py†L1-L164】【F:src/operations/strategy_performance.py†L1-L540】
- Automate regulatory report generation (MiFID transaction files, KYC case exports) using the Timescale journals and publish the
  runbooks alongside telemetry.【F:src/data_foundation/persist/timescale.py†L900-L1265】【F:docs/operations/runbooks/README.md†L1-L19】

## Validation hooks

- **Risk policy regression** – Keep `tests/trading/test_risk_policy.py` and `tests/risk/test_risk_telemetry.py` green so policy
  evaluations and telemetry remain deterministic.【F:tests/trading/test_risk_policy.py†L1-L120】【F:tests/risk/test_risk_telemetry.py†L1-L100】
- **Execution readiness tests** – Exercise fill-rate, latency, and drop-copy scenarios via `tests/operations/test_execution.py`
  and the runtime builder integration test that records execution snapshots.【F:tests/operations/test_execution.py†L1-L110】【F:tests/runtime/test_runtime_builder.py†L240-L360】
- **Compliance journaling** – Ensure Timescale round-trip tests for trade/kyc journals and professional summary assertions stay
  in CI so audit persistence keeps pace with schema updates.【F:tests/data_foundation/test_timescale_compliance_journal.py†L1-L80】【F:tests/runtime/test_professional_app_timescale.py†L200-L320】

## Open questions

1. Which regulatory artefacts (MiFID transaction logs, KYC case exports) should be autogenerated versus authored manually for
   institutional due diligence?【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L639-L648】
2. How do we sequence FIX pilot onboarding with compliance sign-off so telemetry and documentation remain synchronised?【F:docs/roadmap.md†L29-L37】
3. What audit dashboards or evidence logs should ship alongside the runtime to simplify regulator or partner reviews?【F:docs/status/ci_health.md†L150-L240】
