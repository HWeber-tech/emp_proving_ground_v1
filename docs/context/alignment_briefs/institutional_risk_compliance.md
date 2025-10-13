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
  canonical export reflects the real implementation.【F:docs/technical_debt_assessment.md†L33-L80】【F:src/core/__init__.py†L14-L33】【F:docs/reports/CLEANUP_REPORT.md†L71-L104】
- Canonical risk configuration now resides in `src/config/risk/risk_config.py`,
  yet dead-code sweeps still list additional risk and compliance files as unused,
  complicating canonicalisation.【F:src/config/risk/risk_config.py†L1-L72】【F:docs/reports/CLEANUP_REPORT.md†L71-L175】
- Risk manager now returns `0.0` when the risk budget is exhausted or lot sizes
  fall below configured minimums, preventing orphaned orders in depleted paper
  accounts while regression coverage exercises zero-budget flows.【F:src/risk/risk_manager_impl.py†L220-L268】【F:tests/risk/test_risk_manager_impl_additional.py†L27-L70】
- Safety manager now coerces environment-provided confirmation strings via a strict normaliser, raising on unrecognised payloads so live boots cannot bypass the confirmation gate, with regression coverage and the refreshed Phase II completion audit documenting the guardrail and follow-up recommendations.【F:src/governance/safety_manager.py†L1-L120】【F:tests/governance/test_safety_manager.py†L1-L76】【F:docs/reports/risk_governance_phase2_completion_audit.md†L1-L31】
- Policy & code audit Phase II fail-closed refresh now captures the hardened portfolio risk path, updated regression coverage, and governance follow-ups so reviewers inherit the latest remediation evidence in one packet.【docs/reports/policy_code_audit_phase_ii_fail_closed_refresh.md:1】【src/risk/risk_manager_impl.py:845】【tests/risk/test_risk_manager_impl_additional.py:531】【tests/current/test_risk_manager_impl.py:95】
- Policy & code audit Phase II follow-up documents the governance approval de-duplication and risk override coercion fixes with pointers to the ledger and risk implementations plus the guarding regression tests so compliance reviewers inherit the remediation evidence in one packet.【F:docs/audits/policy_code_audit_phase2_followup.md†L1-L19】【F:src/governance/policy_ledger.py†L123-L174】【F:src/risk/risk_manager_impl.py†L50-L980】【F:tests/governance/test_policy_ledger.py†L202-L223】【F:tests/current/test_risk_manager_impl.py†L141-L166】
- Policy & code audit Phase II completion now enumerates audit-log hardening and the locked risk book remediation, pairing fresh guardrail suites so reviewers inherit both the finding narrative and executable proof that corrupt entries are skipped and asynchronous calls cannot race the portfolio snapshot.【F:docs/reports/policy_code_audit_phase2.md†L1-L19】【F:src/governance/audit_logger.py†L203-L338】【F:tests/governance/test_audit_logger.py†L15-L72】【F:src/risk/risk_manager_impl.py†L738-L929】【F:tests/risk/test_risk_manager_impl_additional.py†L648-L694】
- Market-regime detection now fails closed when detectors raise, zeroing multipliers, surfacing blocked telemetry, documenting the guardrail in the Phase II audit refresh, and covering the recovery path under regression tests so governance sees deterministic posture for flaky feeds.【docs/audits/policy_code_audit_phase2.md:1】【src/risk/risk_manager_impl.py:639】【tests/risk/test_risk_manager_impl_additional.py:449】

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
    positions when balances are missing, normalises string payloads, skips
    malformed positions, and enforces mandatory stop losses so CI flags policy
    guardrail drift before it reaches execution.【F:src/trading/risk/risk_policy.py†L29-L238】【F:tests/trading/test_risk_policy.py†L178-L511】
  - Progress: Risk policy warn-threshold coverage asserts leverage and exposure
    checks escalate to warnings before breaching limits, capturing ratios,
    thresholds, and projected exposure metadata so compliance teams can monitor
    approaching guardrails without waiting for outright violations.【F:tests/trading/test_risk_policy.py†L125-L170】
  - Progress: Risk manager now canonicalises position symbols to uppercase keys,
    migrates legacy entries, and aggregates exposure using the normalised
    identifiers so audits and telemetry surfaces share a single exposure view,
    with regression coverage proving mixed-case updates and migrations.【F:src/risk/risk_manager_impl.py†L246-L308】【F:tests/risk/test_risk_manager_impl_additional.py†L200-L250】
- Progress: Guardrail-marked risk policy suite now covers approvals,
  research-mode overrides, minimum size enforcement, closing trades, derived
  equity, and market price fallbacks so institutional limit enforcement remains
  pinned to the `guardrail` CI job.【F:tests/trading/test_risk_policy.py†L1-L511】
- Progress: Core, integration, and trading orchestrators now import `RiskManager` through the consolidated `src.risk` facade, retiring legacy `.manager` entry points so runtime assemblies consume the guarded API under regression coverage.【F:src/core/__init__.py†L1-L52】【F:src/integration/component_integrator.py†L1-L120】【F:src/orchestration/compose.py†L1-L276】【F:tests/current/test_risk_manager_impl.py†L1-L60】
- Progress: Facade guardrails now exercise misconfiguration coercion, sector budget enforcement, and legacy symbol removal via dedicated regression tests and a documented Phase II follow-up audit capturing full trace evidence for roadmap coverage commitments.【F:tests/risk/test_risk_manager_facade.py†L1-L87】【F:tests/risk/test_risk_manager_impl_additional.py†L267-L277】【F:docs/reports/governance_risk_phase2_followup_audit.md†L1-L24】
- Progress: Portfolio risk aggregation now measures absolute position size when computing exposure so short trades consume risk budget instead of collapsing to zero, with regression coverage proving negative positions still report non-zero `risk_amount` and the refreshed Phase II audit logging the fix for governance reviewers.【F:src/risk/risk_manager_impl.py†L228-L236】【F:tests/current/test_risk_manager_impl.py†L234-L242】【F:docs/audits/policy_code_audit_phase2.md†L8-L18】
- Progress: Policy telemetry builders serialise decision snapshots, emit Markdown
  summaries, and publish violation alerts with embedded escalation metadata while
  the trading manager mirrors the feed and the new runbook documents the response,
  giving governance a deterministic alert surface when violations occur.【F:src/trading/risk/policy_telemetry.py†L1-L285】【F:src/trading/trading_manager.py†L1700-L1744】【F:docs/operations/runbooks/risk_policy_violation.md†L1-L51】【F:tests/trading/test_risk_policy_telemetry.py†L1-L422】
- Progress: Risk gateway decisions now cache risk API summaries, embed
  runbook-backed `risk_reference` payloads on approvals and rejections, and
  publish the same enriched metadata through broker events so responders inherit
  a single audited context across telemetry surfaces under regression
  coverage.【F:src/trading/risk/risk_gateway.py†L232-L430】【F:tests/current/test_risk_gateway_validation.py†L1-L213】【F:tests/trading/test_fix_broker_interface_events.py†L15-L152】
- Progress: A Phase II completion refresh audit now captures the new `stop_loss_pips` regressions that prove percentage conversion and floor clamping, recording the closed coverage gap and follow-up telemetry recommendation for ingest teams.【F:docs/reports/governance_risk_phase2_code_audit_refresh.md†L1-L34】【F:tests/current/test_risk_gateway_validation.py†L210-L276】
- Completion: Risk gateway now enforces confidence-notional, leverage, and sector exposure caps derived from `RiskConfig`, rejecting violations with reason-coded telemetry and guardrail coverage for each path.【F:src/trading/risk/risk_gateway.py†L336-L513】【F:tests/current/test_risk_gateway_validation.py†L452-L576】
- Progress: Professional runtime summaries now pin the shared risk API runbook,
  attach runtime metadata, merge resolved interface details, and surface
  structured `RiskApiError` payloads so operators inherit actionable posture
  even when integrations degrade under pytest coverage.【F:src/runtime/predator_app.py†L995-L1063】【F:tests/current/test_runtime_professional_app.py†L304-L364】
- Progress: FIX broker interface captures deterministic risk metadata ahead of
  approvals or rejections, persists the runbook-backed context on order records,
  and emits the same payload on rejection telemetry under regression coverage so
  incident responders observe identical evidence whether they inspect FIX state
  or event bus emissions.【F:src/trading/integration/fix_broker_interface.py†L95-L170】【F:src/trading/integration/fix_broker_interface.py†L278-L366】【F:src/trading/integration/fix_broker_interface.py†L620-L710】【F:tests/trading/test_fix_broker_interface_events.py†L332-L386】
- Progress: Liquidity prober now routes probe tasks through the shared supervisor
  and records deterministic risk metadata or runbook-tagged errors on every run,
  with regression coverage asserting the supervised probes and risk context so
  execution telemetry feeds inherit auditable guardrails.【F:src/trading/execution/liquidity_prober.py†L1-L340】【F:tests/trading/test_execution_liquidity_prober.py†L90-L138】
- Progress: Execution adapters now rely on a shared risk-context helper so paper
  fills, release routing, and trading manager snapshots ingest the canonical
  `build_runtime_risk_metadata` output—and surface runbook-tagged errors—under
  regression coverage that checks provider propagation and describe surfaces.【F:src/trading/execution/_risk_context.py†L1-L120】【F:src/trading/execution/paper_execution.py†L1-L108】【F:src/trading/execution/release_router.py†L1-L154】【F:src/trading/trading_manager.py†L1255-L1409】【F:tests/trading/test_execution_risk_context.py†L38-L165】
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
- Progress: Builder enforcement now refuses to launch when the trading manager
  is missing, recording a `risk_error` payload with the shared runbook and
  raising a deterministic runtime exception so miswired deployments cannot
  bypass mandatory risk controls, with pytest guarding the contract.【F:src/runtime/runtime_builder.py†L690-L739】【F:tests/runtime/test_runtime_builder.py†L268-L294】
- Progress: Deterministic trading risk API continues to centralise config/status
  resolution and surface the shared `RISK_API_RUNBOOK`, while the trading manager
  now merges gateway limits, runtime summaries, and cached decisions into
  `risk_reference` payloads exposed by `get_risk_status()` so supervisors and docs
  consume a single hardened contract under pytest coverage.【F:src/trading/risk/risk_api.py†L1-L158】【F:src/runtime/runtime_builder.py†L323-L353】【F:src/trading/trading_manager.py†L1255-L1343】【F:tests/trading/test_risk_api.py†L90-L152】【F:tests/trading/test_trading_manager_execution.py†L1501-L1566】【F:tests/runtime/test_runtime_builder.py†L200-L234】
- Progress: Trading risk interface telemetry helpers now publish structured
  snapshots and contract-violation alerts with Markdown summaries, update the
  trading manager’s cached posture via `describe_risk_interface()`, and ensure
  bootstrap control center, runtime status, and FIX pilot snapshots merge
  deterministic risk metadata and gateway/interface references via the canonical
  helper so governance receives the shared runbook, risk_config summary, and any
  cached error payload under pytest coverage.【F:src/trading/risk/risk_interface_telemetry.py†L1-L156】【F:src/trading/trading_manager.py†L1370-L1409】【F:src/operations/bootstrap_control_center.py†L232-L511】【F:src/trading/risk/risk_api.py†L201-L238】【F:src/runtime/bootstrap_runtime.py†L210-L334】【F:src/runtime/fix_pilot.py†L22-L318】【F:tests/trading/test_trading_manager_execution.py†L1309-L1541】【F:tests/current/test_bootstrap_control_center.py†L151-L198】【F:tests/runtime/test_fix_pilot.py†L115-L178】
- Progress: Phase II closeout audit now pairs `risk_interface_telemetry` with a
  dedicated regression suite to lock Markdown formatting, snapshot fields, and
  event-bus payloads so governance dashboards receive deterministic evidence for
  risk interface posture.【F:docs/reports/risk_governance_phase2_code_audit_closeout.md†L1-L32】【F:tests/trading/test_risk_interface_telemetry.py†L1-L124】
- Progress: Release-aware execution router now auto-installs across bootstrap
  runtime, control centre, and Predator summaries via the trading manager,
  exposing default stages, engine routing, and forced overrides so compliance
  reviewers inherit audited execution posture under pytest coverage.【F:src/runtime/bootstrap_runtime.py†L214-L492】【F:src/operations/bootstrap_control_center.py†L341-L359】【F:src/runtime/predator_app.py†L1001-L1141】【F:src/trading/trading_manager.py†L140-L267】【F:src/trading/trading_manager.py†L1158-L1230】【F:tests/current/test_bootstrap_runtime_integration.py†L238-L268】【F:tests/trading/test_trading_manager_execution.py†L775-L914】
- Progress: Trading manager release posture now includes the last routed
  execution decision from the release-aware router, capturing forced paper
  routes, drift severity, and audit metadata so dashboards and audits inherit
  the exact enforcement evidence under regression coverage.【F:src/trading/trading_manager.py†L1068-L1134】【F:src/trading/execution/release_router.py†L98-L154】【F:tests/trading/test_execution_risk_context.py†L119-L165】【F:tests/trading/test_trading_manager_execution.py†L874-L999】
- Progress: Execution throughput snapshots now embed `throttle_active`, state, reasons, retry timers, remaining capacity, and scope metadata so compliance dashboards and evidence packs can show exactly why trades were blocked without cross-referencing throttle internals; throttled versus disabled flows are locked under regression coverage.【F:src/trading/trading_manager.py†L521-L839】【F:src/trading/trading_manager.py†L1433-L1450】【F:src/trading/trading_manager.py†L1897-L1940】【F:tests/trading/test_trading_manager_execution.py†L2092-L2160】
- Progress: Release-aware execution router now inspects policy-ledger audit
  posture, merges DriftSentry overrides with audit enforcement, and records the
  forced-reason history plus audit metadata so compliance snapshots explain why
  routes were downgraded to paper under new guardrail coverage.【F:src/trading/execution/release_router.py†L39-L214】【F:src/trading/execution/release_router.py†L260-L332】【F:src/trading/trading_manager.py†L420-L620】【F:tests/trading/test_release_execution_router.py†L1-L240】【F:tests/trading/test_trading_manager_execution.py†L951-L1038】
- Progress: Stage-gate enforcement now forces paper routes for experiment and paper ledger stages, merges drift/audit reasons into routing metadata, and proves via orchestration and gate tests that strategies promoted only to paper cannot execute live even when drift is nominal.【F:src/trading/execution/release_router.py†L80-L141】【F:tests/trading/test_release_execution_router.py†L27-L234】【F:tests/orchestration/test_alpha_trade_loop.py†L24-L162】【F:tests/trading/test_drift_sentry_gate.py†L190-L212】
- Progress: AlphaTrade loop compliance events now embed the release stage and enforcing stage-gate rationale whenever paper routes are imposed, giving auditors stage-aware telemetry without scraping diaries and under regression coverage for the new governance action payloads.【F:src/orchestration/alpha_trade_loop.py†L340-L418】【F:tests/orchestration/test_alpha_trade_loop.py†L324-L392】
- Progress: Drift gate telemetry now publishes structured event-bus payloads and
  Markdown summaries covering severity, forced-paper posture, and routing
  metadata whenever gating decisions fire, and release routing mirrors those
  events with forced-route and audit context so compliance dashboards inherit a
  complete enforcement trail under pytest coverage for the trading manager and
  telemetry helpers.【F:src/trading/gating/telemetry.py†L1-L199】【F:src/trading/trading_manager.py†L360-L612】【F:src/trading/trading_manager.py†L863-L935】【F:tests/trading/test_drift_gate_telemetry.py†L10-L159】【F:tests/trading/test_trading_manager_execution.py†L775-L914】

- Progress: Mock FIX manager coercion helpers now reject non-ASCII payloads,
  guard order-book adapters that raise exceptions, and keep deterministic
  fallbacks under pytest coverage so offline pilots and regression fuzzers
  cannot trigger decoding faults or leak stack traces through operational mock
  flows.【F:src/operational/mock_fix.py†L303-L360】【F:tests/operational/test_mock_fix_security.py†L1-L56】
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
  escalation logs, merges section summaries/statuses into metadata, and derives
  top-level compliance/regulatory/audit badges plus status breakdown counts so
  runtime outages still deliver governance telemetry with the same context the
  cadence delta surfaces, under pytest scenarios covering fallback behaviour.【F:src/operations/governance_reporting.py†L336-L586】【F:src/operations/data/governance_context/compliance_baseline.json†L1-L24】【F:tests/operations/test_governance_reporting.py†L110-L196】
- Progress: Timescale compliance and KYC journals now emit recent-activity counts
  with window metadata, and the governance report flags stale journals while
  recording collection timestamps and strategy scope so reviewers see timely
  evidence under regression coverage.【F:src/data_foundation/persist/timescale.py†L1232-L1322】【F:src/data_foundation/persist/timescale.py†L1617-L1702】【F:src/operations/governance_reporting.py†L336-L620】【F:tests/data_foundation/test_timescale_compliance_journal.py†L103-L117】【F:tests/data_foundation/test_timescale_compliance_journal.py†L199-L210】【F:tests/operations/test_governance_reporting.py†L129-L330】
- Progress: Policy ledger store now enforces staged promotions with diary
  evidence, approval metadata, and threshold overrides, and the rebuild CLI
  regenerates enforceable risk configs plus router guardrails while exporting
  governance workflow snapshots under pytest coverage so compliance reviews can
  trace AlphaTrade tactics from ledger to runtime enforcement without manual
  reconciliation.【F:src/governance/policy_ledger.py†L1-L200】【F:src/governance/policy_rebuilder.py†L1-L141】【F:tools/governance/rebuild_policy.py†L1-L112】【F:tests/governance/test_policy_ledger.py†L33-L181】【F:tests/tools/test_rebuild_policy_cli.py†L11-L41】
- Progress: Ledger persistence now acquires filesystem locks, swaps atomic temp files, and surfaces JSONL/Markdown promotion artifacts through the CLI helpers with concurrency regression coverage so concurrent operators cannot clobber governance state.【F:src/governance/policy_ledger.py†L260-L374】【F:tools/governance/_promotion_helpers.py†L13-L108】【F:tools/governance/promote_policy.py†L122-L335】【F:tests/governance/test_policy_ledger_locking.py†L11-L53】【F:tests/tools/test_promote_policy_cli.py†L1-L180】
  - Progress: Promotion workflows now normalise whitespace around evidence IDs, reject blank submissions, and persist trimmed identifiers across history with guardrail tests covering both governance manager and store surfaces.【F:src/governance/policy_ledger.py†L97-L218】【F:tests/governance/test_policy_ledger.py†L147-L225】【F:tests/governance/test_policy_ledger_locking.py†L51-L65】
  - Progress: Policy ledger record parsing now rejects payloads missing policy or tactic identifiers, normalises approval lists, and records the guardrail in the Phase II audit so malformed governance submissions cannot bypass promotion checks.【F:src/governance/policy_ledger.py†L212-L229】【F:tests/governance/test_policy_ledger.py†L198-L233】【F:docs/audits/policy_code_audit_phase2.md†L8-L25】
- Progress: AlphaTrade graduation CLI now offers an `--apply` mode that promotes
  ledger stages when recommendations clear blockers, annotates JSON/text
  summaries with applied stages, and persists the release via the ledger manager
  so governance cadences can graduate tactics and capture evidence in one run
  under refreshed regression coverage.【F:tools/governance/alpha_trade_graduation.py†L1-L252】【F:tests/governance/test_policy_graduation.py†L253-L335】

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
  - Progress: Runtime builder now publishes the enforced risk configuration as
    telemetry and the professional runtime records the broadcast payload so risk
    summaries mirror the exact configuration emitted to operations dashboards
    under pytest coverage of the event flow and summary surface.【F:src/runtime/runtime_builder.py†L633-L734】【F:src/runtime/predator_app.py†L472-L1009】【F:tests/runtime/test_runtime_builder.py†L340-L420】
  - Progress: Trading manager now merges gateway limit snapshots, resolved risk
    metadata, and runtime summaries into deterministic `risk_reference`
    payloads while surfacing shared runbooks so telemetry, dashboards, and
    audits inherit the same risk configuration under pytest coverage.【F:src/trading/trading_manager.py†L1255-L1744】【F:src/trading/risk/risk_gateway.py†L396-L429】【F:tests/trading/test_trading_manager_execution.py†L1309-L1566】【F:tests/current/test_risk_gateway_validation.py†L391-L460】
- Progress: Release-aware execution router now reconfigures pilot and live
  engines on demand, persists last-route governance metadata, and works with
  the trading manager’s cached pilot/live engines so limited-live promotions
  dispatch orders to the live engine under regression coverage of the stage
  transitions.【F:src/trading/execution/release_router.py†L166-L217】【F:src/trading/trading_manager.py†L141-L287】【F:tests/trading/test_release_execution_router.py†L168-L217】【F:tests/trading/test_trading_manager_execution.py†L1438-L1492】
  - Progress: Trading manager now releases reserved paper positions when broker execution fails, logging diagnostic fallbacks so compliance reviews never encounter phantom exposure from unreleased reservations under guardrail coverage.【F:src/trading/trading_manager.py†L623-L689】
- Progress: Trading manager now installs the PaperBrokerExecutionAdapter as the limited-live engine with stage defaults, risk-context snapshots, and broker failure capture so decision diaries persist the last error payload when simulations falter. Bootstrap can attach a REST `PaperTradingApiAdapter` when extras are supplied, exercising real HTTP round-trips, cleanup callbacks, and failure retries under pytest coverage so paper pilots inherit deterministic routing without bespoke wiring, and a dedicated simulation runner/CLI now drives the bootstrap runtime end-to-end to capture broker orders, errors, and diary counts for roadmap evidence packs; recent hardening timestamps first/last orders, records the most recent error, threads the risk snapshot into bootstrap summaries, and captures residual broker failures even when history buffers are empty while the adapters snapshot the latest broker submission request/response payloads for diaries and HTTP error evidence, with regression coverage asserting the cloned telemetry retains ISO timestamps, status codes, and risk context without mutation.【src/trading/execution/paper_broker_adapter.py:259】【src/trading/execution/paper_broker_adapter.py:457】【src/trading/integration/paper_trading_api.py:321】【src/trading/integration/paper_trading_api.py:503】【src/runtime/bootstrap_runtime.py:461】【src/runtime/paper_simulation.py:193】【tests/runtime/test_bootstrap_paper_broker.py:64】【tests/runtime/test_paper_trading_simulation_runner.py:139】【tests/trading/test_paper_broker_adapter.py:296】【tests/trading/test_paper_trading_api_adapter.py:69】【tests/integration/test_paper_trading_simulation.py:423】 Failover thresholds and cooldowns now propagate from runtime extras and the simulation CLI, blocking additional orders during cooldown windows while metrics surface the failover snapshot and guardrail coverage exercises the block/unblock cycle end to end.【F:src/trading/execution/paper_broker_adapter.py†L57-L379】【F:src/runtime/predator_app.py†L2571-L2649】【F:tools/trading/run_paper_trading_simulation.py†L123-L251】【F:tests/trading/test_paper_broker_adapter.py†L345-L379】
  - Progress: Paper trading REST adapter now retries transient HTTP failures with configurable attempt/backoff extras, exposing the resolved settings in runtime summaries so governance can audit resilience posture, and unit coverage exercises both successful retry promotion and exhaustion paths.【F:src/trading/integration/paper_trading_api.py†L28-L287】【F:tools/trading/run_paper_trading_simulation.py†L21-L162】【F:tests/trading/test_paper_trading_api_adapter.py†L1-L163】
  - Progress: Limited-live diary entries now include serialised `paper_metrics` snapshots sourced from the execution engine, and integration coverage verifies paper simulations persist order counts alongside success/failure tallies for compliance evidence packs.【F:src/orchestration/bootstrap_stack.py†L428-L441】【F:tests/integration/test_paper_trading_simulation.py†L99-L120】
  - Progress: Paper simulation loop now honours an external stop event and the CLI wraps SIGINT/SIGTERM with temporary handlers so drills can shut down gracefully without leaving background tasks, while integration guardrails replay persistent 500 responses to assert PaperTradingApiError telemetry, 0% success ratios, and diary error snapshots land in the evidence pack.【src/runtime/paper_simulation.py:83】【tools/trading/run_paper_trading_simulation.py:25】【tests/runtime/test_paper_trading_simulation_runner.py:177】【tests/integration/test_paper_trading_simulation.py:337】
  - Progress: Paper dry-run evaluations now score simulation reports against latency, failure, throttle, and incident-response budgets, returning structured verdicts plus captured metrics so governance boards can gate promotions on deterministic pass/fail outcomes; pytest coverage exercises budget overrides, missing telemetry, and incident severities.【F:src/operations/paper_trading_dry_run.py†L1-L240】【F:tests/operations/test_paper_trading_dry_run.py†L1-L78】【F:src/runtime/paper_simulation.py†L1-L219】
  - Progress: Execution pipeline now enforces configurable trade throttles via the trading manager, layering scope-aware windows, minimum spacing guards, and countdown metadata so bursty strategies respect governance cooldowns; snapshot payloads now expose `max_trades`, remaining trade credits, bounded utilisation, retry/reset timers, and the resolved scope/context dictionaries (plus `scope_key`) documented in operator guides, with regression coverage spanning throttle helpers, trading-manager snapshots, and docs. Fractional window durations now carry through reason strings and human-readable countdowns (`2.5 seconds` instead of truncating), keeping compliance dashboards precise when sub-second guardrails are configured.【F:src/trading/execution/trade_throttle.py†L123-L344】【F:src/trading/trading_manager.py†L1991-L2031】【F:docs/performance/trading_throughput_monitoring.md†L160-L176】【F:tests/trading/test_trade_throttle.py†L40-L148】【F:tests/trading/test_trading_manager_execution.py†L524-L550】
  - Progress: Throttle snapshots now include bounded `window_utilisation` along with UTC `window_reset_at` and `window_reset_in_seconds` timers so governance dashboards know exactly when a scope regains capacity; regression coverage spans rate-limit, cooldown, and spacing scenarios while the throughput guide documents the telemetry shape.【F:src/trading/execution/trade_throttle.py†L204-L312】【F:tests/trading/test_trade_throttle.py†L15-L205】【F:docs/performance/trading_throughput_monitoring.md†L165-L173】
  - Progress: Trade throttle multipliers now let compliance scale bursty trade quantities instead of blocking them outright, recording `throttle_scaled` experiment events, multiplier counters, and snapshot metadata while docs and guardrail suites cover the intent-scaling path end to end.【F:src/trading/execution/trade_throttle.py†L236-L250】【F:src/trading/trading_manager.py†L538-L999】【F:docs/performance/trading_throughput_monitoring.md†L165-L175】【F:tests/trading/test_trade_throttle.py†L245-L263】【F:tests/trading/test_trading_manager_execution.py†L2127-L2549】
  - Progress: Trade throttle rollback hooks now release rate-limit capacity when execution fails by deleting the recorded timestamp, trimming idle scopes, and updating trading manager stats so compliance dashboards bounce back to `open` immediately after broker errors, with regression coverage spanning throttle + manager guardrails.【src/trading/execution/trade_throttle.py:289】【src/trading/trading_manager.py:1240】【tests/trading/test_trade_throttle.py:313】【tests/trading/test_trading_manager_execution.py:2149】
  - Progress: Professional predator runtime now resolves trade throttle settings from `SystemConfig.extras`, normalises JSON blobs, file hints, and individual env toggles, and injects the result into `BootstrapRuntime` so compliance teams can enforce rate limits without code changes, with regression coverage confirming extras produce the expected scope-aware snapshots.【F:src/runtime/predator_app.py†L1723-L2470】【F:src/runtime/bootstrap_runtime.py†L123-L201】【F:tests/runtime/test_trade_throttle_configuration.py†L1-L55】
  - Progress: Throughput monitoring now returns structured backlog observations, increments aggregate breach counters, emits `backlog_breach` experiment events, and documents the expanded metadata—adding `latest_lag_ms`, `p95_lag_ms`, `breach_rate`, and `max_breach_streak`—so ops can trace lag incidents and trend drift from a single payload under refreshed regression coverage.【F:src/trading/execution/performance_monitor.py†L1-L143】【F:src/trading/execution/backlog_tracker.py†L31-L170】【F:docs/performance/trading_throughput_monitoring.md†L86-L133】【F:tests/trading/execution/test_backlog_tracker.py†L26-L103】【F:tests/trading/execution/test_performance_report.py†L39-L131】
  - Progress: Legacy FIX executor now installs the shared risk-context provider,
    captures metadata/errors each time it reconciles orders, and exposes
    `describe_risk_context()` so FIX pilots surface the same runbook-tagged risk
    evidence as the primary execution engine under pytest coverage.【F:src/trading/execution/fix_executor.py†L65-L152】【F:src/trading/execution/_risk_context.py†L1-L88】【F:tests/current/test_fix_executor.py†L221-L263】
- Progress: Governance cadence runner now persists the last generated timestamp,
  injects strategy and metadata providers, backfills cadence defaults, loads the
  previous payload, and attaches a structured delta (status shifts, section
  additions/removals, summary changes) so the compliance squad can enforce interval
  gating with clear change logs under pytest coverage.【F:src/operations/governance_cadence.py†L130-L245】【F:src/operations/governance_reporting.py†L523-L770】【F:tests/operations/test_governance_cadence.py†L126-L299】【F:tests/operations/test_governance_reporting.py†L386-L441】
- Progress: Governance cadence CLI resolves SystemConfig extras into context
  packs, layers JSON overrides, supports forced runs, renders the delta metadata,
  and emits Markdown/JSON outputs so operators can execute the cadence outside
  the runtime while preserving persisted history and change provenance under
  pytest coverage.【F:tools/governance/run_cadence.py†L1-L368】【F:tests/tools/test_run_governance_cadence.py†L47-L140】
- Progress: Vision alignment report now ingests evolution readiness snapshots,
  adaptive-run telemetry, and champion metadata while the bootstrap control centre
  threads the same readiness payload into runtime status so governance reviewers see
  the gate posture without scraping orchestrator internals. Layer 3 is now labelled
  “Understanding Loop” with a compatibility alias for the legacy intelligence-name,
  keeping governance dashboards aligned with the roadmap vocabulary under expanded
  integration coverage.【F:src/governance/vision_alignment.py†L77-L287】【F:src/operations/bootstrap_control_center.py†L431-L528】【F:tests/current/test_vision_alignment_report.py†L19-L73】【F:tests/runtime/test_bootstrap_runtime_sensory.py†L162-L216】
- Progress: Professional runtime builder now honours `GOVERNANCE_CADENCE_*`
  extras, scheduling the cadence loop, persisting history, and recording the
  latest governance report during live runs so background reporting matches the
  CLI cadence under regression coverage.【F:src/runtime/runtime_builder.py†L2447-L2630】【F:tests/runtime/test_runtime_builder.py†L1356-L1438】
- [x] Wire compliance workflows (KYC, trade surveillance) with markdown exports
  and optional Timescale journaling to satisfy audit requirements.
  - Progress: Compliance workflow evaluation now converts trade,
    KYC, and strategy-registry telemetry into MiFID, Dodd-Frank, audit, and
    governance checklists with Markdown exports and failover-hardened publishes
    under regression coverage.【F:src/compliance/workflow.py†L1-L419】【F:tests/compliance/test_compliance_workflow.py†L1-L182】
  - Progress: Policy ledger release manager records promotions, approvals, and
    adaptive thresholds while the trading manager/runtime builder publish the
    staged governance workflow so release gating and compliance readiness share
    the same evidence trail under pytest coverage.【F:src/governance/policy_ledger.py†L1-L405】【F:src/trading/trading_manager.py†L1094-L1157】【F:src/runtime/runtime_builder.py†L2920-L2987】【F:tests/trading/test_trading_manager_execution.py†L874-L949】
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
