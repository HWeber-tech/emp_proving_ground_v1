# Technical Debt Assessment

This snapshot captures the state of the EMP Proving Ground codebase after the first
modernization waves removed the Kilocode bridge, stabilized CI, and decomposed the
highest-fanin modules. The baseline pipeline (policy → lint → types → pytest) now
passes with 76% coverage, giving the team dependable feedback while we tackle the
remaining hot spots.

Use this document when grooming backlog tickets, preparing review context, or
evaluating whether new work risks destabilizing the current guardrails. The heat map
below summarizes each area at a glance, followed by scorecards that align with the
Phase 6–9 roadmap streams.

## Executive summary

- Formatter rollout is complete: `ruff format --check .` now runs repo-wide in CI,
  the allowlist/script pair has been removed, and Ruff owns formatting/import
  ordering across every package.
- Test coverage is serviceable but brittle around trading execution, risk
  controls, and orchestration wiring; new position lifecycle, data foundation
  config loader, operational metrics sanitization, and FIX mock failure tests
  landed, and the remaining hotspots still need deterministic suites.
- CI observability captures logs, pytest tails, and a flake-telemetry JSON
  artifact; GitHub issue alerts open/close automatically, with the planned
  Slack/webhook mirror and forced-failure validation still pending.
- Weekly dead-code audits generate actionable findings, but the backlog needs
  triage so noise does not accumulate and future scans remain trustworthy.
- Foundational guardrails (pinned toolchain, policy enforcement, configuration
  alignment) are stable and should be protected as the remaining work lands.

## High-risk focus areas (0–3 month horizon)

1. **Runtime entrypoint refactor.** The legacy `main.py` blended configuration
   loading, ingestion orchestration, FIX lifecycle management, and long-lived
   loops without dependency boundaries. `src/runtime/runtime_builder.py` now
   encapsulates ingestion/trading workloads behind `RuntimeApplication`, but the
   CLI, FIX adapters, and shutdown runbooks still need to adopt the new
   abstraction end to end.
   - *Impact*: brittle deployments, no graceful shutdown, high blast radius for
     components that have yet to migrate to the builder.
   - *Immediate actions*: finish wiring CLIs and runbooks through the builder,
     retire ad-hoc task creation in the FIX stack, and document rollback/shutdown
     drills so operations can adopt the new topology.

2. **Async task lifecycle hazards.** Multiple modules call
   `asyncio.create_task` without supervision, and the FIX adapters spin new
   event loops inside callbacks to push messages across threads.
   - *Impact*: background tasks leak, cancellation fails to propagate, and the
     runtime can deadlock or drop work under load.
   - *Immediate actions*: inventory background task creation, migrate into
      structured groups (`asyncio.TaskGroup` or a supervised manager), and add
      regression tests that assert graceful shutdown. The new
      `TaskSupervisor` centralises runtime task tracking and cancellation for the
      Professional Predator app, and FIX adapters plus the event bus worker/fan-out
      loops now register under the supervisor so shutdown sweeps background tasks
      cleanly.【F:src/runtime/task_supervisor.py†L1-L152】【F:src/runtime/predator_app.py†L95-L303】【F:src/core/_event_bus_impl.py†L1-L420】【F:tests/runtime/test_task_supervisor.py†L1-L64】【F:tests/current/test_event_bus_task_supervision.py†L1-L78】

3. **Hollow risk management.** `RiskManager` dynamically imports configuration
   but only validates that `size` and `entry_price` are positive. Tiered limits,
   drawdown guards, leverage controls, and exposure caps already exist in
   `RiskConfig` but are ignored.
   - *Impact*: production runs without safety rails, exposing capital and
     compliance risk.
   - *Immediate actions*: design a deterministic risk API, implement enforcement
     for each `RiskConfig` field, document escalation/override flows, and cover
     the logic with unit/integration tests. TradingManager now instantiates a
     shared risk policy that applies `RiskConfig` thresholds to every trade,
     enriches gateway decisions with policy metadata, streams `telemetry.risk.policy`
     events with Markdown summaries, and ships with regression tests covering
     exposure caps, research-mode relaxations, and integration via the risk
     gateway and runtime surfaces.

4. **Namespace drift and failing validation.** Public exports such as
   `src/core/__init__.py` previously advertised helpers like `get_risk_manager`
   that did not exist while `system_validation_report.json` showed 0/10 checks
   passing.【F:src/core/__init__.py†L17-L56】 The canonical facade now lazily
   exposes only `RiskManager`, regression tests assert the deprecated symbol
   remains absent, and a follow-up audit documents the new evidence; the
   validation backlog still requires dedicated tickets.【F:tests/risk/test_risk_manager_impl_additional.py†L267-L277】【F:docs/reports/governance_risk_phase2_followup_audit.md†L1-L24】
   - *Impact*: historical drift eroded trust in the API surface and left the
     validation suite stalled at 0/10.
   - *Immediate actions*: finish aligning integration docs with the canonical
     facade, then triage each validation failure into an owned ticket with
     acceptance tests.

## Medium-risk & emerging debt (3–6 month horizon)

- **Testing scope divergence.** Pytest only runs `tests/current/`; the quarantined
  suites hide unknown regressions. Action: re-enable suites incrementally with
  feature flags, and capture coverage deltas in `docs/status/ci_health.md`.
- **Dependency surface bloat.** The runtime depends on full Dash/Plotly stacks and
  Pydantic v1 even when unused in production. Action: audit imports, split optional
  extras, and roadmap a Pydantic v2 migration.
- **Documentation void.** The primary `README.md` and operational docs lag the
  evolving runtime, complicating onboarding and governance. Action: align the
  README, architecture reality guide, and operator runbooks with the runtime
  hardening initiative.

## Long-horizon remediation roadmap

| Timeline | Outcomes | Focus areas |
| --- | --- | --- |
| **0–3 months (Stabilise)** | Runtime entrypoints, async lifecycles, and risk enforcement are trustworthy. | Runtime builder design/rollout, supervised task management, system validation triage, and risk policy implementation paired with updated operator docs. |
| **3–6 months (Harden & modularise)** | Runtime services become resilient with first-class telemetry and broader regression coverage. | Replace ad-hoc event loops, expand pytest scope (legacy/integration/load), instrument FIX/orchestrator bridges, and embed governance for safety config reviews. |
| **6–12 months (Evolve)** | The platform supports scale and regulatory scrutiny. | Configuration-as-code for risk, optional dependency rationalisation (Pydantic v2, UI stacks), continuous system validation dashboards/alerts, and SBOM/reporting integration. |

## Heat map (Q3 2025)

| Area | Risk | Signals | Mitigation path | Confidence |
| --- | --- | --- | --- | --- |
| Workflows & automation | 🟩 Low | CI and policy checks share a reusable workflow, pytest tails and logs upload for triage, and scheduled dead-code audits run weekly. | Keep policy guardrails centralized and trim verbose debugging steps if logs approach GitHub limits. | High |
| Dependencies & environment | 🟩 Low | Runtime requirements live in `requirements/base.txt`, dev tooling is pinned in `requirements/dev.txt`, and contributors can verify stacks via `python -m src.system.requirements_check`. | Monitor minimum versions quarterly and decide on a lock file after formatter work stabilizes. | Medium |
| Configuration & policy alignment | 🟩 Low | Default `config.yaml` enforces the FIX-only posture, documentation opens with integration disclaimers, and policy automation blocks unsupported brokers. | Reconfirm policy docs whenever integrations change and mirror updates in setup guides. | Medium |
| Source hygiene | 🟧 Medium | Scratch artifacts are ignored, but formatter rollout and dead-code backlog can reintroduce noise if left uncurated. | Pair formatting slices with guardrail updates and triage audit findings promptly. | Medium |
| Testing & observability | 🟥 High | Coverage sits at 76% with hotspots in trading, risk, data foundation, and sensory modules; flaky-test telemetry and alert delivery remain open. | Execute Phase 7 regression tickets, add flake metrics, and complete the alerting channel rollout. | Medium |
| Runtime realism & mocks | 🟧 Medium | Many subsystems still run against mock shims (e.g., FIX executor) and deprecated tiers, limiting production confidence. | Document limitations in regression tickets and plan real-integration milestones once Phase 6–9 stabilize. | Low |

## Stream scorecards

### Phase 6 – Formatter normalization (Complete)

- **Risk profile** – `ruff format --check .` passes repo-wide and now runs in CI;
  Ruff owns formatting/import ordering and the legacy allowlist/script helpers
  have been removed.
- **Leading indicators** – Formatter trend entries in
  [`tests/.telemetry/ci_metrics.json`](../tests/.telemetry/ci_metrics.json) now
  record `mode="global"`, and [`docs/development/formatter_rollout.md`](development/formatter_rollout.md)
  captures the historical rollout.
- **Immediate actions**
  - Keep CI telemetry fresh via `python -m tools.telemetry.update_ci_metrics --formatter-mode global`.
  - Ensure contributor docs (`docs/development/setup.md`) continue to emphasise
    running `ruff format` locally before commits.
- **Path to done** – Treat formatter enforcement as baseline hygiene; monitor CI
  lint runs and metrics for regressions.
- **Blockers** – None; continue monitoring as part of regular hygiene reviews.

### Phase 7 – Regression depth in trading & risk (High exposure)

- **Risk profile** – Coverage hotspots from the CI baseline (`src/operational/metrics.py`,
  `src/trading/models/position.py`, `src/data_foundation/config/`, and
  `src/sensory/dimensions/why/yield_signal.py`) now have scoped tickets recorded
  in [`docs/status/regression_backlog.md`](status/regression_backlog.md);
  regression suites landed for trading positions, data foundation loaders,
  operational metrics sanitization, and FIX mock failure paths while the sensory
  WHY suite remains queued.
- **Leading indicators** – Coverage deltas per module, number of regression tickets
  delivered, and stability of the mocked FIX execution path.
- **Immediate actions**
  - Maintain the regression backlog table, closing tickets as suites land and
    capturing ownership for the remaining sensory/orchestration work.
  - Extend the FIX execution coverage into orchestration smoke tests while
    integrations remain mocked, leaning on the failure-path tests as guardrails.
  - Keep risk guardrail suites green and expand into the queued sensory signal
    coverage so the backlog can burn down.
- **Path to done** – Land regression suites, capture coverage improvements in the
  CI health dashboard, and establish a steady cadence for new tests when modules change.
- **Blockers** – Lack of real integrations, potential flaky behavior in orchestration
  flows, and limited telemetry for retry diagnostics.

### Phase 8 – Operational telemetry & alerting (Medium exposure)

- **Risk profile** – CI uploads logs and summaries and opens/closes GitHub issue
  alerts automatically, but the Slack/webhook mirror plus a forced-failure drill
  and response metrics still need to be exercised.
- **Progress** – The Slack relay now ships alongside a forced-failure drill
  writer (`tools.telemetry.alert_drill`) and MTTA/MTTR recorder (`tools.telemetry.update_ci_metrics --alert-timeline`), with pytest coverage proving the timeline parsing and metrics wiring so drills produce reproducible evidence.【F:tools/telemetry/alert_drill.py†L1-L143】【F:tools/telemetry/update_ci_metrics.py†L1-L220】【F:tools/telemetry/ci_metrics.py†L1-L520】【F:tests/tools/test_alert_drill.py†L1-L60】【F:tests/tools/test_ci_metrics.py†L1-L600】
- **Leading indicators** – Selected alert channel, successful forced-failure test,
  and freshness of the CI health dashboard.
- **Immediate actions**
  - Document the current GitHub-issue alert workflow and owners in
    [`docs/operations/observability_plan.md`](operations/observability_plan.md)
    while the Slack/webhook mirror is being built.
  - Validate the channel with an intentional failure before declaring victory and
    record MTTA/MTTR baselines.
  - Surface the flake telemetry JSON location alongside formatter/coverage
    metrics in the dashboard so responders know where to look.
- **Path to done** – Keep alert noise low (auto-resolve on green), add pytest flake
  metadata, and review MTTA/MTTR as the signal matures.
- **Blockers** – Team agreement on preferred channel and bandwidth to maintain the
  dashboard.

### Phase 9 – Dead code remediation & modular cleanup (Medium exposure)

- **Risk profile** – Weekly vulture audits surface candidates across strategy
  templates and monitoring helpers; without follow-up they will become noise and
  hide real regressions.
- **Leading indicators** – Audit backlog size, number of deletions per sprint, and
  fan-in metrics for `src/core` modules.
- **Immediate actions**
  - Triage each audit finding as delete, refactor, or intentional keeper and record
    rationale in [`docs/reports/dead_code_audit.md`](reports/dead_code_audit.md).
  - Delete confirmed dead paths and adjust docs/tests accordingly.
  - Plan decomposition work for remaining high-fanin modules.
- **Path to done** – Schedule follow-up audits after each cleanup batch, keep the
  report curated, and capture structural improvements in architecture docs.
- **Blockers** – Hidden dynamic imports, limited time for refactors while formatter
  slices land, and potential coupling to mocked subsystems.

## Guardrails to preserve

- **Workflow reuse** – `.github/workflows/ci.yml` delegates policy enforcement to a
  shared workflow, reducing duplication across entry points.
- **Pinned toolchain** – Ruff, mypy, pytest, coverage, and supporting stubs resolve
  identically across local shells, Docker, and CI.
- **Configuration clarity** – The FIX-only posture is consistent across code,
  configuration, and documentation. Keep policy disclaimers prominent during future
  product explorations.
- **Documentation fidelity** – Architecture guides, setup instructions, and status
  reports now reflect reality. Update them in the same PRs that adjust behavior to
  prevent drift.

## Reporting & next review

- Update [`docs/status/ci_health.md`](status/ci_health.md) after each formatter
  slice, regression batch, or alerting milestone.
- Track formatter telemetry trendlines, coverage deltas, flake counts, and dead-code
  backlog inside weekly debt triage notes.
- Revisit this assessment quarterly (or after completing any Phase 6–9 milestone)
  to record improvements, reprioritize emerging risks, and refresh onboarding
  context.
