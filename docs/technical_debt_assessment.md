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

- Formatter rollout is underway: Stages 0–2 normalized `tests/current/`,
  `src/system/`, `src/core/configuration.py`, `src/trading/execution/`, and
  `src/trading/models/`; Stage 3 covers the entire
  `src/sensory/organs/dimensions/` package (including `__init__.py`, `utils.py`,
  `what_organ.py`, `when_organ.py`, `why_organ.py`, and all previously formatted
  organs), and Stage 4 now enforces the entire `src/sensory/` tree alongside
  `src/data_foundation/config/` via the collapsed allowlist, with ingest/persist
  slices staged next.
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

### Phase 6 – Formatter normalization (High exposure)

- **Risk profile** – `ruff format --check .` still fails across hundreds of files,
  so the pipeline leans on `config/formatter/ruff_format_allowlist.txt` (now
  covering `tests/current/`, `src/system/`, `src/core/configuration.py`,
  `src/trading/execution/`, `src/trading/models/`, the full `src/sensory/`
  tree, and `src/data_foundation/config/` via a handful of directory entries) to
  prevent regressions.
- **Leading indicators** – Allowlist size, directories recorded in
  [`docs/development/formatter_rollout.md`](development/formatter_rollout.md), and
  formatter progress snapshots inside [`docs/status/ci_health.md`](status/ci_health.md).
- **Immediate actions**
  - Broadcast the new `src/sensory/` and `src/data_foundation/config/`
    enforcement and keep `scripts/check_formatter_allowlist.py` green after each
    allowlist update.
  - Prep the `src/data_foundation/ingest/` and `src/data_foundation/persist/`
    slices for formatting, capturing manual edits (if any) and scheduling reviews
    alongside regression coverage owners.
  - Land each Stage 4 package with a paired allowlist update, pytest run, and
    rollout documentation refresh so the guardrail covers newly formatted code
    without masking regressions.
- **Path to done** – Sequence remaining directories by churn, merge slices in quick
  succession, shrink the allowlist to empty, and update contributor guidance so
  Ruff becomes the single formatting tool.
- **Blockers** – Coordination with high-churn branches and generated assets that
  must remain excluded.

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
- Track formatter allowlist size, coverage deltas, flake counts, and dead-code
  backlog inside weekly debt triage notes.
- Revisit this assessment quarterly (or after completing any Phase 6–9 milestone)
  to record improvements, reprioritize emerging risks, and refresh onboarding
  context.
