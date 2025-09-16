# Technical Debt Assessment

This snapshot documents the current state of the EMP Proving Ground codebase after the initial hygiene
pass that retired the Kilocode bridge and restored CI readability. It distills the highest-risk issues
by area so the team can sequence remediation tickets without re-auditing the entire repository.

## Snapshot summary

| Area | Status | Key findings | Next actions |
| --- | --- | --- | --- |
| Workflows & automation | ✅ Stabilized | Policy scanning now routes through a reusable workflow consumed by both `ci.yml` and the standalone policy gate. Logs can still burst GitHub limits if verbose flags are enabled. | Keep policy guardrails centralized and trim overly chatty steps when debugging to avoid log caps. |
| Dependencies & environment | ✅ Stabilized | Runtime dependencies now live in `requirements/base.txt`, and the development toolchain (mypy, Ruff, Black, pytest, coverage, pre-commit, and type stubs) is now fully pinned in `requirements/dev.txt` so CI and local environments agree. Minimum numpy/pandas/scipy versions are documented in the setup guide and enforced by the `python -m src.system.requirements_check` CLI. | Monitor whether a lock file is needed once formatting debt is resolved and revisit minimums as upstream releases land. |
| Configuration & policy alignment | ✅ Stabilized | Default config targets the FIX simulator and documentation now opens with FIX-only policy disclaimers that link to the integration policy. | Keep the policy doc authoritative and update legacy call-outs if the allowed surface changes. |
| Source hygiene | ✅ Stabilized | Repository artifacts (`*.orig`, `changed_files_*.txt`) have been pruned and ignored so they cannot leak back in. | Keep `.gitignore` patterns aligned with future tooling outputs and add pre-commit rules if new artifacts appear. |
| Testing & observability | ✅ Stabilized | End-to-end CI commands now pass (policy, lint, mypy, pytest) with 76% coverage. Regression nets cover `SystemConfig.with_updated`, the scientific stack guard, the FIX parity checker, the legacy `core.configuration` helper, legacy FIX execution flows (initialization guards, quantity/type validation, cancellation fallbacks, and realized PnL), `RiskManagerImpl` edge cases, and the orchestration compose adapters. CI publishes pytest tails + full logs for quicker triage, formatter enforcement is gated by the allowlist guard, automated failure alerts open a tracking issue, and the health snapshot documents key metrics. | Keep refreshing the health snapshot, grow telemetry beyond CI once formatter work stabilizes, and investigate persistent flakes for dashboard automation. |

## Workflows and reusable actions

* **CI (`.github/workflows/ci.yml`)** – Sequential jobs (policy → lint → types → tests → optional backtest) re-use the
  composite `python-setup` action. The policy stage now delegates to the reusable
  [`forbidden-integrations`](../.github/workflows/forbidden-integrations.yml) workflow so the guardrail lives in one
  place.
* **Policy – Block OpenAPI/cTrader** – Continues to run on pushes/PRs to `main` and `cleanup-phase-0`, but now simply
  calls the same reusable workflow as CI, eliminating drift between the two entry points.
* **Composite action (`.github/actions/python-setup`)** – Handles checkout, Python 3.11 installation, and dev dependency
  bootstrapping. Future optimization ideas include splitting lint/type deps from heavy scientific stacks if CI time or
  caching becomes a bottleneck.
* **Dead code audit (`.github/workflows/dead-code-audit.yml`)** – Schedules a weekly vulture run (and allows manual triggers)
  that writes the Markdown snapshot and uploads it as a short-lived artifact so every cleanup batch has an updated
  reference without committing generated reports to the repository.

## Dependency and environment landscape

* Runtime dependencies now live in `requirements/base.txt`, with `requirements/dev.txt` extending that list for typing,
  linting, and test tooling. The root-level `requirements.txt` simply re-exports the canonical manifest for
  compatibility with older scripts.
* Runtime version enforcement still happens in `main.py` (numpy/pandas/scipy guards) and is now mirrored in
  [`docs/development/setup.md`](development/setup.md). Contributors can run `python -m src.system.requirements_check`
  to confirm local environments match the documented floors before running services.
* Tooling versions (mypy, Ruff, Black, pytest/pytest-asyncio/pytest-cov, pre-commit, import-linter, and core type stubs)
  are pinned in `requirements/dev.txt` so automation, local shells, and the mypy Docker image install the same
  versions. Once formatting debt is paid down, consider pruning unused tools like Black if Ruff owns formatting
  completely.

## Configuration & policy notes

* `config.yaml` defaults to the FIX simulator broker with demo credentials placeholders, fully matching the
  OpenAPI/cTrader ban enforced in CI.
* Legacy documentation and archived configs now open with a **Status: Legacy** call-out that points to the
  [`Integration Policy`](policies/integration_policy.md), keeping new contributors on the FIX-only path.

## Source hygiene highlights

* All `.orig` merge remnants and `changed_files_*.txt` scratch outputs were removed from version control and are now
  ignored via `.gitignore`. This prevents inadvertent reintroduction during future conflict resolutions.
* The repo root remains clutter-free; continue scanning for new scratch artifacts when large refactors land and update
  guardrails promptly.
* The initial [dead-code audit](reports/dead_code_audit.md) flagged 16 high-confidence candidates across strategy templates,
  operational metrics, and the trading portfolio monitor. Convert them into targeted tickets and annotate legitimate
  dynamic hooks to keep future scans actionable.
* The trading performance tracker now delegates heavy analytics (Sharpe, Sortino, drawdown, trade summaries) to
  `src/trading/monitoring/performance_metrics.py`, reducing the monolithic class and giving other monitors a reusable helper
  surface.
* The former 350-line `src/core/interfaces.py` hub is now a package with focused modules (`base.py`, `ecosystem.py`,
  `metrics.py`, `analysis.py`) that re-export through `src/core/interfaces/__init__.py`, lowering fan-in on a single file while
  preserving import compatibility.

## Testing & observability gaps

* The Phase 0 CI baseline (2025-09-16) verified that policy, lint, mypy, and pytest jobs succeed with 76% coverage. The legacy
  repository still has 235 unformatted files, but the new allowlist guard confines formatter enforcement to directories that
  have been normalized. See [`docs/ci_baseline_report.md`](ci_baseline_report.md) for the full command log and coverage hotspots.
* Targeted regression tests now protect `SystemConfig.with_updated` conversions, the scientific stack guard, the FIX parity checker, the legacy `core.configuration` accessors, the deprecated FIX executor (including validation errors and realized PnL accounting), `RiskManagerImpl` sizing/limits, and the orchestration compose adapters so configuration drift is caught.
* CI appends pytest tails to the Step Summary, uploads the full log as an artifact, and now raises issue-based alerts whenever the workflow fails. The alerts auto-close after a successful rerun so the backlog only reflects active problems.
* [`docs/status/ci_health.md`](status/ci_health.md) surfaces the most recent coverage snapshot, formatter rollout progress, and pointers to triage resources.

## Recommended next actions

1. **Stage 0 formatter rollout** – Finish formatting `tests/current/`, expand the formatter allowlist, and ensure CI enforces the new slice without regressing test stability.
2. **Prepare Stage 1 formatting** – Dry-run formatting for `src/system/` and `src/core/configuration.py`, capture any manual cleanups, and plan the allowlist expansion once Stage 0 lands.
3. **Target coverage hotspots** – Spin up regression tickets for `src/operational/metrics.py`, `src/trading/models/position.py`, the `src/data_foundation/config/` modules, and `src/sensory/dimensions/why/yield_signal.py` so the weakest areas from the baseline report gain protection.
4. **Dead-code audit follow-up** – Convert the remaining findings in [Dead code audit – 2025-09-16](reports/dead_code_audit.md) into delete/refactor work, annotating intentional dynamic hooks to keep future scans clean.
5. **Flake telemetry** – Extend the pytest job to emit failure metadata artifacts and publish a lightweight dashboard to track recurring flakes, matching the observability plan.

Revisit this assessment after the backlog above lands (or at least quarterly) so improvements are captured and emerging risks can be reprioritized.
