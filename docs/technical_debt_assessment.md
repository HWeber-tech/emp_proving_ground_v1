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
| Testing & observability | ⚠️ Needs attention | End-to-end CI commands now pass (policy, lint, mypy, pytest) with 76% coverage. Regression nets now cover `SystemConfig.with_updated`, the scientific stack guard, the FIX parity checker, and the legacy `core.configuration` helper. CI publishes pytest tails + full logs for quicker triage, but the formatter gate still fails on 235 files. | Plan an incremental formatting rollout, expand coverage into trading/risk hot spots called out in the CI baseline report, and implement the alerting options captured in `docs/operations/observability_plan.md`. |

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

## Testing & observability gaps

* The Phase 0 CI baseline (2025-09-16) verified that policy, lint, mypy, and pytest jobs succeed with 76% coverage, while
  `ruff format --check .` still fails on 235 files. See [`docs/ci_baseline_report.md`](ci_baseline_report.md) for the full command log and coverage hotspots.
* Targeted regression tests now protect `SystemConfig.with_updated` conversions, the scientific stack guard, the FIX parity checker, and the legacy `core.configuration` accessors so configuration drift is caught.
* CI appends pytest tails to the Step Summary and uploads the full log as an artifact, restoring lightweight observability without the Kilocode relay.

## Recommended execution order

1. **Phase 6 – Formatter normalization** – Finalize the staged `ruff format` rollout plan, execute it in reviewable slices,
   and tighten CI once each directory lands.
2. **Phase 7 – Regression depth** – Convert the coverage hotspots outlined in `docs/ci_baseline_report.md` into trading,
   risk, and orchestration regression suites.
3. **Phase 8 – Operational telemetry** – Implement the selected alerting channel and expose lightweight health dashboards so
   failures surface without manual log digging.
4. **Phase 9 – Dead code remediation** – Work through the findings in
   [Dead code audit – 2025-09-16](reports/dead_code_audit.md), deleting unused paths and scheduling follow-up audits as the
   codebase evolves.

Revisit this assessment after completing the Phase&nbsp;6–9 roadmap milestones so improvements can be reflected and new risks can be prioritized.
