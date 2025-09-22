# CI & Workflow Technical Debt TODO

This backlog captures the workflow debt identified in the latest CI review and
turns each finding into an actionable task. Update the checkboxes as work
progresses and add links to issues or PRs once they are filed.

## Immediate (next 1–2 sprints)

- [x] **Broaden pytest execution scope**
  Expand automation beyond `tests/current/` so quarantined suites under `tests/`
  run on a cadence (full run, labeled PR job, or scheduled workflow) and prevent
  regressions from hiding in legacy directories. (_Implemented via CI `tests-full`
  job with opt-in `ci:full-tests` label_)
- [x] **Extend forbidden integration scanning**
  Update `.github/workflows/forbidden-integrations.yml` to traverse additional
  code-bearing directories (`scripts/`, archived tests, docs snippets) so policy
  violations cannot slip through unscanned paths. (_Default scan now includes
  `tests/`, `scripts/`, `docs/`, and `tools/`_)
- [x] **Introduce least-privilege permissions**
  Add an explicit `permissions` block to `.github/workflows/ci.yml`, granting
  only the scopes each job requires to reduce blast radius if a job is
  compromised. (_CI workflow now requests read-only contents and minimal actions
  access_)
- [x] **Re-enable backtest validation before merge**
  Run the backtest job for pull requests via a label, manual opt-in, or
  conditional matrix so contributors can validate changes before they land on
  `main`. (_Apply the `ci:backtest` label to run this job on PRs_)

## Near term (quarter scale)

- [x] **Right-size dependency installation**
  Split the composite `python-setup` action or add caching so `requirements/dev.txt` and heavy scientific stacks are not rebuilt from scratch on every job. (_Composite action now restores a cached virtual environment_)
- [x] **Remove redundant repository checkouts**
  Trim extra `actions/checkout` steps from workflows that already invoke the
  composite setup action to cut per-run overhead. (_Dead-code audit workflow now
  relies on the composite checkout_)
- [x] **Alert on nightly typing regressions**
  Enhance the scheduled typing workflow to create issues, send notifications, or
  otherwise surface failures when the nightly type-check run breaks. (_Nightly
  typing workflow automatically opens/closes alert issues_)

## Tracking

- Assign owners by appending `@handle` next to each task once staffed.
- Link to GitHub issues (e.g., `GH-1234`) beside the checkbox when tickets are
  created.
- Review this list during sprint planning to keep workflow maintenance visible.
