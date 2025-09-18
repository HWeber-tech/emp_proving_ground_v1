# RFC: Formatter allowlist retirement

## Summary

Stage 4 formatting now enforces `src/sensory/`, all data foundation packages, and
the operational/performance/data_integration directories. This RFC codifies the
plan for removing `config/formatter/ruff_format_allowlist.txt` and switching CI
to a repository-wide `ruff format --check .` guard.

## Motivation

* Reduce maintenance overhead from keeping the allowlist in sync.
* Align local developer workflows with CI by making the formatter mandatory.
* Simplify future onboarding and tooling documentation once the allowlist is gone.

## Proposed steps

1. **Freeze final directories** – Confirm no outstanding formatting PRs are queued
   for `scripts/` helpers or other tooling-only paths.
2. **Format remaining helpers** – Run `ruff format` across `scripts/check_formatter_allowlist.py`
   and associated analysis helpers so they can be removed from the allowlist.
3. **Expand CI guard** – Update `.github/workflows/ci.yml` to run `ruff format --check .`
   instead of invoking `scripts/check_formatter_allowlist.py`.
4. **Remove allowlist file** – Delete `config/formatter/ruff_format_allowlist.txt`
   and drop the script from the repository.
5. **Refresh documentation** – Update `docs/development/setup.md`,
   `docs/development/formatter_rollout.md`, and `docs/status/ci_health.md` to describe
   the new default behaviour.
6. **Post-merge monitoring** – For the first week, run `python -m ruff format --check .`
   locally before pushing to ensure the guard is stable, and capture any regressions
   in the modernization sync notes.

## Rollout considerations

* Coordinate with teams touching `scripts/` to avoid merge conflicts during the
  final formatting sweep.
* Flag the CI change in the developer newsletter so contributors refresh their
  local `ruff` versions before the guard flips.
* Keep the RFC open for comments until the tooling helpers land; update the
  checklist below as work completes.

## Checklist

- [ ] Format `scripts/check_formatter_allowlist.py` and helper modules.
- [ ] Verify repo-wide `ruff format --check .` locally.
- [ ] Update CI workflow.
- [ ] Remove allowlist file and script.
- [ ] Refresh contributor documentation.
- [ ] Monitor the first week of runs for regressions.

## Status

Draft – circulated with Platform, Tooling, and Reliability teams on 2025-09-25.
