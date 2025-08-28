# Title
Batch10: typing/import hygiene progress and CI enhancements

## Summary
- Batch10 fix3–fix14 toward zero-error typing baseline
- Consolidate and protect zero-error baseline
- Add nightly typing job and full-repo PR job

## Links and artifacts
- Latest clean snapshot: [mypy_snapshots/mypy_summary_2025-08-27T09-43-50Z.txt](mypy_snapshots/mypy_summary_2025-08-27T09-43-50Z.txt:1)
- PR typing workflow: [typing.yml](.github/workflows/typing.yml:1)
- Nightly typing workflow: [typing-nightly.yml](.github/workflows/typing-nightly.yml:1)
- Reference docs: [typing_recipes.md](docs/development/typing_recipes.md:1), [pre_commit.md](docs/development/pre_commit.md:1)

## Validation checklist
- [ ] pre-commit run clean (ruff, black, isort, mypy changed-files)
- [ ] typing.yml: changed-files job green
- [ ] typing.yml: full-repo mypy-full job green with artifacts uploaded
- [ ] static-analysis workflow green (ruff/import-linter)
- [ ] tests workflow green

## Risk and rollout
- Behavior-preserving edits only (≤5 edits/file typical) and import hygiene adjustments
- Rollback/mitigation: revert PR; nightly typing continues to guard baseline

## Post-merge actions
- [ ] Enable branch protection with required checks (see [branch_protection.md](docs/development/branch_protection.md:1))
- [ ] Tag release (see [release_checklist.md](docs/development/release_checklist.md:1))