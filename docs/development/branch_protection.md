# Branch protection: enforce zero-error typing baseline

Purpose
- Require CI gates to protect the zero-error typing baseline.

Steps (GitHub UI)
- Go to Repository Settings → Branches → Branch protection rules → Add rule for main
- Enable “Require a pull request before merging”
- Enable “Require status checks to pass before merging”, then select:
  - typing workflow checks (changed-files job and “Mypy (full repo)”)
  - static analysis workflow checks
  - tests workflow checks
- Optionally enable “Require branches to be up to date”
- Optional: Require conversation resolution
- Save the rule

Notes
- Job names may appear only after the first green run; re-open the rule to select them
- Nightly typing workflow is informational; PRs must be blocked by the typing PR workflow
- References: [typing.yml](../../.github/workflows/typing.yml:1), [typing-nightly.yml](../../.github/workflows/typing-nightly.yml:1)