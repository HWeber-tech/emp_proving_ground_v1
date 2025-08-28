# Release checklist: codify clean typing baseline and hardened CI

Goal
- Tag a release that codifies the zero-error typing baseline and hardened CI.

Checklist
- [ ] Ensure main is green with all required checks
- [ ] Confirm latest clean snapshot exists: [mypy_snapshots/mypy_summary_2025-08-27T09-43-50Z.txt](../../mypy_snapshots/mypy_summary_2025-08-27T09-43-50Z.txt:1)
- [ ] Verify nightly typing has run within past 24–48h
- [ ] Review [docs/development/typing_recipes.md](docs/development/typing_recipes.md:1) and [docs/development/pre_commit.md](docs/development/pre_commit.md:1) for currency
- [ ] Tag and create GitHub release (annotated) summarizing:
  - Zero-error baseline
  - CI gates (typing PR, nightly, static analysis, tests)
  - Import hygiene posture and stubs policy
- [ ] After tagging, verify branch protection rule still lists the required checks

Notes
- Consider adding pyright as non-blocking in a future iteration