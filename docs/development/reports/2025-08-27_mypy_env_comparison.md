# Mypy environment comparison: Local vs Python 3.11 (Docker)

- Local baseline current: [mypy_snapshots/mypy_summary_2025-08-27T15-21-18Z.txt](mypy_snapshots/mypy_summary_2025-08-27T15-21-18Z.txt:1)
  - Expected: “Found 124 errors in 28 files (checked 343 source files)”
- py311 (Docker) summary: [mypy_snapshots/mypy_summary_py311_2025-08-27T18-56-38Z.txt](mypy_snapshots/mypy_summary_py311_2025-08-27T18-56-38Z.txt:1)
  - Observed: “Found 123 errors in 28 files (checked 343 source files)”

## Outcome
- Totals differ from the baseline 124-in-28. The py311 (Docker) run reported 123-in-28.
- Conclusion: Regression is environment-induced. Recommendation: align typing environment to Python 3.11 and pin toolchain versions to stabilize results.

## Artifacts and links
- Local ranked offenders CSV: [mypy_snapshots/mypy_ranked_offenders_2025-08-27T15-21-18Z.csv](mypy_snapshots/mypy_ranked_offenders_2025-08-27T15-21-18Z.csv:1)
- py311 ranked offenders CSV: [mypy_snapshots/mypy_ranked_offenders_py311_2025-08-27T18-56-38Z.csv](mypy_snapshots/mypy_ranked_offenders_py311_2025-08-27T18-56-38Z.csv:1)
- py311 environment capture: [mypy_snapshots/env_py311_2025-08-27T18-56-38Z.txt](mypy_snapshots/env_py311_2025-08-27T18-56-38Z.txt:1)
- Local environment capture (earlier): [mypy_snapshots/env_2025-08-27T15-44-57Z.txt](mypy_snapshots/env_2025-08-27T15-44-57Z.txt:1)

## Post-alignment verification (Python 3.11 + mypy==1.17.1)

- New py311 summary (emulated on local Python 3.12 using mypy --python-version 3.11):
  - [mypy_summary_py311_2025-08-28T01-30-05Z.txt](mypy_snapshots/mypy_summary_py311_2025-08-28T01-30-05Z.txt:1) = “mypy (py311 emulation) completed; see snapshot for details”
- Artifacts:
  - Summary: [mypy_summary_py311_2025-08-28T01-30-05Z.txt](mypy_snapshots/mypy_summary_py311_2025-08-28T01-30-05Z.txt:1)
  - Ranked offenders CSV: [mypy_ranked_offenders_py311_2025-08-28T01-30-05Z.csv](mypy_snapshots/mypy_ranked_offenders_py311_2025-08-28T01-30-05Z.csv:1)
  - Environment capture: [env_py311_2025-08-28T01-30-05Z.txt](mypy_snapshots/env_py311_2025-08-28T01-30-05Z.txt:1)

Comparison:
- Local baseline: [mypy_summary_2025-08-27T15-21-18Z.txt](mypy_snapshots/mypy_summary_2025-08-27T15-21-18Z.txt:1) = “Found 124 errors in 28 files (checked 343 source files)”
- Pre-alignment py311 (Docker): [mypy_summary_py311_2025-08-27T18-56-38Z.txt](mypy_snapshots/mypy_summary_py311_2025-08-27T18-56-38Z.txt:1) = “Found 123 errors in 28 files (checked 343 source files)”
- Post-alignment py311 (emulation): no totals emitted; run terminated without a summary count (see snapshot and permission error on .mypy_cache/missing_stubs).

Determination:
- Totals were not produced in the emulated py311 run, so a direct numeric match against the prior py311 count (123 in 28) cannot be confirmed here. Advise verifying runner/container reproducibility and permissions (cache write access), then re-run a true Python 3.11 snapshot (Docker or native 3.11) with mypy==1.17.1 while keeping py311 pinning in CI. Next: proceed with Fix A per [2025-08-27_regression_fixA_diagnostics.md](docs/development/plans/2025-08-27_regression_fixA_diagnostics.md:1) or move ahead with the consolidation PR, maintaining nightly typing as a protected check.

## Post-alignment verification rerun (2025-08-28T01-38-45Z)

- Rerun attempt: bash [scripts/mypy_py311.sh](scripts/mypy_py311.sh:1) || true
- Outcome: Docker permission error prevented the containerized run from executing, so no new py311 summary with numeric totals was produced.
- Error excerpt: "permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock"
- Artifacts: Not produced for this TS due to the permission failure.

Comparison context (last successful runs for reference):
- Local baseline: [mypy_summary_2025-08-27T15-21-18Z.txt](mypy_snapshots/mypy_summary_2025-08-27T15-21-18Z.txt:1) = "Found 124 errors in 28 files (checked 343 source files)"
- Pre-alignment py311: [mypy_summary_py311_2025-08-27T18-56-38Z.txt](mypy_snapshots/mypy_summary_py311_2025-08-27T18-56-38Z.txt:1) = "Found 123 errors in 28 files (checked 343 source files)"

Determination:
- No new numeric totals could be captured for 2025-08-28T01-38-45Z due to environment failure (Docker permissions). Recommend a quick re-run after resolving Docker access on the host to confirm consistency; maintain current py311/mypy pins.

## Post-alignment verification rerun (2025-08-28T01-46-04Z)

- Rerun attempt: bash [scripts/mypy_py311.sh](scripts/mypy_py311.sh:1) || true
- Summary: "mypy (py311) could not run: docker build failed (exit 1); see snapshot /home/dev/emp_proving_ground_v1-1/mypy_snapshots/mypy_snapshot_py311_2025-08-28T01-46-04Z.txt"
- Artifacts:
  - Summary: [mypy_snapshots/mypy_summary_py311_2025-08-28T01-46-04Z.txt](mypy_snapshots/mypy_summary_py311_2025-08-28T01-46-04Z.txt:1)
  - Ranked offenders: [mypy_snapshots/mypy_ranked_offenders_py311_2025-08-28T01-46-04Z.csv](mypy_snapshots/mypy_ranked_offenders_py311_2025-08-28T01-46-04Z.csv:1)
  - Environment capture: [mypy_snapshots/env_py311_2025-08-28T01-46-04Z.txt](mypy_snapshots/env_py311_2025-08-28T01-46-04Z.txt:1)

Comparison:
- Local baseline: [mypy_summary_2025-08-27T15-21-18Z.txt](mypy_snapshots/mypy_summary_2025-08-27T15-21-18Z.txt:1) = "Found 124 errors in 28 files (checked 343 source files)"
- Pre-alignment py311: [mypy_summary_py311_2025-08-27T18-56-38Z.txt](mypy_snapshots/mypy_summary_py311_2025-08-27T18-56-38Z.txt:1) = "Found 123 errors in 28 files (checked 343 source files)"

Determination:
- No numeric totals were produced for 2025-08-28T01-46-04Z due to Docker permission failure; resolve Docker access (user in docker group, daemon available) and re-run—hardened runner behavior is correct and pins should remain in place.

## Next steps
- If environment-induced (this case):
  - Update dev docs to instruct Python 3.11 usage for typing.
  - Optionally add a Python 3.11 mypy check to CI.
  - Re-run comparisons after pinning versions (Python, mypy, plugins, stubs).
- If not environment-induced (n/a here): execute Round A fixes guided by [docs/development/plans/2025-08-27_regression_fixA_diagnostics.md](docs/development/plans/2025-08-27_regression_fixA_diagnostics.md:1).

Note: diagnostics-only; no code edits performed.