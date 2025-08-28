# Mypy Python 3.11 Environment Alignment (Reproducible Runner)

Purpose
- Reproduce mypy runs under Python 3.11 to align with CI and differentiate environment-induced regressions from code regressions.
- Runs entirely in Docker so Python 3.11 does not need to be installed on the host.
- Installs developer dependencies from [requirements/dev.txt](requirements/dev.txt:1) to match project tooling expectations.

Usage
- Build and run the snapshot generator:
  - bash [scripts/mypy_py311.sh](scripts/mypy_py311.sh:1)
- Force a rebuild to pick up dependency changes (for example after editing [requirements/dev.txt](requirements/dev.txt:1)):
  - MYRUN_REBUILD=1 bash [scripts/mypy_py311.sh](scripts/mypy_py311.sh:1)

What the runner does
- Builds a Docker image from [docker/mypy311/Dockerfile](docker/mypy311/Dockerfile:1) using python:3.11-slim with minimal build tools (build-essential, git, ca-certificates), upgrades pip, and installs [requirements/dev.txt](requirements/dev.txt:1).
- Runs mypy in a container with the repo mounted at /workspace and working directory /workspace.
- Cleans .mypy_cache inside the container prior to the run.
- Captures environment details and mypy output to timestamped snapshot artifacts.

20 | Outputs
21 | Artifacts are written to [mypy_snapshots](mypy_snapshots:1) with UTC timestamps, for example:
22 | - Full mypy snapshot: [mypy_snapshots/mypy_snapshot_py311_*.txt](mypy_snapshots:1)
23 | - Summary line: [mypy_snapshots/mypy_summary_py311_*.txt](mypy_snapshots:1)
24 | - Ranked offenders CSV: [mypy_snapshots/mypy_ranked_offenders_py311_*.csv](mypy_snapshots:1)
25 | - Environment capture: [mypy_snapshots/env_py311_*.txt](mypy_snapshots:1)
26 | 
27 | Docker troubleshooting for py311 mypy runner
28 | Preconditions
29 | - Docker is installed on the host.
30 | - The Docker daemon is running.
31 | - Your user is a member of the “docker” group (so you can run docker without sudo).
32 | 
33 | Common errors and remedies
34 | - Permission denied or cannot connect to Docker daemon:
35 |   - Add your current user to the docker group, then start a new login session (use newgrp docker or log out and log back in).
36 |   - Restart the Docker daemon to refresh permissions/state.
37 |   - Verify Docker works by running “docker run hello-world”.
38 | - Corporate-managed host blocks Docker:
39 |   - Use CI fallback (see Fallback strategy when local Docker is unavailable).
40 | 
41 | Verification steps
42 | - Run the hardened runner [scripts/mypy_py311.sh](scripts/mypy_py311.sh:1).
43 | - Confirm artifacts under [mypy_snapshots/](mypy_snapshots:1), including a summary file with a numeric totals line like “Found N errors in M files (checked K source files)”.
44 | - If totals are missing or failures persist:
45 |   - Inspect the snapshot file path printed by the runner and review the full output inside it.
46 |   - Check the summary file content for a failure message emitted by the runner.
47 | 
48 | Fallback strategy when local Docker is unavailable
49 | - Manually dispatch the nightly typing workflow [typing-nightly.yml](.github/workflows/typing-nightly.yml:1) in GitHub Actions (Actions UI → Typing Nightly → Run workflow).
50 | - After the run completes, download artifacts (summary, ranked offenders, environment capture).
51 | - Add links to these artifacts in the comparison report [2025-08-27_mypy_env_comparison.md](docs/development/reports/2025-08-27_mypy_env_comparison.md:1) for cross-environment tracking.
52 | 
53 | Note about environment alignment
54 | - All typing runs should use Python 3.11 with mypy==1.17.1 for consistency with CI (pinned in [requirements/dev.txt](requirements/dev.txt:1)).
55 | - If you temporarily use a different local environment, acknowledge the variance in any shared reports or PR notes.
56 | 
57 | Quick commands checklist
58 | - Check Docker status and your group membership.
59 | - Add your user to the docker group and refresh your session.
60 | - Restart the Docker daemon.
61 | - Verify Docker works with hello-world.
62 | - Re-run [scripts/mypy_py311.sh](scripts/mypy_py311.sh:1) and confirm the summary shows numeric totals.
27 | Notes
28 | - The Docker image installs dependencies from [requirements/dev.txt](requirements/dev.txt:1). If these change, rebuild with:
29 |   - MYRUN_REBUILD=1 bash [scripts/mypy_py311.sh](scripts/mypy_py311.sh:1)
30 | - This tooling is diagnostics-only: no application code or CI pipelines are modified.
31 | - Default Docker entrypoint is bash as defined in [docker/mypy311/Dockerfile](docker/mypy311/Dockerfile:1), and the build context excludes typical junk via [docker/mypy311/.dockerignore](docker/mypy311/.dockerignore:1).