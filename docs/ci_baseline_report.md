# CI Baseline – 2025-09-16

This report captures the initial end-to-end pipeline run required to complete Phase 0 of the modernization roadmap. Each
command mirrors the jobs wired into `.github/workflows/ci.yml` so future executions can compare results against this snapshot.

## Command results

| Command | Outcome | Key findings |
| --- | --- | --- |
| `./scripts/check_forbidden_integrations.sh` | ✅ | No OpenAPI, FastAPI, or cTrader references detected in `src/` or `tests/current/`. |
| `ruff check .` | ✅ | Linting succeeded with the current Ruff ruleset. |
| `ruff format --check .` | ✅ | Formatter guard passes; repo-wide Ruff enforcement replaced the staged allowlist rollout. |
| `mypy --config-file mypy.ini src` | ✅ | Static typing passed; only informational reminders about unchecked bodies appeared for legacy functions. |
| `pytest tests/current --cov=src --cov-report=term -q` | ✅ | 100 tests passed (2 skipped). Overall coverage sits at 76.04%, clearing the 70% threshold in the configuration. |

## Coverage observations

Even though the suite passed, several modules remain lightly exercised:

* `src/operational/metrics.py` – 60.66% coverage with large untested control paths, particularly around metrics wiring and
  registry integration.
* `src/trading/models/position.py` – 70.71% coverage; gap clusters around lifecycle helpers and serialization routines.
* `src/data_foundation` configuration modules – each sits between 64–69% coverage, leaving many validation branches unverified.
* `src/sensory/dimensions/why/yield_signal.py` – 61.90% coverage with sparse branch assertions on signal construction.

Documenting these hotspots now allows later roadmap phases to target the weakest regression nets without re-running the entire
suite from scratch.

## Formatting debt follow-up

The staged formatter rollout concluded with repo-wide Ruff enforcement. Use the
historical playbook in [`docs/development/formatter_rollout.md`](development/formatter_rollout.md)
if future modules need similar treatment, and keep CI telemetry fresh via
`python -m tools.telemetry.update_ci_metrics --formatter-mode global`.
