# Typing & CI Onboarding Guide

Welcome to the EMP engineering team! This guide summarises the expectations, guardrails, and local setup needed to keep the mypy checks healthy after the CI recovery effort.

## 1. Local Environment Checklist
- Install dev dependencies: `pip install -r requirements/dev.txt` (or use the project Docker image).
- Install pre-commit hooks: `pre-commit install --hook-type pre-push`.
- Ensure your IDE understands the local `stubs/` directory (configure `PYTHONPATH=stubs` if required).

## 2. Day-to-Day Workflow
1. **Run the strict-on-touch script before pushing**
   ```bash
   poetry run python tools/run_mypy_strict_on_changed.py
   ```
   This mirrors the CI gate and fails fast if new untyped definitions slip in.
2. **Run the full suite when touching shared modules**
   ```bash
   mypy --config-file mypy.ini src
   pytest -q
   ```
3. Capture a snapshot in `mypy_snapshots/` whenever you land a large typing sweep so the weekly log stays current.

## 3. Coding Standards
- Prefer `Mapping`/`Sequence` in public APIs; use concrete `dict`/`list` only for local mutation.
- Annotate all function returns explicitly (`-> None` when no value is returned).
- Use shared coercion helpers from `src/core/coercion.py` for numeric conversions.
- Add protocol or stub definitions under `stubs/` for third-party dependencies instead of sprinkling `type: ignore`.

## 4. CI Expectations
- The `types` job blocks merges and runs `mypy --config-file mypy.ini src` with `check_untyped_defs = True`.
- Nightly CI reruns the same job and publishes the report artifact. Investigate failures immediately.
- The pre-push hook executes mypy on staged Python files; keep it enabled to avoid noisy CI failures.

## 5. When You Hit an Error
1. Check the [mypy playbooks](mypy_playbooks.md) for remediation recipes by error class.
2. If a stub is missing, consult the [type stub intake checklist](mypy_dependency_checklist.md).
3. Document significant fixes in the [weekly status log](mypy_status_log.md) and update the CI recovery plan if scope changes.

## 6. Getting Help
- #eng-infra Slack channel for CI/type checking support.
- Pair with package owners listed in `docs/ci_recovery_plan.md` when introducing new modules or dependencies.
- Open draft PRs early to leverage the strict-on-touch automation feedback.

Keeping these practices in mind ensures the mypy backlog stays at zero while the team continues shipping features.
