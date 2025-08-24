Contributing Guide (Typing, Linting, and Workflow)

Overview
- This repository enforces strict static typing and safety-focused linting.
- All contributors must run the local checks before committing.
- Heavy dependencies (numpy, sklearn, torch, prometheus_client, etc.) are isolated via local stubs and lazy imports to keep import-time safe.

Authoritative configuration
- Mypy: see [mypy.ini](../../../mypy.ini:1) and [pyproject.toml](../../../pyproject.toml:53)
- Lint/format: ruff/black in [pyproject.toml](../../../pyproject.toml:1)
- Pre-commit hooks: [.pre-commit-config.yaml](../../../.pre-commit-config.yaml:1)
- CI pipeline: [.github/workflows/ci.yaml](../../../.github/workflows/ci.yaml:1)

Local development environment (Python 3.11)
1) Install Python 3.11
- Any OS: install Python 3.11 using your system package manager or pyenv.
- Debian/Ubuntu: apt install python3.11 python3.11-venv python3.11-distutils
- macOS: brew install python@3.11
- Windows: install from python.org or use uv/pyenv-win

2) Create a virtual environment
- python3.11 -m venv .venv
- Linux/macOS: source .venv/bin/activate
- Windows (PowerShell): .venv\Scripts\Activate.ps1

3) Upgrade pip and install tooling
- python -m pip install --upgrade pip
- python -m pip install mypy "ruff==0.6.*" "black==24.*" pre-commit
- Optional project stubs for local runs:
  - python -m pip install types-PyYAML pandas-stubs types-psutil

4) Install pre-commit hooks
- pre-commit install
- pre-commit run --all-files
  - This runs ruff (with auto-fix), black (format), and mypy (type checks) per [.pre-commit-config.yaml](../../../.pre-commit-config.yaml:1)

If your system forbids pip in the base interpreter (PEP 668)
- Use a venv as shown above (recommended)
- Or use pipx: pipx install mypy ruff black pre-commit
- Or install system packages (Debian/Ubuntu):
  - apt install mypy ruff black
  - Use pre-commit via pipx or a venv

How to run checks manually
- Type check:
  - python -m mypy src --config-file mypy.ini
- Lint (no changes):
  - python -m ruff check .
- Lint (auto-fix):
  - python -m ruff check . --fix
- Format:
  - python -m black .

CI pipeline
- GitHub Actions workflow: [.github/workflows/ci.yaml](../../../.github/workflows/ci.yaml:1)
  - Python 3.11, pip cache, mypy cache
  - Steps:
    - mypy src --config-file mypy.ini
    - ruff check .
    - black --check .
- CI must be green before merging:
  - Zero mypy errors
  - Lint and format checks must pass

Typing standards
- Strict mypy flags enabled; see [mypy.ini](../../../mypy.ini:1) and [pyproject.toml](../../../pyproject.toml:53)
- Public surfaces should avoid concrete Dict/Mutable types; prefer Mapping[...] for inputs and precise types for returns.
- Avoid Any. Use:
  - Protocols for structural typing
  - TypedDict/dataclasses for payloads
  - numpy.typing.NDArray for arrays (under TYPE_CHECKING)
- Keep heavy third-party imports localized at call sites and/or behind TYPE_CHECKING
  - Example patterns in [src/intelligence/competitive_intelligence.py](../../../src/intelligence/competitive_intelligence.py:1)

Local stubs policy
- Stubs live under stubs/ and are sourced by mypy via mypy_path:
  - See [stubs/README.md](../../../stubs/README.md:1)
  - Examples:
    - [stubs/sklearn/cluster.pyi](../../../stubs/sklearn/cluster.pyi:1)
    - [stubs/torch/nn.pyi](../../../stubs/torch/nn.pyi:1)
    - [stubs/prometheus_client/__init__.pyi](../../../stubs/prometheus_client/__init__.pyi:1)
- If a third-party type is missing or too heavy to import at runtime, prefer a narrow stub that matches only what we use.

Metrics/telemetry adapters (non-raising, lazy)
- Do not import prometheus_client at module import time; use lazy registry and/or runtime checks.
- Reference implementation:
  - Registry: [src/operational/metrics_registry.py](../../../src/operational/metrics_registry.py:1)
  - Facade: [src/operational/metrics.py](../../../src/operational/metrics.py:1)
- Telemetry port and optional registration:
  - Runtime sink registration via src.core.telemetry
  - Stubs for typing only: [stubs/src/core/telemetry.pyi](../../../stubs/src/core/telemetry.pyi:1)

Patterns to follow
- Use TYPE_CHECKING blocks for imports strictly used for types
- Use Protocols instead of subclassing Any-typed interfaces
- Normalize imports to src.* across repository
- Convert numpy scalars to Python float when returning scalar values
- Avoid dict invariance issues in public signatures (prefer Mapping for inputs, concrete Dict for internal structures only)
- Use cast as a last resort, with comments explaining why the cast is safe

Submitting a PR
- Ensure local pre-commit hooks pass on all files:
  - pre-commit run --all-files
- Ensure no new mypy errors are introduced
- Keep changes minimally invasive per module; prefer small, focused PRs:
  - For typing conversions, aim for no runtime behavior changes
- Reference the TODO checkpoints in the PR description:
  - Mark which tasks progressed (see roadmap below)

Roadmap checkpoints (typing burn-down)
- Intelligence modules:
  - Complete strict typing in portfolio_evolution (done)
  - Add typed wrappers and stubs for sklearn/torch where used
- Operational modules:
  - Keep metrics typed, lazy, and non-raising; registry memoization returns protocol types
- Orchestration and ecosystem:
  - Replace Any payloads with Protocols, TypedDicts, or dataclasses
  - Ensure consistent async signatures where appropriate
- Governance and configs:
  - Remove legacy ignores and dynamic tricks; prefer explicit, typed models

Common pitfalls
- Unused type: ignore comments
- Import-time heavy dependencies causing runtime failures in minimal environments
- Returning numpy scalar types instead of Python float
- Dict variance and Optional handling (no_implicit_optional is on)
- Hidden Any through legacy imports; replace with Protocols/TypedDicts

Questions
- Open an issue and link the relevant file and configuration:
  - Example references:
    - [mypy.ini](../../../mypy.ini:1)
    - [pyproject.toml](../../../pyproject.toml:53)
    - [src/core/interfaces.py](../../../src/core/interfaces.py:1)
    - [src/intelligence/portfolio_evolution.py](../../../src/intelligence/portfolio_evolution.py:1)

## Pre-commit

Set up the dev tooling environment:
- python -m venv .venv_tools
- source .venv_tools/bin/activate
- pip install -r requirements/dev.txt

Install Git hooks:
- pre-commit install

Run all hooks across the repository (useful before pushing):
- pre-commit run --all-files

PEP 668 note:
- If your system Python is “externally managed” and blocks pip installs, use the .venv_tools environment above or install pre-commit via pipx:
  - pipx install pre-commit

CI:
- The CI runs pre-commit against the entire repository on each push and pull request. Fix reported issues locally, re-run pre-commit, and push again.
## Typing PR Acceptance Criteria (Interim Profile A)

Effective immediately for all pull requests that modify Python code:

- Linting and formatting gates (changed files):
  - Run: ruff check, black --check, isort --check-only.
- Mypy typing gates (changed files):
  - Base config: no new mypy errors against the repository baseline using [mypy.ini](mypy.ini).
  - Strict-on-touch: changed files must satisfy at least L2 strictness via CI flags (see Strictness Ladder below). New modules should target L3.
- Annotation requirements:
  - All new public functions and methods must be fully annotated (parameters and return).
  - Collections must be parameterized (no bare list/dict/set/tuple).
- Type ignores and missing types:
  - Any `# type: ignore[CODE]` must include a specific error code and a brief justification in the PR description.
  - Prefer adding/refining stubs under [stubs/](stubs/) over `ignore_missing_imports`. If a stub is added/updated, reference it in the PR description.
- Transitional dynamic patterns:
  - If dynamic/untyped patterns must remain, add a TODO with an owner and a linked issue. Document the intended replacement (e.g., Protocol, TypedDict, dataclass, or generic).

### Strictness Ladder

- L1 (entry bar for touched files):
  - --disallow-untyped-defs, --check-untyped-defs
- L2 (strict-on-touch minimum):
  - L1 plus --disallow-incomplete-defs, --no-implicit-optional
- L3 (target for new modules and pilot package src/sensory/utils):
  - L2 plus --disallow-any-generics, --warn-unused-ignores, --warn-redundant-casts, --warn-return-any, --strict-equality
  - Per-package config may also set ignore_missing_imports=False and implicit_reexport=False
- L4 (strict parity):
  - Equivalent to mypy --strict unless otherwise documented

See CI workflow for enforcement: [.github/workflows/typing.yml](.github/workflows/typing.yml)