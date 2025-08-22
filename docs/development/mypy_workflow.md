# MyPy Workflow: Fast-Track Strict Remediation

This repository includes tooling to accelerate type quality improvements with actionable triage, safe auto-rewrites, and editor/CLI integration.

## Prerequisites

- Local venv for mypy at `./.venv_mypy` with mypy and libcst installed:
  - `./.venv_mypy/bin/pip install mypy libcst`
- A `mypy.ini` at the repo root configured for your project layout.

## Quick checks

- One-click in VSCode:
  - Task: “mypy: quick” — runs strict mypy with error codes and logs to `scripts/analysis/out/last_mypy.txt`.
- CLI:
  - `bash scripts/analysis/mypy_quick.sh`

Artifacts:
- Raw output snapshot: `scripts/analysis/out/last_mypy.txt`

## Daemon (dmypy) checks

- VSCode Tasks:
  - “mypy: dmypy start” — starts daemon and warms caches
  - “mypy: dmypy status” — shows status
  - “mypy: dmypy check” — runs a fast incremental check
  - “mypy: dmypy stop” — stops daemon
- CLI wrapper:
  - `bash scripts/analysis/mypy_daemon.sh start|status|check|stop`
- Make targets:
  - `make dmypy-start`, `make dmypy-status`, `make dmypy-check`, `make dmypy-stop`

Notes:
- Environment ensures `MYPYPATH=stubs:src:$MYPYPATH` if `stubs/` exists.

## Summaries and triage

Generate machine-readable artifacts and a concise console summary:

- VSCode Task:
  - “mypy: summary”
- CLI:
  - `./.venv_mypy/bin/python scripts/analysis/run_mypy_summary.py`
- Make:
  - `make type-summary`

Outputs:
- `scripts/analysis/out/mypy_report.json` — raw items
- `scripts/analysis/out/mypy_report.csv` — CSV: file,line,code,message
- Console summary includes:
  - Total errors
  - Top 10 error codes
  - Top 15 files by error count
  - Earliest 20 errors (stable order)

## Auto-rewriter for trivial explicit Any

The rewriter uses LibCST for safe, types-only edits. It operates in dry-run by default (prints unified diffs).

Patterns:
- `Dict[str, Any] → dict[str, object]`
- `List[Any] → list[object]`
- `Sequence[Any] → Sequence[object]`
- `Mapping[str, Any] → Mapping[str, object]`
- `Optional[Dict[str, Any]] → Optional[dict[str, object]]`
- `Callable[..., Any] → Callable[..., None]` only when the annotated name indicates a handler/callback (parameter/variable contains “handler” or “callback”)
- `*args: Any → *args: object`
- `**kwargs: Any → **kwargs: object`

Scope controls:
- Default paths: `src/`
- Default exclude regex: `^stubs/|/tests?/|^tests/`
- Optional: `--include REGEX`, `--exclude REGEX`
- Optional: `--backup` to write `.bak` before applying

Usage:
- Dry-run (default):  
  `./.venv_mypy/bin/python scripts/cleanup/explicit_any_rewriter.py`
- Apply changes:  
  `./.venv_mypy/bin/python scripts/cleanup/explicit_any_rewriter.py --apply`
- With backups:  
  `./.venv_mypy/bin/python scripts/cleanup/explicit_any_rewriter.py --apply --backup`
- Make targets:
  - `make type-explicit-any-dry`
  - `make type-explicit-any-apply`

Safeguards:
- Only modifies annotation syntax trees (never strings/comments).
- Only targets the specified trivial patterns. No behavior/runtime changes.

Review workflow:
1) Run dry-run
2) Inspect printed unified diffs
3) Re-run with `--apply` (optionally `--backup`)
4) Run `mypy: quick` or `dmypy check` to verify impact

## VSCode integration

Tasks (`.vscode/tasks.json`):
- “mypy: quick”
- “mypy: summary”
- “mypy: dmypy start/status/check/stop”
- Problem matcher parses `file:line: error` lines for quick navigation.

Settings (`.vscode/settings.json`):
- `python.analysis.typeCheckingMode: "basic"` (switch to `"strict"` when ready)
- Mypy Type Checker extension config:
  - `mypy-type-checker.path: [".venv_mypy/bin/mypy"]`
  - `mypy-type-checker.args: ["--config-file","mypy.ini","--namespace-packages","--explicit-package-bases"]`
- Error Lens quality-of-life options enabled

Recommended extensions:
- Mypy Type Checker: `ms-python.mypy-type-checker`
- Error Lens: `usernamehw.errorlens`
- Pylance: `ms-python.vscode-pylance`

## High-impact triage flow

1) Generate summary:
   - `make type-summary`
2) Prioritize by:
   - Top error codes (e.g., `no-untyped-def`, `var-annotated`, `call-arg`)
   - Top offending files by count
3) Choose “shortest-path” modules:
   - Focus on modules dominated by trivial `Any` patterns (explicit container Any, args/kwargs Any)
   - Prefer modules with fewer cross-module dependencies
4) Apply safe rewrites:
   - `make type-explicit-any-dry` → review → `make type-explicit-any-apply`
5) Iterate with dmypy:
   - `make dmypy-check` to get fast feedback between edits

## Validation sanity checks

- Reporter artifacts:
  - `./.venv_mypy/bin/python scripts/analysis/run_mypy_summary.py`
- Rewriter dry-run:
  - `make type-explicit-any-dry` (should print diffs but not write)
- VSCode JSON (basic sanity via JSON load):
  - Both `.vscode/tasks.json` and `.vscode/settings.json` should parse as valid JSON.