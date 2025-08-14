# PR-4: App/Operational/UI/Integration hardening

This PR implements Phase 4 of the import normalization and typing posture hardening for application-facing areas: operational, integration, data_integration, ui, and data_sources.

Scope summary
- Normalized imports using the existing rewriter/guard flow. Strict dry-run found no unresolved legacy plain-root imports for the targeted areas; no mapping changes were required.
- Removed mypy overrides and shrank the mypy exclude regex for the targeted areas in [pyproject.toml](pyproject.toml:1).
- Kept Ruff configuration unchanged globally; ran a targeted probe to assess readiness to drop excludes. Probe surfaced violations, so no excludes were removed.

Files changed
- [pyproject.toml](pyproject.toml:1)
  - [tool.mypy] exclude now: ^src/(?:evolution|domain)/ ... (kept only evolution and domain; removed data_integration|operational|ui|data_sources|integration).
  - [[tool.mypy.overrides]] removed ignore_errors entries for:
    - "src.data_integration.*"
    - "src.operational.*"
    - "src.ui.*"
    - "src.data_sources.*"
    - "src.integration.*"
- [docs/development/import_rewrite_map.yaml](docs/development/import_rewrite_map.yaml:1) — unchanged (no additional star mappings were required).

Commands executed and results

1) Rewriter strict dry-run (sanity check)
- Command: python [scripts/cleanup/rewrite_imports.py](scripts/cleanup/rewrite_imports.py:1) --root src --map [docs/development/import_rewrite_map.yaml](docs/development/import_rewrite_map.yaml:1) --dry-run --strict --verbose
- Result: No files changed.
- Per-mapping hit counts: mapping[5]=3, mapping[12]=3, mapping[13]=2, mapping[14]=43, mapping[15]=13
- Unresolved star-imports: 0
- Unresolved legacy plain imports: 0

2) Applied rewriter with backups (idempotence check)
- Command: python [scripts/cleanup/rewrite_imports.py](scripts/cleanup/rewrite_imports.py:1) --root src --map [docs/development/import_rewrite_map.yaml](docs/development/import_rewrite_map.yaml:1) --backup --verbose
- Result: No files changed.

3) Legacy import guard
- Command: python [scripts/cleanup/check_legacy_imports.py](scripts/cleanup/check_legacy_imports.py:1) --root src --map [docs/development/import_rewrite_map.yaml](docs/development/import_rewrite_map.yaml:1) --fail --verbose
- Result: No legacy import violations detected. Allowed/ignored files were reported as expected.

4) Ruff targeted probe (non-blocking; configuration unchanged)
- Command: ruff check --select E9,F63,F7,F82,F401,E402,I001,F403,F405 src/operational src/integration src/data_integration src/ui src/data_sources
- Result: Found 63 errors; 52 fixable with --fix. Representative issues include:
  - E402/I001: import order and module-level import position in src/data_integration/__init__.py
  - I001/F401: unsorted and unused imports in src/data_integration/data_fusion.py and src/operational/md_capture.py
  - I001/F401: similar patterns in src/operational/health_monitor.py, src/ui/cli/main_cli.py, src/ui/ui_manager.py
- Decision: Do not drop any Ruff excludes in [pyproject.toml](pyproject.toml:1) for this PR. Excludes retention recorded; follow-up PRs can address file-by-file fixes and de-scope excludes safely.

Acceptance checklist
- [x] App/operational mypy overrides removed in [pyproject.toml](pyproject.toml:1)
- [x] [tool.mypy] exclude regex updated (kept evolution and domain only)
- [x] Rewriter strict dry-run/apply: no changes; per-mapping hits captured; no unresolved items
- [x] Legacy import guard: green (“No legacy import violations detected.”)
- [x] Ruff probe executed; excludes retained due to violations; decision recorded

Risk and rollback
- Risk: Low. Only [pyproject.toml](pyproject.toml:1) was modified; code rewrites were a no-op. Mypy now includes targeted app areas; because ignore_missing_imports remains true, CI risk is limited.
- Rollback: Revert the [pyproject.toml](pyproject.toml:1) changes in a single commit if necessary. No source files were altered by the rewriter.

Next steps (Phase 5 preview)
- Remove remaining mypy overrides (e.g., governance/domain/etc.) and converge on full typing coverage.
- Flip [tool.mypy] ignore_missing_imports to false after stubs/gaps are addressed.
- Tackle Ruff findings in targeted modules, then prune specific excludes in [pyproject.toml](pyproject.toml:1) once zero-violation is demonstrated consistently.

References
- Rewriter: [scripts/cleanup/rewrite_imports.py](scripts/cleanup/rewrite_imports.py:1)
- Guard: [scripts/cleanup/check_legacy_imports.py](scripts/cleanup/check_legacy_imports.py:1)
- Mapping: [docs/development/import_rewrite_map.yaml](docs/development/import_rewrite_map.yaml:1)
- Config: [pyproject.toml](pyproject.toml:1)