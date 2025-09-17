# Ruff Formatter Rollout Plan

The repository still contains hundreds of legacy files that do not pass
`ruff format`. Rather than landing one giant, review-hostile reformat, we will
normalize the tree in staged slices. Each slice enrolls a directory (or focused
module) in formatter enforcement once it has been manually cleaned up.

## Slicing strategy

| Stage | Scope | Gating tasks | Owner |
| --- | --- | --- | --- |
| 0 | `tests/current/` | Land mechanical formatting PRs for all current regression tests and update fixtures as needed. | QA Guild |
| 1 | `src/system/` & `src/core/configuration.py` | Confirm tests stay green after formatting and document any manual adjustments to config helpers. | Platform |
| 2 | `src/trading/execution/` & `src/trading/models/` | Pair formatting with targeted regression coverage from Phase 7 to keep behavior visible. | Trading |
| 3 | `src/sensory/organs/dimensions/` (per organ) | Format one organ at a time, logging tricky lint suppressions directly in PR descriptions. | Sensory |
| 4 | Remaining packages | Continue package-by-package once the high-traffic areas above are stable. | All |

Stages can proceed in parallel as long as each owner keeps their PRs focused and
communicates any mechanical churn that will affect neighbors (for example shared
fixtures or import paths).

## Opting a directory into enforcement

1. Run `ruff format <path>` locally and ensure the diff only contains mechanical
   whitespace/import updates.
2. Re-run the relevant tests (`pytest tests/current -q`) to catch accidental
   behavior changes.
3. Add the directory (or specific files) to
   `config/formatter/ruff_format_allowlist.txt`, keeping the list alphabetized.
4. Commit the formatting diff and the allowlist change in the same PR so CI will
   enforce the new slice immediately.

The CI workflow now calls `scripts/check_formatter_allowlist.py`, which reads the
allowlist and executes `ruff format --check` for every entry. As the allowlist
grows, the guardrail automatically expands.

## Contributor workflow updates

* Run `ruff format` before opening a PR when you touch a file that already lives
  in the allowlist. CI will fail if the formatter drifted.
* When you work outside the allowlist, prefer `ruff check --select I` to nudge
  import ordering toward the formatter output ahead of time.
* Use review comments or PR descriptions to flag any surprising manual fixes so
  later stages can avoid repeat work.

Check the modernization [roadmap](../roadmap.md) for the current stage status
and ownership expectations.

## Stage status – 2025-09-16 update

- **Stage 0 – `tests/current/`**: Completed. `ruff format` was applied across the
  regression suite, `tests/current/` is now listed in
  `config/formatter/ruff_format_allowlist.txt`, and the test suite passed in
  strict asyncio mode to confirm no behavioral drift.
- **Stage 1 – `src/system/` & `src/core/configuration.py`**: Dry-run formatting
  surfaced only cosmetic rewrites in `src/system/requirements_check.py`
  (collapsing multi-line string appends). No manual guardrails were required,
  so this slice is ready for the mechanical formatting PR immediately after
  Stage 0 lands.

### Stage 1 preparation notes

- The requirements check CLI already enforces 88-character-friendly strings;
  the diff produced by `ruff format` is limited to combining short `message.append`
  calls into single lines.
- No handwritten fixtures or generated assets live under `src/system/`; once
  formatted, CI guardrails can extend the allowlist without additional skips.
