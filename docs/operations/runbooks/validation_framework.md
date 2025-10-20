# Validation framework failure response

The async validation suite in `src/validation/validation_framework.py` exercises integration, data quality, performance, and governance guardrails before deployments proceed.【F:src/validation/validation_framework.py†L23-L514】 Results are standardised via `ValidationResult`, which records pass/fail state, numeric thresholds, metadata, and timestamps for downstream tooling.【F:src/core/validation_models.py†L7-L39】 Use this runbook to operate the suite and respond when checks fail.

## Run the suite

1. From the repository root run `python -m src.validation.validation_framework`. The CLI initialises the validator registry, runs each check, prints a summary table, and exits non-zero if the success rate falls below 75%.【F:src/validation/validation_framework.py†L31-L510】 The exit code lets CI and promotion gates fail fast on weak validation coverage.
2. Inspect `validation_report.json` generated in the working directory for an auditable snapshot. The report captures scores, thresholds, free-form details, and any metadata emitted by individual validators, and it tolerates up to two failures before marking the overall status as `FAILED`.【F:src/validation/validation_framework.py†L425-L471】 Archive the file with release notes or attach it to the operational readiness packet when the suite underpins a deployment decision.

## Validation checks

- `component_integration` ensures `ComponentIntegratorImpl` boots, enumerates canonical components, and records any missing services in the metadata payload.【F:src/validation/validation_framework.py†L44-L107】 A failure typically means a service alias resolved incorrectly or an integration prerequisite is offline.
- `data_integrity` validates required fields, type coercion, numeric ranges, and integer coercion on the canonical payload; violations are reported field-by-field in the `invalid_fields` map.【F:src/validation/validation_framework.py†L109-L199】 Use the metadata list to confirm which schema keys were validated before feeding new data sources.
- `performance_metrics` calculates rolling returns, volatility, and Sharpe ratio, bounding them inside expected operational ranges.【F:src/validation/validation_framework.py†L213-L249】 Deviations signal drift in the performance calculator or unexpected return magnitudes.
- `error_handling` simulates a raised `ValidationException` to confirm framework-level catching and reporting remain intact.【F:src/validation/validation_framework.py†L251-L279】 Failing here implies the exception wiring changed or an unexpected error type leaked through.
- `security_compliance` checks required secrets and API keys are present before remote calls execute.【F:src/validation/validation_framework.py†L281-L314】 Missing keys usually trace back to misconfigured environment variables or secret rotation scripts.
- `business_logic` asserts baseline trading or risk rules still return truthy results after recent edits.【F:src/validation/validation_framework.py†L316-L351】 Investigate rule definitions when the pass count drops below the expected total.
- `system_stability` fans out 100 concurrent async operations and expects 95%+ success to guard against event-loop regressions.【F:src/validation/validation_framework.py†L353-L387】 Failures often indicate blocking work on the loop or resource exhaustion.
- `regulatory_compliance` validates that required governance artefacts stay registered, signalling that audit and retention hooks remain live.【F:src/validation/validation_framework.py†L389-L423】 Extend the checklist when new regulator commitments land.

## Failure handling

- Review the `details` and `metadata` for each failed validator inside `validation_report.json` to pinpoint the offending component or field. Metadata is especially rich for component and data integrity checks.【F:src/validation/validation_framework.py†L69-L191】【F:src/core/validation_models.py†L29-L39】
- Check the framework logs: a caught exception emits `Validator <name> failed` with the original error string, which highlights unexpected crashes or dependency import issues.【F:src/validation/validation_framework.py†L431-L446】 Re-run with `PYTHONASYNCIODEBUG=1` when concurrency issues are suspected.
- Address the failing contract:
  - Fix missing integrations by verifying `ComponentIntegratorImpl.list_components()` output and ensuring each dependency initialises cleanly before rerunning the suite.【F:src/validation/validation_framework.py†L65-L107】
  - For schema breaches, reproduce the payload through the offending ingest path, correct the transformation, and confirm the `invalid_fields` map is empty on rerun.【F:src/validation/validation_framework.py†L133-L199】
  - For score-based checks (`performance_metrics`, `system_stability`, `regulatory_compliance`), capture the recorded metrics and correlate with recent code or infrastructure changes before accepting risk.【F:src/validation/validation_framework.py†L213-L423】
- Only proceed with deployment once the report shows the intended success rate (≥75%) and the summary status reads `PASSED`. Persist the fixed report in version control or the release artifact store so auditors can trace the remediation history.
