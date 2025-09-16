# Modernization Roadmap

This roadmap captures the next steps for reducing technical debt and restoring confidence in the EMP Professional Predator codebase and delivery pipelines. Items are grouped by phase so the team can focus on one layer of stability at a time while keeping downstream work unblocked.

## Phase 0 – Immediate hygiene (Week 1)
- **Retire redundant automation**: Remove the deprecated Kilocode CI Bridge workflow (done) and confirm no remaining secrets or labels reference the integration.
- **Unblock CI parsing**: Resolve the merge-conflict markers in `.github/workflows/ci.yml` so GitHub can execute the workflow again.
- **Baseline pipeline health**: Run the fixed CI workflow end-to-end (policy, lint, mypy, pytest) and capture current failure modes to inform later phases.

## Phase 1 – Policy enforcement consolidation (Week 2)
- **Single source of truth for OpenAPI/cTrader ban**: Merge the duplicated policy checks (`policy` job inside `ci.yml` and `policy-openapi-block.yml`) into one reusable action or shared job.
- **Configuration alignment**: Audit `config.yaml` and documentation to remove or clearly flag cTrader/OpenAPI placeholders so runtime defaults match the enforced FIX-only posture.

## Phase 2 – Dependency and environment hygiene (Weeks 3–4)
- **Rationalize requirements**: Decide on a primary dependency manifest (e.g., `requirements.txt` + extras or a lock file) and update workflows, docs, and dev onboarding accordingly.
- **Pin critical tooling**: Align versions for linting, typing, and testing tools across local dev and CI to eliminate "works on my machine" discrepancies.
- **Scientific stack verification**: Keep the runtime version checks in `main.py` authoritative by documenting the minimum supported versions and adding pre-flight validation scripts if needed.

## Phase 3 – Repository cleanup (Weeks 4–5)
- **Prune stray artifacts**: Delete `.orig` backups, `changed_files_*.txt`, and other scratch files from version control.
- **Establish guardrails**: Add `.gitignore` updates or pre-commit hooks to prevent regenerated artifacts from re-entering the repo.
- **Audit for dead code**: Use static analysis and runtime metrics to identify modules that can be archived or removed entirely.

## Phase 4 – Test coverage and observability (Weeks 6–7)
- **Strengthen regression nets**: Expand pytest coverage around the event bus, trading pipelines, and configuration loaders to reduce reliance on manual smoke tests.
- **Type safety focus areas**: Use the nightly mypy reports to target high-churn modules and drive them toward clean type annotations.
- **Operational insights**: Evaluate lightweight telemetry or structured logging so failures surface without external services like Kilocode.

## Phase 5 – Strategic refactors (Weeks 8+)
- **Subsystem decomposition**: Prioritize modules with the highest coupling (e.g., `core` ↔ `trading`) for refactor spikes once the pipeline is green.
- **Documentation refresh**: Produce system architecture and runbook documentation reflecting the stabilized workflows and policies.
- **Plan for future automation**: Revisit alerting and failure triage needs after CI is reliable, considering native GitHub features or self-hosted tooling if necessary.

### Tracking and review
- Review progress in weekly debt triage meetings.
- Keep a shared checklist linked to this roadmap so contributors can claim items and record findings.
- Revisit phases quarterly to update priorities based on new discoveries or product needs.
