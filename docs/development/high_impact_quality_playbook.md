# High-Impact Delivery Quality Playbook

## Purpose
Codify the non-negotiable quality activities that accompany every high-impact
roadmap story. The playbook is referenced by engineering leads during planning
and enforced through CI guard rails.

## Mandatory deliverables per story
1. **Automated tests**
   - Provide unit, integration, and—where complexity justifies—property-based
     tests covering the change surface.
   - Record new tests under the relevant package in `tests/` and update
     regression manifests when data fixtures change.
2. **CI pipeline updates**
   - Ensure GitHub Actions or internal CI executes the new tests and publishes
     artefacts (risk reports, GA metrics, dashboards).
   - Update workflow documentation in `docs/ci_baseline_report.md` when new jobs
     or artefacts are introduced.
3. **Documentation refresh**
   - Update affected runbooks, encyclopedia cross-references, and roadmap status
     pages.
   - Record knowledge transfer notes in `docs/reports/` or the relevant
     subsystem directory.
4. **Code quality enforcement**
   - Run `ruff`, `mypy`, and domain-specific validators locally before raising a
     pull request.
   - Add CI checks or pre-commit hooks if the change introduces new tooling.
5. **Retrospective inputs**
   - Capture defects, alerts, or operational surprises in the sprint retro and
     feed them into `docs/research/research_debt_register.md` or the operations
     backlog as appropriate.

## Governance
- Engineering managers audit adherence weekly using PR templates and the roadmap
  tooling in `tools/roadmap/high_impact.py`.
- Deviations require explicit sign-off recorded in the PR description and
  backfilled into the next retro.
- Update this playbook whenever tooling or compliance requirements evolve.
