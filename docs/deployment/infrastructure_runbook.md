# Infrastructure Runbook

This runbook satisfies the roadmap requirement to "capture
infrastructure-as-code runbook" and documents how we deploy EMP to Oracle
Cloud (or equivalent environments) with appropriate guardrails.

## Environments

| Environment | Purpose | Provisioning |
| --- | --- | --- |
| `local` | Developer workstations, CI replay | `docker-compose` (see `docker/`) |
| `paper` | Continuous paper trading | Terraform workspace `paper` |
| `staging` | Pre-production smoke environment | Terraform workspace `staging` |
| `production` | Live trading | Terraform workspace `prod` |

Each environment is expressed as Terraform workspaces referencing the same
module stack. Environment overlays live in `k8s/overlays/<env>` and are
consumed by ArgoCD or Flux pipelines.

## Deployment Workflow

1. **Plan** – Run `make terraform-plan ENV=<env>` to generate an execution
   plan. Commit the plan artifact for review.
2. **Review** – Peers review the plan for drift, secrets handling, and cost
   deltas. Approval is required before apply.
3. **Apply** – Execute `make terraform-apply ENV=<env>` to provision or
   update infrastructure. Output logs are stored under
   `artifacts/deployment/<env>/`.
4. **Smoke Test** – Trigger `make deployment-smoke ENV=<env>` which runs the
   paper-trading dry run and dependency audit.
5. **Promote** – After staging smoke tests pass, promote manifests by
   tagging the release and updating the GitOps repository pointer.

## Secrets Management

- Secrets are injected via OCI Vault and synchronized using
  `tools/secrets/pull_vault_secrets.py`.
- Local development uses `.env` files stored in `env_templates/` with
  placeholder values; never commit real credentials.
- Secrets rotation cadence: monthly for FIX credentials, quarterly for data
  vendors.

## Rollback Procedures

1. Run `make terraform-plan ENV=<env> TARGET=rollback` to preview the
   rollback state.
2. Execute `make terraform-apply ENV=<env> TARGET=rollback` to revert to the
   last stable snapshot.
3. Trigger the Ops Command checklist (see companion document) to validate
   FIX connectivity, data feeds, and risk services post-rollback.

## Observability

- Deployment jobs publish OpenTelemetry traces to the observability
  collector defined in `config/observability/otel_collector.yaml`.
- Dependency audits run automatically within CI (`dependency-audit` job) and
  must be green before promotion.
- Smoke tests emit health reports stored under `artifacts/deployment/` for
  audit.

### Dependency Audit Exemptions

- Approved exemptions live in
  `config/governance/dependency_allowlist.json`.
- Each entry records an identifier, justification, and expiry date.
- Update the allowlist via pull request and link to the security ticket that
  tracks remediation.
- The CI job persists the raw report to
  `artifacts/dependency_audit/report.json` for review.

## Disaster Recovery Drills

- Quarterly drills restore the PostgreSQL and Redis backups to a fresh
  staging environment. Document outcomes in
  `docs/deployment/drills/<YYYY-MM-DD>.md`.
- Recovery time objectives (RTO): 30 minutes for paper, 10 minutes for
  production.

## Change Logging

- Every infrastructure change references a JIRA ticket and the High-Impact
  Development Roadmap section it closes.
- Use the PR template's "Operational Impact" section to summarize changes.
- Store generated plans and apply logs under
  `artifacts/deployment/<env>/history/` for auditors.

Maintaining this runbook ensures contributors can execute deployments
without relying on institutional memory, preventing the recurrence of
operations-related technical debt.
