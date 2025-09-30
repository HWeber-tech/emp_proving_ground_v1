# Infrastructure-as-code deployment runbook

This runbook translates the High-Impact Development Roadmap guidance into a
repeatable infrastructure-as-code workflow that mirrors the encyclopedia’s
operational posture. It assumes the institutional stack is managed via Terraform
(or an equivalent provisioning tool) with deployment artefacts stored in
`infrastructure/`.

## Goals

- Codify baseline infrastructure (VPC, Kubernetes clusters, databases, observability)
  as declarative manifests.
- Provide deterministic promotion workflows between staging and production.
- Capture operational guardrails for secrets, environment overrides, and drift
  detection.

## Source layout

```
infra/
  environments/
    staging/
      main.tf
      variables.tf
    production/
      main.tf
  modules/
    networking/
    compute/
    observability/
  scripts/
    plan.sh
    apply.sh
```

The repository root contains a `Makefile` entry point (`make infra-plan` and
`make infra-apply`) that wraps the scripts with environment selection and remote
state configuration.

## Environment overlays

Each environment directory applies the shared modules with environment-specific
inputs. Secrets are sourced from the platform’s secret manager (AWS Secrets
Manager, GCP Secret Manager, or Vault) and injected via Terraform variables at
plan time. Never commit secrets or environment-specific credentials to source
control.

## Plan and apply workflow

1. Operators trigger `make infra-plan ENV=staging` (or `production`).
2. The helper script configures remote state, selects the workspace, and runs
   `terraform plan` with the appropriate variable files.
3. Plans are archived under `artifacts/infra/<env>/YYYY-MM-DD/plan.txt` and
   attached to deployment tickets.
4. After approval, `make infra-apply ENV=staging` executes the change with the
   same variable set. Apply logs are stored alongside the plan artefacts.
5. Drift detection runs nightly via `terraform plan -detailed-exitcode`; results
   feed into the operational readiness dashboard.

## Secrets and configuration management

- Use environment variables or encrypted variable files for credentials.
- Rotate access tokens quarterly and document the rotation window in
  `docs/operations/runbooks/redis_cache_outage.md` and the new ops checklist.
- Maintain a `secrets.auto.tfvars` template with placeholders; operators copy
  the template, fill values locally, and ensure it is excluded via `.gitignore`.

## Testing and validation

- Lint Terraform with `terraform fmt` and `terraform validate` prior to plan.
- For Kubernetes manifests, run `kubectl diff --server-side` against the target
  cluster in staging.
- Capture integration smoke tests in `scripts/deployment/verify_stack.py` to
  validate ingress, database connectivity, and observability before flipping
  traffic.

## Rollback procedure

1. Execute `make infra-plan ENV=production` to confirm the desired rollback
   state.
2. Apply the rollback by checking out the last known-good tag and running
   `make infra-apply ENV=production`.
3. Validate via the disaster recovery drill and observability dashboards.

## Audit logging

- All plan/apply invocations emit structured logs via `scripts/deployment/log_run.py`.
- Logs are forwarded to the observability stack and retained for 90 days.
- The runbook `docs/deployment/drills/disaster_recovery_drill.md` links each
  drill to the corresponding infrastructure change tickets.

## Related tooling

- `tools/security/pip_audit_runner.py` for dependency scanning.
- `scripts/run_disaster_recovery_drill.py` for simulated failover validation.
- `docs/deployment/ops_command_checklist.md` for the daily operations cadence.
