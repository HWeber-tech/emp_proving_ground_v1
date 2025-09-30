# Oracle Cloud Smoke Test Playbook

This playbook operationalises the roadmap requirement to gate Oracle Cloud
deployments behind automated smoke tests and an explicit rollback command.

## Components

- **Plan file** — `config/deployment/oracle_smoke_plan.yaml` lists each command to
  run after applying Kubernetes manifests.  Tests cover API health, FIX gateway
  connectivity, and metrics endpoints.
- **Executor** — `scripts/deployment/run_oracle_smoke_tests.py` loads the plan,
  executes commands sequentially, and emits JSON or human-readable summaries.
- **Rollback helper** — `scripts/deployment/rollback_oracle_release.sh` wraps
  `kubectl rollout undo` with logic to target the previous revision.

## Running the suite

```bash
# Execute the default plan and print a console summary
scripts/deployment/run_oracle_smoke_tests.py

# Emit JSON for CI pipelines and capture the exit status
scripts/deployment/run_oracle_smoke_tests.py --json > smoke_results.json
```

Critical failures exit with status code 1 so CI can abort the rollout and invoke
`rollback_oracle_release.sh` automatically.  Non-critical checks (e.g., metrics
surface probes) report failures but do not fail the deployment.

## Extending the plan

- Add new commands to the YAML file to cover recently provisioned services
  (Redis, Prometheus, risk API, etc.).
- Provide custom environment variables for individual tests by specifying an
  `env` mapping alongside the command.
- Adjust timeouts per test to accommodate regions with higher latency.

## Operational workflow

1. Apply the desired Kustomize overlay (`kustomize build k8s/overlays/prod ...`).
2. Run the smoke-test script and verify output.  The JSON summary can be pushed to
   object storage for auditing.
3. If a critical test fails, run the rollback script immediately and open an
   incident according to the Ops Command checklist.
4. Attach the smoke-test summary to the deployment ticket for regulatory audit
   evidence.
