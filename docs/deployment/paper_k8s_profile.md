# Kubernetes paper trading profile

The paper trading environment mirrors the production stack while keeping
non-destructive defaults. The new Kustomize overlay `k8s/overlays/paper`
packages runtime tuning, sealed secrets, and replay automation so the roadmap's
paper-ready gate can be exercised without bespoke YAML edits.

## Prerequisites

1. Install the Bitnami SealedSecrets controller in the target cluster.
2. Install KEDA (or remove the ScaledJob prior to apply).
3. Ensure the shared storage classes backing `emp-reports-pvc` exist in the
   cluster. The replay jobs persist their artefacts on this claim.

## Sealed secrets workflow

1. Create a temporary secret with the required keys:
   ```bash
   kubectl create secret generic emp-secrets-paper \
     --namespace emp-system-paper \
     --from-literal=POSTGRES_PASSWORD='s3cret' \
     --from-literal=REDIS_PASSWORD='redis-pass' \
     --from-literal=API_KEY='paper-api-key'
   ```
2. Pipe it through `kubeseal` using the controller public key:
   ```bash
   kubeseal --format yaml \
     --namespace emp-system-paper \
     --name emp-secrets-paper \
     < secret.yaml > k8s/overlays/paper/sealedsecret.yaml
   ```
3. Delete the temporary secret and commit the sealed secret to source control.

The placeholder ciphertext committed in the repository is intentionally invalid;
`kubeseal` must be re-run before deploying to a real cluster.

## Replay automation

- `emp-nightly-replay` runs daily at 03:00 UTC with a `--min-confidence 0.65`
  override to align with the paper governance thresholds.
- `emp-replay-autoscaler` scales between 1 and 8 workers as Redis list
  `emp:replay:paper` grows. Populate the list with pending evaluation payloads
  (JSON describing datasets, ledger overrides, etc.) to fan out replay runs
  without manual intervention. Each run writes its evidence bundle to
  `/app/artifacts/nightly_replay` on the shared reports PVC, which is surfaced
  to compliance dashboards.

## Deployment checklist

1. Regenerate the sealed secret and commit it.
2. Render the manifests and review diff:
   ```bash
   kustomize build k8s/overlays/paper > /tmp/paper.yaml
   kubectl diff -f /tmp/paper.yaml || true
   ```
3. Apply to the cluster:
   ```bash
   kubectl apply -f /tmp/paper.yaml
   ```
4. Confirm resources:
   ```bash
   kubectl get deployments,cronjobs,scaledjobs -n emp-system-paper
   kubectl logs job/<replay-job> -n emp-system-paper
   ```
5. Validate artefacts land under the shared reports bucket and that governance
   evidence references the `emp-system-paper` namespace.

## Rollback

- Delete the ScaledJob if replay traffic needs to be halted (`kubectl delete
  scaledjob emp-replay-autoscaler-paper -n emp-system-paper`).
- Scale the deployment back to zero replicas for maintenance windows.
- Re-apply the previous sealed secret from git history if credentials need to be
  rolled back.

The profile aligns with the operational readiness brief by ensuring the paper
stack shares deterministic automation and secrets management with production
while remaining isolated from live capital.
