# Kubernetes Deployment Layout

The manifests follow a Kustomize-first structure so environment-specific
configuration, sealed secrets, and replay automation can be added without
forking the base definitions.

```
k8s/
├── base/
│   ├── configmap.yaml
│   ├── deployment.yaml
│   ├── ingress.yaml
│   ├── kustomization.yaml
│   ├── namespace.yaml
│   ├── replay-cronjob.yaml
│   ├── replay-scaledjob.yaml
│   ├── replay-triggerauth.yaml
│   ├── service.yaml
│   ├── serviceaccount.yaml
│   └── (secret.template.md) ← guidance only, secrets are provided per overlay
├── overlays/
│   ├── dev/
│   │   ├── dev-secrets.env.example
│   │   ├── kustomization.yaml
│   │   ├── patch-configmap.yaml
│   │   ├── patch-cronjob.yaml
│   │   ├── patch-deployment.yaml
│   │   ├── patch-ingress.yaml
│   │   ├── patch-scaledjob.yaml
│   │   └── secret-generator.yaml
│   ├── paper/
│   │   ├── kustomization.yaml
│   │   ├── patch-configmap.yaml
│   │   ├── patch-cronjob.yaml
│   │   ├── patch-deployment.yaml
│   │   ├── patch-ingress.yaml
│   │   ├── patch-scaledjob.yaml
│   │   └── sealedsecret.yaml (placeholder — regenerate with `kubeseal`)
│   └── prod/
│       ├── kustomization.yaml
│       ├── patch-configmap.yaml
│       ├── patch-cronjob.yaml
│       ├── patch-deployment.yaml
│       ├── patch-ingress.yaml
│       ├── patch-scaledjob.yaml
│       ├── prod-secrets.env.example
│       └── secret-generator.yaml
└── emp-deployment.yaml (legacy flat manifest retained for downstream tooling)
```

## Replay automation

`base/` now ships a nightly CronJob (`emp-nightly-replay`) and a KEDA `ScaledJob`
(`emp-replay-autoscaler`) that launches replay workers when the Redis backlog
(`emp:replay:*`) grows. Each overlay retunes image tags, polling thresholds, and
resource envelopes so dev, paper, and production can scale independently.

- The CronJob writes artefacts under `/app/artifacts/nightly_replay` mounted on
  the reports PVC so governance reviews can consume the evidence packs.
- The ScaledJob uses the shared `emp-replay-runner` service account and
  `TriggerAuthentication` bound to `emp-secrets` for Redis credentials. Install
  the KEDA operator in the target cluster before applying the manifests.

## Secrets management

- **Development** continues to rely on a local `.env` that is transformed into a
  Kubernetes `Secret` via `SecretGenerator`. Copy `dev-secrets.env.example` to
  `dev-secrets.env`, fill in credentials, and keep the file untracked.
- **Paper** and **production** expect Bitnami SealedSecrets. The committed
  manifests contain obvious placeholders (`AgAAAA...`). Replace them with cluster
  specific ciphertext by running `kubeseal --format yaml` against a temporary
  Secret that holds the real values. The generated file should be committed in
  place of the placeholder to keep paper/prod in sync.
- The base layer no longer includes a plaintext secret; all references resolve to
  environment overlays so we avoid accidental credential drift.

## Deploying

```bash
# Development sandbox (local credentials via SecretGenerator)
kustomize build k8s/overlays/dev | kubectl apply -f -

# Paper trading stack (requires sealed secret generated with the paper controller key)
kustomize build k8s/overlays/paper | kubectl apply -f -

# Production (replace prod-secrets.env with your secure workflow or sealed secret)
kustomize build k8s/overlays/prod | kubectl apply -f -
```

If you cannot install KEDA in an environment, remove `replay-scaledjob.yaml`
from the rendered manifest by deleting it from the overlay or leveraging
`kustomize build ... | yq 'del(select(.kind == "ScaledJob"))'` during apply.

The legacy single-file manifest remains for older automation and will be removed
once consumers migrate to the layered layout.
