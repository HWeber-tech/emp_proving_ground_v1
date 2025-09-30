# Kubernetes Deployment Layout

This directory now follows a Kustomize-friendly structure to reduce merge
conflicts and encode environment-specific overrides without duplicating base
manifests.

```
k8s/
├── base/
│   ├── deployment.yaml
│   ├── configmap.yaml
│   ├── ingress.yaml
│   ├── namespace.yaml
│   ├── secret.yaml
│   ├── service.yaml
│   └── kustomization.yaml
├── overlays/
│   ├── dev/
│   │   ├── dev-secrets.env.example
│   │   ├── kustomization.yaml
│   │   ├── patch-configmap.yaml
│   │   ├── patch-deployment.yaml
│   │   ├── patch-ingress.yaml
│   │   └── secret-generator.yaml
│   └── prod/
│       ├── kustomization.yaml
│       ├── patch-configmap.yaml
│       ├── patch-deployment.yaml
│       ├── patch-ingress.yaml
│       ├── prod-secrets.env.example
│       └── secret-generator.yaml
└── emp-deployment.yaml (legacy flat manifest for reference)
```

## Secrets management

The overlays use Kustomize `SecretGenerator` definitions with `.env` files that
**must not be committed**.  Copy the provided `*.env.example` files to
`dev-secrets.env` or `prod-secrets.env`, populate them with values from your
secret store (Vault, Oracle OCI Vault, SOPS, SealedSecrets), and keep them out
of version control.  The generator uses `behavior: replace` so environment
overrides do not leak fallback secrets from the base layer.

For production we recommend replacing the generated secret with a SealedSecret
or external secret controller.  The base manifest intentionally keeps the data
plane minimal for local testing.

## Deploying

```bash
# Development sandbox
kustomize build k8s/overlays/dev | kubectl apply -f -

# Production (requires prod-secrets.env populated locally or injected via CI)
kustomize build k8s/overlays/prod | kubectl apply -f -
```

The legacy `emp-deployment.yaml` file is retained to avoid breaking downstream
automation.  It will be removed once all pipelines migrate to the kustomize
layout.
