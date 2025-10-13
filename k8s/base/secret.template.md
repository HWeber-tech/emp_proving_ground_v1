# Secret template

The base kustomization intentionally omits `emp-secrets` so overlays can supply
credentials through their preferred controller. If you need to stand up a rapid
sandbox without SealedSecrets or SecretGenerator, create a throwaway manifest:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: emp-secrets
  namespace: emp-system
stringData:
  POSTGRES_PASSWORD: changeme
  REDIS_PASSWORD: changeme
  API_KEY: changeme
```

Do not commit the file. Instead, `kubectl apply -f secret.yaml` locally and
rely on the overlay-specific solution (SecretGenerator for `dev`, SealedSecret
for `paper`, CI-managed secret for `prod`) for real environments.
