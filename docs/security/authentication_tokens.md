# Authentication Tokens

Authentication to operator-facing endpoints (runtime health, metrics, or
internal tooling) relies on short-lived bearer tokens signed with a shared
HMAC secret. This guide covers configuring the secret, issuing tokens with the
required roles, validating the payload, and presenting the token to guarded
services.

## Configure the shared secret and roles

- The runtime builder refuses to start health endpoints unless
  `RUNTIME_HEALTHCHECK_AUTH_SECRET` is populated in the system extras. Update
your deployment profile (`config/deployment/runtime_*.yaml`) or container
environment so the runtime and any issuance scripts share the same, randomly
generated secret (32+ bytes recommended).【F:src/runtime/runtime_builder.py#L5145】【F:config/deployment/runtime_dev.yaml#L24】
- Default role requirements are `runtime.health:read` for `/health` and both
  `runtime.health:read` + `runtime.metrics:read` for `/metrics`. Override via
  `RUNTIME_HEALTHCHECK_HEALTH_ROLES` or `RUNTIME_HEALTHCHECK_METRICS_ROLES`
  when a different role model is required.【F:src/runtime/runtime_builder.py#L5164】【F:src/runtime/healthcheck.py#L605】
- If you enable an audience check with `RUNTIME_HEALTHCHECK_TOKEN_AUDIENCE`,
  the issued token must embed a matching `aud` claim.【F:src/runtime/runtime_builder.py#L5171】【F:src/security/auth_tokens.py#L88】

## Issue a runtime access token

Use the `create_access_token` helper to mint HMAC-SHA256 bearer tokens with
custom claims and expirations.【F:src/security/auth_tokens.py#L69】 The script 
below emits a token valid for one hour with health and metrics permissions:

```bash
export RUNTIME_HEALTHCHECK_AUTH_SECRET="$(pass runtime/health-secret)"  # replace with your secret source
python - <<'PY'
from datetime import timedelta
import os
from src.security import create_access_token

secret = os.environ["RUNTIME_HEALTHCHECK_AUTH_SECRET"]
token = create_access_token(
    subject="ops-engineer",
    secret=secret,
    roles=["runtime.health:read", "runtime.metrics:read"],
    expires_in=timedelta(hours=1),
    audience="runtime-health",
)
print(token)
PY
```

Key parameters:
- `subject` identifies the operator or automation account.
- `roles` must cover every endpoint you plan to call; duplicate or empty values
  are stripped automatically.【F:src/security/auth_tokens.py#L91】
- `expires_in` should stay short (minutes or hours). Tokens inherit the runtime
  clock and expire once the embedded epoch passes.【F:src/security/auth_tokens.py#L94】
- `audience` is optional unless your deployment opts in via 
  `RUNTIME_HEALTHCHECK_TOKEN_AUDIENCE`.

## Validate and inspect a token

Before distributing a token, decode it with the same helper to confirm the
claims and expiry window:

```bash
export TOKEN="$GENERATED_TOKEN"
python - <<'PY'
import os
from pprint import pprint
from src.security import decode_access_token

secret = os.environ["RUNTIME_HEALTHCHECK_AUTH_SECRET"]
pprint(decode_access_token(token=os.environ["TOKEN"], secret=secret))
PY
```

Export `TOKEN` to point at the bearer value you just minted before invoking the
snippet. A successful
decode logs "Validated access token" and returns the payload so you can audit
roles and audience claims.【F:src/security/auth_tokens.py#L128】 Failed
verification raises `InvalidTokenError` or `ExpiredTokenError`, matching the
runtime's behaviour when a request arrives with a bad bearer header.【F:src/runtime/healthcheck.py#L624】

## Call protected endpoints

Pass the token in the `Authorization` header when querying the runtime:

```bash
TOKEN="$(python issue_runtime_token.py)"  # replace with the issuance command you use
curl \
  --header "Authorization: Bearer ${TOKEN}" \
  --header "Accept: application/json" \
  https://runtime-host:8000/health

curl \
  --header "Authorization: Bearer ${TOKEN}" \
  https://runtime-host:8000/metrics
```

Use `--cacert` (or `-k` only in disposable lab environments) so TLS validation
succeeds. A `401` response indicates a missing/invalid token, while a `403`
means the token is valid but lacks the required role.【F:src/runtime/healthcheck.py#L611】

## Rotation, revocation, and audit

- Rotate `RUNTIME_HEALTHCHECK_AUTH_SECRET` on the same cadence as other
  infrastructure secrets; changing the secret instantly invalidates every
  existing token.【F:src/runtime/runtime_builder.py#L5158】
- For finer-grained control, use the governance `TokenManager` to mint tokens
  with per-user metadata and explicit revocation. `TokenManager.revoke_token`
  flags specific tokens while leaving the shared secret untouched, and logs the
  revocation for audit trails.【F:src/governance/token_manager.py#L124】【F:src/governance/token_manager.py#L170】
- Never commit tokens or secrets to source control. Keep issuance scripts in
  operational runbooks and ensure logs redact the bearer values (only hashes are
  recorded by the helpers).【F:src/security/auth_tokens.py#L137】
