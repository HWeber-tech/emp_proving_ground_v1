# Integration Policy

This policy documents the enforced boundaries for third-party market integrations.
It ensures configuration defaults, documentation, and automation stay aligned with
our **FIX-only** execution posture.

## Current stance

| Surface | Status | Notes |
| --- | --- | --- |
| FIX protocol connections (IC Markets) | ✅ Supported | Active implementations live in `config/fix/` and `src/operational/`.
| FIX simulators for testing | ✅ Supported | Default configs point at the simulator to keep local runs safe.
| cTrader OpenAPI / REST | ❌ Forbidden | Blocked by CI and runtime guards. Legacy docs retained only for history.
| FastAPI / OpenAPI powered services | ❌ Forbidden | The `forbidden-integrations` workflow fails if these paths reappear.
| Proprietary broker SDKs without review | 🚫 Requires RFC | Open a design doc and security review before adding new dependencies.

## Enforcement

* [`./scripts/check_forbidden_integrations.sh`](../../scripts/check_forbidden_integrations.sh)
  centralises the keyword scan used by CI.
* Both `ci.yml` and `policy-openapi-block.yml` import the reusable
  [forbidden-integrations workflow](../../.github/workflows/forbidden-integrations.yml)
  so the policy executes consistently in every pipeline.
* Runtime startup enforces FIX-only brokers through `SystemConfig` defaults.

When violations are detected the build fails immediately and the offending paths
are printed in the workflow log.

## Documentation & configuration expectations

* Default configs (`config.yaml`, `.env` templates) must not reference
  prohibited providers outside clearly marked legacy sections.
* Legacy guides that describe the cTrader OpenAPI live in `archive/legacy/` and must
  start with a **Status: Legacy** disclaimer that links back to this policy.
* New documentation should emphasise FIX workflows and explicitly call out any
  reliance on blocked providers as historical context only.

## Contributing changes

Before submitting a change that touches integrations:

1. Confirm the change aligns with the table above.
2. Update relevant documentation or samples to highlight the allowed path.
3. Run `./scripts/check_forbidden_integrations.sh` locally.
4. Include rationale in the PR description if you are proposing an exception.

Requests for new third-party integrations require a lightweight RFC that
covers security review, failure modes, and rollback plans. Exceptions are rare
and must be signed off by the maintainers before implementation work begins.
