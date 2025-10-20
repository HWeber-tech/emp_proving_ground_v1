# Microstructure Verification (OpenAPI) – Retired

The microstructure verification playbook previously targeted the deprecated
OpenAPI integration. The project has standardised on FIX-only connectivity, so
the detailed instructions, scripts, and sample outputs have been removed.

For current verification coverage:

- Use `scripts/verify_fix_connection.py` and related FIX smoke tests.
- Follow the guidance in [`docs/policies/integration_policy.md`](policies/integration_policy.md).
- Coordinate with the FIX API documentation in `docs/fix_api/` for
  environment-specific steps.

The original OpenAPI content is intentionally withheld to prevent confusion
about the supported integration surface.
