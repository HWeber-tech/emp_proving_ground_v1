# Microstructure Verification README (Legacy)

This README used to walk through the OpenAPI-based verification scripts. Those
scripts have been retired alongside the connectivity stack they exercised.

To validate market data and execution flows today:

1. Configure the FIX environment as outlined in `docs/fix_api/`.
2. Run the FIX smoke tests (`scripts/verify_fix_connection.py`,
   `scripts/verify_fix_sensory_integration.py`, etc.).
3. Capture results in the FIX reporting templates under `docs/fix_api/`.

The removed content is intentionally excluded to avoid implying continued
support for the legacy OpenAPI integration.
