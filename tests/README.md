Test scopes
===========

- tests/current: maintained, green scope for ongoing development (FIX mock, backtest CLI, sensory layer features)
- tests/legacy: quarantined suites targeting deprecated modules (e.g., cTrader OpenAPI, old sensory paths). Not executed by default.

Use `pytest -q` to run current scope. To run all, temporarily set `testpaths=tests` in `pytest.ini` or pass an explicit path.

