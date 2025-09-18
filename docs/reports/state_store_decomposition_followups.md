# State store decomposition follow-ups

## Context

`src/core/state_store.py` now strictly defines the protocol while the in-memory
implementation lives under `src/operational/state_store/`. The next clean-up
passes target legacy helpers that were previously co-located with the protocol.

## Planned work

| Area | Action | Notes |
| --- | --- | --- |
| `src/core/performance/` | Delete `market_data_cache` adapters that proxy to the operational state store once consumers switch to the new registry helpers. | Coordinate with performance analytics owners before removing the shim. |
| `src/core/performance/` | Move `performance/state_cache.py` to an operational package or drop in favour of the event-bus backed cache. | Requires confirming no orchestration modules import the legacy helper. |
| `src/core/risk/` | Remove deprecated risk cache utilities that duplicated `RiskManagerImpl` storage. | Replace imports with the protocol from `src.core.state_store`. |
| `src/core/risk/` | Collapse unused state snapshots exported solely for tests. | Ensure regression suites rely on the operational state store instead. |

## Status tracking

- [ ] Confirm cache shims in `src/core/performance/` are unused and schedule deletions.
- [ ] Update risk modules to consume the operational state store adapter directly.
- [ ] Regenerate the dead-code audit after removals to verify no stale helpers remain.
