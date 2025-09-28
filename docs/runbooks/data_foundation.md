# Data Foundation Runbook

## Reference Data Loader Overview

The reference data loader hydrates instruments, trading sessions, and holiday calendars
from the JSON manifests stored in `config/reference_data`. This satisfies the Phase 1D
roadmap requirement for Tier-0 bootstrap datasets that can be hydrated without manual
intervention.

### Files

| File | Purpose |
| --- | --- |
| `config/reference_data/instruments.json` | Canonical instrument catalogue with venue, tick size, and session mapping. |
| `config/reference_data/sessions.json` | Trading sessions with timezone aware open/close windows and trading days. |
| `config/reference_data/holidays.json` | Venue-aware holiday calendars used to halt trading when markets are closed. |

### Python API

```python
from src.data_foundation.reference import ReferenceDataLoader

loader = ReferenceDataLoader()
loader.instruments()  # -> tuple[InstrumentRecord, ...]
loader.sessions()     # -> tuple[TradingSession, ...]
loader.holidays()     # -> tuple[HolidayRecord, ...]
```

Each accessor returns immutable dataclasses that mirror the encyclopedia's Layer 1
specification. The loader caches results and exposes helpers like `session_for_instrument`
and `is_holiday` so downstream components (e.g. sensors, backtesters) can align their
behaviour with the official calendars.

### Operational Notes

- The JSON manifests are intentionally small and dependency-free so CI environments can
  hydrate reference data without network access.
- Add new instruments or sessions by editing the JSON files and extending the accompanying
  tests in `tests/current/test_reference_data_loader.py`.
- If a venue operates multiple sessions per day, add separate entries with unique
  `session_id` values. The loader's `window_for_date` helper produces timezone-aware
  windows that respect cross-midnight sessions.
