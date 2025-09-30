# Dead Code Cleanup Log

## 2025-10-29

- Removed the deprecated `src/core/strategy/templates` package so legacy
  mean-reversion/momentum stubs no longer shadow the production trading
  strategies.
- Retired `scripts/verify_complete_system.py`, which depended on the deleted
  template shims and no longer reflected the supervised runtime surface.
