# Dead-code backlog triage snapshot (2025-09-22)

This snapshot captures the current classification for the first 100 candidates
called out in `docs/reports/CLEANUP_REPORT.md`.  Use the updated
`tools.cleanup.dead_code_tracker` helpers to regenerate the view as new cleanup
passes land.

## Status breakdown

Command: `python tools/cleanup/dead_code_tracker.py --report docs/reports/CLEANUP_REPORT.md`

- Total candidates scanned: 100
- Present on disk: 63
- Missing/retired: 37
- Shim exports still surfacing legacy imports: 4
- Module-not-found stubs providing migration guidance: 8
- Active implementations that still need ownership decisions: 51

## Notes

- Missing entries already resolved during earlier cleanup batches and only
  linger in the published audit list.
- Module-not-found stubs actively block legacy imports while giving developers a
  guided redirect; once downstream repositories complete their migrations, these
  files can be deleted outright.
- Shim exports (`src/core/event_bus.py`, `src/core/exceptions.py`,
  `src/core/sensory_organ.py`, `src/governance/strategy_registry.py`) serve as
  canonical facades.  They should not be removed until the dependency map shows
  zero consumers pointing at the shim path.
- Anything labelled `active` should be reviewed with respective domain owners to
  confirm whether the implementation stays, moves under a canonical package, or
  can be sunset.

Re-run the tracker after each deletion batch and commit the refreshed snapshot to
keep the roadmap evidence aligned with reality.
