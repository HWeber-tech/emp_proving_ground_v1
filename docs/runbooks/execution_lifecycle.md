# Execution Lifecycle Runbook

## Overview

This runbook maps the end-to-end execution lifecycle implemented in the
repository to the EMP Encyclopedia (Chapters 10 & 24). It documents the
components responsible for order state transitions, persistence, and
position accounting so operators can troubleshoot paper-trading and live
simulation flows.

## Architectural Components

1. **FIX Broker Interface (`src/trading/integration/fix_broker_interface.py`)**
   - Normalises execution reports, order cancel rejects, and propagates
     structured callbacks for order lifecycle events.
   - Emits canonical payloads for `acknowledged`, `partial_fill`, `filled`,
     `cancelled`, and `rejected` events.
2. **Order Lifecycle Processor (`src/trading/order_management/lifecycle_processor.py`)**
   - Bridges FIX callbacks to the deterministic state machine.
   - Persists execution outcomes into the append-only order event journal.
   - Updates the shared `PositionTracker` for real-time exposure & PnL.
3. **Order State Machine (`src/trading/order_management/order_state_machine.py`)**
   - Enforces Chapter 10 lifecycle parity
     (`New → Acknowledged → Partially Filled → Filled/Cancelled/Rejected`).
   - Guards against overfills, invalid regressions, and ensures quantity
     conservation.
4. **Position Tracker (`src/trading/order_management/position_tracker.py`)**
   - Maintains FIFO/LIFO inventory, realised and unrealised PnL, and
     account-level exposure.
5. **Order Event Journal (`data_foundation/events/order_events.parquet`)**
   - Append-only journal for replay, audit parity, and dry-run tooling.
6. **PnL & Exposure Dashboard (`scripts/pnl_exposure_dashboard.py`)**
   - Replays the journal to present snapshot exposure totals per Chapter 24
     operational guidelines.
7. **Fix Integration Pilot (`src/runtime/fix_pilot.py`)**
   - Automatically attaches the lifecycle processor and position tracker to the
     FIX broker interface.
   - Publishes aggregated open order, exposure, and journal metadata for
     operational telemetry.

## Sequence Diagram

```
Client Strategy
    |
    | (1) submit_order
    v
FIXBrokerInterface --(2)--> FIX Venue
    ^                        |
    | (3) execution_report   |
    |                        v
    +--> OrderLifecycleProcessor
            |
            |--(4)--> OrderStateMachine.validate_transition
            |
            |--(5)--> OrderEventJournal.append
            |
            |--(6)--> PositionTracker.record_fill
            v
    Monitoring / Dashboard (PnL, Exposure, Alerts)
```

**Legend**
1. Strategy submits an order through `FIXBrokerInterface.place_market_order`.
2. FIX interface transmits NewOrderSingle to the venue.
3. Venue responds with execution reports (ack/fill/cancel/reject).
4. Lifecycle processor normalises payloads into `OrderExecutionEvent` and
   enforces state transition rules.
5. Journal receives the immutable event + snapshot pair for replay and
   compliance review.
6. Position tracker updates inventory and PnL, exposing totals to the
   dashboard and risk engines.

## Operational Playbook

1. **Order Intake Validation**
   - Confirm the order is registered via `OrderLifecycleProcessor.state_machine.has_order`.
   - Use `order_lifecycle_dry_run.py` to replay suspect FIX logs before
     forwarding to live services.
2. **Execution Monitoring**
   - Tail the JSONL fallback journal (`order_events.parquet.jsonl`) during
     paper trading to validate event continuity.
   - Run `scripts/pnl_exposure_dashboard.py` after major sessions to verify
     exposures align with broker statements.
3. **Reconciliation**
   - Invoke `PositionTracker.generate_reconciliation_report` nightly using
     broker-provided balances.
   - Run `scripts/reconcile_positions.py --broker nightly_broker.json` to
     replay the journal and emit discrepancies for operator review.
   - Investigate discrepancies surfaced in the dashboard totals and journal
     events.
4. **Incident Response**
   - For malformed or rejected events, inspect the journal dead-letter log
     (`order_events.parquet.deadletter.jsonl`).
   - Cross-reference sequence numbers/time-stamps with FIX logs to identify
     root causes per Chapter 24 guidance.

## Exit Criteria Alignment

- 100% of FIX lifecycle events emit journal entries with symbol, side,
  account, and price context.
- `PositionTracker` snapshots reflect real-time filled quantity, realised
  PnL, and exposure suitable for risk guardrails.
- Dashboard output and reconciliation reports provide auditable artefacts
  for nightly operational reviews.
