# EMP Proving Ground v1

EMP Proving Ground is a truth-first research harness for building an algorithmic
trading stack. The repository now centres on a rich **mock FIX simulator and
bridge** so developers can exercise order lifecycles without touching live
brokers. Strategy, risk, and evolution layers remain architectural scaffolding.

## Current Reality

| Area | Reality |
| --- | --- |
| Execution | `MockFIXManager` and `FIXConnectionManager` drive all order flows. No live broker connectivity ships in-tree. |
| Data | Market data is synthetic, published by the mock FIX stack; there are no production market data feeds. |
| Strategies / Risk | Framework classes exist, but the behaviours are placeholders with logging or `pass` blocks. |
| Evolution & Intelligence | Interfaces and feature-flag wiring only; no genetic or adaptive loops run. |
| Observability | Logging, telemetry shims, and status snapshots cover the simulator runtime. Live venue dashboards are not implemented. |

Refer to `docs/DEVELOPMENT_STATUS.md` for the authoritative status ledger. Any
module that talks to real infrastructure is expected to reside in a downstream,
credential-bearing fork.

## Working With the Mock Stack

1. Create a virtual environment and install dependencies: `pip install -r requirements.txt`.
2. Run the regression suite that exercises the FIX simulator: `pytest tests/current`.
3. Launch the runtime with synthetic data (safe to kill at any time):

   ```bash
   python -m main --skip-ingest
   ```

   The process wires the runtime, spins up the mock FIX manager, and streams
   simulated fills into the trading scaffolding for inspection.

## Contributing

- Keep documentation honest about mock versus real capabilities.
- Expand tests around the simulator or clearly flag TODOs when stubbing features.
- Update `docs/DEVELOPMENT_STATUS.md` whenever you graduate a component from mock
  to production-grade behaviour.

This is not a production trading system. Treat the repository as a sandbox until
real integrations replace the mocks.
