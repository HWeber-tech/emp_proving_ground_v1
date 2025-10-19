# EMP Proving Ground v1

EMP Proving Ground is a truth-first research harness for building an algorithmic
trading stack. The public build ships a **mock FIX simulator and bridge** so
developers can exercise order lifecycles without touching live brokers,
alongside an opt-in Tier-0 ingest that pulls real Yahoo Finance bars into DuckDB
for sensors and strategy prototypes. Strategy and evolution layers are still
under active development, but risk policies, portfolio monitoring, and
paper/live broker adapters are implemented and wired for research mode.

## Current Reality

| Area | Reality |
| --- | --- |
| Execution | **Defaults to mock.** `MockFIXManager` and `FIXConnectionManager` drive order flows. A `LiveBrokerExecutionAdapter` ships for downstream deployments but requires external broker modules and credentials, so public runs stay in paper/sim mode. |
| Market data (runtime) | **Mock.** Quote and fill events originate from the simulator; no streaming venue connectivity is active in the public repo. |
| Market data (ingest) | **Real, opt-in.** Tier-0 ingest fetches Yahoo Finance daily/intraday bars (`fetch_daily_bars`, `fetch_intraday_trades`) and writes them to DuckDB. Running `python -m main` without `--skip-ingest` will hit Yahoo's APIs. |
| Institutional data backbone | **Feature-flagged, secrets required.** Timescale/Redis/Kafka connectors ship in this repo but stay inert without credentials; default runs fall back to the Tier-0 DuckDB/Yahoo path. |
| Strategies / Risk | **Scaffolding.** Framework classes exist, but behaviours are placeholders with logging or `pass` blocks. |
| Evolution & Intelligence | **Scaffolding.** Interfaces and feature-flag wiring only; no genetic or adaptive loops run. |
| Observability | **Partial.** Structured logging, telemetry shims, and status snapshots cover the simulator, ingest runtime, and risk adapters. Live venue dashboards are not implemented. |

Refer to `docs/DEVELOPMENT_STATUS.md` for the authoritative status ledger. Any
module that talks to real infrastructure is expected to reside in a downstream,
credential-bearing fork.

## Working With the Mock Stack

1. Create a virtual environment and install dependencies: `pip install -r requirements.txt`.
2. Run the regression suite that exercises the FIX simulator: `pytest tests/current`.
3. (Optional, **real data**) Launch Tier-0 ingest + runtime:

   ```bash
   python -m main
   ```

   This fetches Yahoo Finance bars over the network, stores them in DuckDB, and
   then wires the runtime. Supply `--skip-ingest` to avoid the network call.
4. Launch the runtime with synthetic order flow only (safe to kill at any time):

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
