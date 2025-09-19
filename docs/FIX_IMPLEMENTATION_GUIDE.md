# EMP v4.0 FIX Implementation Guide

## Overview
The EMP proving ground now routes FIX connectivity through a compatibility layer that can speak to a genuine IC Markets implementation when available but defaults to a fully scripted simulator. `FIXConnectionManager` owns session start-up, queue bridging, and initiator helpers, while `MockFIXManager` provides configurable market data and order lifecycles for regression workloads. Shared typing lives in `src/operational/fix_types.py` so downstream components can remain agnostic to whether the simulator or a genuine bridge is driving callbacks.【F:src/operational/fix_connection_manager.py†L162-L360】【F:src/operational/mock_fix.py†L495-L756】【F:src/operational/fix_types.py†L1-L106】

## Architecture
### Components
1. **MockFIXManager** – emits scripted market data, configurable execution plans, identifier sequencing, and manual lifecycle hooks so tests can exercise every branch of order processing without real connectivity.【F:src/operational/mock_fix.py†L495-L756】【F:src/operational/mock_fix.py†L2630-L2880】  
2. **FIXConnectionManager** – chooses between the simulator and optional genuine manager, then adapts callbacks into asyncio queues expected by sensory and trading components.【F:src/operational/fix_connection_manager.py†L162-L360】  
3. **FIX typing layer** – `fix_types` defines the market-data entries, execution payloads, and manager protocols shared between both implementations.【F:src/operational/fix_types.py†L1-L106】  
4. **Optional genuine bridge** – if the real IC Markets manager/config modules import successfully, the connection manager will instantiate them when credentials are present; otherwise the simulator remains in control.【F:src/operational/fix_connection_manager.py†L40-L118】【F:src/operational/fix_connection_manager.py†L172-L264】

### Message Flow
- Market data callbacks are re-encoded into FIX-style snapshot messages (`35=W`) and pushed into an asyncio queue via `_FIXApplicationAdapter` for consumers such as the sensory organ.【F:src/operational/fix_connection_manager.py†L120-L319】  
- Order callbacks are translated into ExecutionReport dictionaries (`35=8`) carrying identifiers, quantities, pricing, metadata overrides, and timestamps so downstream pipelines see realistic FIX tags.【F:src/operational/fix_connection_manager.py†L320-L518】

## Session Selection & Environment Flags
`FIXConnectionManager.start_sessions()` inspects environment variables and configuration to decide whether to run the mock. Setting `EMP_USE_MOCK_FIX=1` forces the simulator; otherwise the presence of both price and trade credentials triggers the genuine implementation when the optional modules are importable.【F:src/operational/fix_connection_manager.py†L172-L264】 When the simulator is chosen, any default metadata exposed by the provided config object (account, order type, commissions, settlement, capacity) is forwarded to the mock so generated ExecutionReports inherit those defaults.【F:src/operational/fix_connection_manager.py†L200-L263】【F:tests/current/test_fix_smoke.py†L88-L143】

## Quick Start
1. **Instantiate and start the manager**
   ```python
   from src.operational.fix_connection_manager import FIXConnectionManager

   class Config:
       environment = "test"
       account_number = "000"
       password = "demo"

   manager = FIXConnectionManager(Config())
   assert manager.start_sessions()
   ```
   The example above mirrors the smoke and lifecycle tests that drive the simulator when credentials are absent.【F:tests/current/test_fix_smoke.py†L17-L60】【F:tests/current/test_fix_lifecycle.py†L26-L45】

2. **Attach queues and send orders**
   ```python
   import asyncio

   trade_q = asyncio.Queue()
   price_q = asyncio.Queue()

   manager.get_application("trade").set_message_queue(trade_q)
   manager.get_application("price").set_message_queue(price_q)

   initiator = manager.get_initiator("trade")

   class Order:
       cl_ord_id = "ABC123"
       quantity = 3.0
       price = 1.5
       account = "SIM"
       order_type = "2"
       time_in_force = "1"

   assert initiator.send_message(Order())
   ```
   Downstream code can `await` the queues to receive ExecutionReports and market data snapshots emitted by the simulator.【F:tests/current/test_fix_smoke.py†L31-L84】【F:tests/current/test_fix_lifecycle.py†L47-L159】

3. **Stop sessions when finished**
   ```python
   manager.stop_sessions()
   ```
   Stopping resets queue metrics and clears the cached manager/initiator handles.【F:src/operational/fix_connection_manager.py†L518-L547】

## Driving the Simulator Directly
For unit tests that need deterministic control, instantiate `MockFIXManager` without the connection manager:
```python
from types import SimpleNamespace
from src.operational.mock_fix import MockFIXManager, MockExecutionStep

manager = MockFIXManager(symbol="GBPUSD", execution_interval=0.0, synchronous_order_flows=True)
manager.add_order_callback(lambda info: print(info.exec_id))
manager.start()
manager.trade_connection.send_message_and_track(
    SimpleNamespace(
        cl_ord_id="ORD1",
        quantity=4.0,
        price=1.2345,
        mock_execution_plan=[
            MockExecutionStep("1", quantity=2.0, delay=0.0),
            MockExecutionStep("F", quantity=2.0, delay=0.0),
        ],
    )
)
manager.wait_for_idle(timeout=1.0)
```
The simulator supports configurable market data plans, overrideable order defaults, identifier sequencing, manual emissions, and synchronous execution for deterministic regression flows.【F:src/operational/mock_fix.py†L2630-L2880】【F:tests/current/test_mock_fix.py†L17-L159】

## Telemetry & Introspection
Use the following helpers to assert behaviour after scripted lifecycles:
- `snapshot_telemetry()` – returns recorded events such as `market_data_snapshot`, `order_execution`, and `order_complete` for assertion-friendly inspection.【F:src/operational/mock_fix.py†L2703-L2834】  
- `get_order_history(cl_ord_id)` / `get_last_order_info(cl_ord_id)` – provide cloned execution payloads and latest order info snapshots so tests can validate cumulative quantities, leaves, identifiers, and metadata without mutating internal state.【F:src/operational/mock_fix.py†L1834-L1852】【F:tests/current/test_mock_fix.py†L100-L145】

## Troubleshooting
- **Simulator fails to start** – The connection manager logs an error and returns `False` if the injected manager cannot start sessions. Unit tests cover this path so harnesses can assert the fallback behaviour.【F:tests/current/test_fix_manager_failures.py†L15-L37】  
- **No trade connection** – Sending through the initiator without an initialized trade connection logs an error and returns `False`, preventing silent drops.【F:src/operational/fix_connection_manager.py†L148-L160】【F:tests/current/test_fix_manager_failures.py†L40-L46】  
- **Queue backpressure** – `_FIXApplicationAdapter` tracks delivered/dropped counts and logs when queues are full; ensure downstream consumers drain messages promptly.【F:src/operational/fix_connection_manager.py†L120-L145】

## Next Steps
Integrating a production FIX flow requires replacing the simulator with the genuine manager once credentials and modules are available, then extending regression coverage to include broker round-trips. Until that happens, rely on the simulator to validate strategy plumbing, order lifecycle handling, and telemetry instrumentation without needing external connectivity.【F:src/operational/fix_connection_manager.py†L40-L264】【F:tests/current/test_fix_smoke.py†L17-L143】
