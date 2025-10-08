from __future__ import annotations

import os
import time

import pytest

from src.governance.policy_ledger import PolicyLedgerStage, PolicyLedgerStore


def test_policy_ledger_store_upsert_times_out_when_lock_held(tmp_path) -> None:
    ledger_path = tmp_path / "ledger.json"
    store = PolicyLedgerStore(ledger_path, lock_timeout=0.1, stale_lock_timeout=10.0)

    lock_ctx = store._exclusive_lock()
    lock_ctx.__enter__()
    try:
        with pytest.raises(TimeoutError):
            store.upsert(policy_id="alpha", tactic_id="t-1", stage=PolicyLedgerStage.EXPERIMENT)
    finally:
        lock_ctx.__exit__(None, None, None)


def test_policy_ledger_store_recovers_from_stale_lock(tmp_path) -> None:
    ledger_path = tmp_path / "ledger.json"
    store = PolicyLedgerStore(ledger_path, lock_timeout=0.2, stale_lock_timeout=0.05)

    lock_path = store._lock_path()
    lock_path.write_text("stale")
    stale_time = time.time() - 1
    os.utime(lock_path, (stale_time, stale_time))

    store.upsert(policy_id="alpha", tactic_id="t-1", stage=PolicyLedgerStage.EXPERIMENT)

    assert not lock_path.exists()
    assert store.get("alpha") is not None


def test_policy_ledger_store_merges_sequential_writers(tmp_path) -> None:
    ledger_path = tmp_path / "ledger.json"
    store_one = PolicyLedgerStore(ledger_path)
    store_two = PolicyLedgerStore(ledger_path)

    store_one.upsert(policy_id="alpha", tactic_id="t-1", stage=PolicyLedgerStage.EXPERIMENT)
    store_two.upsert(policy_id="beta", tactic_id="t-2", stage=PolicyLedgerStage.PAPER)

    refreshed = PolicyLedgerStore(ledger_path)
    records = {record.policy_id: record.stage for record in refreshed.iter_records()}

    assert records == {
        "alpha": PolicyLedgerStage.EXPERIMENT,
        "beta": PolicyLedgerStage.PAPER,
    }
