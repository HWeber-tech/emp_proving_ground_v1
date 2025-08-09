import os
import json
from datetime import datetime, timedelta


def write_jsonl(path, events):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")


def test_backtest_runs_minimal(tmp_path):
    md_path = tmp_path / "md.jsonl"
    macro_path = tmp_path / "macro.jsonl"
    yields_path = tmp_path / "yields.jsonl"

    now = datetime.utcnow()
    # Minimal MD: two snapshots
    md_events = [
        {"timestamp": (now).isoformat(), "symbol": "EURUSD", "bids": [[1.1000, 1000]], "asks": [[1.1002, 1000]]},
        {"timestamp": (now + timedelta(seconds=1)).isoformat(), "symbol": "EURUSD", "bids": [[1.1001, 1000]], "asks": [[1.1003, 1000]]},
    ]
    # Macro: one event
    macro_events = [
        {"timestamp": (now + timedelta(minutes=10)).isoformat(), "calendar": "unit", "event": "CPI"}
    ]
    # Yields: two tenors
    yield_events = [
        {"timestamp": now.isoformat(), "curve": "UST", "tenor": "2Y", "value": 4.50},
        {"timestamp": now.isoformat(), "curve": "UST", "tenor": "10Y", "value": 4.65},
    ]

    write_jsonl(str(md_path), md_events)
    write_jsonl(str(macro_path), macro_events)
    write_jsonl(str(yields_path), yield_events)

    out_dir = tmp_path / "reports"

    import subprocess, sys
    cmd = [sys.executable, "scripts/backtest_report.py", "--file", str(md_path), "--macro-file", str(macro_path), "--yields-file", str(yields_path), "--out-dir", str(out_dir)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    # Check outputs exist
    assert (out_dir / "report.json").exists()
    assert (out_dir / "BACKTEST_SUMMARY.md").exists()


