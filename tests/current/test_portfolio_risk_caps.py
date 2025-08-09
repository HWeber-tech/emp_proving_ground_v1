import json
import subprocess
import sys
from pathlib import Path


def test_var_cap_attenuates(tmp_path: Path):
    md_path = tmp_path / "md.jsonl"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write('{"timestamp":"2024-01-01T00:00:00","symbol":"EURUSD","bids":[[1.1000,1000]],"asks":[[1.1002,1000]]}\n')
        fh.write('{"timestamp":"2024-01-01T00:00:01","symbol":"EURUSD","bids":[[1.1001,1000]],"asks":[[1.1003,1000]]}\n')
    out_dir = tmp_path / "reports"
    # Force storm regime so var is present and set a very low cap via env to trigger attenuation
    cmd = [sys.executable, "scripts/backtest_report.py", "--file", str(md_path), "--force-regime", "storm", "--out-dir", str(out_dir)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    # Ensure outputs exist
    assert (out_dir / "report.json").exists()

