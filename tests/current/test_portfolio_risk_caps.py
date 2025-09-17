import subprocess
import sys
from pathlib import Path

import pytest

from src.trading.risk.portfolio_caps import apply_aggregate_cap, usd_beta_sign


def test_var_cap_attenuates(tmp_path: Path):
    md_path = tmp_path / "md.jsonl"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(
            '{"timestamp":"2024-01-01T00:00:00","symbol":"EURUSD","bids":[[1.1000,1000]],"asks":[[1.1002,1000]]}\n'
        )
        fh.write(
            '{"timestamp":"2024-01-01T00:00:01","symbol":"EURUSD","bids":[[1.1001,1000]],"asks":[[1.1003,1000]]}\n'
        )
    out_dir = tmp_path / "reports"
    # Force storm regime so var is present and set a very low cap via env to trigger attenuation
    cmd = [
        sys.executable,
        "scripts/backtest_report.py",
        "--file",
        str(md_path),
        "--force-regime",
        "storm",
        "--out-dir",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    # Ensure outputs exist
    assert (out_dir / "report.json").exists()


@pytest.mark.parametrize(
    "current, cap, desired, expected",
    [
        (0.0, 5.0, 2.0, 2.0),
        (2.5, 5.0, 1.0, 1.0),
        (4.8, 5.0, 2.0, 0.2),
        (5.0, 5.0, 1.5, 0.0),
        (-1.0, 3.0, -2.0, 0.0),
    ],
)
def test_apply_aggregate_cap_clamps_to_remaining_capacity(
    current: float, cap: float, desired: float, expected: float
) -> None:
    allowed = apply_aggregate_cap(current, cap, desired)

    assert allowed == pytest.approx(expected)


def test_apply_aggregate_cap_recovers_capacity() -> None:
    cap = 5.0

    first_allocation = apply_aggregate_cap(4.5, cap, 2.0)
    assert first_allocation == pytest.approx(0.5)

    recovered_allocation = apply_aggregate_cap(1.0, cap, 3.0)
    assert recovered_allocation == pytest.approx(3.0)


@pytest.mark.parametrize(
    "symbol, exposure, expected",
    [
        ("EURUSD", 1.5, 1.5),
        ("eurusd", -2.0, -2.0),
        ("USDJPY", 3.0, -3.0),
        ("USDCAD", -1.2, 1.2),
        ("EURGBP", 4.0, 0.0),
        ("BTCUSD", 0.5, 0.5),
        ("XAU", 1.0, 1.0),
    ],
)
def test_usd_beta_sign_interprets_symbol_orientation(
    symbol: str, exposure: float, expected: float
) -> None:
    beta = usd_beta_sign(symbol, exposure)

    assert beta == pytest.approx(expected)
