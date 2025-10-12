from __future__ import annotations

from emp.cli import emp_cycle


def test_progress_primary_and_risk_gate():
    baseline = {"sharpe": 0.5, "max_dd": 25.0, "winrate": 0.5}
    metrics = {"sharpe": 0.7, "max_dd": 20.0, "winrate": 0.6}
    cfg = emp_cycle.ProgressCfg(primary_metric="sharpe", threshold=None, risk_max_dd=25.0, secondary=[])
    assert emp_cycle._is_progress(metrics, baseline, cfg)

    metrics_bad_dd = {"sharpe": 1.2, "max_dd": 30.0}
    assert not emp_cycle._is_progress(metrics_bad_dd, baseline, cfg)


def test_progress_threshold_and_secondary():
    baseline = {"return": 0.1, "winrate": 0.5}
    metrics = {"return": 0.08, "winrate": 0.6}
    cfg = emp_cycle.ProgressCfg(
        primary_metric="return",
        threshold=0.09,
        risk_max_dd=None,
        secondary=[emp_cycle.SecondaryConstraint(name="winrate", op=">=", value=0.55)],
    )
    assert not emp_cycle._is_progress(metrics, baseline, cfg)

    improved = {"return": 0.11, "winrate": 0.6, "max_dd": 10}
    assert emp_cycle._is_progress(improved, baseline, cfg)
