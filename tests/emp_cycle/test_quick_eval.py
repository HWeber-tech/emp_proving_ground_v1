from __future__ import annotations

from types import SimpleNamespace

from emp.core import quick_eval


class MockStrategy:
    def __init__(self, profit: float, loss: float, drawdown: float, sharpe: float):
        self.result = SimpleNamespace(
            gross_profit=profit,
            gross_loss=loss,
            max_drawdown=drawdown,
            sharpe=sharpe,
        )

    def backtest(self, data_slice):
        return self.result


def test_quick_eval_metrics_and_threshold():
    strategy = MockStrategy(profit=100.0, loss=40.0, drawdown=-10.0, sharpe=1.5)
    metrics = quick_eval.quick_eval(strategy, data_slice={"days": 10})
    assert metrics["profit_factor"] == 2.5
    assert metrics["max_drawdown_abs"] == 10.0
    assert metrics["sharpe"] == 1.5
    assert quick_eval.passes_quick_threshold(metrics, threshold=0.5)

    failing_strategy = MockStrategy(profit=10.0, loss=80.0, drawdown=-30.0, sharpe=-0.5)
    failing_metrics = quick_eval.quick_eval(failing_strategy, data_slice={})
    assert not quick_eval.passes_quick_threshold(failing_metrics, threshold=1.0)
