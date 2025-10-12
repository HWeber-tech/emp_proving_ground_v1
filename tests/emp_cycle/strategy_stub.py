"""Test-only strategy stub for the experimentation cycle smoke tests."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict


class StubStrategy:
    def __init__(self, params: Dict[str, Any]):
        self.params = params

    def _scale(self) -> float:
        weight = float(self.params.get("weight", 1.0))
        return max(0.2, min(2.0, weight))

    def backtest(self, data_slice: Any) -> Any:
        scale = self._scale()
        gross_profit = 120.0 * scale
        gross_loss = 40.0 * scale
        max_drawdown = -15.0 * scale
        sharpe = 0.8 * scale + 0.2 * float(self.params.get("bias", 0.0))
        total_return = 20.0 * scale
        winrate = 0.55 + 0.05 * min(1.0, scale)
        return SimpleNamespace(
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            max_drawdown=max_drawdown,
            sharpe=sharpe,
            total_return=total_return,
            winrate=winrate,
        )

    def full_backtest(self) -> Any:
        scale = self._scale() * 1.1
        return SimpleNamespace(
            gross_profit=150.0 * scale,
            gross_loss=50.0 * scale,
            max_drawdown=-18.0 * scale,
            sharpe=0.9 * scale,
            total_return=25.0 * scale,
            winrate=0.58 + 0.04 * min(1.0, scale),
        )


def make_strategy(params: Dict[str, Any]) -> StubStrategy:
    return StubStrategy(params)
