import importlib


def test_strategy_engine_register_unregister() -> None:
    tse = importlib.import_module("src.trading.strategy_engine")
    assert hasattr(tse, "StrategyEngine")

    from src.core.strategy.engine import (  # type: ignore
        BaseStrategy,
        StrategyEngine,
        StrategyPerformance,
    )

    class DummyStrategy(BaseStrategy):  # minimal stub
        async def generate_signal(self, market_data, symbol: str):
            return None

    engine = StrategyEngine()
    strat = DummyStrategy(strategy_id="s1", symbols=["AAPL"])

    assert engine.register(strat) is True
    perf = engine.performance("s1")
    assert isinstance(perf, StrategyPerformance)

    assert engine.unregister("s1") is True
    assert engine.performance("s1") is None
