# Import directly from module to avoid pulling heavy execution package
from src.trading.execution.execution_model import ExecContext, estimate_slippage_bps, estimate_commission_bps
from src.data_foundation.config.execution_config import ExecutionConfig, SlippageModel, FeeModel


def test_slippage_estimator_increases_with_inputs():
    cfg = ExecutionConfig(slippage=SlippageModel(base_bps=0.2, spread_coef=10.0, imbalance_coef=1.0, sigma_coef=10.0, size_coef=1.0),
                          fees=FeeModel(commission_bps=0.05))
    low = ExecContext(spread=0.0, top_imbalance=0.0, sigma_ann=0.0, size_ratio=0.0)
    high = ExecContext(spread=0.001, top_imbalance=1.0, sigma_ann=0.2, size_ratio=1.0)
    assert estimate_slippage_bps(low, cfg) < estimate_slippage_bps(high, cfg)
    assert estimate_commission_bps(cfg) == 0.05

