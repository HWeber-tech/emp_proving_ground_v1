from __future__ import annotations

import pytest

from src.thinking.models.backbone_ssm import StructuredStateSpaceDropIn


torch = pytest.importorskip("torch")


class _DummySSD(torch.nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(features, features)

    def forward(self, x, state=None):  # noqa: ANN001,D401 - torch convention
        out = self.linear(x)
        if state is None:
            next_state = torch.zeros_like(out)
        else:
            next_state = state.to(dtype=out.dtype, device=out.device)
        return out, next_state + 1.0


class _BadSSD(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x, state=None):  # noqa: ANN001 - torch convention
        out = self.linear(x)
        return out, state


def test_drop_in_matches_reference_semantics() -> None:
    mlp = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.GELU(),
        torch.nn.Linear(8, 4),
    )
    ssd = _DummySSD(4)
    adapter = StructuredStateSpaceDropIn(mlp=mlp, ssd_block=ssd)

    x = torch.randn(6, 4)
    state = torch.randn(6, 4)

    output, next_state = adapter.forward(x, state)

    assert output.shape == torch.Size([6, 4])
    assert output.dtype == x.dtype
    assert output.device == x.device

    assert isinstance(next_state, torch.Tensor)
    assert next_state.shape == state.shape
    assert next_state.dtype == x.dtype
    assert next_state.device == x.device

    x2 = torch.randn(3, 4)
    output2, state2 = adapter.forward(x2)

    assert output2.shape == torch.Size([3, 4])
    assert output2.dtype == x2.dtype
    assert output2.device == x2.device
    assert isinstance(state2, torch.Tensor)
    assert state2.shape == torch.Size([3, 4])
    assert state2.dtype == x2.dtype
    assert state2.device == x2.device


def test_drop_in_respects_reference_dtype() -> None:
    mlp = torch.nn.Sequential(torch.nn.Linear(4, 4)).double()
    ssd = _DummySSD(4).double()
    adapter = StructuredStateSpaceDropIn(mlp=mlp, ssd_block=ssd)

    x = torch.randn(2, 4, dtype=torch.float64)
    output, state = adapter.forward(x)

    assert output.dtype == torch.float64
    assert isinstance(state, torch.Tensor)
    assert state.dtype == torch.float64


def test_drop_in_detects_feature_mismatch() -> None:
    mlp = torch.nn.Sequential(torch.nn.Linear(4, 4))
    ssd = _BadSSD(4, 5)
    adapter = StructuredStateSpaceDropIn(mlp=mlp, ssd_block=ssd)

    with pytest.raises(ValueError):
        adapter.forward(torch.randn(1, 4))
