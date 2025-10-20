from __future__ import annotations

from collections.abc import Iterable as IterableABC
import pytest

from src.thinking.learning import SequenceChunk, TrainerChunker

try:  # torch is optional in some environments
    import torch  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover - torch absent
    torch = None  # type: ignore[assignment]


class DummyTensor:
    """Lightweight tensor stub with detach semantics for testing."""

    def __init__(self, *values: float, requires_grad: bool = False) -> None:
        if len(values) == 1 and isinstance(values[0], IterableABC):
            values = tuple(values[0])
        self.values = tuple(float(v) for v in values)
        self.requires_grad = requires_grad

    def detach(self) -> "DummyTensor":
        return DummyTensor(self.values, requires_grad=False)

    def requires_grad_(self, flag: bool = True) -> "DummyTensor":
        self.requires_grad = bool(flag)
        return self

    def __repr__(self) -> str:  # pragma: no cover - diagnostic helper
        return f"DummyTensor(values={self.values}, requires_grad={self.requires_grad})"


def test_trainer_chunker_carries_state_across_chunks() -> None:
    sequence = list(range(50))
    chunker = TrainerChunker(burn_in=4, train_len=10)

    iterator = iter(chunker.iter_chunks(sequence))

    first = next(iterator)
    assert isinstance(first, SequenceChunk)
    assert first.index == 0
    assert first.burn_in_length == 4
    assert first.burn_in == sequence[:4]
    assert first.train == sequence[4:14]
    assert first.initial_state is None

    carried0 = first.carry_state(DummyTensor(1.0, 2.0).requires_grad_())
    assert isinstance(carried0, DummyTensor)
    assert carried0.values == (1.0, 2.0)
    assert carried0.requires_grad is False

    second = next(iterator)
    assert second.index == 1
    assert second.burn_in is None
    assert second.burn_in_length == 0
    assert second.train == sequence[14:24]
    assert isinstance(second.initial_state, DummyTensor)
    assert second.initial_state.values == (1.0, 2.0)
    carried1 = second.carry_state(DummyTensor(3.0, 4.0).requires_grad_())
    assert carried1.values == (3.0, 4.0)

    third = next(iterator)
    assert third.index == 2
    assert third.burn_in is None
    assert third.train == sequence[24:34]
    assert isinstance(third.initial_state, DummyTensor)
    assert third.initial_state.values == (3.0, 4.0)


def test_trainer_chunker_resets_when_state_not_carried() -> None:
    sequence = list(range(40))
    chunker = TrainerChunker(burn_in=4, train_len=10)

    iterator = iter(chunker.iter_chunks(sequence))

    first = next(iterator)
    first.carry_state(DummyTensor(-1.0))

    second = next(iterator)
    # Intentionally skip carry_state on second chunk to force a reset.
    third = next(iterator)

    assert third.burn_in is not None
    assert third.burn_in_length == 4
    assert third.burn_in == sequence[24:28]
    assert third.initial_state is None
    assert third.train == sequence[28:38]


def test_trainer_chunker_detaches_nested_state() -> None:
    sequence = list(range(30))
    chunker = TrainerChunker(burn_in=4, train_len=10)

    nested_state = {
        "a": DummyTensor(0.1, 0.2).requires_grad_(),
        "b": (
            DummyTensor(0.3).requires_grad_(),
            DummyTensor(0.4, 0.5).requires_grad_(),
        ),
    }

    iterator = iter(chunker.iter_chunks(sequence, initial_state=nested_state))
    first = next(iterator)

    assert isinstance(first.initial_state, dict)
    assert first.burn_in is None  # initial state provided, so burn-in skipped
    assert first.initial_state is not nested_state
    assert first.initial_state["a"].requires_grad is False
    assert all(component.requires_grad is False for component in first.initial_state["b"])

    new_state = {
        "a": DummyTensor(1.0).requires_grad_(),
        "b": (DummyTensor(2.0).requires_grad_(),),
    }
    carried = first.carry_state(new_state)
    assert carried["a"].requires_grad is False
    assert carried["b"][0].requires_grad is False

    second = next(iterator)
    assert isinstance(second.initial_state, dict)
    assert second.initial_state["a"].values == (1.0,)
    assert second.burn_in is None


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_trainer_chunker_with_torch_state() -> None:
    assert torch is not None  # for type checkers
    sequence = torch.arange(30, dtype=torch.float32)
    chunker = TrainerChunker(burn_in=4, train_len=10)

    iterator = iter(chunker.iter_chunks(sequence))
    first = next(iterator)
    state = torch.ones(2, requires_grad=True)
    detached = first.carry_state(state)
    assert torch.equal(detached, torch.ones(2))
    assert detached.requires_grad is False

    second = next(iterator)
    assert torch.equal(second.initial_state, torch.ones(2))
    assert second.burn_in is None
