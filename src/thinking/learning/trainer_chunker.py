"""Chunked TBPTT utilities for trainer burn-in scheduling (C.3.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

try:  # torch is an optional dependency in several test environments
    import torch
except ModuleNotFoundError:  # pragma: no cover - torch may not be installed
    torch = None  # type: ignore[assignment]


def _supports_len(obj: object) -> bool:
    return hasattr(obj, "__len__")


def _sequence_length(sequence: object) -> int:
    if hasattr(sequence, "shape") and sequence.shape:  # numpy / torch tensors
        return int(sequence.shape[0])
    if _supports_len(sequence):
        return len(sequence)  # type: ignore[arg-type]
    raise TypeError("Sequence must define __len__ or a shape[0] attribute")


def _slice_sequence(sequence: Any, slc: slice) -> Any:
    try:
        return sequence[slc]
    except TypeError as exc:  # pragma: no cover - defensive guard
        raise TypeError("Sequence must support slicing") from exc


def _detach_state(state: Any) -> Any:
    if state is None:
        return None

    if torch is not None and isinstance(state, torch.Tensor):
        result = state.detach()
        result.requires_grad_(False)
        return result

    if hasattr(state, "detach") and callable(state.detach):  # generic tensor-like
        try:
            result = state.detach()
        except TypeError:  # pragma: no cover - defensive
            result = state
        if hasattr(result, "requires_grad") and callable(result.requires_grad_):
            result.requires_grad_(False)
        return result

    if isinstance(state, Mapping):
        return type(state)((key, _detach_state(value)) for key, value in state.items())

    if isinstance(state, tuple):
        return tuple(_detach_state(item) for item in state)

    if isinstance(state, list):
        return [_detach_state(item) for item in state]

    if isinstance(state, (set, frozenset)):
        return type(state)(_detach_state(item) for item in state)

    return state


@dataclass(slots=True)
class SequenceChunk:
    """Descriptor for a TBPTT training chunk."""

    index: int
    start: int
    stop: int
    burn_in_slice: slice
    train_slice: slice
    burn_in: Any | None
    train: Any
    initial_state: Any | None
    _carry_callback: Callable[[Any | None], Any | None]
    _carried: bool = False

    @property
    def train_length(self) -> int:
        return self.train_slice.stop - self.train_slice.start

    @property
    def burn_in_length(self) -> int:
        return self.burn_in_slice.stop - self.burn_in_slice.start

    def carry_state(self, state: Any | None) -> Any | None:
        """Record the state to be passed into the next chunk.

        The provided ``state`` is detached before storage so gradients do not
        cross chunk boundaries.
        """

        result = self._carry_callback(state)
        self._carried = True
        return result


class TrainerChunker:
    """Chunk long sequences for truncated BPTT with burn-in warmup."""

    def __init__(self, *, burn_in: int = 512, train_len: int = 2048) -> None:
        if burn_in < 0:
            raise ValueError("burn_in must be non-negative")
        if train_len <= 0:
            raise ValueError("train_len must be positive")
        self._burn_in = int(burn_in)
        self._train_len = int(train_len)

    @property
    def burn_in(self) -> int:
        return self._burn_in

    @property
    def train_len(self) -> int:
        return self._train_len

    def iter_chunks(
        self,
        sequence: Any,
        *,
        initial_state: Any | None = None,
    ) -> Iterable[SequenceChunk]:
        """Yield :class:`SequenceChunk` objects across ``sequence``.

        ``sequence`` must support ``len(sequence)`` and slicing via ``sequence[slice]``.
        The iterator maintains a carried recurrent state across chunks and detaches
        it between iterations so backpropagation stays within each chunk.
        """

        length = _sequence_length(sequence)
        if length == 0:
            return

        state = _detach_state(initial_state)
        cursor = 0
        chunk_index = 0

        while cursor < length:
            burn_start = cursor
            if state is None:
                burn_end = min(burn_start + self._burn_in, length)
                train_start = burn_end
            else:
                burn_end = burn_start
                train_start = cursor

            train_end = min(train_start + self._train_len, length)
            if train_end <= train_start:
                break

            burn_slice = slice(burn_start, burn_end)
            train_slice = slice(train_start, train_end)

            burn_data = _slice_sequence(sequence, burn_slice) if burn_end > burn_start else None
            train_data = _slice_sequence(sequence, train_slice)

            next_state: dict[str, Any | None] = {"value": None, "set": False}

            def _carry(new_state: Any | None, store=next_state) -> Any | None:
                store["value"] = _detach_state(new_state)
                store["set"] = True
                return store["value"]

            chunk = SequenceChunk(
                index=chunk_index,
                start=cursor,
                stop=train_end,
                burn_in_slice=burn_slice,
                train_slice=train_slice,
                burn_in=burn_data,
                train=train_data,
                initial_state=_detach_state(state),
                _carry_callback=_carry,
            )

            yield chunk

            state = next_state["value"] if next_state["set"] else None
            cursor = train_end
            chunk_index += 1

    def chunk(self, sequence: Any, *, initial_state: Any | None = None) -> list[SequenceChunk]:
        """Convenience helper returning a list of chunks."""

        return list(self.iter_chunks(sequence, initial_state=initial_state))
