"""Novelty archive for strategy genotypes.

This module builds a compact novelty archive that fingerprints strategy
Genotypes via a deterministic signature and lightweight probe vector.  The
archive tracks the most recent entries up to a fixed capacity and exposes
helpers to score and register new genotypes.  Novelty is measured using cosine
folk distance between probe vectors and clipped to ``[0, 1]`` for ease of
interpretation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .strategy_contracts import StrategyGenotype

__all__ = [
    "NoveltyArchive",
    "NoveltyProbe",
    "compute_genotype_signature",
    "compute_probe_vector",
]


@dataclass(frozen=True)
class NoveltyProbe:
    """Snapshot describing a genotype's novelty against the archive."""

    signature: str
    vector: tuple[int, ...]
    novelty: float

    def as_dict(self) -> Mapping[str, object]:
        return {
            "signature": self.signature,
            "vector": list(self.vector),
            "novelty": float(self.novelty),
        }


def compute_genotype_signature(genotype: StrategyGenotype) -> str:
    """Return a deterministic signature for ``genotype``.

    The signature is a SHA-1 hash of the canonical JSON representation emitted
    by :meth:`~StrategyGenotype.as_dict`.
    """

    payload = genotype.as_dict()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()


def compute_probe_vector(
    genotype: StrategyGenotype,
    *,
    dimensions: int = 16,
) -> tuple[int, ...]:
    """Vectorise ``genotype`` into a lightweight probe vector."""

    tokens = list(_flatten_payload(genotype.as_dict()))
    return _vectorize(tokens, dimensions=dimensions)


class NoveltyArchive:
    """Keep a bounded archive of genotype probes and expose novelty scoring."""

    def __init__(
        self,
        *,
        capacity: int = 2048,
        dimensions: int = 16,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self._capacity = int(capacity)
        self._dimensions = int(dimensions)
        self._entries: deque[NoveltyProbe] = deque()
        self._index: dict[str, NoveltyProbe] = {}

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def __len__(self) -> int:  # pragma: no cover - trivial container protocol
        return len(self._entries)

    def __contains__(self, signature: str) -> bool:  # pragma: no cover - trivial container protocol
        return signature in self._index

    def probe(self, genotype: StrategyGenotype) -> NoveltyProbe:
        """Compute novelty for ``genotype`` without mutating the archive."""

        signature = compute_genotype_signature(genotype)
        vector = compute_probe_vector(genotype, dimensions=self._dimensions)
        novelty = self._compute_novelty(vector, signature)
        return NoveltyProbe(signature=signature, vector=vector, novelty=novelty)

    def record(self, genotype: StrategyGenotype) -> NoveltyProbe:
        """Compute novelty for ``genotype`` and insert the probe into the archive."""

        probe = self.probe(genotype)
        if probe.signature in self._index:
            existing = self._index[probe.signature]
            return NoveltyProbe(signature=existing.signature, vector=existing.vector, novelty=0.0)
        self._append(probe)
        return probe

    def _append(self, probe: NoveltyProbe) -> None:
        if len(self._entries) >= self._capacity:
            oldest = self._entries.popleft()
            self._index.pop(oldest.signature, None)
        self._entries.append(probe)
        self._index[probe.signature] = probe

    def _compute_novelty(self, vector: Sequence[int], signature: str) -> float:
        if signature in self._index:
            return 0.0
        min_distance: float | None = None
        for entry in self._entries:
            distance = _cosine_distance(vector, entry.vector)
            if min_distance is None or distance < min_distance:
                min_distance = distance
                if min_distance == 0.0:
                    break
        if min_distance is None:
            return 1.0
        return float(_clip_novelty(min_distance))


def _clip_novelty(value: float) -> float:
    return max(0.0, min(1.0, value))


def _json_default(value: object) -> object:
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _flatten_payload(payload: object, prefix: str = "") -> Iterable[str]:
    if isinstance(payload, Mapping):
        for key in sorted(payload):
            yield from _flatten_payload(payload[key], f"{prefix}{key}.")
        return
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for idx, item in enumerate(payload):
            yield from _flatten_payload(item, f"{prefix}{idx}.")
        return
    token = f"{prefix[:-1]}={_normalise_value(payload)}"
    yield token


def _normalise_value(value: object) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isfinite(value):
        return json.dumps(value, separators=(",", ":"))
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return json.dumps(value, default=_json_default, sort_keys=True, separators=(",", ":"))


def _vectorize(tokens: Iterable[str], *, dimensions: int) -> tuple[int, ...]:
    vector = [0] * dimensions
    for token in sorted(tokens):
        h = 0
        for char in token:
            h = (h * 33 + ord(char)) & 0xFFFFFFFF
        bucket = h % dimensions
        weight = (h // dimensions) % 7 + 1
        vector[bucket] += weight
    return tuple(vector)


def _cosine_distance(vec_a: Sequence[int], vec_b: Sequence[int]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    similarity = dot / (norm_a * norm_b)
    similarity = max(-1.0, min(1.0, similarity))
    return 1.0 - similarity
