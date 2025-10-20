from __future__ import annotations

import logging
import time
from typing import Any, Callable, Mapping, MutableMapping, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class BackboneImplementation(Protocol):
    """Minimal protocol for backbone implementations."""

    name: str

    def forward(self, x: Any, state: Any | None = None) -> tuple[Any, Any | None]:
        ...


class _BaseBackbone:
    def __init__(self, name: str) -> None:
        self.name = name

    def forward(self, x: Any, state: Any | None = None) -> tuple[Any, Any | None]:
        return x, state


class Mamba2Backbone(_BaseBackbone):
    def __init__(self) -> None:
        super().__init__("mamba2")


class Mamba3Backbone(_BaseBackbone):
    def __init__(self) -> None:
        super().__init__("mamba3")


class BackboneSSM:
    """Select between state-space model implementations with latency fallback."""

    DEFAULT_IMPL = "mamba3"
    DEFAULT_FALLBACK_IMPL = "mamba2"
    DEFAULT_LATENCY_THRESHOLD_MS = 0.35

    _DEFAULT_FACTORIES: Mapping[str, Callable[[], BackboneImplementation]] = {
        "mamba2": Mamba2Backbone,
        "mamba3": Mamba3Backbone,
    }

    def __init__(
        self,
        *,
        impl: str | None = None,
        fallback_impl: str | None = None,
        latency_threshold_ms: float | int | str | None = None,
        implementations: Mapping[str, BackboneImplementation | Callable[[], BackboneImplementation]] | None = None,
    ) -> None:
        factories: MutableMapping[str, Callable[[], BackboneImplementation]] = {
            name: factory for name, factory in self._DEFAULT_FACTORIES.items()
        }
        if implementations:
            for name, provider in implementations.items():
                factories[str(name).strip().lower()] = self._as_factory(provider)

        self._implementations: dict[str, BackboneImplementation] = {}
        for name, factory in factories.items():
            self._implementations[name] = self._coerce_instance(name, factory)

        self._primary_impl_name = (impl or self.DEFAULT_IMPL).strip().lower()
        if self._primary_impl_name not in self._implementations:
            raise ValueError(f"Unknown SSM implementation: {self._primary_impl_name}")

        self._fallback_impl_name = (fallback_impl or self.DEFAULT_FALLBACK_IMPL).strip().lower()
        if self._fallback_impl_name not in self._implementations:
            raise ValueError(f"Unknown fallback implementation: {self._fallback_impl_name}")

        self._active_impl_name = self._primary_impl_name
        self._last_used_impl_name: str | None = None
        self._last_latency_ms: float | None = None
        self._latency_failures = 0
        self._last_fallback_reason: str | None = None

        threshold = self._coerce_latency_threshold(latency_threshold_ms)
        if threshold is None:
            threshold = self.DEFAULT_LATENCY_THRESHOLD_MS
        if threshold <= 0:
            threshold = None
        self._latency_threshold_ms = threshold

    def forward(self, x: Any, state: Any | None = None) -> tuple[Any, Any | None]:
        impl = self._implementations[self._active_impl_name]
        self._last_used_impl_name = impl.name

        start = time.perf_counter()
        result = impl.forward(x, state)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._last_latency_ms = elapsed_ms

        if (
            self._latency_threshold_ms is not None
            and elapsed_ms > self._latency_threshold_ms
            and self._active_impl_name != self._fallback_impl_name
        ):
            logger.warning(
                "BackboneSSM latency %.3fms breached %.3fms; switching to %s",
                elapsed_ms,
                self._latency_threshold_ms,
                self._fallback_impl_name,
            )
            self._latency_failures += 1
            self._active_impl_name = self._fallback_impl_name
            self._last_fallback_reason = "latency_threshold"
        else:
            self._latency_failures = 0

        return result

    @property
    def active_impl_name(self) -> str:
        return self._active_impl_name

    @property
    def fallback_impl_name(self) -> str:
        return self._fallback_impl_name

    @property
    def primary_impl_name(self) -> str:
        return self._primary_impl_name

    @property
    def last_used_impl_name(self) -> str | None:
        return self._last_used_impl_name

    @property
    def last_latency_ms(self) -> float | None:
        return self._last_latency_ms

    @property
    def latency_threshold_ms(self) -> float | None:
        return self._latency_threshold_ms

    @property
    def latency_failures(self) -> int:
        return self._latency_failures

    @property
    def last_fallback_reason(self) -> str | None:
        return self._last_fallback_reason

    def reset_to_primary(self) -> None:
        self._active_impl_name = self._primary_impl_name
        self._latency_failures = 0
        self._last_fallback_reason = None

    def use_impl(self, name: str) -> None:
        lowered = name.strip().lower()
        if lowered not in self._implementations:
            raise KeyError(f"Backbone implementation not registered: {name}")
        self._active_impl_name = lowered
        self._latency_failures = 0
        if lowered != self._fallback_impl_name:
            self._last_fallback_reason = None

    def register_implementation(
        self,
        name: str,
        implementation: BackboneImplementation | Callable[[], BackboneImplementation],
    ) -> None:
        lowered = name.strip().lower()
        factory = self._as_factory(implementation)
        self._implementations[lowered] = self._coerce_instance(lowered, factory)

    def has_implementation(self, name: str) -> bool:
        return name.strip().lower() in self._implementations

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | Any,
        *,
        implementations: Mapping[str, BackboneImplementation | Callable[[], BackboneImplementation]] | None = None,
        default_latency_threshold_ms: float | int | str | None = None,
    ) -> "BackboneSSM":
        sources: list[Mapping[str, Any]] = []

        if isinstance(config, Mapping):
            sources.append(config)
            model_section = config.get("model")
            if isinstance(model_section, Mapping):
                sources.append(model_section)

        extras = getattr(config, "extras", None)
        if isinstance(extras, Mapping):
            sources.append(extras)

        impl = cls._extract_first(sources, (
            "MODEL_SSM_IMPL",
            "MODEL_PRIMARY_IMPL",
            "model.ssm_impl",
            "ssm_impl",
            "SSM_IMPL",
        ))
        fallback = cls._extract_first(sources, (
            "MODEL_FALLBACK_IMPL",
            "MODEL_SECONDARY_IMPL",
            "model.fallback_impl",
            "fallback_impl",
            "FALLBACK_IMPL",
        ))
        latency_raw = cls._extract_first(sources, (
            "MODEL_LATENCY_THRESHOLD_MS",
            "model.latency_threshold_ms",
            "latency_threshold_ms",
        ))

        latency_threshold: float | int | str | None = latency_raw
        if latency_threshold is None:
            latency_threshold = default_latency_threshold_ms

        return cls(
            impl=impl or cls.DEFAULT_IMPL,
            fallback_impl=fallback or cls.DEFAULT_FALLBACK_IMPL,
            latency_threshold_ms=latency_threshold,
            implementations=implementations,
        )

    @staticmethod
    def _extract_first(sources: list[Mapping[str, Any]], keys: tuple[str, ...]) -> str | None:
        for source in sources:
            for key in keys:
                if key in source:
                    value = source[key]
                    if value is None:
                        continue
                    text = str(value).strip()
                    if text:
                        return text
        return None

    @staticmethod
    def _coerce_latency_threshold(value: float | int | str | None) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _as_factory(
        provider: BackboneImplementation | Callable[[], BackboneImplementation]
    ) -> Callable[[], BackboneImplementation]:
        if callable(provider) and not isinstance(provider, BackboneImplementation):
            return provider  # type: ignore[return-value]

        def _factory() -> BackboneImplementation:
            if callable(provider):
                instance = provider()  # type: ignore[operator]
            else:
                instance = provider
            return instance

        return _factory

    @staticmethod
    def _coerce_instance(
        name: str,
        factory: Callable[[], BackboneImplementation],
    ) -> BackboneImplementation:
        instance = factory()
        if not isinstance(instance, BackboneImplementation):  # type: ignore[arg-type]
            if not hasattr(instance, "forward"):
                raise TypeError(f"Backbone implementation must expose forward(): {name}")
        if not getattr(instance, "name", None):
            setattr(instance, "name", name)
        return instance


__all__ = [
    "BackboneSSM",
    "BackboneImplementation",
    "Mamba2Backbone",
    "Mamba3Backbone",
]
