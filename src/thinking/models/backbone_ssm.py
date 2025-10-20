from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableMapping, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

try:  # Optional dependency for tensor semantics
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    np = None  # type: ignore[assignment]

try:  # Optional dependency for tensor semantics
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


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


class StructuredStateSpaceDropIn(_BaseBackbone):
    """Wrap an SSD block so it can replace an existing MLP backbone."""

    def __init__(
        self,
        *,
        mlp: Callable[[Any], Any],
        ssd_block: Callable[..., Any],
        name: str = "ssd",
    ) -> None:
        if mlp is None:
            raise ValueError("Reference MLP module is required for SSD drop-in calibration")
        if ssd_block is None:
            raise ValueError("SSD block is required for drop-in wrapper")

        super().__init__(name)
        self._reference_mlp = mlp
        self._ssd_block = ssd_block
        self._validated = False
        self._input_metadata: _TensorMetadata | None = None
        self._output_metadata: _TensorMetadata | None = None
        self._ssd_prev_device: Any | None = None
        self._ssd_prev_dtype: Any | None = None
        self._mlp_prev_device: Any | None = None
        self._mlp_prev_dtype: Any | None = None

    def forward(self, x: Any, state: Any | None = None) -> tuple[Any, Any | None]:
        # Reset metadata when tensor semantics change (e.g. new device or dtype).
        input_meta = _tensor_metadata_or_none(x)
        if self._input_metadata is not None and input_meta is not None:
            if not self._input_metadata.same_tensor_semantics(input_meta):
                self._validated = False
                self._input_metadata = None
                self._output_metadata = None

        self._prepare_for_input(x)

        ssd_output, next_state = self._call_ssd_block(x, state)

        if not self._validated:
            reference_output = self._run_reference_mlp(x, state)
            self._initialise_metadata(input_meta, ssd_output, reference_output)
            self._validated = True

        aligned_output = self._align_output(ssd_output)
        aligned_state = self._align_state(next_state)
        return aligned_output, aligned_state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_for_input(self, x: Any) -> None:
        if torch is None or not isinstance(x, torch.Tensor):
            return

        device = x.device
        dtype = x.dtype if torch.is_floating_point(x) else None
        self._move_torch_module(self._ssd_block, device, dtype, is_reference=False)
        self._move_torch_module(self._reference_mlp, device, dtype, is_reference=True)

    def _move_torch_module(
        self,
        module: Any,
        device: Any | None,
        dtype: Any | None,
        *,
        is_reference: bool,
    ) -> None:
        if torch is None or module is None or not isinstance(module, torch.nn.Module):
            return

        device_attr = "_mlp_prev_device" if is_reference else "_ssd_prev_device"
        dtype_attr = "_mlp_prev_dtype" if is_reference else "_ssd_prev_dtype"
        prev_device = getattr(self, device_attr)
        prev_dtype = getattr(self, dtype_attr)

        to_kwargs: dict[str, Any] = {}
        if device is not None and device != prev_device:
            to_kwargs["device"] = device
        if dtype is not None and dtype != prev_dtype:
            to_kwargs["dtype"] = dtype

        if to_kwargs:
            module.to(**to_kwargs)
            setattr(self, device_attr, device)
            setattr(self, dtype_attr, dtype)
        else:
            if prev_device is None:
                setattr(self, device_attr, device)
            if prev_dtype is None and dtype is not None:
                setattr(self, dtype_attr, dtype)

    def _call_ssd_block(self, x: Any, state: Any | None) -> tuple[Any, Any | None]:
        block = self._ssd_block
        last_error: TypeError | None = None
        for attempt in self._call_attempts(block, x, state):
            try:
                result = attempt()
            except TypeError as exc:
                if not self._is_signature_error(exc):
                    raise
                last_error = exc
                continue
            return self._normalise_forward_result(result, state)

        if last_error is not None:
            raise last_error
        raise RuntimeError("Unable to invoke SSD block forward() with provided inputs")

    def _run_reference_mlp(self, x: Any, state: Any | None) -> Any | None:
        mlp = self._reference_mlp
        context = nullcontext()
        if torch is not None and isinstance(x, torch.Tensor):
            context = torch.no_grad()

        with context:
            last_error: TypeError | None = None
            for attempt in self._call_attempts(mlp, x, state):
                try:
                    result = attempt()
                except TypeError as exc:
                    if not self._is_signature_error(exc):
                        raise
                    last_error = exc
                    continue
                return result[0] if isinstance(result, tuple) and result else result

        if last_error is not None:
            raise last_error
        raise RuntimeError("Unable to invoke reference MLP forward() with provided inputs")

    def _initialise_metadata(
        self,
        input_meta: _TensorMetadata | None,
        ssd_output: Any,
        reference_output: Any | None,
    ) -> None:
        if input_meta is not None:
            self._input_metadata = input_meta

        try:
            ssd_meta = _TensorMetadata.from_value(ssd_output, treat_batch_as_dynamic=True)
        except TypeError as exc:  # pragma: no cover - defensive
            raise TypeError("SSD block returned unsupported output type") from exc

        baseline_meta = ssd_meta
        if reference_output is not None:
            baseline_meta = _TensorMetadata.from_value(reference_output, treat_batch_as_dynamic=True)

        if not baseline_meta.compatible(ssd_meta):
            raise ValueError(
                "SSD block output feature shape does not match MLP baseline: "
                f"expected {baseline_meta.feature_shape}, got {ssd_meta.feature_shape}",
            )

        self._output_metadata = baseline_meta

    def _align_output(self, value: Any) -> Any:
        metadata = self._output_metadata
        if metadata is None:
            return value
        metadata.validate(value, role="SSD output")
        return metadata.align(value)

    def _align_state(self, state: Any) -> Any:
        metadata = self._input_metadata
        if metadata is None or state is None:
            return state
        return _coerce_structure(state, metadata)

    @staticmethod
    def _normalise_forward_result(result: Any, fallback_state: Any | None) -> tuple[Any, Any | None]:
        if isinstance(result, tuple):
            if not result:
                raise ValueError("SSD block forward() returned an empty tuple")
            if len(result) == 1:
                return result[0], fallback_state
            return result[0], result[1]
        return result, fallback_state

    @staticmethod
    def _is_signature_error(exc: TypeError) -> bool:
        message = str(exc)
        indicators = (
            "positional argument",
            "keyword argument",
            "unexpected keyword",
            "required positional",
            "missing 1 required",
            "multiple values for argument",
        )
        return any(term in message for term in indicators)

    @staticmethod
    def _call_attempts(
        fn: Callable[..., Any],
        x: Any,
        state: Any | None,
    ) -> list[Callable[[], Any]]:
        attempts: list[Callable[[], Any]] = []
        attempts.append(lambda: fn(x, state=state))
        attempts.append(lambda: fn(x, state))
        attempts.append(lambda: fn(x))
        return attempts


@dataclass(slots=True)
class _TensorMetadata:
    library: str
    ndim: int
    feature_shape: tuple[int, ...]
    dtype: Any | None
    device: Any | None

    @classmethod
    def from_value(cls, value: Any, *, treat_batch_as_dynamic: bool) -> "_TensorMetadata":
        if torch is not None and isinstance(value, torch.Tensor):
            shape = tuple(int(dim) for dim in value.shape)
            if treat_batch_as_dynamic and len(shape) >= 2:
                feature_shape = shape[1:]
            else:
                feature_shape = shape
            return cls("torch", len(shape), feature_shape, value.dtype, value.device)
        if np is not None and isinstance(value, np.ndarray):
            shape = tuple(int(dim) for dim in value.shape)
            if treat_batch_as_dynamic and len(shape) >= 2:
                feature_shape = shape[1:]
            else:
                feature_shape = shape
            return cls("numpy", len(shape), feature_shape, value.dtype, None)

        shape_attr = getattr(value, "shape", None)
        if shape_attr is not None:
            shape_tuple = tuple(int(dim) for dim in shape_attr)
            if treat_batch_as_dynamic and len(shape_tuple) >= 2:
                feature_shape = shape_tuple[1:]
            else:
                feature_shape = shape_tuple
            dtype = getattr(value, "dtype", None)
            device = getattr(value, "device", None)
            return cls(type(value).__name__, len(shape_tuple), feature_shape, dtype, device)

        raise TypeError(f"Unsupported tensor-like value {type(value)!r}")

    def compatible(self, other: "_TensorMetadata") -> bool:
        return (
            self.library == other.library
            and self.ndim == other.ndim
            and self.feature_shape == other.feature_shape
        )

    def validate(self, value: Any, *, role: str) -> None:
        candidate = _TensorMetadata.from_value(value, treat_batch_as_dynamic=True)
        if self.library != candidate.library:
            raise ValueError(f"{role} backend mismatch: expected {self.library}, got {candidate.library}")
        if self.ndim != candidate.ndim:
            raise ValueError(f"{role} rank mismatch: expected {self.ndim}, got {candidate.ndim}")
        if self.feature_shape != candidate.feature_shape:
            raise ValueError(
                f"{role} feature shape mismatch: expected {self.feature_shape}, got {candidate.feature_shape}"
            )

    def same_tensor_semantics(self, other: "_TensorMetadata") -> bool:
        if self.library != other.library:
            return False
        if self.ndim != other.ndim:
            return False
        if self.feature_shape != other.feature_shape:
            return False
        if self.dtype is not None and other.dtype is not None and self.dtype != other.dtype:
            return False
        if self.device is not None and other.device is not None and self.device != other.device:
            return False
        return True

    def align(self, value: Any) -> Any:
        if self.library == "torch" and torch is not None and isinstance(value, torch.Tensor):
            to_kwargs: dict[str, Any] = {}
            if self.device is not None and value.device != self.device:
                to_kwargs["device"] = self.device
            if self.dtype is not None and value.dtype != self.dtype:
                to_kwargs["dtype"] = self.dtype
            if to_kwargs:
                value = value.to(**to_kwargs)
            return value

        if self.library == "numpy" and np is not None and isinstance(value, np.ndarray):
            if self.dtype is not None and value.dtype != self.dtype:
                value = value.astype(self.dtype, copy=False)
            return value

        return value


def _tensor_metadata_or_none(value: Any) -> _TensorMetadata | None:
    try:
        return _TensorMetadata.from_value(value, treat_batch_as_dynamic=True)
    except TypeError:
        return None


def _coerce_structure(value: Any, metadata: _TensorMetadata) -> Any:
    if torch is not None and isinstance(value, torch.Tensor) and metadata.library == "torch":
        to_kwargs: dict[str, Any] = {}
        if metadata.device is not None and value.device != metadata.device:
            to_kwargs["device"] = metadata.device
        if metadata.dtype is not None and value.dtype != metadata.dtype:
            to_kwargs["dtype"] = metadata.dtype
        if to_kwargs:
            value = value.to(**to_kwargs)
        return value

    if np is not None and isinstance(value, np.ndarray) and metadata.library == "numpy":
        if metadata.dtype is not None and value.dtype != metadata.dtype:
            value = value.astype(metadata.dtype, copy=False)
        return value

    if isinstance(value, dict):
        return {key: _coerce_structure(item, metadata) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_coerce_structure(item, metadata) for item in value)
    if isinstance(value, list):
        return [_coerce_structure(item, metadata) for item in value]
    return value


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
    "StructuredStateSpaceDropIn",
]
