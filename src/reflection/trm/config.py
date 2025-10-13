"""Runtime configuration helpers for the TRM production runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

_CONFIG_DEFAULT_PATH = Path("config/reflection/rim.config.example.yml")


@dataclass(slots=True)
class TRMParams:
    """Recursive loop parameters recorded for telemetry."""

    K_outer: int = 4
    n_inner: int = 3
    halt_enabled: bool = True


@dataclass(slots=True)
class TelemetryConfig:
    """Telemetry and logging configuration."""

    log_dir: Path = Path("artifacts/rim_logs")


@dataclass(slots=True)
class RedactConfig:
    """Redaction policy for emitted suggestions."""

    fields: tuple[str, ...] = ()
    mode: str = "hash"


@dataclass(slots=True)
class ModelConfig:
    """Model repository configuration."""

    path: Path | None = None
    temperature: float = 1.0


@dataclass(slots=True)
class AutoApplySettings:
    """Configuration for the TRM governance auto-apply safeguard."""

    enabled: bool = False
    uplift_threshold: float = 0.0
    max_risk_hits: int = 0
    min_budget_remaining: float = 0.0
    max_budget_utilisation: float | None = None
    require_budget_metrics: bool = True
    default_budget_limit: float = 50.0
    strategy_budget_limits: dict[str, float] = field(default_factory=dict)

    def budget_limit_for(self, strategy_id: str) -> float:
        """Return the configured budget limit for the supplied strategy."""

        if self.strategy_budget_limits:
            limit = self.strategy_budget_limits.get(strategy_id)
            if isinstance(limit, (int, float)):
                return float(limit)
        return float(self.default_budget_limit)


@dataclass(slots=True)
class RIMRuntimeConfig:
    """Resolved configuration for the production TRM runner."""

    diaries_dir: Path = Path("artifacts/diaries")
    diary_glob: str = "diaries-*.jsonl"
    window_minutes: int = 1440
    min_entries: int = 100
    suggestion_cap: int = 10
    confidence_floor: float = 0.65
    enable_governance_gate: bool = True
    publish_channel: str = "file://artifacts/rim_suggestions"
    kill_switch: bool = False
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    trm_params: TRMParams = field(default_factory=TRMParams)
    redact: RedactConfig = field(default_factory=RedactConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lock_path: Path = Path("artifacts/locks/rim.lock")
    governance_queue_path: Path = Path("artifacts/governance/reflection_queue.jsonl")
    governance_digest_path: Path = Path("artifacts/governance/reflection_digest.json")
    governance_markdown_path: Path = Path("artifacts/governance/reflection_digest.md")
    auto_apply: AutoApplySettings | None = None


@dataclass(slots=True)
class RuntimeConfigBundle:
    """Container for config and metadata hashes."""

    config: RIMRuntimeConfig
    config_hash: str
    source_path: Path


def _coerce_path(value: Any) -> Path:
    if isinstance(value, Path):
        return value
    return Path(str(value))


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_trm_params(mapping: dict[str, Any] | None) -> TRMParams:
    if not mapping:
        return TRMParams()
    return TRMParams(
        K_outer=int(mapping.get("K_outer", TRMParams.K_outer)),
        n_inner=int(mapping.get("n_inner", TRMParams.n_inner)),
        halt_enabled=bool(mapping.get("halt_enabled", TRMParams.halt_enabled)),
    )


def _build_telemetry(mapping: dict[str, Any] | None) -> TelemetryConfig:
    if not mapping:
        return TelemetryConfig()
    return TelemetryConfig(log_dir=_coerce_path(mapping.get("log_dir", TelemetryConfig.log_dir)))


def _build_redact(mapping: dict[str, Any] | None) -> RedactConfig:
    if not mapping:
        return RedactConfig()
    fields = tuple(str(field) for field in mapping.get("fields", ()))
    mode = str(mapping.get("mode", RedactConfig.mode)).lower() or "hash"
    return RedactConfig(fields=fields, mode=mode)


def _build_model(mapping: dict[str, Any] | None) -> ModelConfig:
    if not mapping:
        return ModelConfig()
    path_value = mapping.get("path")
    path = _coerce_path(path_value) if path_value else None
    temperature = float(mapping.get("temperature", 1.0))
    return ModelConfig(path=path, temperature=temperature)


def _build_auto_apply(mapping: dict[str, Any] | None) -> AutoApplySettings | None:
    if not mapping:
        return None

    enabled = bool(mapping.get("enabled", True))
    uplift_threshold = float(mapping.get("uplift_threshold", 0.0))
    max_risk_hits = int(mapping.get("max_risk_hits", 0))
    min_budget_remaining = float(mapping.get("min_budget_remaining", 0.0))
    max_budget_utilisation = _coerce_optional_float(mapping.get("max_budget_utilisation"))
    require_budget_metrics = bool(mapping.get("require_budget_metrics", True))
    default_budget_limit = float(mapping.get("default_budget_limit", 50.0))

    raw_limits = mapping.get("strategy_budget_limits")
    limits: dict[str, float] = {}
    if isinstance(raw_limits, dict):
        for key, value in raw_limits.items():
            if not key:
                continue
            try:
                limits[str(key)] = float(value)
            except (TypeError, ValueError):
                continue

    return AutoApplySettings(
        enabled=enabled,
        uplift_threshold=uplift_threshold,
        max_risk_hits=max_risk_hits,
        min_budget_remaining=min_budget_remaining,
        max_budget_utilisation=max_budget_utilisation,
        require_budget_metrics=require_budget_metrics,
        default_budget_limit=default_budget_limit,
        strategy_budget_limits=limits,
    )


def _load_config_dict(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text()
    digest = sha256(text.encode("utf-8")).hexdigest()
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Configuration at {path} must be a mapping")
    return data, digest


def load_runtime_config(path: Path | None = None) -> RuntimeConfigBundle:
    """Load and normalise runtime configuration from YAML."""

    resolved_path = path if path and path.exists() else _CONFIG_DEFAULT_PATH
    if not resolved_path.exists():
        raise FileNotFoundError(f"Unable to locate TRM config at {resolved_path}")

    raw_config, digest = _load_config_dict(resolved_path)

    telemetry = _build_telemetry(_ensure_mapping(raw_config.get("telemetry")))
    trm_params = _build_trm_params(_ensure_mapping(raw_config.get("trm_params")))
    redact = _build_redact(_ensure_mapping(raw_config.get("redact")))
    model = _build_model(_ensure_mapping(raw_config.get("model")))
    governance = _ensure_mapping(raw_config.get("governance"))

    diaries_dir = _coerce_path(raw_config.get("diaries_dir", RIMRuntimeConfig.diaries_dir))
    diary_glob = str(raw_config.get("diary_glob", RIMRuntimeConfig.diary_glob))
    window_minutes = int(raw_config.get("window_minutes", RIMRuntimeConfig.window_minutes))
    min_entries = int(raw_config.get("min_entries", RIMRuntimeConfig.min_entries))
    suggestion_cap = int(raw_config.get("suggestion_cap", RIMRuntimeConfig.suggestion_cap))
    confidence_floor = float(raw_config.get("confidence_floor", RIMRuntimeConfig.confidence_floor))
    enable_governance_gate = bool(
        raw_config.get("enable_governance_gate", RIMRuntimeConfig.enable_governance_gate)
    )
    publish_channel = str(raw_config.get("publish_channel", RIMRuntimeConfig.publish_channel))
    kill_switch = bool(raw_config.get("kill_switch", RIMRuntimeConfig.kill_switch))
    lock_path = _coerce_path(raw_config.get("lock_path", RIMRuntimeConfig.lock_path))
    governance_queue_path = _coerce_path(
        (governance or {}).get(
            "queue_path",
            RIMRuntimeConfig.governance_queue_path,
        )
    )
    governance_digest_path = _coerce_path(
        (governance or {}).get(
            "digest_path",
            RIMRuntimeConfig.governance_digest_path,
        )
    )
    governance_markdown_path = _coerce_path(
        (governance or {}).get(
            "markdown_path",
            RIMRuntimeConfig.governance_markdown_path,
        )
    )
    auto_apply = _build_auto_apply(_ensure_mapping((governance or {}).get("auto_apply")))

    config = RIMRuntimeConfig(
        diaries_dir=diaries_dir,
        diary_glob=diary_glob,
        window_minutes=window_minutes,
        min_entries=min_entries,
        suggestion_cap=suggestion_cap,
        confidence_floor=confidence_floor,
        enable_governance_gate=enable_governance_gate,
        publish_channel=publish_channel,
        kill_switch=kill_switch,
        telemetry=telemetry,
        trm_params=trm_params,
        redact=redact,
        model=model,
        lock_path=lock_path,
        governance_queue_path=governance_queue_path,
        governance_digest_path=governance_digest_path,
        governance_markdown_path=governance_markdown_path,
        auto_apply=auto_apply,
    )

    return RuntimeConfigBundle(config=config, config_hash=digest, source_path=resolved_path)


def _ensure_mapping(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("Expected mapping")
    return value


__all__ = [
    "AutoApplySettings",
    "ModelConfig",
    "RIMRuntimeConfig",
    "RuntimeConfigBundle",
    "TRMParams",
    "load_runtime_config",
]
