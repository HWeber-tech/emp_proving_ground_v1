"""Primary entrypoints for institutional data integration helpers.

The :mod:`src.data_integration` package rejuvenates the historical bootstrap
stubs with the real Timescale/Redis/Kafka-backed data backbone described in the
roadmap.  Consumers import :class:`RealDataManager` (an implementation of the
:class:`src.core.market_data.MarketDataGateway` protocol) alongside a lightweight
:class:`DataSourceConfig` helper that normalises environment or
:class:`~src.governance.system_config.SystemConfig` extras into declarative
connection settings.

The legacy provider symbols remain available for compatibility; they now point
to deliberate placeholders so importing modules receive descriptive guidance
rather than ``pass`` skeletons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from src.governance.system_config import SystemConfig
from src.validation.models import ValidationResult

from src.data_foundation.cache.redis_cache import (
    RedisCachePolicy,
    RedisConnectionSettings,
)
from src.data_foundation.persist.timescale import TimescaleConnectionSettings
from src.data_foundation.streaming.kafka_stream import KafkaConnectionSettings

from .real_data_integration import (
    BackboneConnectivityReport,
    ConnectivityProbeSnapshot,
    RealDataManager,
)


@dataclass(frozen=True)
class DataSourceConfig:
    """Normalised connector settings for the institutional data backbone."""

    timescale: TimescaleConnectionSettings
    redis: RedisConnectionSettings
    kafka: KafkaConnectionSettings
    cache_policy: RedisCachePolicy = field(
        default_factory=RedisCachePolicy.institutional_defaults
    )
    extras: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, str] | None,
        *,
        fallback_policy: RedisCachePolicy | None = None,
    ) -> "DataSourceConfig":
        payload = mapping or {}
        timescale = TimescaleConnectionSettings.from_mapping(payload)
        redis = RedisConnectionSettings.from_mapping(payload)
        kafka = KafkaConnectionSettings.from_mapping(payload)
        cache_policy = RedisCachePolicy.from_mapping(
            payload,
            fallback=fallback_policy or RedisCachePolicy.institutional_defaults(),
        )
        return cls(
            timescale=timescale,
            redis=redis,
            kafka=kafka,
            cache_policy=cache_policy,
            extras={str(k): str(v) for k, v in payload.items()},
        )

    @classmethod
    def from_system_config(
        cls,
        config: SystemConfig,
        *,
        fallback_policy: RedisCachePolicy | None = None,
    ) -> "DataSourceConfig":
        extras = config.extras if isinstance(config.extras, dict) else {}
        return cls.from_mapping(extras, fallback_policy=fallback_policy)


class _PlaceholderProvider:
    """Compatibility shim for historical provider symbols.

    The legacy stack imported ``YahooFinanceDataProvider``-style classes.  The
    roadmap intentionally replaces those brittle adapters with governed
    Timescale/Kafka flows.  By raising a descriptive error we keep import-time
    behaviour predictable while nudging callers toward ``RealDataManager``.
    """

    _message = (
        "The institutional data backbone replaces direct provider adapters. "
        "Use RealDataManager or the market data fabric connectors instead."
    )

    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - defensive
        raise RuntimeError(self._message)


class YahooFinanceDataProvider(_PlaceholderProvider):
    pass


class AlphaVantageDataProvider(_PlaceholderProvider):
    pass


class FREDDataProvider(_PlaceholderProvider):
    pass


class NewsAPIDataProvider(_PlaceholderProvider):
    pass


class MarketDataValidator(_PlaceholderProvider):
    pass


class DataConsistencyChecker(_PlaceholderProvider):
    pass


class DataQualityMonitor(_PlaceholderProvider):
    pass


class ValidationLevel(_PlaceholderProvider):
    pass


class DataQualityThresholds(_PlaceholderProvider):
    pass


class DataIssue(_PlaceholderProvider):
    pass


ADVANCED_PROVIDERS_AVAILABLE = False

__all__ = [
    "RealDataManager",
    "DataSourceConfig",
    "ValidationResult",
    "YahooFinanceDataProvider",
    "AlphaVantageDataProvider",
    "FREDDataProvider",
    "NewsAPIDataProvider",
    "MarketDataValidator",
    "DataConsistencyChecker",
    "DataQualityMonitor",
    "ValidationLevel",
    "DataQualityThresholds",
    "DataIssue",
    "ADVANCED_PROVIDERS_AVAILABLE",
    "BackboneConnectivityReport",
    "ConnectivityProbeSnapshot",
]
