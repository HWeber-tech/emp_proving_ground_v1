"""Kafka streaming helpers aligned with the institutional data backbone roadmap."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable, Mapping, MutableMapping, Protocol, Sequence, cast

from src.core.event_bus import Event, EventBus, get_global_bus

from ..ingest.health import IngestHealthReport
from ..ingest.metrics import IngestMetricsSnapshot
from ..ingest.quality import IngestQualityReport
from ..persist.timescale import TimescaleConnectionSettings, TimescaleIngestResult
from ..persist.timescale_reader import TimescaleQueryResult, TimescaleReader

logger = logging.getLogger(__name__)


def _normalise_env(mapping: Mapping[str, str] | None) -> MutableMapping[str, str]:
    if mapping is None:
        return {k: v for k, v in os.environ.items() if isinstance(v, str)}
    return {str(k): str(v) for k, v in mapping.items()}


def _coerce_int(payload: Mapping[str, str], key: str, default: int) -> int:
    raw = payload.get(key)
    if raw is None:
        return default
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _coerce_bool(payload: Mapping[str, str], key: str, default: bool) -> bool:
    raw = payload.get(key)
    if raw is None:
        return default
    normalized = str(raw).strip().lower().replace("-", "_")
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _coerce_optional_float(
    payload: Mapping[str, str], key: str, default: float | None
) -> float | None:
    """Coerce a mapping entry into a float while accepting sentinels."""

    raw = payload.get(key)
    if raw is None:
        return default

    normalized = str(raw).strip().lower()
    if normalized in {"none", "null", "off", "disable", "disabled"}:
        return None

    try:
        return float(normalized)
    except (TypeError, ValueError):
        return default


def _normalise_dimension(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def ingest_topic_config_from_mapping(
    mapping: Mapping[str, str] | None,
) -> tuple[dict[str, str], str | None]:
    """Extract ingest topic configuration from environment-style mappings.

    The roadmap’s Kafka milestone calls for a predictable mapping between
    Timescale ingest dimensions (daily bars, intraday trades, macro events)
    and Kafka topics. Operators can configure topics in two ways:

    - ``KAFKA_INGEST_TOPICS": "daily_bars:topic.daily,intraday_trades:topic.intraday"``
    - Dedicated keys such as ``KAFKA_INGEST_TOPIC_DAILY_BARS``.

    A ``KAFKA_INGEST_DEFAULT_TOPIC`` (or ``*:topic`` entry) acts as a fallback
    for dimensions that do not have an explicit mapping.
    """

    payload = _normalise_env(mapping)
    topic_map: dict[str, str] = {}

    combined = payload.get("KAFKA_INGEST_TOPICS")
    if combined:
        for entry in str(combined).split(","):
            item = entry.strip()
            if not item:
                continue
            if ":" in item:
                key, topic = item.split(":", 1)
            elif "=" in item:
                key, topic = item.split("=", 1)
            else:
                continue
            dimension = _normalise_dimension(key)
            topic_name = topic.strip()
            if not topic_name:
                continue
            if dimension == "*":
                topic_map.setdefault("*", topic_name)
            else:
                topic_map[dimension] = topic_name

    prefix = "KAFKA_INGEST_TOPIC_"
    reserved_suffixes = (
        "_PARTITIONS",
        "_REPLICATION_FACTOR",
        "_CONFIG",
    )
    for key, value in payload.items():
        if not key.startswith(prefix):
            continue
        if any(key.endswith(suffix) for suffix in reserved_suffixes):
            continue
        dimension = _normalise_dimension(key[len(prefix) :])
        topic_name = value.strip()
        if not topic_name:
            continue
        topic_map[dimension] = topic_name

    default_topic = payload.get("KAFKA_INGEST_DEFAULT_TOPIC")
    default_topic = default_topic.strip() if default_topic else None

    if "*" in topic_map and not default_topic:
        default_topic = topic_map["*"]

    return topic_map, default_topic


@dataclass(frozen=True)
class KafkaTopicSpec:
    """Specification for Kafka topics that should exist for ingest streams."""

    name: str
    partitions: int = 1
    replication_factor: int = 1
    config: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not self.name or not str(self.name).strip():
            raise ValueError("Kafka topic name must not be empty")
        if int(self.partitions) <= 0:
            raise ValueError("Kafka topic partitions must be positive")
        if int(self.replication_factor) <= 0:
            raise ValueError("Kafka topic replication_factor must be positive")

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "partitions": int(self.partitions),
            "replication_factor": int(self.replication_factor),
        }
        if self.config:
            payload["config"] = dict(self.config)
        return payload


@dataclass(frozen=True)
class KafkaTopicProvisioningSummary:
    """Summary of Kafka topic provisioning attempts."""

    requested: tuple[str, ...] = field(default_factory=tuple)
    existing: tuple[str, ...] = field(default_factory=tuple)
    created: tuple[str, ...] = field(default_factory=tuple)
    failed: Mapping[str, str] = field(default_factory=dict)
    dry_run: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "requested": list(self.requested),
            "existing": list(self.existing),
            "created": list(self.created),
            "dry_run": self.dry_run,
        }
        if self.failed:
            payload["failed"] = dict(self.failed)
        if self.notes:
            payload["notes"] = list(self.notes)
        return payload


def _safe_int(value: object | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(float(text))
        except ValueError:
            return None
    return None


def _normalise_timestamp(value: object | None, *, fallback: datetime | None = None) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        else:
            value = value.astimezone(UTC)
        return value.isoformat()

    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds > 1e12:
            seconds /= 1000.0
        return datetime.fromtimestamp(seconds, tz=UTC).isoformat()

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return _normalise_timestamp(None, fallback=fallback)
        try:
            seconds = float(text)
        except ValueError:
            try:
                parsed = datetime.fromisoformat(text)
            except ValueError:
                return text
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            else:
                parsed = parsed.astimezone(UTC)
            return parsed.isoformat()
        return _normalise_timestamp(seconds, fallback=fallback)

    if fallback is None:
        fallback = datetime.now(tz=UTC)
    return fallback.isoformat()


def _first_present(mapping: Mapping[str, object], *keys: str) -> object | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


@dataclass(frozen=True)
class KafkaPartitionLag:
    """Lag metrics for a single Kafka topic partition."""

    topic: str
    partition: int
    current_offset: int | None = None
    end_offset: int | None = None
    lag: int | None = None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "topic": self.topic,
            "partition": int(self.partition),
        }
        if self.current_offset is not None:
            payload["current_offset"] = int(self.current_offset)
        if self.end_offset is not None:
            payload["end_offset"] = int(self.end_offset)
        if self.lag is not None:
            payload["lag"] = int(self.lag)
        return payload


@dataclass(frozen=True)
class KafkaConsumerLagSnapshot:
    """Structured view of Kafka consumer lag across subscribed partitions."""

    partitions: tuple[KafkaPartitionLag, ...]
    recorded_at: str
    total_lag: int | None = None
    max_lag: int | None = None
    topic_lag: Mapping[str, int] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "recorded_at": self.recorded_at,
            "partitions": [partition.as_dict() for partition in self.partitions],
        }
        if self.total_lag is not None:
            payload["total_lag"] = int(self.total_lag)
        if self.max_lag is not None:
            payload["max_lag"] = int(self.max_lag)
        if self.topic_lag:
            payload["topic_lag"] = {key: int(value) for key, value in self.topic_lag.items()}
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


def _snapshot_from_metrics(metrics: Mapping[str, object]) -> KafkaConsumerLagSnapshot | None:
    topics_raw = metrics.get("topics")
    if not isinstance(topics_raw, Mapping):
        return None

    partitions: list[KafkaPartitionLag] = []
    topic_totals: dict[str, int] = {}

    for topic, topic_data in topics_raw.items():
        if not isinstance(topic, str):
            topic = str(topic)
        if not isinstance(topic_data, Mapping):
            continue
        partitions_raw = topic_data.get("partitions")
        if not isinstance(partitions_raw, Mapping):
            continue
        for partition_key, partition_data in partitions_raw.items():
            if not isinstance(partition_data, Mapping):
                continue
            partition_id = _safe_int(partition_key)
            if partition_id is None:
                partition_id = _safe_int(partition_data.get("partition"))
            if partition_id is None:
                continue

            current_offset = _safe_int(
                _first_present(
                    partition_data,
                    "consumer_offset",
                    "committed_offset",
                    "current_offset",
                    "offset",
                )
            )
            end_offset = _safe_int(
                _first_present(
                    partition_data,
                    "hi_offset",
                    "end_offset",
                    "log_end_offset",
                    "watermark_high",
                )
            )
            lag = _safe_int(_first_present(partition_data, "consumer_lag", "lag"))
            if lag is None and current_offset is not None and end_offset is not None:
                lag = max(end_offset - current_offset, 0)

            partition = KafkaPartitionLag(
                topic=topic,
                partition=int(partition_id),
                current_offset=current_offset,
                end_offset=end_offset,
                lag=lag,
            )
            partitions.append(partition)
            if lag is not None:
                topic_totals[topic] = topic_totals.get(topic, 0) + lag

    if not partitions:
        return None

    known_lags = [p.lag for p in partitions if p.lag is not None]
    total_lag = sum(int(lag) for lag in known_lags) if known_lags else None
    max_lag = max(int(lag) for lag in known_lags) if known_lags else None

    recorded_at = _normalise_timestamp(
        metrics.get("time") or metrics.get("timestamp"),
        fallback=datetime.now(tz=UTC),
    )

    metadata: dict[str, object] = {}
    for key in ("client_id", "group_id", "name", "cluster_id"):
        value = metrics.get(key)
        if value is not None:
            metadata[key] = value

    if topic_totals:
        metadata.setdefault("topic_count", len(topic_totals))

    return KafkaConsumerLagSnapshot(
        partitions=tuple(sorted(partitions, key=lambda item: (item.topic, item.partition))),
        recorded_at=recorded_at,
        total_lag=total_lag,
        max_lag=max_lag,
        topic_lag=topic_totals,
        metadata=metadata,
    )


def capture_consumer_lag(consumer: KafkaConsumerLike) -> KafkaConsumerLagSnapshot | None:
    """Capture consumer lag metrics if the Kafka client exposes them."""

    metrics_fn = getattr(consumer, "metrics", None)
    if not callable(metrics_fn):
        return None

    try:
        metrics = metrics_fn()
    except Exception:  # pragma: no cover - defensive guard for flaky clients
        logger.exception("Kafka consumer lag probe failed to retrieve metrics")
        return None

    if not isinstance(metrics, Mapping):
        return None

    return _snapshot_from_metrics(metrics)


class KafkaAdminClientLike(Protocol):
    """Subset of :mod:`confluent_kafka` admin client behaviour used for provisioning."""

    def list_topics(
        self, timeout: float | None = None
    ) -> Any:  # pragma: no cover - protocol definition
        ...

    def create_topics(
        self, new_topics: Sequence[Any], request_timeout: float | None = None
    ) -> Mapping[str, Any] | Sequence[Any]:  # pragma: no cover - protocol definition
        ...


KafkaAdminFactory = Callable[[Mapping[str, Any]], KafkaAdminClientLike]
KafkaNewTopicFactory = Callable[[KafkaTopicSpec], Any]


def _extract_topic_names(metadata: Any) -> set[str]:
    topics: set[str] = set()
    if metadata is None:
        return topics
    raw_topics = getattr(metadata, "topics", None)
    if isinstance(raw_topics, Mapping):
        topics.update(str(name) for name in raw_topics.keys())
    elif isinstance(raw_topics, Sequence):
        topics.update(str(item) for item in raw_topics)
    return topics


def _parse_topic_config_payload(raw: str | None, label: str) -> dict[str, str]:
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Invalid JSON payload for %s; skipping", label)
        return {}
    if not isinstance(payload, Mapping):
        logger.warning(
            "Expected %s to decode to a JSON object; got %s", label, type(payload).__name__
        )
        return {}
    return {str(key): str(value) for key, value in payload.items()}


def _iter_topic_results(result: Mapping[str, Any] | Sequence[Any]) -> list[tuple[str, Any]]:
    if isinstance(result, Mapping):
        return [(str(name), future) for name, future in result.items()]
    ordered: list[tuple[str, Any]] = []
    for item in result:
        name = getattr(item, "topic", None)
        if name is None:
            continue
        ordered.append((str(name), item))
    return ordered


class KafkaTopicProvisioner:
    """Ensure ingest topics exist before publishers/consumers start."""

    def __init__(
        self,
        settings: KafkaConnectionSettings,
        *,
        admin_factory: KafkaAdminFactory | None = None,
        new_topic_factory: KafkaNewTopicFactory | None = None,
        list_timeout: float = 5.0,
        create_timeout: float = 10.0,
    ) -> None:
        self._settings = settings
        self._admin_factory = admin_factory
        self._new_topic_factory = new_topic_factory
        self._list_timeout = list_timeout
        self._create_timeout = create_timeout
        self._logger = logging.getLogger(f"{__name__}.KafkaTopicProvisioner")

    def ensure_topics(
        self,
        specs: Sequence[KafkaTopicSpec],
        *,
        dry_run: bool = False,
    ) -> KafkaTopicProvisioningSummary:
        if not specs:
            return KafkaTopicProvisioningSummary()

        unique_specs: dict[str, KafkaTopicSpec] = {}
        ordered_specs: list[KafkaTopicSpec] = []
        for spec in specs:
            if spec.name not in unique_specs:
                unique_specs[spec.name] = spec
                ordered_specs.append(spec)

        requested = tuple(spec.name for spec in ordered_specs)
        notes: list[str] = []

        if not self._settings.configured:
            notes.append("Kafka connection not configured; skipping topic provisioning")
            return KafkaTopicProvisioningSummary(requested=requested, notes=tuple(notes))

        admin_factory = self._admin_factory
        new_topic_factory = self._new_topic_factory
        if admin_factory is None or new_topic_factory is None:
            try:
                from confluent_kafka.admin import AdminClient, NewTopic
            except Exception:  # pragma: no cover - optional dependency
                notes.append("confluent_kafka unavailable; cannot provision topics")
                self._logger.warning(
                    "Kafka topic provisioning requested but confluent_kafka is not installed"
                )
                return KafkaTopicProvisioningSummary(
                    requested=requested,
                    notes=tuple(notes),
                )

            if admin_factory is None:
                def _default_admin_factory(config: Mapping[str, Any]) -> KafkaAdminClientLike:
                    return cast(KafkaAdminClientLike, AdminClient(config))

                admin_factory = _default_admin_factory
            if new_topic_factory is None:
                def _default_new_topic_factory(spec: KafkaTopicSpec) -> Any:
                    return NewTopic(
                        spec.name,
                        num_partitions=int(spec.partitions),
                        replication_factor=int(spec.replication_factor),
                        config=dict(spec.config or {}),
                    )

                new_topic_factory = _default_new_topic_factory

        admin = admin_factory(self._settings.admin_config())

        try:
            metadata = admin.list_topics(timeout=self._list_timeout)
        except Exception as exc:  # pragma: no cover - defensive guard
            self._logger.warning("Failed to list Kafka topics: %s", exc)
            notes.append(f"list_topics failed: {exc}")
            return KafkaTopicProvisioningSummary(requested=requested, notes=tuple(notes))

        existing = _extract_topic_names(metadata)
        existing_names = tuple(name for name in requested if name in existing)
        to_create = [spec for spec in ordered_specs if spec.name not in existing]

        if not to_create:
            return KafkaTopicProvisioningSummary(
                requested=requested,
                existing=existing_names,
                notes=tuple(notes),
            )

        if dry_run:
            notes.append("dry_run=True; topics not created")
            created_names = tuple(spec.name for spec in to_create)
            return KafkaTopicProvisioningSummary(
                requested=requested,
                existing=existing_names,
                created=created_names,
                dry_run=True,
                notes=tuple(notes),
            )

        futures = admin.create_topics(
            [new_topic_factory(spec) for spec in to_create],
            request_timeout=self._create_timeout,
        )

        created: list[str] = []
        failed: dict[str, str] = {}
        for name, future in _iter_topic_results(futures):
            try:
                result_fn = getattr(future, "result", None)
                if callable(result_fn):
                    result_fn()
                created.append(name)
            except Exception as exc:  # pragma: no cover - defensive logging
                failed[name] = str(exc)
                self._logger.warning("Failed to create Kafka topic %s: %s", name, exc)

        # Ensure we report topics that were requested but not acknowledged by the admin client
        remaining = {spec.name for spec in to_create} - set(created) - set(failed.keys())
        for name in remaining:
            failed[name] = "no result returned"

        created_names = tuple(created)
        return KafkaTopicProvisioningSummary(
            requested=requested,
            existing=existing_names,
            created=created_names,
            failed=failed,
            notes=tuple(notes),
        )


def resolve_ingest_topic_specs(
    mapping: Mapping[str, str] | None,
    *,
    default_partitions: int = 1,
    default_replication_factor: int = 1,
) -> list[KafkaTopicSpec]:
    """Build topic specifications for configured ingest dimensions."""

    payload = _normalise_env(mapping)
    topic_map, default_topic = ingest_topic_config_from_mapping(payload)
    if not topic_map and not default_topic:
        return []

    base_partitions = max(
        1, _coerce_int(payload, "KAFKA_INGEST_TOPIC_PARTITIONS", default_partitions)
    )
    base_replication = max(
        1, _coerce_int(payload, "KAFKA_INGEST_TOPIC_REPLICATION_FACTOR", default_replication_factor)
    )
    global_config = _parse_topic_config_payload(
        payload.get("KAFKA_INGEST_TOPIC_CONFIG"), "KAFKA_INGEST_TOPIC_CONFIG"
    )

    specs_by_topic: dict[str, KafkaTopicSpec] = {}
    ordered_topics: list[str] = []

    def _dimension_key(dimension: str) -> str:
        return "DEFAULT" if dimension in {"*", "default"} else dimension.upper()

    def _topic_config_for(dimension: str) -> dict[str, str]:
        dimension_key = _dimension_key(dimension)
        config = dict(global_config)
        label = (
            "KAFKA_INGEST_DEFAULT_TOPIC_CONFIG"
            if dimension_key == "DEFAULT"
            else f"KAFKA_INGEST_TOPIC_{dimension_key}_CONFIG"
        )
        config.update(_parse_topic_config_payload(payload.get(label), label))
        return {k: v for k, v in config.items() if v is not None}

    def _partitions_for(dimension: str) -> int:
        dimension_key = _dimension_key(dimension)
        label = (
            "KAFKA_INGEST_DEFAULT_TOPIC_PARTITIONS"
            if dimension_key == "DEFAULT"
            else f"KAFKA_INGEST_TOPIC_{dimension_key}_PARTITIONS"
        )
        return max(1, _coerce_int(payload, label, base_partitions))

    def _replication_for(dimension: str) -> int:
        dimension_key = _dimension_key(dimension)
        label = (
            "KAFKA_INGEST_DEFAULT_TOPIC_REPLICATION_FACTOR"
            if dimension_key == "DEFAULT"
            else f"KAFKA_INGEST_TOPIC_{dimension_key}_REPLICATION_FACTOR"
        )
        return max(1, _coerce_int(payload, label, base_replication))

    def _register_spec(dimension: str, topic_name: str) -> None:
        if not topic_name:
            return
        if topic_name in specs_by_topic:
            return
        config = _topic_config_for(dimension)
        spec = KafkaTopicSpec(
            name=topic_name,
            partitions=_partitions_for(dimension),
            replication_factor=_replication_for(dimension),
            config=config or None,
        )
        specs_by_topic[topic_name] = spec
        ordered_topics.append(topic_name)

    for dimension, topic_name in topic_map.items():
        _register_spec(dimension, topic_name)

    if default_topic:
        _register_spec("*", default_topic)

    return [specs_by_topic[name] for name in ordered_topics]


def should_auto_create_topics(mapping: Mapping[str, str] | None) -> bool:
    """Check whether ingest topic provisioning should run automatically."""

    payload = _normalise_env(mapping)
    if not payload:
        return False
    if _coerce_bool(payload, "KAFKA_INGEST_AUTO_CREATE_TOPICS", False):
        return True
    return _coerce_bool(payload, "KAFKA_AUTO_CREATE_TOPICS", False)


def create_ingest_event_publisher(
    settings: "KafkaConnectionSettings",
    mapping: Mapping[str, str] | None = None,
    *,
    producer_factory: KafkaProducerFactory | None = None,
    serializer: Callable[[Mapping[str, object]], bytes] | None = None,
) -> "KafkaIngestEventPublisher" | None:
    """Instantiate a :class:`KafkaIngestEventPublisher` when topics are defined.

    The helper keeps runtime wiring terse: it reads the ingest topic mapping,
    falls back gracefully when Kafka credentials are incomplete, and attempts to
    import ``confluent_kafka`` when a custom producer factory is not provided.
    """

    if not settings.configured:
        logger.debug("Kafka ingest publisher skipped: no brokers configured")
        return None

    payload = _normalise_env(mapping)
    topic_map, default_topic = ingest_topic_config_from_mapping(payload)
    if not topic_map and not default_topic:
        logger.warning(
            "Kafka configured (%s) but no ingest topics defined; set "
            "KAFKA_INGEST_TOPICS or KAFKA_INGEST_DEFAULT_TOPIC",
            settings.summary(redacted=True),
        )
        return None

    if producer_factory is None:
        try:
            from confluent_kafka import Producer
        except Exception:  # pragma: no cover - depends on optional package
            logger.warning(
                "Kafka ingest requested (%s) but confluent_kafka is not installed;"
                " skipping publisher setup",
                settings.summary(redacted=True),
            )
            return None

        def _default_producer_factory(config: Mapping[str, Any]) -> KafkaProducerLike:
            return cast(KafkaProducerLike, Producer(config))

        producer_factory = _default_producer_factory

    producer = settings.create_producer(factory=producer_factory)
    if producer is None:
        logger.warning(
            "Kafka ingest settings present (%s) but producer factory returned None",
            settings.summary(redacted=True),
        )
        return None

    flush_timeout = _coerce_optional_float(payload, "KAFKA_INGEST_FLUSH_TIMEOUT", 2.0)
    return KafkaIngestEventPublisher(
        producer,
        topic_map={k: v for k, v in topic_map.items() if k != "*"},
        default_topic=default_topic,
        serializer=serializer,
        flush_timeout=flush_timeout,
    )


class KafkaProducerLike(Protocol):
    """Minimal protocol describing the operations used by the publisher."""

    def produce(self, topic: str, value: bytes, key: str | bytes | None = None) -> Any: ...

    def flush(self, timeout: float | None = None) -> Any:  # pragma: no cover - optional at runtime
        ...


KafkaProducerFactory = Callable[[Mapping[str, Any]], KafkaProducerLike]


class KafkaMessageLike(Protocol):
    """Protocol describing the subset of confluent_kafka.Message we rely on."""

    def error(self) -> object | None:  # pragma: no cover - runtime guard only
        ...

    def value(self) -> bytes | bytearray | str | None: ...

    def topic(self) -> str: ...

    def key(self) -> bytes | str | None: ...


class KafkaConsumerLike(Protocol):
    """Minimal protocol describing the Kafka consumer surface used by the bridge."""

    def subscribe(self, topics: Sequence[str]) -> None: ...

    def poll(self, timeout: float | None = None) -> KafkaMessageLike | None: ...

    def commit(
        self,
        message: KafkaMessageLike | None = None,
        asynchronous: bool = False,
    ) -> Any:  # pragma: no cover - runtime guard
        ...

    def close(self) -> None:  # pragma: no cover - depends on runtime consumer
        ...


KafkaConsumerFactory = Callable[[Mapping[str, Any]], KafkaConsumerLike]


@dataclass(frozen=True)
class KafkaConnectionSettings:
    """Connection details required to instantiate a Kafka producer."""

    bootstrap_servers: str
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: str | None = None
    username: str | None = None
    password: str | None = None
    client_id: str = "emp-institutional-ingest"
    acks: str = "all"
    linger_ms: int = 20
    batch_size: int = 32768
    enable_idempotence: bool = True
    compression_type: str | None = None

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, str] | None = None) -> "KafkaConnectionSettings":
        data = _normalise_env(mapping)

        bootstrap = (
            data.get("KAFKA_BROKERS")
            or data.get("KAFKA_BOOTSTRAP_SERVERS")
            or data.get("KAFKA_URL")
            or ""
        )
        bootstrap = ",".join(part.strip() for part in bootstrap.split(",") if part.strip())

        username = data.get("KAFKA_USERNAME") or data.get("KAFKA_USER")
        username = username.strip() if username else None

        password = data.get("KAFKA_PASSWORD") or data.get("KAFKA_PASS")
        password = password.strip() if password else None

        security_protocol = data.get("KAFKA_SECURITY_PROTOCOL")
        if security_protocol:
            security_protocol = security_protocol.strip().upper()
        elif username and password:
            security_protocol = "SASL_SSL"
        else:
            security_protocol = "PLAINTEXT"

        sasl_mechanism = data.get("KAFKA_SASL_MECHANISM")
        sasl_mechanism = sasl_mechanism.strip().upper() if sasl_mechanism else None
        if username and password and sasl_mechanism is None:
            sasl_mechanism = "PLAIN"

        client_id = data.get("KAFKA_CLIENT_ID") or "emp-institutional-ingest"
        client_id = client_id.strip()

        acks = data.get("KAFKA_ACKS") or "all"
        acks = acks.strip()

        linger_ms = _coerce_int(data, "KAFKA_LINGER_MS", 20)
        batch_size = _coerce_int(data, "KAFKA_BATCH_SIZE", 32768)
        enable_idempotence = _coerce_bool(data, "KAFKA_ENABLE_IDEMPOTENCE", True)

        compression_type = data.get("KAFKA_COMPRESSION_TYPE")
        compression_type = compression_type.strip().lower() if compression_type else None

        return cls(
            bootstrap_servers=bootstrap,
            security_protocol=security_protocol,
            sasl_mechanism=sasl_mechanism,
            username=username,
            password=password,
            client_id=client_id,
            acks=acks,
            linger_ms=linger_ms,
            batch_size=batch_size,
            enable_idempotence=enable_idempotence,
            compression_type=compression_type,
        )

    @classmethod
    def from_env(cls) -> "KafkaConnectionSettings":
        return cls.from_mapping(os.environ)

    @property
    def configured(self) -> bool:
        return bool(self.bootstrap_servers)

    def _base_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {"bootstrap.servers": self.bootstrap_servers}
        if self.security_protocol:
            config["security.protocol"] = self.security_protocol
        if self.sasl_mechanism:
            config["sasl.mechanism"] = self.sasl_mechanism
        if self.username is not None:
            config["sasl.username"] = self.username
        if self.password is not None:
            config["sasl.password"] = self.password
        return config

    def client_config(self) -> dict[str, Any]:
        config = self._base_config()
        config.update(
            {
                "client.id": self.client_id,
                "acks": self.acks,
                "linger.ms": self.linger_ms,
                "batch.size": self.batch_size,
            }
        )
        if self.enable_idempotence:
            config["enable.idempotence"] = True
        if self.compression_type:
            config["compression.type"] = self.compression_type
        return config

    def admin_config(self) -> dict[str, Any]:
        """Return configuration suitable for Kafka admin clients."""

        return self._base_config()

    def consumer_config(
        self,
        *,
        group_id: str,
        auto_offset_reset: str = "latest",
        enable_auto_commit: bool = True,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        config = self._base_config()
        config.update(
            {
                "group.id": group_id,
                "client.id": self.client_id,
                "auto.offset.reset": auto_offset_reset,
            }
        )
        if not enable_auto_commit:
            config["enable.auto.commit"] = False
        if extra:
            config.update({str(k): v for k, v in extra.items()})
        return config

    def create_producer(
        self, *, factory: KafkaProducerFactory | None = None
    ) -> KafkaProducerLike | None:
        if not self.configured:
            logger.debug("Kafka connection not configured; skipping producer creation")
            return None
        if factory is None:
            raise ValueError("A Kafka producer factory must be provided")
        config = self.client_config()
        logger.debug("Creating Kafka producer with config: %s", self.summary(redacted=True))
        return factory(config)

    def create_consumer(
        self,
        *,
        factory: KafkaConsumerFactory | None = None,
        group_id: str,
        auto_offset_reset: str = "latest",
        enable_auto_commit: bool = True,
        extra: Mapping[str, Any] | None = None,
    ) -> KafkaConsumerLike | None:
        if not self.configured:
            logger.debug("Kafka connection not configured; skipping consumer creation")
            return None
        if factory is None:
            raise ValueError("A Kafka consumer factory must be provided")
        config = self.consumer_config(
            group_id=group_id,
            auto_offset_reset=auto_offset_reset,
            enable_auto_commit=enable_auto_commit,
            extra=extra,
        )
        logger.debug(
            "Creating Kafka consumer for group %s (auto_offset_reset=%s)",
            group_id,
            auto_offset_reset,
        )
        return factory(config)

    def summary(self, *, redacted: bool = False) -> str:
        if not self.configured:
            return "Kafka connection not configured"
        parts = [
            f"bootstrap={self.bootstrap_servers}",
            f"security={self.security_protocol}",
            f"acks={self.acks}",
            f"client_id={self.client_id}",
        ]
        if self.sasl_mechanism:
            parts.append(f"sasl={self.sasl_mechanism}")
        if self.username:
            parts.append("username=***" if redacted else f"username={self.username}")
        if self.password:
            parts.append("password=***" if redacted else "password=<set>")
        return "Kafka(" + ", ".join(parts) + ")"


def _sanitise_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    def _coerce(value: object) -> object:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, datetime):
            ts = value if value.tzinfo else value.replace(tzinfo=UTC)
            return ts.astimezone(UTC).isoformat()
        if isinstance(value, Mapping):
            return {str(k): _coerce(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_coerce(v) for v in value]
        return str(value)

    return {str(k): _coerce(v) for k, v in metadata.items()}


def _default_serializer(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _default_deserializer(data: bytes | bytearray | str) -> Mapping[str, object]:
    if isinstance(data, (bytes, bytearray)):
        text = data.decode("utf-8")
    else:
        text = data
    payload = json.loads(text)
    if not isinstance(payload, Mapping):
        raise ValueError("Kafka ingest payload must decode to a mapping")
    return dict(payload)


class KafkaIngestEventPublisher:
    """Publish Timescale ingest outcomes to Kafka topics."""

    def __init__(
        self,
        producer: KafkaProducerLike,
        *,
        topic_map: Mapping[str, str],
        default_topic: str | None = None,
        serializer: Callable[[Mapping[str, object]], bytes] | None = None,
        flush_timeout: float | None = 2.0,
    ) -> None:
        self._producer = producer
        self._topic_map = {str(k): str(v) for k, v in topic_map.items() if str(v).strip()}
        self._default_topic = default_topic or self._topic_map.get("*")
        self._serializer = serializer or _default_serializer
        self._flush_timeout = flush_timeout
        self._logger = logging.getLogger(f"{__name__}.KafkaIngestEventPublisher")

    def _resolve_topic(self, dimension: str) -> str | None:
        topic = self._topic_map.get(dimension)
        if topic:
            return topic
        return self._default_topic

    def publish(
        self,
        result: TimescaleIngestResult,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        topic = self._resolve_topic(result.dimension)
        if not topic:
            self._logger.debug(
                "No Kafka topic configured for dimension %s; skipping", result.dimension
            )
            return

        payload: dict[str, object] = {
            "emitted_at": datetime.now(tz=UTC).isoformat(),
            "result": result.as_dict(),
        }
        if metadata:
            payload["metadata"] = _sanitise_metadata(metadata)

        key = result.symbols[0] if result.symbols else result.dimension
        try:
            message = self._serializer(payload)
            self._producer.produce(topic, value=message, key=key)
        except Exception:
            self._logger.exception(
                "Failed to publish ingest result for %s to Kafka topic %s", result.dimension, topic
            )
            return

        if self._flush_timeout is None:
            return

        flush = getattr(self._producer, "flush", None)
        if callable(flush):
            try:
                flush(self._flush_timeout)
            except Exception:  # pragma: no cover - safety net for runtime failures
                self._logger.exception("Kafka producer flush failed for topic %s", topic)


@dataclass(frozen=True)
class KafkaIngestBackfillSummary:
    """Summary of a Timescale → Kafka ingest backfill run."""

    dimension: str
    topic: str
    rows: int
    symbols: tuple[str, ...]
    start_ts: datetime | None
    end_ts: datetime | None
    freshness_seconds: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "topic": self.topic,
            "rows": self.rows,
            "symbols": list(self.symbols),
            "start_ts": self.start_ts.isoformat() if self.start_ts else None,
            "end_ts": self.end_ts.isoformat() if self.end_ts else None,
            "freshness_seconds": self.freshness_seconds,
        }


def _canonical_dimension(dimension: str) -> str:
    normalized = _normalise_dimension(dimension)
    mapping = {
        "daily": "daily_bars",
        "daily_bar": "daily_bars",
        "daily_bars": "daily_bars",
        "intraday": "intraday_trades",
        "intraday_trade": "intraday_trades",
        "intraday_trades": "intraday_trades",
        "macro": "macro_events",
        "macro_event": "macro_events",
        "macro_events": "macro_events",
    }
    try:
        return mapping[normalized]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unsupported ingest dimension: {dimension}") from exc


def _fetch_timescale_dimension(
    reader: TimescaleReader,
    *,
    dimension: str,
    identifiers: Sequence[str] | None,
    start: datetime | None,
    end: datetime | None,
    limit: int | None,
) -> TimescaleQueryResult:
    if dimension == "daily_bars":
        return reader.fetch_daily_bars(symbols=identifiers, start=start, end=end, limit=limit)
    if dimension == "intraday_trades":
        return reader.fetch_intraday_trades(symbols=identifiers, start=start, end=end, limit=limit)
    if dimension == "macro_events":
        return reader.fetch_macro_events(calendars=identifiers, start=start, end=end, limit=limit)
    raise ValueError(f"Unsupported ingest dimension: {dimension}")


def backfill_ingest_dimension_to_kafka(
    *,
    dimension: str,
    timescale_settings: TimescaleConnectionSettings,
    kafka_settings: "KafkaConnectionSettings",
    mapping: Mapping[str, str] | None = None,
    identifiers: Sequence[str] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int | None = None,
    producer_factory: KafkaProducerFactory | None = None,
    serializer: Callable[[Mapping[str, object]], bytes] | None = None,
) -> KafkaIngestBackfillSummary:
    """Replay a Timescale dimension into the configured Kafka ingest topic.

    The roadmap's Kafka follow-ups require tooling that can republish historical
    Timescale ingest results into Kafka so downstream consumers (and new
    environments) inherit the same telemetry history.  This helper reads from the
    Timescale tables using :class:`TimescaleReader`, wraps the results in a
    :class:`TimescaleIngestResult`, and emits it through the standard Kafka
    ingest publisher with explicit ``backfill`` metadata.
    """

    canonical_dimension = _canonical_dimension(dimension)
    payload = _normalise_env(mapping)
    topic_map, default_topic = ingest_topic_config_from_mapping(payload)
    topic = topic_map.get(canonical_dimension) or default_topic
    if not topic:
        raise ValueError(f"No Kafka ingest topic configured for dimension {canonical_dimension}")

    publisher = create_ingest_event_publisher(
        kafka_settings,
        payload,
        producer_factory=producer_factory,
        serializer=serializer,
    )
    if publisher is None:
        raise RuntimeError("Kafka ingest publisher is not configured; cannot run backfill")

    query_started = time.perf_counter()
    engine = timescale_settings.create_engine()
    try:
        reader = TimescaleReader(engine)
        query_result = _fetch_timescale_dimension(
            reader,
            dimension=canonical_dimension,
            identifiers=identifiers,
            start=start,
            end=end,
            limit=limit,
        )
    finally:
        engine.dispose()

    duration = time.perf_counter() - query_started
    reference = datetime.now(tz=UTC)
    freshness = query_result.freshness_age_seconds(reference=reference)

    ingest_result = TimescaleIngestResult(
        rows_written=query_result.rowcount,
        symbols=query_result.symbols,
        start_ts=query_result.start_ts,
        end_ts=query_result.end_ts,
        ingest_duration_seconds=duration,
        freshness_seconds=freshness,
        dimension=canonical_dimension,
        source="timescale_backfill",
    )

    metadata: dict[str, object] = {
        "backfill": True,
        "requested_filters": {
            "identifiers": list(identifiers) if identifiers else None,
            "start": start.isoformat() if start else None,
            "end": end.isoformat() if end else None,
            "limit": limit,
        },
    }

    publisher.publish(ingest_result, metadata=metadata)

    return KafkaIngestBackfillSummary(
        dimension=canonical_dimension,
        topic=topic,
        rows=query_result.rowcount,
        symbols=query_result.symbols,
        start_ts=query_result.start_ts,
        end_ts=query_result.end_ts,
        freshness_seconds=freshness,
    )


class KafkaIngestHealthPublisher:
    """Publish ingest health reports to a Kafka telemetry topic."""

    def __init__(
        self,
        producer: KafkaProducerLike,
        *,
        topic: str,
        key: str | None = "ingest_health",
        serializer: Callable[[Mapping[str, object]], bytes] | None = None,
        flush_timeout: float | None = 2.0,
    ) -> None:
        self._producer = producer
        self._topic = str(topic).strip()
        self._key = key
        self._serializer = serializer or _default_serializer
        self._flush_timeout = flush_timeout
        self._logger = logging.getLogger(f"{__name__}.KafkaIngestHealthPublisher")

    @property
    def topic(self) -> str:
        return self._topic

    @property
    def key(self) -> str | None:
        return self._key

    @property
    def flush_timeout(self) -> float | None:
        return self._flush_timeout

    def summary(self) -> str:
        key = self._key if self._key is not None else "<auto>"
        if self._flush_timeout is None:
            flush_desc = "flush=disabled"
        else:
            flush_desc = f"flush_timeout={self._flush_timeout:.2f}s"
        return f"Kafka ingest health topic {self._topic} (key={key}, {flush_desc})"

    def publish(
        self,
        report: IngestHealthReport,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        payload: dict[str, object] = {
            "emitted_at": datetime.now(tz=UTC).isoformat(),
            "report": report.as_dict(),
        }
        if metadata:
            payload["metadata"] = _sanitise_metadata(metadata)

        try:
            message = self._serializer(payload)
            self._producer.produce(self._topic, value=message, key=self._key)
        except Exception:
            self._logger.exception(
                "Failed to publish ingest health report to Kafka topic %s",
                self._topic,
            )
            return

        if self._flush_timeout is None:
            return

        flush = getattr(self._producer, "flush", None)
        if callable(flush):
            try:
                flush(self._flush_timeout)
            except Exception:  # pragma: no cover - runtime guard
                self._logger.exception(
                    "Kafka producer flush failed for ingest health topic %s",
                    self._topic,
                )


class KafkaIngestMetricsPublisher:
    """Publish ingest metrics snapshots to a Kafka telemetry topic."""

    def __init__(
        self,
        producer: KafkaProducerLike,
        *,
        topic: str,
        key: str | None = "ingest_metrics",
        serializer: Callable[[Mapping[str, object]], bytes] | None = None,
        flush_timeout: float | None = 2.0,
    ) -> None:
        self._producer = producer
        self._topic = str(topic).strip()
        self._key = key
        self._serializer = serializer or _default_serializer
        self._flush_timeout = flush_timeout
        self._logger = logging.getLogger(f"{__name__}.KafkaIngestMetricsPublisher")

    @property
    def topic(self) -> str:
        return self._topic

    @property
    def key(self) -> str | None:
        return self._key

    @property
    def flush_timeout(self) -> float | None:
        return self._flush_timeout

    def summary(self) -> str:
        key = self._key if self._key is not None else "<auto>"
        if self._flush_timeout is None:
            flush_desc = "flush=disabled"
        else:
            flush_desc = f"flush_timeout={self._flush_timeout:.2f}s"
        return f"Kafka ingest metrics topic {self._topic} (key={key}, {flush_desc})"

    def publish(
        self,
        snapshot: IngestMetricsSnapshot,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        payload: dict[str, object] = {
            "emitted_at": datetime.now(tz=UTC).isoformat(),
            "metrics": snapshot.as_dict(),
        }
        if metadata:
            payload["metadata"] = _sanitise_metadata(metadata)

        try:
            message = self._serializer(payload)
            self._producer.produce(self._topic, value=message, key=self._key)
        except Exception:
            self._logger.exception(
                "Failed to publish ingest metrics snapshot to Kafka topic %s",
                self._topic,
            )
            return

        if self._flush_timeout is None:
            return

        flush = getattr(self._producer, "flush", None)
        if callable(flush):
            try:
                flush(self._flush_timeout)
            except Exception:  # pragma: no cover - runtime guard
                self._logger.exception(
                    "Kafka producer flush failed for ingest metrics topic %s",
                    self._topic,
                )


class KafkaIngestQualityPublisher:
    """Publish ingest quality reports to a Kafka telemetry topic."""

    def __init__(
        self,
        producer: KafkaProducerLike,
        *,
        topic: str,
        key: str | None = "ingest_quality",
        serializer: Callable[[Mapping[str, object]], bytes] | None = None,
        flush_timeout: float | None = 2.0,
    ) -> None:
        self._producer = producer
        self._topic = str(topic).strip()
        self._key = key
        self._serializer = serializer or _default_serializer
        self._flush_timeout = flush_timeout
        self._logger = logging.getLogger(f"{__name__}.KafkaIngestQualityPublisher")

    @property
    def topic(self) -> str:
        return self._topic

    @property
    def key(self) -> str | None:
        return self._key

    @property
    def flush_timeout(self) -> float | None:
        return self._flush_timeout

    def summary(self) -> str:
        key = self._key if self._key is not None else "<auto>"
        if self._flush_timeout is None:
            flush_desc = "flush=disabled"
        else:
            flush_desc = f"flush_timeout={self._flush_timeout:.2f}s"
        return f"Kafka ingest quality topic {self._topic} (key={key}, {flush_desc})"

    def publish(
        self,
        report: IngestQualityReport,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        payload: dict[str, object] = {
            "emitted_at": datetime.now(tz=UTC).isoformat(),
            "quality": report.as_dict(),
        }
        if metadata:
            payload["metadata"] = _sanitise_metadata(metadata)

        try:
            message = self._serializer(payload)
            self._producer.produce(self._topic, value=message, key=self._key)
        except Exception:
            self._logger.exception(
                "Failed to publish ingest quality report to Kafka topic %s",
                self._topic,
            )
            return

        if self._flush_timeout is None:
            return

        flush = getattr(self._producer, "flush", None)
        if callable(flush):
            try:
                flush(self._flush_timeout)
            except Exception:  # pragma: no cover - runtime guard
                self._logger.exception(
                    "Kafka producer flush failed for ingest quality topic %s",
                    self._topic,
                )


def create_ingest_health_publisher(
    settings: "KafkaConnectionSettings",
    mapping: Mapping[str, str] | None = None,
    *,
    producer_factory: KafkaProducerFactory | None = None,
    serializer: Callable[[Mapping[str, object]], bytes] | None = None,
) -> KafkaIngestHealthPublisher | None:
    """Instantiate a :class:`KafkaIngestHealthPublisher` when configured."""

    if not settings.configured:
        logger.debug("Kafka ingest health publisher skipped: no brokers configured")
        return None

    payload = _normalise_env(mapping)
    enabled = _coerce_bool(payload, "KAFKA_INGEST_HEALTH_ENABLED", True)
    if not enabled:
        logger.info("Kafka ingest health publisher disabled via configuration")
        return None

    topic = payload.get("KAFKA_INGEST_HEALTH_TOPIC")
    topic = topic.strip() if topic else None
    if not topic:
        fallback = payload.get("KAFKA_INGEST_DEFAULT_TOPIC")
        topic = fallback.strip() if fallback else None

    if not topic:
        logger.warning(
            "Kafka ingest health publisher skipped: no KAFKA_INGEST_HEALTH_TOPIC configured",
        )
        return None

    if producer_factory is None:
        try:
            from confluent_kafka import Producer
        except Exception:  # pragma: no cover - optional dependency
            logger.warning(
                "Kafka ingest health requested (%s) but confluent_kafka is not installed; skipping",
                settings.summary(redacted=True),
            )
            return None

        def _default_producer_factory(config: Mapping[str, Any]) -> KafkaProducerLike:
            return cast(KafkaProducerLike, Producer(config))

        producer_factory = _default_producer_factory

    producer = settings.create_producer(factory=producer_factory)
    if producer is None:
        logger.warning(
            "Kafka ingest health settings present (%s) but producer factory returned None",
            settings.summary(redacted=True),
        )
        return None

    flush_timeout = _coerce_optional_float(payload, "KAFKA_INGEST_HEALTH_FLUSH_TIMEOUT", 2.0)
    key = payload.get("KAFKA_INGEST_HEALTH_KEY")
    key = key.strip() if isinstance(key, str) and key.strip() else None
    return KafkaIngestHealthPublisher(
        producer,
        topic=topic,
        key=key or "ingest_health",
        serializer=serializer,
        flush_timeout=flush_timeout,
    )


def create_ingest_metrics_publisher(
    settings: "KafkaConnectionSettings",
    mapping: Mapping[str, str] | None = None,
    *,
    producer_factory: KafkaProducerFactory | None = None,
    serializer: Callable[[Mapping[str, object]], bytes] | None = None,
) -> KafkaIngestMetricsPublisher | None:
    """Instantiate a :class:`KafkaIngestMetricsPublisher` when configured."""

    if not settings.configured:
        logger.debug("Kafka ingest metrics publisher skipped: no brokers configured")
        return None

    payload = _normalise_env(mapping)
    enabled = _coerce_bool(payload, "KAFKA_INGEST_METRICS_ENABLED", True)
    if not enabled:
        logger.info("Kafka ingest metrics publisher disabled via configuration")
        return None

    topic = payload.get("KAFKA_INGEST_METRICS_TOPIC")
    topic = topic.strip() if topic else None
    if not topic:
        fallback = payload.get("KAFKA_INGEST_DEFAULT_TOPIC")
        topic = fallback.strip() if fallback else None

    if not topic:
        logger.warning(
            "Kafka ingest metrics publisher skipped: no KAFKA_INGEST_METRICS_TOPIC configured",
        )
        return None

    if producer_factory is None:
        try:
            from confluent_kafka import Producer
        except Exception:  # pragma: no cover - optional dependency
            logger.warning(
                "Kafka ingest metrics requested (%s) but confluent_kafka is not installed; skipping",
                settings.summary(redacted=True),
            )
            return None

        def _default_producer_factory(config: Mapping[str, Any]) -> KafkaProducerLike:
            return cast(KafkaProducerLike, Producer(config))

        producer_factory = _default_producer_factory

    producer = settings.create_producer(factory=producer_factory)
    if producer is None:
        logger.warning(
            "Kafka ingest metrics settings present (%s) but producer factory returned None",
            settings.summary(redacted=True),
        )
        return None

    default_flush = _coerce_optional_float(payload, "KAFKA_INGEST_FLUSH_TIMEOUT", 2.0)
    flush_timeout = _coerce_optional_float(
        payload, "KAFKA_INGEST_METRICS_FLUSH_TIMEOUT", default_flush
    )
    key = payload.get("KAFKA_INGEST_METRICS_KEY")
    key = key.strip() if isinstance(key, str) and key.strip() else "ingest_metrics"
    return KafkaIngestMetricsPublisher(
        producer,
        topic=topic,
        key=key,
        serializer=serializer,
        flush_timeout=flush_timeout,
    )


def create_ingest_quality_publisher(
    settings: "KafkaConnectionSettings",
    mapping: Mapping[str, str] | None = None,
    *,
    producer_factory: KafkaProducerFactory | None = None,
    serializer: Callable[[Mapping[str, object]], bytes] | None = None,
) -> KafkaIngestQualityPublisher | None:
    """Instantiate a :class:`KafkaIngestQualityPublisher` when configured."""

    if not settings.configured:
        logger.debug("Kafka ingest quality publisher skipped: no brokers configured")
        return None

    payload = _normalise_env(mapping)
    enabled = _coerce_bool(payload, "KAFKA_INGEST_QUALITY_ENABLED", True)
    if not enabled:
        logger.info("Kafka ingest quality publisher disabled via configuration")
        return None

    topic = payload.get("KAFKA_INGEST_QUALITY_TOPIC")
    topic = topic.strip() if topic else None
    if not topic:
        fallback = payload.get("KAFKA_INGEST_DEFAULT_TOPIC")
        topic = fallback.strip() if fallback else None

    if not topic:
        logger.warning(
            "Kafka ingest quality publisher skipped: no KAFKA_INGEST_QUALITY_TOPIC configured",
        )
        return None

    if producer_factory is None:
        try:
            from confluent_kafka import Producer
        except Exception:  # pragma: no cover - optional dependency
            logger.warning(
                "Kafka ingest quality requested (%s) but confluent_kafka is not installed; skipping",
                settings.summary(redacted=True),
            )
            return None

        def _default_producer_factory(config: Mapping[str, Any]) -> KafkaProducerLike:
            return cast(KafkaProducerLike, Producer(config))

        producer_factory = _default_producer_factory

    producer = settings.create_producer(factory=producer_factory)
    if producer is None:
        logger.warning(
            "Kafka ingest quality settings present (%s) but producer factory returned None",
            settings.summary(redacted=True),
        )
        return None

    default_flush = _coerce_optional_float(payload, "KAFKA_INGEST_FLUSH_TIMEOUT", 2.0)
    flush_timeout = _coerce_optional_float(
        payload, "KAFKA_INGEST_QUALITY_FLUSH_TIMEOUT", default_flush
    )
    key = payload.get("KAFKA_INGEST_QUALITY_KEY")
    key = key.strip() if isinstance(key, str) and key.strip() else "ingest_quality"
    return KafkaIngestQualityPublisher(
        producer,
        topic=topic,
        key=key,
        serializer=serializer,
        flush_timeout=flush_timeout,
    )


class KafkaIngestEventConsumer:
    """Consume ingest events from Kafka and replay them onto the runtime event bus."""

    def __init__(
        self,
        consumer: KafkaConsumerLike,
        *,
        topics: Sequence[str],
        event_bus: EventBus,
        event_type: str = "telemetry.ingest",
        source: str = "timescale_ingest.kafka",
        poll_timeout: float = 1.0,
        deserializer: Callable[[bytes | bytearray | str], Mapping[str, object]] | None = None,
        idle_sleep: float = 0.5,
        commit_offsets: bool = False,
        commit_asynchronously: bool = False,
        publish_consumer_lag: bool = False,
        consumer_lag_event_type: str = "telemetry.kafka.lag",
        consumer_lag_source: str | None = None,
        consumer_lag_interval: float | None = 30.0,
    ) -> None:
        self._consumer = consumer
        self._topics = tuple(dict.fromkeys(topic for topic in topics if topic))
        self._event_bus = event_bus
        self._event_type = event_type
        self._source = source
        self._poll_timeout = max(0.0, float(poll_timeout))
        self._deserializer = deserializer or _default_deserializer
        self._idle_sleep = max(0.0, float(idle_sleep))
        self._logger = logging.getLogger(f"{__name__}.KafkaIngestEventConsumer")
        self._started = False
        self._closed = False
        self._commit_offsets = bool(commit_offsets)
        self._commit_asynchronously = bool(commit_asynchronously)
        self._commit_warning_logged = False
        self._publish_consumer_lag = bool(publish_consumer_lag)
        lag_source = consumer_lag_source or f"{source}.lag"
        self._consumer_lag_source = lag_source.strip() or source
        self._consumer_lag_event_type = consumer_lag_event_type.strip() or "telemetry.kafka.lag"
        if consumer_lag_interval is None:
            self._consumer_lag_interval = 0.0
        else:
            self._consumer_lag_interval = max(0.0, float(consumer_lag_interval))
        self._last_consumer_lag_publish: float | None = None

        if not self._topics:
            raise ValueError("Kafka ingest consumer requires at least one topic")

    @property
    def topics(self) -> tuple[str, ...]:
        return self._topics

    def summary(self) -> str:
        return (
            f"KafkaIngestEventConsumer(topics={list(self._topics)}, event_type={self._event_type})"
        )

    def start(self) -> None:
        if self._started:
            return
        self._consumer.subscribe(list(self._topics))
        self._started = True
        self._logger.debug("Kafka ingest consumer subscribed to %s", list(self._topics))

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._consumer.close()
        except Exception:  # pragma: no cover - defensive logging
            self._logger.exception("Kafka ingest consumer close failed")
        finally:
            self._closed = True

    def _publish_event(
        self, *, event_type: str, payload: Mapping[str, object], source: str
    ) -> None:
        event = Event(
            type=event_type,
            payload=dict(payload),
            source=source,
        )

        publish_from_sync = getattr(self._event_bus, "publish_from_sync", None)
        is_running = getattr(self._event_bus, "is_running", lambda: False)
        if callable(publish_from_sync) and callable(is_running) and is_running():
            try:
                publish_from_sync(event)
                return
            except Exception:  # pragma: no cover - runtime guard
                self._logger.exception(
                    "Kafka ingest consumer failed to publish %s on local event bus",
                    event_type,
                )

        try:
            topic_bus = get_global_bus()
            topic_bus.publish_sync(event.type, event.payload, source=event.source)
        except Exception:  # pragma: no cover - runtime guard
            self._logger.debug(
                "Kafka ingest consumer publish for %s skipped", event_type, exc_info=True
            )

    def _publish(self, payload: Mapping[str, object]) -> None:
        self._publish_event(
            event_type=self._event_type,
            payload=payload,
            source=self._source,
        )

    def _prepare_payload(
        self, payload: Mapping[str, object], message: KafkaMessageLike
    ) -> dict[str, object]:
        data = dict(payload)
        data.setdefault("kafka_topic", message.topic())
        key = message.key()
        if key is not None and "kafka_key" not in data:
            if isinstance(key, (bytes, bytearray)):
                try:
                    data["kafka_key"] = key.decode("utf-8")
                except UnicodeDecodeError:
                    data["kafka_key"] = key.hex()
            else:
                data["kafka_key"] = str(key)
        data.setdefault("kafka_received_at", datetime.now(tz=UTC).isoformat())
        return data

    def _handle_message(self, message: KafkaMessageLike) -> bool:
        error_fn = getattr(message, "error", None)
        if callable(error_fn):
            err = error_fn()
            if err:
                self._logger.warning("Kafka ingest consumer received error: %s", err)
                return False

        value = message.value()
        if value is None:
            self._logger.debug(
                "Kafka ingest consumer received empty message from %s", message.topic()
            )
            return False

        try:
            payload = self._deserializer(value)
        except Exception:
            self._logger.exception(
                "Kafka ingest consumer failed to decode payload from %s", message.topic()
            )
            return False

        if not isinstance(payload, Mapping):
            self._logger.warning(
                "Kafka ingest consumer expected mapping payload but received %s", type(payload)
            )
            return False

        prepared = self._prepare_payload(payload, message)
        self._publish(prepared)
        self._commit(message)
        self._maybe_publish_consumer_lag()
        return True

    def poll_once(self) -> bool:
        if not self._started:
            self.start()
        message = self._consumer.poll(self._poll_timeout)
        if message is None:
            return False
        return self._handle_message(message)

    async def run_forever(self, stop_event: asyncio.Event | None = None) -> None:
        stop = stop_event or asyncio.Event()
        self.start()
        try:
            while not stop.is_set():
                processed = await asyncio.to_thread(self.poll_once)
                if not processed and self._idle_sleep > 0:
                    await asyncio.sleep(self._idle_sleep)
        except asyncio.CancelledError:  # pragma: no cover - runtime guard
            self._logger.debug("Kafka ingest consumer task cancelled")
            raise
        finally:
            self.close()

    def _commit(self, message: KafkaMessageLike) -> None:
        if not self._commit_offsets:
            return

        commit = getattr(self._consumer, "commit", None)
        if not callable(commit):
            if not self._commit_warning_logged:
                self._logger.warning(
                    "Kafka ingest consumer commit requested but consumer does not expose commit()"
                )
                self._commit_warning_logged = True
            return

        attempts: tuple[tuple[tuple[object, ...], dict[str, object]], ...] = (
            ((), {"message": message, "asynchronous": self._commit_asynchronously}),
            ((), {"message": message}),
            ((message, self._commit_asynchronously), {}),
            ((message,), {}),
            ((), {}),
        )

        commit_fn = cast(Callable[..., object], commit)

        for args, kwargs in attempts:
            try:
                commit_fn(*args, **kwargs)
                return
            except TypeError:
                continue
            except Exception:
                self._logger.exception("Kafka ingest consumer failed to commit offsets")
                return

        if not self._commit_warning_logged:
            self._logger.warning(
                "Kafka ingest consumer failed to commit offsets: unsupported commit signature"
            )
            self._commit_warning_logged = True

    def _maybe_publish_consumer_lag(self) -> None:
        if not self._publish_consumer_lag:
            return

        now = time.monotonic()
        if (
            self._last_consumer_lag_publish is not None
            and self._consumer_lag_interval > 0
            and now - self._last_consumer_lag_publish < self._consumer_lag_interval
        ):
            return

        snapshot = capture_consumer_lag(self._consumer)
        if snapshot is None:
            return

        payload = snapshot.as_dict()
        payload.setdefault("topics", list(self._topics))

        self._publish_event(
            event_type=self._consumer_lag_event_type,
            payload=payload,
            source=self._consumer_lag_source,
        )
        self._last_consumer_lag_publish = now


def create_ingest_event_consumer(
    settings: "KafkaConnectionSettings",
    mapping: Mapping[str, str] | None,
    *,
    event_bus: EventBus,
    consumer_factory: KafkaConsumerFactory | None = None,
    deserializer: Callable[[bytes | bytearray | str], Mapping[str, object]] | None = None,
) -> KafkaIngestEventConsumer | None:
    """Instantiate a :class:`KafkaIngestEventConsumer` when topics are available."""

    if not settings.configured:
        logger.debug("Kafka ingest consumer skipped: no brokers configured")
        return None

    payload = _normalise_env(mapping)
    enabled = _coerce_bool(payload, "KAFKA_INGEST_CONSUMER_ENABLED", True)
    if not enabled:
        logger.info("Kafka ingest consumer disabled via configuration")
        return None

    topic_map, default_topic = ingest_topic_config_from_mapping(payload)
    topics: set[str] = {topic for topic in topic_map.values() if topic}
    if default_topic:
        topics.add(default_topic)

    raw_consumer_topics = payload.get("KAFKA_INGEST_CONSUMER_TOPICS")
    if raw_consumer_topics:
        for entry in str(raw_consumer_topics).split(","):
            topic = entry.strip()
            if topic:
                topics.add(topic)

    if not topics:
        logger.warning(
            "Kafka ingest consumer skipped: no topics configured via KAFKA_INGEST_TOPICS or KAFKA_INGEST_CONSUMER_TOPICS",
        )
        return None

    group_id = payload.get("KAFKA_INGEST_CONSUMER_GROUP") or "emp-ingest-bridge"
    group_id = group_id.strip()

    auto_reset = (
        payload.get("KAFKA_INGEST_CONSUMER_OFFSET_RESET")
        or payload.get("KAFKA_INGEST_CONSUMER_AUTO_OFFSET_RESET")
        or "latest"
    )
    auto_reset = auto_reset.strip().lower()

    enable_auto_commit = _coerce_bool(payload, "KAFKA_INGEST_CONSUMER_AUTO_COMMIT", True)

    poll_timeout = _coerce_optional_float(payload, "KAFKA_INGEST_CONSUMER_POLL_TIMEOUT", 1.0)
    if poll_timeout is None or poll_timeout <= 0:
        poll_timeout = 1.0

    idle_sleep = _coerce_optional_float(payload, "KAFKA_INGEST_CONSUMER_IDLE_SLEEP", 0.5)
    if idle_sleep is None or idle_sleep < 0:
        idle_sleep = 0.5

    commit_on_publish = _coerce_bool(
        payload,
        "KAFKA_INGEST_CONSUMER_COMMIT_ON_PUBLISH",
        not enable_auto_commit,
    )
    commit_async = _coerce_bool(
        payload,
        "KAFKA_INGEST_CONSUMER_COMMIT_ASYNC",
        False,
    )

    publish_consumer_lag = _coerce_bool(
        payload,
        "KAFKA_INGEST_CONSUMER_PUBLISH_LAG",
        False,
    )
    consumer_lag_event_type = (
        payload.get("KAFKA_INGEST_CONSUMER_LAG_EVENT_TYPE") or "telemetry.kafka.lag"
    ).strip()
    consumer_lag_source = payload.get("KAFKA_INGEST_CONSUMER_LAG_EVENT_SOURCE")
    consumer_lag_interval = _coerce_optional_float(
        payload,
        "KAFKA_INGEST_CONSUMER_LAG_INTERVAL",
        30.0,
    )

    event_type = payload.get("KAFKA_INGEST_CONSUMER_EVENT_TYPE") or "telemetry.ingest"
    event_type = event_type.strip()
    event_source = payload.get("KAFKA_INGEST_CONSUMER_EVENT_SOURCE") or "timescale_ingest.kafka"
    event_source = event_source.strip()

    if consumer_factory is None:
        try:
            from confluent_kafka import Consumer
        except Exception:  # pragma: no cover - optional dependency guard
            logger.warning(
                "Kafka ingest consumer requested (%s) but confluent_kafka is not installed; skipping",
                settings.summary(redacted=True),
            )
            return None

        def _default_consumer_factory(config: Mapping[str, Any]) -> KafkaConsumerLike:
            return cast(KafkaConsumerLike, Consumer(config))

        consumer_factory = _default_consumer_factory

    consumer = settings.create_consumer(
        factory=consumer_factory,
        group_id=group_id,
        auto_offset_reset=auto_reset,
        enable_auto_commit=enable_auto_commit,
    )
    if consumer is None:
        logger.warning(
            "Kafka ingest consumer settings present (%s) but consumer factory returned None",
            settings.summary(redacted=True),
        )
        return None

    bridge = KafkaIngestEventConsumer(
        consumer,
        topics=sorted(topics),
        event_bus=event_bus,
        event_type=event_type,
        source=event_source,
        poll_timeout=poll_timeout,
        deserializer=deserializer,
        idle_sleep=idle_sleep,
        commit_offsets=commit_on_publish,
        commit_asynchronously=commit_async,
        publish_consumer_lag=publish_consumer_lag,
        consumer_lag_event_type=consumer_lag_event_type,
        consumer_lag_source=consumer_lag_source,
        consumer_lag_interval=consumer_lag_interval,
    )

    logger.info(
        "Kafka ingest consumer ready: group=%s topics=%s (event_type=%s)",
        group_id,
        list(bridge.topics),
        event_type,
    )
    return bridge


__all__ = [
    "KafkaConnectionSettings",
    "KafkaConsumerFactory",
    "KafkaConsumerLike",
    "KafkaConsumerLagSnapshot",
    "KafkaProducerLike",
    "KafkaProducerFactory",
    "KafkaIngestBackfillSummary",
    "KafkaIngestEventPublisher",
    "KafkaIngestEventConsumer",
    "KafkaIngestHealthPublisher",
    "KafkaIngestMetricsPublisher",
    "KafkaPartitionLag",
    "KafkaTopicProvisioner",
    "KafkaTopicProvisioningSummary",
    "KafkaTopicSpec",
    "capture_consumer_lag",
    "backfill_ingest_dimension_to_kafka",
    "create_ingest_event_publisher",
    "create_ingest_event_consumer",
    "create_ingest_health_publisher",
    "create_ingest_metrics_publisher",
    "ingest_topic_config_from_mapping",
    "resolve_ingest_topic_specs",
    "should_auto_create_topics",
]
