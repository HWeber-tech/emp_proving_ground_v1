"""Smoke tests for the developer data backbone services.

The checks verify that the docker-compose services for TimescaleDB, Redis, and
Kafka are reachable using the connection settings exposed via SystemConfig
extras. Each probe skips automatically when the corresponding service or client
library is not available, enabling CI to gate on the tests only when the stack
is provisioned locally.
"""

from __future__ import annotations

import uuid
from contextlib import suppress

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from src.data_foundation.cache.redis_cache import RedisConnectionSettings
from src.data_foundation.persist.timescale import TimescaleConnectionSettings
from src.data_foundation.streaming.kafka_stream import KafkaConnectionSettings


@pytest.mark.integration
def test_timescale_dev_service_round_trip() -> None:
    """Ensure a Timescale connection can create, write, and read a table."""

    settings = TimescaleConnectionSettings.from_env()
    if not settings.configured or not settings.is_postgres():
        pytest.skip("Timescale dev service not configured")

    try:
        engine = settings.create_engine()
    except OperationalError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Timescale engine unavailable: {exc}")

    table = f"emp_dev_probe_{uuid.uuid4().hex}"
    value: str | None = None

    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    f"CREATE TABLE IF NOT EXISTS {table} (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)"
                )
            )
            connection.execute(
                text(f"INSERT INTO {table} (id, payload) VALUES (1, :payload)"),
                {"payload": "ping"},
            )
            value = connection.execute(
                text(f"SELECT payload FROM {table} WHERE id = 1")
            ).scalar_one()
    except OperationalError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Timescale transaction failed: {exc}")
    finally:
        with suppress(Exception):
            with engine.begin() as connection:
                connection.execute(text(f"DROP TABLE IF EXISTS {table}"))
        engine.dispose()

    assert value == "ping"


@pytest.mark.integration
def test_redis_dev_service_round_trip() -> None:
    """Ensure the Redis cache accepts read/write operations."""

    settings = RedisConnectionSettings.from_env()
    if not settings.configured:
        pytest.skip("Redis dev service not configured")

    try:
        client = settings.create_client()
    except Exception as exc:  # pragma: no cover - optional dependency guard
        pytest.skip(f"Redis client unavailable: {exc}")

    key = f"emp:dev:probe:{uuid.uuid4().hex}"

    try:
        assert client.ping() is True
        client.set(key, "pong", ex=30)
        value = client.get(key)
        assert value == b"pong"
    finally:
        with suppress(Exception):
            client.delete(key)
        close = getattr(client, "close", None)
        if callable(close):
            close()


@pytest.mark.integration
def test_kafka_dev_service_round_trip() -> None:
    """Ensure Kafka producer/consumer loop works against the dev broker."""

    settings = KafkaConnectionSettings.from_env()
    if not settings.configured:
        pytest.skip("Kafka dev service not configured")
    if settings.security_protocol and settings.security_protocol != "PLAINTEXT":
        pytest.skip("Kafka smoke test only supports PLAINTEXT endpoints")

    try:
        from kafka import KafkaConsumer, KafkaProducer  # type: ignore[import]
        from kafka.errors import NoBrokersAvailable  # type: ignore[import]
    except Exception:  # pragma: no cover - optional dependency guard
        pytest.skip("kafka-python library not installed")

    bootstrap_servers = [
        entry.strip()
        for entry in settings.bootstrap_servers.split(",")
        if entry.strip()
    ]
    if not bootstrap_servers:
        pytest.skip("Kafka bootstrap servers not configured")

    topic = f"emp_dev_probe_{uuid.uuid4().hex}"

    try:
        producer = KafkaProducer(bootstrap_servers=bootstrap_servers)
    except NoBrokersAvailable as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Kafka broker unavailable: {exc}")

    try:
        producer.send(topic, key=b"probe", value=b"ping")
        producer.flush()
    finally:
        producer.close()

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        consumer_timeout_ms=2000,
    )
    try:
        payloads = [record.value for record in consumer]
    finally:
        consumer.close()

    assert b"ping" in payloads
