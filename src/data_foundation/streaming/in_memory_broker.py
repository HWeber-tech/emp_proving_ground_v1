"""In-memory Kafka-compatible broker for regression and integration tests.

The institutional roadmap expects Kafka-backed ingest telemetry, but unit and
integration tests should remain hermetic.  This module provides a lightweight
in-process broker that satisfies the ``KafkaProducerLike`` and
``KafkaConsumerLike`` protocols used by :mod:`src.data_foundation.streaming`
without requiring an external Kafka cluster.  It lets regression suites exercise
the full store→cache→stream cycle using exactly the same publisher/consumer
plumbing that production wiring relies on.

``InMemoryKafkaBroker`` keeps an append-only log per topic.  Producers append
serialised payloads, while consumers maintain offsets to read messages in
order.  Polling uses a condition variable so background streaming loops can wait
for new events without busy-waiting.  The implementation intentionally mirrors a
subset of Kafka semantics (single consumer group, at-least-once delivery) while
staying dependency free.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .kafka_stream import KafkaConsumerLike, KafkaMessageLike, KafkaProducerLike


@dataclass(slots=True)
class _StoredMessage:
    topic: str
    value: bytes
    key: bytes | str | None

    def to_message(self) -> "InMemoryKafkaMessage":
        return InMemoryKafkaMessage(self.topic, self.value, self.key)


class InMemoryKafkaMessage(KafkaMessageLike):
    """Minimal message wrapper returned by :class:`InMemoryKafkaConsumer`."""

    __slots__ = ("_topic", "_value", "_key")

    def __init__(self, topic: str, value: bytes, key: bytes | str | None) -> None:
        self._topic = topic
        self._value = value
        self._key = key

    def error(self) -> None:  # pragma: no cover - symmetry with real clients
        return None

    def value(self) -> bytes:
        return self._value

    def topic(self) -> str:
        return self._topic

    def key(self) -> bytes | str | None:
        return self._key


class InMemoryKafkaBroker:
    """Simple append-only broker shared by in-memory producers and consumers."""

    def __init__(self) -> None:
        self._topics: dict[str, deque[_StoredMessage]] = {}
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)

    # ------------------------------------------------------------------
    # Producer helpers
    # ------------------------------------------------------------------

    def create_producer(self) -> "InMemoryKafkaProducer":
        return InMemoryKafkaProducer(self)

    def publish(self, topic: str, value: bytes, key: bytes | str | None) -> None:
        if not topic:
            raise ValueError("topic must not be empty")
        if not isinstance(value, (bytes, bytearray)):
            raise TypeError("Kafka payloads must be bytes")
        payload = bytes(value)
        with self._condition:
            log = self._topics.setdefault(topic, deque())
            log.append(_StoredMessage(topic=topic, value=payload, key=key))
            self._condition.notify_all()

    # ------------------------------------------------------------------
    # Consumer helpers
    # ------------------------------------------------------------------

    def create_consumer(self) -> "InMemoryKafkaConsumer":
        return InMemoryKafkaConsumer(self)

    def poll(
        self,
        topics: Sequence[str],
        offsets: Mapping[str, int],
        timeout: float | None = None,
    ) -> InMemoryKafkaMessage | None:
        deadline = None if timeout is None else time.monotonic() + max(timeout, 0.0)
        with self._condition:
            while True:
                for topic in topics:
                    log = self._topics.get(topic)
                    if not log:
                        continue
                    index = offsets.get(topic, 0)
                    if index < len(log):
                        message = log[index].to_message()
                        offsets[topic] = index + 1  # type: ignore[index]
                        return message

                if timeout is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._condition.wait(remaining)
                else:
                    self._condition.wait()

    # ------------------------------------------------------------------
    # Introspection helpers for tests
    # ------------------------------------------------------------------

    def topic_names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._topics))

    def snapshot(self) -> tuple[_StoredMessage, ...]:
        with self._lock:
            messages: list[_StoredMessage] = []
            for topic in sorted(self._topics):
                messages.extend(self._topics[topic])
            return tuple(messages)

    def metadata(self) -> dict[str, Any]:
        """Return a lightweight structure compatible with ``list_topics``."""

        return {"topics": {name: {} for name in self.topic_names()}}


class InMemoryKafkaProducer(KafkaProducerLike):
    """Producer adapter that records messages in the in-memory broker."""

    def __init__(self, broker: InMemoryKafkaBroker) -> None:
        self._broker = broker

    def produce(self, topic: str, value: bytes, key: bytes | str | None = None) -> None:
        self._broker.publish(topic, value, key)

    def flush(self, timeout: float | None = None) -> None:  # pragma: no cover - noop
        return None

    def close(self) -> None:  # pragma: no cover - symmetry with kafka-python
        return None


class InMemoryKafkaConsumer(KafkaConsumerLike):
    """Consumer adapter that replays messages from the in-memory broker."""

    def __init__(self, broker: InMemoryKafkaBroker) -> None:
        self._broker = broker
        self._topics: tuple[str, ...] = ()
        self._offsets: dict[str, int] = {}
        self._closed = False

    def subscribe(self, topics: Iterable[str]) -> None:
        subscribed = tuple(dict.fromkeys(str(topic) for topic in topics if str(topic).strip()))
        if not subscribed:
            raise ValueError("At least one topic must be provided")
        self._topics = subscribed
        self._offsets = {topic: 0 for topic in subscribed}

    def poll(self, timeout: float | None = None) -> InMemoryKafkaMessage | None:
        if self._closed or not self._topics:
            return None
        return self._broker.poll(self._topics, self._offsets, timeout)

    def commit(
        self,
        message: KafkaMessageLike | None = None,
        asynchronous: bool = False,
    ) -> None:  # pragma: no cover - offset management is implicit
        return None

    def close(self) -> None:
        self._closed = True

    def list_topics(self, timeout: float | None = None) -> Mapping[str, Any]:
        return self._broker.metadata()


__all__ = [
    "InMemoryKafkaBroker",
    "InMemoryKafkaConsumer",
    "InMemoryKafkaMessage",
    "InMemoryKafkaProducer",
]
