"""Helpers for securely publishing operational telemetry events.

These utilities centralise the guarded publish-from-runtime → fall back to global
bus pattern so individual modules do not need to implement their own blanket
``except Exception`` handling.  The helpers log contextual warnings for expected
failure modes and raise a typed :class:`EventPublishError` when an unexpected
exception bubbles up.  This keeps the telemetry path explicit while satisfying
the security roadmap requirement to replace broad exception handlers in
operational modules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from src.core.event_bus import Event, EventBus, TopicBus, get_global_bus

__all__ = [
    "EventPublishError",
    "publish_event_with_failover",
]


@dataclass(slots=True)
class EventPublishError(RuntimeError):
    """Raised when an event cannot be published via the runtime or global bus."""

    stage: str
    event_type: str

    def __post_init__(self) -> None:
        RuntimeError.__init__(
            self,
            f"Failed to publish event {self.event_type!r} via {self.stage} bus",
        )


def publish_event_with_failover(
    event_bus: EventBus,
    event: Event,
    *,
    logger: logging.Logger,
    runtime_fallback_message: str,
    runtime_unexpected_message: str,
    runtime_none_message: str,
    global_not_running_message: str,
    global_unexpected_message: str,
    global_bus_factory: Callable[[], TopicBus] | None = None,
) -> None:
    """Publish ``event`` using ``event_bus`` with a deterministic fallback.

    The helper mirrors the operational telemetry contract used across multiple
    modules: attempt to publish via the runtime bus when available and fall back
    to the global topic bus if the runtime path rejects the event.  Unexpected
    exceptions from either bus are wrapped in :class:`EventPublishError` to keep
    the calling code free from blanket ``except Exception`` clauses.
    """

    publish_from_sync = getattr(event_bus, "publish_from_sync", None)
    if callable(publish_from_sync) and event_bus.is_running():
        try:
            result = publish_from_sync(event)
        except RuntimeError as exc:
            logger.warning(runtime_fallback_message, exc_info=exc)
        except Exception as exc:  # pragma: no cover - defensive guardrail
            logger.exception(runtime_unexpected_message, exc_info=exc)
            raise EventPublishError("runtime", event.type) from exc
        else:
            if result is None:
                logger.warning(runtime_none_message)
            else:
                return

    factory = global_bus_factory or get_global_bus

    try:
        topic_bus = factory()
        topic_bus.publish_sync(event.type, event.payload, source=event.source)
    except RuntimeError as exc:
        logger.error(global_not_running_message, exc_info=exc)
        raise EventPublishError("global", event.type) from exc
    except Exception as exc:  # pragma: no cover - defensive guardrail
        logger.exception(global_unexpected_message, exc_info=exc)
        raise EventPublishError("global", event.type) from exc
