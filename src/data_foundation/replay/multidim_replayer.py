"""Multi-dimensional replay helpers used by the ingestion regression suite."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, TypedDict, cast

from src.core.types import JSONObject

logger = logging.getLogger(__name__)

ReplayKind = Literal["md", "macro", "yield"]


class ReplayEvent(TypedDict, total=False):
    """Typed representation of a replay event entry."""

    timestamp: str
    _kind: ReplayKind


def _parse_jsonl(path: str, kind: ReplayKind) -> Iterator[ReplayEvent]:
    """Yield JSON payloads from ``path`` tagged with ``kind``."""

    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = cast(JSONObject, json.loads(line))
                except json.JSONDecodeError:
                    logger.debug("Skipping malformed JSONL line in %s", path, exc_info=True)
                    continue
                event: ReplayEvent = {"_kind": kind}
                if "timestamp" in payload and isinstance(payload["timestamp"], str):
                    event["timestamp"] = payload["timestamp"]
                event.update(cast(dict[str, object], payload))
                yield event
    except OSError:
        logger.warning("Replay source %s is unavailable", path)


def _timestamp_key(event: ReplayEvent) -> datetime:
    raw = event.get("timestamp")
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            logger.debug("Invalid timestamp on replay event: %s", raw, exc_info=True)
    return datetime.min


@dataclass(slots=True)
class MultiDimReplayer:
    """Merge market data, macro, and yield event streams from JSONL inputs."""

    md_path: str | None = None
    macro_path: str | None = None
    yields_path: str | None = None

    def _iter_events(self) -> Iterable[ReplayEvent]:
        if self.md_path:
            yield from _parse_jsonl(self.md_path, "md")
        if self.macro_path:
            yield from _parse_jsonl(self.macro_path, "macro")
        if self.yields_path:
            yield from _parse_jsonl(self.yields_path, "yield")

    def replay(
        self,
        *,
        on_md: Callable[[ReplayEvent], None] | None = None,
        on_macro: Callable[[ReplayEvent], None] | None = None,
        on_yield: Callable[[ReplayEvent], None] | None = None,
        limit: int | None = None,
    ) -> int:
        """Replay events ordered by timestamp, invoking the provided callbacks."""

        events = sorted(self._iter_events(), key=_timestamp_key)
        callbacks: dict[ReplayKind, Callable[[ReplayEvent], None] | None] = {
            "md": on_md,
            "macro": on_macro,
            "yield": on_yield,
        }

        emitted = 0
        for event in events:
            if limit is not None and emitted >= limit:
                break
            kind = event.get("_kind")
            if kind not in callbacks:
                continue
            callback = callbacks[cast(ReplayKind, kind)]
            if callback is None:
                continue
            callback(event)
            if kind == "md":
                emitted += 1

        return emitted


__all__ = ["MultiDimReplayer", "ReplayEvent", "ReplayKind"]
