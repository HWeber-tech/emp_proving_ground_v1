from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Mapping
from contextlib import suppress
from typing import Final


logger = logging.getLogger(__name__)


class JsonlWriterError(RuntimeError):
    """Raised when JSONL persistence fails."""


_JSON_SEPARATORS: Final[tuple[str, str]] = (",", ":")


def _normalise_events(events: Iterable[Mapping[str, object] | dict[str, object]]) -> list[dict[str, object]]:
    """Copy events into ``dict`` instances to ensure JSON serialisation stability."""

    normalised: list[dict[str, object]] = []
    for event in events:
        if isinstance(event, dict):
            normalised.append(event)
            continue
        normalised.append({str(key): value for key, value in event.items()})
    return normalised


def write_events_jsonl(
    events: Iterable[Mapping[str, object] | dict[str, object]], out_path: str
) -> str:
    """Persist iterable ``events`` to ``out_path`` in JSON Lines format.

    The implementation avoids swallowing unexpected exceptions so operational
    tooling sees genuine filesystem/serialisation failures instead of silently
    receiving an empty path.
    """

    serialisable_events = _normalise_events(events)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    try:
        with open(out_path, "w", encoding="utf-8") as fh:
            for event in serialisable_events:
                try:
                    payload = json.dumps(event, separators=_JSON_SEPARATORS)
                except (TypeError, ValueError) as exc:  # defensive against unsafe payloads
                    logger.error(
                        "Event payload is not JSON serialisable for %s: %s", out_path, exc
                    )
                    raise JsonlWriterError("Event payload is not JSON serialisable") from exc
                fh.write(f"{payload}\n")
    except JsonlWriterError:
        with suppress(OSError):
            os.remove(out_path)
        raise
    except OSError as exc:
        logger.error("Failed to write events JSONL to %s: %s", out_path, exc)
        raise JsonlWriterError(f"Failed to write JSONL to {out_path}") from exc

    return out_path
