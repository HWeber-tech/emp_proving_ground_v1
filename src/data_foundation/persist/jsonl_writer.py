from __future__ import annotations

import json
import logging
import os
from contextlib import suppress


logger = logging.getLogger(__name__)


class JsonlWriterError(RuntimeError):
    """Raised when JSONL persistence fails."""


def write_events_jsonl(events: list[dict[str, object]], out_path: str) -> str:
    """Persist a list of event dictionaries to a JSONL file.

    The implementation avoids swallowing unexpected exceptions so operational
    tooling sees genuine filesystem/serialisation failures instead of silently
    receiving an empty path.
    """

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    try:
        with open(out_path, "w", encoding="utf-8") as fh:
            for event in events:
                try:
                    payload = json.dumps(event)
                except (TypeError, ValueError) as exc:  # defensive against unsafe payloads
                    logger.error("Event payload is not JSON serialisable for %s: %s", out_path, exc)
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
