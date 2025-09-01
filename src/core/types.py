"""
Canonical JSON type aliases for the project.

Use these instead of dict[str, Any] when passing JSON-like payloads between
modules, across adapters, or to/from external systems.

Examples:
    from src.core.types import JSONObject, JSONValue

    def handle_payload(payload: JSONObject) -> JSONValue:
        ...

These aliases are intentionally minimal and align with RFC 8259 JSON shapes.
"""

from __future__ import annotations

from typing import Union

# Atomic JSON primitives (RFC 8259)
JSONPrimitive = Union[str, int, float, bool, None]

# Recursive JSON structure
JSONValue = Union[JSONPrimitive, "JSONArray", "JSONObject"]
JSONArray = list["JSONValue"]
JSONObject = dict[str, JSONValue]

__all__ = ["JSONPrimitive", "JSONValue", "JSONArray", "JSONObject"]
