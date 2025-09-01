from __future__ import annotations

from typing import Any

class Response:
    status_code: int
    content: bytes

class Session:
    headers: dict[str, str]
    def __init__(self) -> None: ...
    def get(self, url: str, timeout: float | int = ...) -> Response: ...
    def close(self) -> None: ...