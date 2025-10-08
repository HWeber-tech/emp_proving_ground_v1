"""Asynchronous client for paper trading REST APIs used in limited-live routing."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, MutableMapping
from urllib.parse import urljoin
import uuid

import aiohttp

logger = logging.getLogger(__name__)

__all__ = [
    "PaperTradingApiError",
    "PaperTradingApiSettings",
    "PaperTradingApiAdapter",
]


class PaperTradingApiError(RuntimeError):
    """Raised when the paper trading REST API rejects or fails a request."""


@dataclass(slots=True)
class PaperTradingApiSettings:
    """Configuration payload for the REST paper trading adapter."""

    base_url: str
    api_key: str | None = None
    api_secret: str | None = None
    api_key_header: str = "APCA-API-KEY-ID"
    api_secret_header: str = "APCA-API-SECRET-KEY"
    order_endpoint: str = "/v2/orders"
    order_id_field: str = "id"
    time_in_force: str = "day"
    verify_ssl: bool = True
    request_timeout: float | None = 10.0
    client_order_prefix: str = "alpha"
    account_id: str | None = None
    extra_headers: Mapping[str, str] = field(default_factory=dict)

    def build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            str(key): str(value)
            for key, value in (self.extra_headers or {}).items()
            if key and value is not None
        }
        if self.api_key:
            headers.setdefault(self.api_key_header, self.api_key)
        if self.api_secret:
            headers.setdefault(self.api_secret_header, self.api_secret)
        return headers

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PaperTradingApiSettings":
        base_url_raw = payload.get("PAPER_TRADING_API_URL")
        if not base_url_raw or not str(base_url_raw).strip():
            raise ValueError("PAPER_TRADING_API_URL must be provided for paper trading integration")

        base_url = str(base_url_raw).strip()
        api_key = cls._coerce_optional_str(payload.get("PAPER_TRADING_API_KEY"))
        api_secret = cls._coerce_optional_str(payload.get("PAPER_TRADING_API_SECRET"))
        api_key_header = cls._coerce_optional_str(
            payload.get("PAPER_TRADING_API_KEY_HEADER"), default="APCA-API-KEY-ID"
        )
        api_secret_header = cls._coerce_optional_str(
            payload.get("PAPER_TRADING_API_SECRET_HEADER"), default="APCA-API-SECRET-KEY"
        )
        order_endpoint = cls._coerce_optional_str(
            payload.get("PAPER_TRADING_ORDER_ENDPOINT"), default="/v2/orders"
        )
        order_id_field = cls._coerce_optional_str(
            payload.get("PAPER_TRADING_ORDER_ID_FIELD"), default="id"
        )
        time_in_force = cls._coerce_optional_str(
            payload.get("PAPER_TRADING_TIME_IN_FORCE"), default="day"
        )
        verify_ssl = cls._coerce_optional_bool(
            payload.get("PAPER_TRADING_VERIFY_SSL"), default=True
        )
        timeout_value = payload.get("PAPER_TRADING_ORDER_TIMEOUT")
        request_timeout = None
        if timeout_value is not None:
            try:
                request_timeout = float(timeout_value)
                if request_timeout <= 0:
                    request_timeout = None
            except (TypeError, ValueError):
                request_timeout = None
        else:
            request_timeout = 10.0

        client_order_prefix = cls._coerce_optional_str(
            payload.get("PAPER_TRADING_CLIENT_ORDER_PREFIX"), default="alpha"
        )
        account_id = cls._coerce_optional_str(payload.get("PAPER_TRADING_ACCOUNT_ID"))
        extra_headers = cls._parse_headers(payload.get("PAPER_TRADING_EXTRA_HEADERS"))

        return cls(
            base_url=base_url,
            api_key=api_key,
            api_secret=api_secret,
            api_key_header=api_key_header,
            api_secret_header=api_secret_header,
            order_endpoint=order_endpoint,
            order_id_field=order_id_field,
            time_in_force=time_in_force,
            verify_ssl=verify_ssl,
            request_timeout=request_timeout,
            client_order_prefix=client_order_prefix,
            account_id=account_id,
            extra_headers=extra_headers,
        )

    @staticmethod
    def _coerce_optional_str(value: Any, *, default: str | None = None) -> str | None:
        if value is None:
            return default
        text = str(value).strip()
        if not text:
            return default
        return text

    @staticmethod
    def _coerce_optional_bool(value: Any, *, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        normalized = str(value).strip().lower()
        if normalized in {"true", "1", "yes", "y", "on", "enable", "enabled"}:
            return True
        if normalized in {"false", "0", "no", "n", "off", "disable", "disabled"}:
            return False
        return default

    @staticmethod
    def _parse_headers(raw: Any) -> Mapping[str, str]:
        if raw is None:
            return {}
        if isinstance(raw, Mapping):
            return {
                str(key): str(value)
                for key, value in raw.items()
                if key and value is not None
            }
        text = str(raw).strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, Mapping):
            return {
                str(key): str(value)
                for key, value in parsed.items()
                if key and value is not None
            }

        headers: dict[str, str] = {}
        for token in text.split(";"):
            if not token.strip():
                continue
            if "=" in token:
                name, value = token.split("=", 1)
            elif ":" in token:
                name, value = token.split(":", 1)
            else:
                continue
            name = name.strip()
            value = value.strip()
            if name and value:
                headers[name] = value
        return headers


class PaperTradingApiAdapter:
    """Minimal async client translating execution intents into REST API calls."""

    def __init__(
        self,
        *,
        settings: PaperTradingApiSettings,
        session_factory: Callable[[], aiohttp.ClientSession] | None = None,
    ) -> None:
        self._settings = settings
        self._session_factory = session_factory
        self._session: aiohttp.ClientSession | None = None

    @property
    def settings(self) -> PaperTradingApiSettings:
        return self._settings

    async def close(self) -> None:
        session = self._session
        if session is not None and not session.closed:
            await session.close()
        self._session = None

    async def place_market_order(self, symbol: str, side: str, quantity: float) -> str:
        session = await self._ensure_session()
        url = self._build_orders_url()
        payload = self._build_market_payload(symbol, side, quantity)
        headers = self._settings.build_headers()
        timeout = self._build_timeout()

        logger.debug(
            "paper_trading_api_submit",
            extra={
                "url": url,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
            },
        )

        try:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                ssl=self._settings.verify_ssl,
                timeout=timeout,
            ) as response:
                body_text = await response.text()
                if response.status >= 400:
                    raise PaperTradingApiError(
                        f"Paper trading API responded with {response.status}: {body_text}"
                    )
                try:
                    body = await response.json()
                except aiohttp.ContentTypeError as exc:
                    raise PaperTradingApiError(
                        f"Paper trading API returned non-JSON payload: {body_text}"
                    ) from exc
        except asyncio.TimeoutError as exc:
            raise PaperTradingApiError("Paper trading API request timed out") from exc
        except aiohttp.ClientError as exc:
            raise PaperTradingApiError(f"Paper trading API request failed: {exc}") from exc

        order_id = self._extract_order_id(body)
        if not order_id:
            raise PaperTradingApiError(
                "Paper trading API response is missing an order identifier"
            )
        return order_id

    async def _ensure_session(self) -> aiohttp.ClientSession:
        session = self._session
        if session is not None and not session.closed:
            return session
        if self._session_factory is not None:
            session = self._session_factory()
        else:
            timeout = self._build_timeout()
            session = aiohttp.ClientSession(timeout=timeout)
        self._session = session
        return session

    def _build_orders_url(self) -> str:
        base = self._settings.base_url.rstrip("/") + "/"
        endpoint = self._settings.order_endpoint.lstrip("/")
        return urljoin(base, endpoint)

    def _build_market_payload(self, symbol: str, side: str, quantity: float) -> Mapping[str, Any]:
        normalized_side = self._normalise_side(side)
        payload: dict[str, Any] = {
            "symbol": str(symbol).upper(),
            "qty": self._format_quantity(quantity),
            "side": normalized_side,
            "type": "market",
            "time_in_force": self._settings.time_in_force,
        }
        if self._settings.account_id:
            payload["account_id"] = self._settings.account_id
        client_prefix = self._settings.client_order_prefix
        if client_prefix:
            payload["client_order_id"] = f"{client_prefix}-{uuid.uuid4().hex[:18]}"
        return payload

    def _normalise_side(self, side: str) -> str:
        candidate = str(side or "").strip().lower()
        if candidate in {"buy", "b"}:
            return "buy"
        if candidate in {"sell", "s"}:
            return "sell"
        raise PaperTradingApiError(f"Unsupported order side: {side!r}")

    def _format_quantity(self, quantity: float) -> str:
        try:
            qty = float(quantity)
        except (TypeError, ValueError):
            raise PaperTradingApiError(f"Invalid quantity for paper trade: {quantity!r}")
        if qty <= 0:
            raise PaperTradingApiError("Quantity must be positive for paper trading orders")
        if qty.is_integer():
            return str(int(qty))
        return f"{qty:.8f}".rstrip("0").rstrip(".")

    def _build_timeout(self) -> aiohttp.ClientTimeout | None:
        if self._settings.request_timeout is None:
            return None
        return aiohttp.ClientTimeout(total=self._settings.request_timeout)

    def _extract_order_id(self, body: Any) -> str | None:
        if isinstance(body, Mapping):
            # Support nested responses by checking top-level first, then metadata block.
            order_id = body.get(self._settings.order_id_field)
            if isinstance(order_id, (str, int)) and str(order_id).strip():
                return str(order_id)
            metadata = body.get("order")
            if isinstance(metadata, Mapping):
                nested = metadata.get(self._settings.order_id_field)
                if isinstance(nested, (str, int)) and str(nested).strip():
                    return str(nested)
        return None

    def describe(self) -> Mapping[str, Any]:
        """Return a serialisable snapshot of the adapter configuration."""

        payload: MutableMapping[str, Any] = {
            "base_url": self._settings.base_url,
            "order_endpoint": self._settings.order_endpoint,
            "order_id_field": self._settings.order_id_field,
            "time_in_force": self._settings.time_in_force,
            "verify_ssl": self._settings.verify_ssl,
            "timeout": self._settings.request_timeout,
        }
        if self._settings.account_id:
            payload["account_id"] = self._settings.account_id
        if self._settings.extra_headers:
            payload["extra_headers"] = dict(self._settings.extra_headers)
        return payload

