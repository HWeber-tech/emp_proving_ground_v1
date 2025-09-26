"""Redis cache configuration helpers aligned with the roadmap."""

from __future__ import annotations

import logging
import os
import time
from importlib import import_module
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Protocol, cast
from urllib.parse import urlparse

_redis_mod: object | None
try:  # pragma: no cover - redis optional in bootstrap environments
    _redis_mod = import_module("redis")
except Exception:  # pragma: no cover - keep module importable without redis
    _redis_mod = None


class _RedisModule(Protocol):
    def from_url(self, url: str, **options: Any) -> Any: ...

    def Redis(self, **options: Any) -> Any: ...


redis: _RedisModule | None = cast("_RedisModule | None", _redis_mod)

logger = logging.getLogger(__name__)

ClientFactory = Callable[["RedisConnectionSettings"], Any]
TimeFn = Callable[[], float]


def _normalise_env(mapping: Mapping[str, str] | None) -> MutableMapping[str, str]:
    if mapping is None:
        return {k: v for k, v in os.environ.items() if isinstance(v, str)}
    return {str(k): str(v) for k, v in mapping.items()}


def _parse_csv(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return tuple()
    parts = [segment.strip() for segment in str(raw).split(",")]
    return tuple(part for part in parts if part)


def _coerce_int(payload: Mapping[str, str], key: str, default: int) -> int:
    raw = payload.get(key)
    if raw is None:
        return default
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _coerce_optional_int(payload: Mapping[str, str], key: str, default: int | None) -> int | None:
    raw = payload.get(key)
    if raw is None:
        return default
    normalized = str(raw).strip().lower()
    if normalized in {"", "none", "null", "off", "disabled"}:
        return None
    try:
        return int(normalized)
    except (TypeError, ValueError):
        return default


def _coerce_float(payload: Mapping[str, str], key: str, default: float | None) -> float | None:
    raw = payload.get(key)
    if raw is None:
        return default
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _coerce_bool(payload: Mapping[str, str], key: str, default: bool) -> bool:
    raw = payload.get(key)
    if raw is None:
        return default
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


@dataclass(frozen=True)
class RedisConnectionSettings:
    """Connection information required to instantiate a Redis client."""

    url: str | None = None
    host: str | None = None
    port: int = 6379
    db: int = 0
    username: str | None = None
    password: str | None = None
    ssl: bool = False
    client_name: str = "emp-professional-cache"
    socket_timeout: float | None = 5.0
    health_check_interval: float = 0.0
    retry_on_timeout: bool = True

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, str] | None = None) -> "RedisConnectionSettings":
        data = _normalise_env(mapping)

        url = data.get("REDIS_URL") or data.get("REDIS_URI") or data.get("REDISCLOUD_URL")
        url = url.strip() if url else None

        host = data.get("REDIS_HOST") or data.get("CACHE_HOST") or None
        host = host.strip() if host else None

        port = _coerce_int(data, "REDIS_PORT", 6379)
        db = _coerce_int(data, "REDIS_DB", 0)

        username = data.get("REDIS_USERNAME") or data.get("REDIS_USER")
        username = username.strip() if username else None

        password = data.get("REDIS_PASSWORD") or data.get("REDIS_PASS")
        password = password.strip() if password else None

        ssl = _coerce_bool(data, "REDIS_SSL", False)

        client_name = data.get("REDIS_CLIENT_NAME") or data.get("REDIS_APP_NAME")
        client_name = client_name.strip() if client_name else "emp-professional-cache"

        socket_timeout = _coerce_float(data, "REDIS_SOCKET_TIMEOUT", 5.0)
        health_check = _coerce_float(data, "REDIS_HEALTH_CHECK_INTERVAL", 5.0)
        if health_check is None:
            health_check = 0.0

        retry_on_timeout = _coerce_bool(data, "REDIS_RETRY_ON_TIMEOUT", True)

        parsed = urlparse(url) if url else None
        if parsed:
            if parsed.hostname and not host:
                host = parsed.hostname
            if parsed.port:
                port = parsed.port
            if parsed.path and parsed.path.strip("/"):
                try:
                    db = int(parsed.path.strip("/"))
                except ValueError:
                    pass
            if parsed.username and not username:
                username = parsed.username
            if parsed.password and not password:
                password = parsed.password
            if parsed.scheme and parsed.scheme.lower().startswith("rediss"):
                ssl = True

        return cls(
            url=url,
            host=host,
            port=port,
            db=db,
            username=username,
            password=password,
            ssl=ssl,
            client_name=client_name,
            socket_timeout=socket_timeout,
            health_check_interval=health_check,
            retry_on_timeout=retry_on_timeout,
        )

    @classmethod
    def from_env(cls) -> "RedisConnectionSettings":
        return cls.from_mapping(os.environ)

    @property
    def configured(self) -> bool:
        return bool(self.url or self.host)

    def connection_url(self) -> str:
        if self.url:
            return self.url
        if not self.host:
            raise ValueError("Redis host not configured")
        scheme = "rediss" if self.ssl else "redis"
        auth: str = ""
        if self.username and self.password:
            auth = f"{self.username}:{self.password}@"
        elif self.password:
            auth = f":{self.password}@"
        return f"{scheme}://{auth}{self.host}:{self.port}/{self.db}"

    def create_client(self, *, factory: ClientFactory | None = None) -> Any:
        """Instantiate a Redis client using the stored settings."""

        if not self.configured:
            raise RuntimeError("Redis settings are not configured")

        if factory is not None:
            return factory(self)

        if redis is None:  # pragma: no cover - depends on optional dependency
            raise RuntimeError("redis package is not installed")

        options: dict[str, Any] = {
            "socket_timeout": self.socket_timeout,
            "health_check_interval": self.health_check_interval,
            "client_name": self.client_name,
            "retry_on_timeout": self.retry_on_timeout,
        }
        if self.username:
            options["username"] = self.username
        if self.password:
            options["password"] = self.password

        if self.url:
            options["ssl"] = self.ssl
            options["db"] = self.db
            return redis.from_url(self.url, **options)

        if not self.host:
            raise RuntimeError("Redis host not configured")

        options.update(
            {
                "host": self.host,
                "port": self.port,
                "db": self.db,
                "ssl": self.ssl,
            }
        )
        return redis.Redis(**options)

    def summary(self, *, redacted: bool = False) -> str:
        if not self.configured:
            return "Redis: not configured"
        try:
            url = self.connection_url()
        except ValueError:
            url = self.url or "redis://<incomplete>"
        if redacted and self.password:
            redacted_url = url.replace(self.password, "***")
        else:
            redacted_url = url.replace(self.password or "", "***") if self.password else url
        suffix = " (ssl)" if self.ssl else ""
        return f"Redis endpoint {redacted_url}{suffix}"


@dataclass(frozen=True)
class RedisCachePolicy:
    """Declarative caching policy for Redis-backed hot symbol caches."""

    ttl_seconds: int | None = 900
    max_keys: int | None = 512
    namespace: str = "emp:cache"
    invalidate_prefixes: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def bootstrap_defaults(cls) -> "RedisCachePolicy":
        return cls(ttl_seconds=1800, max_keys=256, namespace="emp:bootstrap")

    @classmethod
    def institutional_defaults(cls) -> "RedisCachePolicy":
        return cls(ttl_seconds=900, max_keys=1024, namespace="emp:cache")

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, str] | None = None,
        *,
        fallback: "RedisCachePolicy" | None = None,
    ) -> "RedisCachePolicy":
        payload = _normalise_env(mapping)
        defaults = fallback or cls.institutional_defaults()

        ttl = _coerce_optional_int(payload, "REDIS_CACHE_TTL_SECONDS", defaults.ttl_seconds)
        max_keys = _coerce_optional_int(payload, "REDIS_CACHE_MAX_KEYS", defaults.max_keys)
        namespace = payload.get("REDIS_CACHE_NAMESPACE", defaults.namespace).strip()
        if namespace.endswith(":"):
            namespace = namespace[:-1]

        invalidate_raw = payload.get("REDIS_CACHE_INVALIDATE_PREFIXES")
        invalidate = _parse_csv(invalidate_raw) or defaults.invalidate_prefixes

        return cls(
            ttl_seconds=ttl,
            max_keys=max_keys,
            namespace=namespace,
            invalidate_prefixes=invalidate,
        )

    def namespace_key(self, key: str) -> str:
        namespace = self.namespace.strip()
        if not namespace:
            return key
        if key.startswith(f"{namespace}:"):
            return key
        return f"{namespace}:{key}"


class CacheMetrics(dict[str, int | str]):
    """Dictionary payload describing cache telemetry counts."""


class InMemoryRedis:
    """Tiny in-memory Redis replacement used for bootstrap and tests."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}
        self._hits = 0
        self._misses = 0
        self._sets = 0

    def get(self, key: str) -> Any | None:
        if key in self._store:
            self._hits += 1
        else:
            self._misses += 1
        return self._store.get(key)

    def set(self, key: str, value: Any) -> Any:
        self._store[key] = value
        self._sets += 1
        return value

    def delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            if key in self._store:
                removed += 1
                del self._store[key]
        return removed

    def metrics(self, *, reset: bool = False) -> CacheMetrics:
        snapshot = CacheMetrics(
            hits=self._hits,
            misses=self._misses,
            evictions=0,
            expirations=0,
            invalidations=0,
            namespace="emp:inmemory",
            sets=self._sets,
            keys=len(self._store),
        )
        if reset:
            self._hits = self._misses = self._sets = 0
        return snapshot


class ManagedRedisCache:
    """Redis client decorator applying :class:`RedisCachePolicy` and telemetry."""

    def __init__(
        self,
        client: Any,
        policy: RedisCachePolicy,
        *,
        time_fn: TimeFn = time.monotonic,
    ) -> None:
        self._client = client
        self.policy = policy
        self._time_fn = time_fn
        self._key_order: "OrderedDict[str, float]" = OrderedDict()
        self._expiry: dict[str, float | None] = {}
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0
        self._invalidations = 0

    @property
    def raw_client(self) -> Any:
        return self._client

    def _purge_expired(self) -> None:
        now = self._time_fn()
        expired: list[str] = []
        for key, expiry in list(self._expiry.items()):
            if expiry is not None and expiry <= now:
                expired.append(key)
        if not expired:
            return
        delete = getattr(self._client, "delete", None)
        for key in expired:
            self._expiry.pop(key, None)
            self._key_order.pop(key, None)
            if callable(delete):
                try:
                    delete(key)
                except Exception:  # pragma: no cover - defensive logging
                    logger.debug("Failed to delete expired key %s", key)
        self._expirations += len(expired)

    def get(self, key: str) -> Any:
        namespaced = self.policy.namespace_key(key)
        self._purge_expired()
        try:
            value = self._client.get(namespaced)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Redis get failed for %s: %s", namespaced, exc)
            return None

        if value is None:
            self._misses += 1
            self._key_order.pop(namespaced, None)
            self._expiry.pop(namespaced, None)
            return None

        self._hits += 1
        if namespaced in self._key_order:
            self._key_order.move_to_end(namespaced)
        else:
            self._key_order[namespaced] = self._time_fn()
        return value

    def set(self, key: str, value: Any) -> Any:
        namespaced = self.policy.namespace_key(key)
        ttl = self.policy.ttl_seconds
        expiry = None if ttl is None or ttl <= 0 else self._time_fn() + ttl
        try:
            result = self._client.set(namespaced, value)
        except TypeError:
            result = self._client.set(namespaced, value)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Redis set failed for %s: %s", namespaced, exc)
            return None

        if expiry is not None:
            expire_fn = getattr(self._client, "expire", None)
            if callable(expire_fn):
                try:
                    expire_fn(namespaced, ttl)
                except Exception:  # pragma: no cover
                    logger.debug("Redis expire failed for %s", namespaced)

        self._expiry[namespaced] = expiry
        self._key_order[namespaced] = self._time_fn()
        self._key_order.move_to_end(namespaced)
        self._enforce_capacity()
        return result

    def delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            namespaced = self.policy.namespace_key(key)
            delete = getattr(self._client, "delete", None)
            if callable(delete):
                try:
                    delete(namespaced)
                except Exception:  # pragma: no cover
                    logger.debug("Redis delete failed for %s", namespaced)
            if namespaced in self._key_order:
                removed += 1
                self._key_order.pop(namespaced, None)
            self._expiry.pop(namespaced, None)
        return removed

    def invalidate(self, prefixes: Iterable[str] | None = None) -> int:
        targets = tuple(prefixes) if prefixes is not None else self.policy.invalidate_prefixes
        if not targets:
            return 0

        removed = 0
        delete = getattr(self._client, "delete", None)
        for prefix in targets:
            namespaced_prefix = self.policy.namespace_key(prefix)
            keys = [key for key in list(self._key_order) if key.startswith(namespaced_prefix)]
            if not keys:
                continue
            removed += len(keys)
            for key in keys:
                if callable(delete):
                    try:
                        delete(key)
                    except Exception:  # pragma: no cover
                        logger.debug("Redis delete failed during invalidate for %s", key)
                self._key_order.pop(key, None)
                self._expiry.pop(key, None)
        if removed:
            self._invalidations += removed
        return removed

    def metrics(self, *, reset: bool = False) -> CacheMetrics:
        snapshot: CacheMetrics = CacheMetrics(
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
            expirations=self._expirations,
            invalidations=self._invalidations,
            namespace=self.policy.namespace,
        )
        if reset:
            self._hits = self._misses = self._evictions = 0
            self._expirations = self._invalidations = 0
        return snapshot

    def _enforce_capacity(self) -> None:
        if self.policy.max_keys is None or self.policy.max_keys <= 0:
            return

        while len(self._key_order) > self.policy.max_keys:
            oldest, _ = self._key_order.popitem(last=False)
            self._expiry.pop(oldest, None)
            delete = getattr(self._client, "delete", None)
            if callable(delete):
                try:
                    delete(oldest)
                except Exception:  # pragma: no cover
                    logger.debug("Redis delete failed during eviction for %s", oldest)
            self._evictions += 1

    def __getattr__(self, item: str) -> Any:
        return getattr(self._client, item)


def configure_redis_client(
    settings: RedisConnectionSettings,
    *,
    factory: ClientFactory | None = None,
    ping: bool = True,
) -> Any | None:
    """Create and validate a Redis client, returning ``None`` on failure."""

    if not settings.configured:
        logger.debug("Redis settings not configured; skipping client creation")
        return None

    try:
        client = settings.create_client(factory=factory)
    except Exception as exc:
        logger.warning("Failed to instantiate Redis client: %s", exc)
        return None

    if ping:
        try:
            response = client.ping()
            logger.debug("Redis ping response: %s", response)
        except Exception as exc:  # pragma: no cover - network failure handling
            logger.warning("Redis ping failed: %s", exc)
            return None

    logger.info("Redis cache configured for %s", settings.summary())
    return client


def wrap_managed_cache(
    client: Any | None,
    *,
    policy: RedisCachePolicy | None = None,
    bootstrap: bool = False,
) -> ManagedRedisCache:
    """Ensure callers receive a :class:`ManagedRedisCache` with telemetry."""

    resolved_policy = policy or (
        RedisCachePolicy.bootstrap_defaults()
        if bootstrap
        else RedisCachePolicy.institutional_defaults()
    )
    base_client = client or InMemoryRedis()
    if isinstance(base_client, ManagedRedisCache):
        return base_client
    return ManagedRedisCache(base_client, resolved_policy)


__all__ = [
    "RedisConnectionSettings",
    "RedisCachePolicy",
    "InMemoryRedis",
    "ManagedRedisCache",
    "configure_redis_client",
    "wrap_managed_cache",
]
