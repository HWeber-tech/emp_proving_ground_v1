from __future__ import annotations

from typing import Any

import pytest

from src.data_foundation.persist.timescale import TimescaleConnectionSettings


class _DummyEngine:
    def __init__(self, dispose_called: list[bool]) -> None:
        self._dispose_called = dispose_called

    def dispose(self) -> None:  # pragma: no cover - defensive; not used
        self._dispose_called.append(True)


def test_timescale_connection_settings_applies_pool_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    extras = {
        "TIMESCALEDB_URL": "postgresql://user:pass@localhost:5432/emp",
        "TIMESCALEDB_APP": "test-app",
        "TIMESCALEDB_POOL_SIZE": "5",
        "TIMESCALEDB_MAX_OVERFLOW": "7",
        "TIMESCALEDB_POOL_TIMEOUT": "12.5",
        "TIMESCALEDB_POOL_RECYCLE": "600",
        "TIMESCALEDB_POOL_PRE_PING": "false",
        "TIMESCALEDB_STATEMENT_TIMEOUT_MS": "15000",
        "TIMESCALEDB_CONNECT_TIMEOUT": "9",
    }

    settings = TimescaleConnectionSettings.from_mapping(extras)

    captured: dict[str, Any] = {}

    def _fake_create_engine(url: str, **kwargs: Any) -> _DummyEngine:
        captured["url"] = url
        captured["kwargs"] = kwargs
        return _DummyEngine([])

    monkeypatch.setattr(
        "src.data_foundation.persist.timescale.create_engine",
        _fake_create_engine,
    )

    settings.create_engine()

    assert captured["url"] == extras["TIMESCALEDB_URL"]
    params = captured["kwargs"]
    assert params["pool_size"] == 5
    assert params["max_overflow"] == 7
    assert params["pool_timeout"] == pytest.approx(12.5)
    assert params["pool_recycle"] == 600
    assert params["pool_pre_ping"] is False
    assert "connect_args" in params
    assert params["connect_args"]["application_name"] == "test-app"
    assert params["connect_args"]["connect_timeout"] == 9
    assert params["connect_args"]["options"].endswith("statement_timeout=15000")


def test_timescale_connection_settings_skips_pool_overrides_for_sqlite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extras = {
        "TIMESCALEDB_URL": "sqlite:///tmp/test.db",
        "TIMESCALEDB_POOL_SIZE": "12",
        "TIMESCALEDB_POOL_PRE_PING": "0",
    }

    settings = TimescaleConnectionSettings.from_mapping(extras)

    captured: dict[str, Any] = {}

    def _fake_create_engine(url: str, **kwargs: Any) -> _DummyEngine:
        captured["url"] = url
        captured["kwargs"] = kwargs
        return _DummyEngine([])

    monkeypatch.setattr(
        "src.data_foundation.persist.timescale.create_engine",
        _fake_create_engine,
    )

    settings.create_engine()

    assert captured["url"] == extras["TIMESCALEDB_URL"]
    params = captured["kwargs"]
    assert "pool_size" not in params
    assert "max_overflow" not in params
    assert params["pool_pre_ping"] is False
    assert "connect_args" not in params


def test_timescale_connection_settings_builds_url_from_components() -> None:
    extras = {
        "TIMESCALEDB_HOST": "timescale.prod.internal",
        "TIMESCALEDB_PORT": "6543",
        "TIMESCALEDB_DB": "alpha_trade",
        "TIMESCALEDB_USER": "alpha_user",
        "TIMESCALEDB_PASSWORD": "s3cr3t!",
        "TIMESCALEDB_SSLMODE": "require",
        "TIMESCALEDB_SSLROOTCERT": "/etc/timescale/root.pem",
        "TIMESCALEDB_DRIVER": "psycopg2",
    }

    settings = TimescaleConnectionSettings.from_mapping(extras)

    assert (
        settings.url
        == "postgresql+psycopg2://alpha_user:s3cr3t%21@timescale.prod.internal:6543/alpha_trade"
        "?sslmode=require&sslrootcert=/etc/timescale/root.pem"
    )
    assert settings.configured is True


def test_timescale_connection_settings_prefers_explicit_url() -> None:
    extras = {
        "TIMESCALEDB_URL": "postgresql://primary/db",
        "TIMESCALEDB_HOST": "ignored.example.com",
    }

    settings = TimescaleConnectionSettings.from_mapping(extras)

    assert settings.url == "postgresql://primary/db"
