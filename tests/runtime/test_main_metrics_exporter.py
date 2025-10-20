from __future__ import annotations

from types import SimpleNamespace

import pytest

import main
from src.governance.system_config import EmpEnvironment, SystemConfig


@pytest.fixture(autouse=True)
def logger_stub(monkeypatch):
    stub_logger = SimpleNamespace(info=[], warning=[], exception=[])

    def _record(level):
        sequence = getattr(stub_logger, level)

        def _inner(message, *args, **kwargs):
            sequence.append((message, args, kwargs))

        return _inner

    monkeypatch.setattr(main, "logger", SimpleNamespace(  # type: ignore[assignment]
        info=_record("info"),
        warning=_record("warning"),
        exception=_record("exception"),
    ))
    yield stub_logger


def test_metrics_exporter_uses_tls_and_port(monkeypatch):
    calls: dict[str, tuple[object | None, object | None, object | None]] = {}

    def _fake_start(port=None, cert_path=None, key_path=None):
        calls["args"] = (port, cert_path, key_path)

    monkeypatch.setattr(main, "start_metrics_server", _fake_start)

    main._maybe_start_metrics_exporter(
        {
            "METRICS_EXPORTER_PORT": "9200",
            "METRICS_EXPORTER_TLS_CERT_PATH": "/certs/runtime.pem",
            "METRICS_EXPORTER_TLS_KEY_PATH": "/certs/runtime.key",
        }
    )

    assert calls["args"] == (9200, "/certs/runtime.pem", "/certs/runtime.key")


def test_metrics_exporter_disabled(monkeypatch, logger_stub):
    called = False

    def _fake_start(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(main, "start_metrics_server", _fake_start)

    main._maybe_start_metrics_exporter({"METRICS_EXPORTER_ENABLED": "false"})

    assert not called
    assert logger_stub.info and "disabled" in logger_stub.info[0][0]


def test_metrics_exporter_invalid_port(monkeypatch, logger_stub):
    captured: dict[str, object | None] = {}

    def _fake_start(port=None, **_kwargs):
        captured["port"] = port

    monkeypatch.setattr(main, "start_metrics_server", _fake_start)

    extras = {"METRICS_EXPORTER_PORT": "invalid"}
    main._maybe_start_metrics_exporter(extras)

    assert captured.get("port") is None
    assert logger_stub.warning and "Invalid METRICS_EXPORTER_PORT" in logger_stub.warning[0][0]


def test_warns_for_production_without_tls(logger_stub):
    config = SystemConfig().with_updated(
        environment=EmpEnvironment.production,
        extras={},
    )

    main._warn_if_production_tls_disabled(config, config.extras)

    assert logger_stub.warning, "expected TLS warning for production without TLS"


def test_skips_warning_when_production_tls_present(logger_stub):
    config = SystemConfig().with_updated(
        environment=EmpEnvironment.production,
        extras={"SECURITY_TLS_VERSIONS": "TLS1.2,TLS1.3"},
    )

    main._warn_if_production_tls_disabled(config, config.extras)

    assert not logger_stub.warning, "unexpected TLS warning when TLS versions configured"
