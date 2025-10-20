from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Mapping

import pytest

from src.operational.fix_connection_manager import FIXConnectionManager
from src.operational.live_broker_secrets import load_live_broker_secrets


def test_load_live_broker_secrets_selects_active_profile() -> None:
    now = datetime.now(timezone.utc)
    mapping = {
        "LIVE_BROKER_SANDBOX_PRICE_SENDER_COMP_ID": "demo.price",
        "LIVE_BROKER_SANDBOX_PRICE_USERNAME": "demo_price_user",
        "LIVE_BROKER_SANDBOX_PRICE_PASSWORD": "demo_price_pw",
        "LIVE_BROKER_SANDBOX_TRADE_SENDER_COMP_ID": "demo.trade",
        "LIVE_BROKER_SANDBOX_TRADE_USERNAME": "demo_trade_user",
        "LIVE_BROKER_SANDBOX_TRADE_PASSWORD": "demo_trade_pw",
        "LIVE_BROKER_SANDBOX_ROTATED_AT": (now - timedelta(days=5)).isoformat(),
        "LIVE_BROKER_PROD_PRICE_SENDER_COMP_ID": "prod.price",
        "LIVE_BROKER_PROD_PRICE_USERNAME": "prod_price_user",
        "LIVE_BROKER_PROD_PRICE_PASSWORD": "prod_price_pw",
        "LIVE_BROKER_PROD_TRADE_SENDER_COMP_ID": "prod.trade",
        "LIVE_BROKER_PROD_TRADE_USERNAME": "prod_trade_user",
        "LIVE_BROKER_PROD_TRADE_PASSWORD": "prod_trade_pw",
        "LIVE_BROKER_PROD_ROTATED_AT": (now - timedelta(days=2)).isoformat(),
        "LIVE_BROKER_PROD_SECRET_NAME": "vault/emp/prod",
    }

    secrets = load_live_broker_secrets(mapping, environment="production")

    profile = secrets.active_profile
    assert profile is not None
    assert profile.price is not None and profile.price.username == "prod_price_user"
    assert profile.trade is not None and profile.trade.username == "prod_trade_user"
    assert secrets.healthy is True

    summary = secrets.describe()
    assert summary["healthy"] is True
    assert summary["profiles"]["prod"]["price"]["password"] == "***"


def test_load_live_broker_secrets_falls_back_to_legacy_fix_keys() -> None:
    mapping = {
        "FIX_PRICE_SENDER_COMP_ID": "legacy.price",
        "FIX_PRICE_USERNAME": "legacy_price_user",
        "FIX_PRICE_PASSWORD": "legacy_price_pw",
        "FIX_TRADE_SENDER_COMP_ID": "legacy.trade",
        "FIX_TRADE_USERNAME": "legacy_trade_user",
        "FIX_TRADE_PASSWORD": "legacy_trade_pw",
    }

    secrets = load_live_broker_secrets(mapping, environment="demo")

    profile = secrets.active_profile
    assert profile is not None
    assert profile.is_complete() is True
    assert profile.price is not None and profile.price.username == "legacy_price_user"
    assert profile.trade is not None and profile.trade.username == "legacy_trade_user"


def test_fix_connection_manager_describes_live_broker_secrets(monkeypatch) -> None:
    monkeypatch.setenv("EMP_USE_MOCK_FIX", "1")
    extras = {
        "LIVE_BROKER_SANDBOX_PRICE_SENDER_COMP_ID": "demo.price",
        "LIVE_BROKER_SANDBOX_PRICE_USERNAME": "demo_price_user",
        "LIVE_BROKER_SANDBOX_PRICE_PASSWORD": "demo_price_pw",
        "LIVE_BROKER_SANDBOX_TRADE_SENDER_COMP_ID": "demo.trade",
        "LIVE_BROKER_SANDBOX_TRADE_USERNAME": "demo_trade_user",
        "LIVE_BROKER_SANDBOX_TRADE_PASSWORD": "demo_trade_pw",
        "LIVE_BROKER_SANDBOX_ROTATED_AT": (
            datetime.now(timezone.utc) - timedelta(days=1)
        ).isoformat(),
    }

    class _Cfg:
        environment = "demo"
        account_number = None
        password = None

        def __getattr__(self, item: str) -> object:
            return None

    cfg = _Cfg()
    cfg.extras = extras

    manager = FIXConnectionManager(cfg)
    try:
        assert manager.start_sessions() is True
        summary = manager.describe_live_broker_secrets()
        assert summary is not None
        assert summary["healthy"] is True
        assert summary["profiles"]["sandbox"]["complete"] is True
    finally:
        manager.stop_sessions()


def test_load_live_broker_secrets_aws_secrets_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    secret_payload = {
        "LIVE_BROKER_PROD_PRICE_SENDER_COMP_ID": "aws.prod.price",
        "LIVE_BROKER_PROD_PRICE_USERNAME": "aws_price_user",
        "LIVE_BROKER_PROD_PRICE_PASSWORD": "aws_price_pw",
        "LIVE_BROKER_PROD_TRADE_SENDER_COMP_ID": "aws.prod.trade",
        "LIVE_BROKER_PROD_TRADE_USERNAME": "aws_trade_user",
        "LIVE_BROKER_PROD_TRADE_PASSWORD": "aws_trade_pw",
        "LIVE_BROKER_PROD_SECRET_REF": "aws/prod/live",
    }

    class _StubClient:
        def __init__(self, payload: Mapping[str, str]) -> None:
            self._payload = payload

        def get_secret_value(self, SecretId: str) -> Mapping[str, object]:
            assert SecretId == "arn:aws:secretsmanager:us-east-1:123456789012:secret:live"
            return {"SecretString": json.dumps(self._payload)}

    class _StubSession:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def client(self, service_name: str, region_name: str | None = None) -> _StubClient:
            assert service_name == "secretsmanager"
            assert region_name == "us-east-1"
            return _StubClient(secret_payload)

    def _stub_client(service_name: str, region_name: str | None = None) -> _StubClient:
        assert service_name == "secretsmanager"
        return _StubClient(secret_payload)

    stub_boto3 = SimpleNamespace(
        session=SimpleNamespace(Session=_StubSession),
        client=_stub_client,
    )

    monkeypatch.setitem(sys.modules, "boto3", stub_boto3)

    mapping = {
        "LIVE_BROKER_SECRET_MANAGER": "aws_secrets_manager",
        "LIVE_BROKER_SECRET_ID": "arn:aws:secretsmanager:us-east-1:123456789012:secret:live",
        "LIVE_BROKER_SECRET_REGION": "us-east-1",
    }

    secrets = load_live_broker_secrets(mapping, environment="production")

    profile = secrets.active_profile
    assert profile is not None
    assert profile.price is not None and profile.price.username == "aws_price_user"
    assert profile.trade is not None and profile.trade.username == "aws_trade_user"
    assert profile.secret_reference == "aws/prod/live"


def test_load_live_broker_secrets_vault(monkeypatch: pytest.MonkeyPatch) -> None:
    secret_payload = {
        "LIVE_BROKER_SANDBOX_PRICE_SENDER_COMP_ID": "vault.sandbox.price",
        "LIVE_BROKER_SANDBOX_PRICE_USERNAME": "vault_price_user",
        "LIVE_BROKER_SANDBOX_PRICE_PASSWORD": "vault_price_pw",
        "LIVE_BROKER_SANDBOX_TRADE_SENDER_COMP_ID": "vault.sandbox.trade",
        "LIVE_BROKER_SANDBOX_TRADE_USERNAME": "vault_trade_user",
        "LIVE_BROKER_SANDBOX_TRADE_PASSWORD": "vault_trade_pw",
        "LIVE_BROKER_SANDBOX_SECRET_PATH": "secret/data/demo",
    }

    class _StubKV2:
        def read_secret_version(self, *, path: str, mount_point: str, **_: object) -> Mapping[str, object]:
            assert path == "kv/data/emp/demo"
            assert mount_point == "secret"
            return {"data": {"data": secret_payload}}

    class _StubClient:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.secrets = SimpleNamespace(kv=SimpleNamespace(v2=_StubKV2()))

    stub_hvac = SimpleNamespace(Client=_StubClient)

    monkeypatch.setitem(sys.modules, "hvac", stub_hvac)

    mapping = {
        "LIVE_BROKER_SECRET_MANAGER_PROVIDER": "vault",
        "LIVE_BROKER_VAULT_ADDR": "https://vault.example.test",
        "LIVE_BROKER_VAULT_TOKEN": "vault-token",
        "LIVE_BROKER_VAULT_PATH": "kv/data/emp/demo",
        "LIVE_BROKER_VAULT_MOUNT": "secret",
    }

    secrets = load_live_broker_secrets(mapping, environment="sandbox")

    profile = secrets.active_profile
    assert profile is not None
    assert profile.price is not None and profile.price.username == "vault_price_user"
    assert profile.trade is not None and profile.trade.username == "vault_trade_user"
    assert profile.secret_reference == "secret/data/demo"
