from __future__ import annotations

import json
import sys
import types

from src.operations import secrets_manager


def test_resolve_secret_reference_parses_aws_secret_json(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _Client:
        def __init__(self, **kwargs: object) -> None:
            captured["client_kwargs"] = kwargs

        def get_secret_value(self, *, SecretId: str) -> dict[str, object]:
            captured["secret_id"] = SecretId
            payload = {
                "price": {"username": "aws_price_user", "password": "aws_price_pw"},
                "trade": {"username": "aws_trade_user", "password": "aws_trade_pw"},
                "metadata": {"rotation_approver": "aws-team"},
            }
            return {"SecretString": json.dumps(payload)}

    class _Session:
        def __init__(self, **kwargs: object) -> None:
            captured["session_kwargs"] = kwargs

        def client(self, service_name: str, **kwargs: object) -> _Client:
            captured["service_name"] = service_name
            return _Client(**kwargs)

    fake_boto3 = types.SimpleNamespace(session=types.SimpleNamespace(Session=_Session))
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    payload = secrets_manager.resolve_secret_reference(
        "aws://emp/prod/credentials?region=us-east-1",
        env={},
    )

    assert payload is not None
    assert payload["PRICE_USERNAME"] == "aws_price_user"
    assert payload["TRADE_PASSWORD"] == "aws_trade_pw"
    assert payload["METADATA_ROTATION_APPROVER"] == "aws-team"

    assert captured["session_kwargs"] == {"region_name": "us-east-1"}
    assert captured["service_name"] == "secretsmanager"
    assert captured["secret_id"] == "emp/prod/credentials"


def test_resolve_secret_reference_parses_vault_secret(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _KV:
        def __init__(self) -> None:
            self.v2 = types.SimpleNamespace(read_secret_version=self._read_secret_version)

        @staticmethod
        def _read_secret_version(**kwargs: object) -> dict[str, object]:
            captured["kv_kwargs"] = kwargs
            return {
                "data": {
                    "data": {
                        "price": {
                            "username": "vault_price_user",
                            "password": "vault_price_pw",
                        },
                        "trade": {
                            "username": "vault_trade_user",
                            "password": "vault_trade_pw",
                        },
                    }
                }
            }

    class _Secrets:
        def __init__(self) -> None:
            self.kv = _KV()

    class _VaultClient:
        def __init__(
            self,
            *,
            url: str,
            token: str,
            namespace: str | None,
            verify: object,
        ) -> None:
            captured["client_kwargs"] = {
                "url": url,
                "token": token,
                "namespace": namespace,
                "verify": verify,
            }
            self.secrets = _Secrets()

    fake_hvac = types.SimpleNamespace(Client=_VaultClient)
    monkeypatch.setitem(sys.modules, "hvac", fake_hvac)

    env = {"VAULT_ADDR": "https://vault.example", "VAULT_TOKEN": "root"}
    payload = secrets_manager.resolve_secret_reference(
        "vault://kv/data/emp/trading?field=price",
        env=env,
    )

    assert payload is not None
    assert payload["PRICE_USERNAME"] == "vault_price_user"
    assert payload["PRICE_PASSWORD"] == "vault_price_pw"
    assert "TRADE_USERNAME" not in payload

    assert captured["client_kwargs"]["url"] == "https://vault.example"
    assert captured["kv_kwargs"]["path"] == "kv/data/emp/trading"
    assert captured["kv_kwargs"]["mount_point"] == "secret"
