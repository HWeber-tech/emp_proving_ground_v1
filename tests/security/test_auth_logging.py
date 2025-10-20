from __future__ import annotations

import logging

import pytest

from src.governance.token_manager import TokenManager, TokenValidationError
from src.security.auth_tokens import (
    InvalidTokenError,
    create_access_token,
    decode_access_token,
)


def _tamper(token: str) -> str:
    """Return a token with the final character flipped for invalidation."""

    tail = token[-1]
    replacement = "A" if tail != "A" else "B"
    return f"{token[:-1]}{replacement}"


def test_token_manager_logs_issue_and_validation(caplog: pytest.LogCaptureFixture) -> None:
    manager = TokenManager("secret-key")

    with caplog.at_level(logging.INFO):
        issued = manager.issue_token("user-1", claims={"scope": "ops"})

    assert any(record.message == "Issued auth token" for record in caplog.records)

    caplog.clear()

    with caplog.at_level(logging.INFO):
        manager.decode_token(issued.token)

    assert any(record.message == "Validated auth token" for record in caplog.records)


def test_token_manager_logs_validation_failure(caplog: pytest.LogCaptureFixture) -> None:
    manager = TokenManager("secret-key")
    issued = manager.issue_token("user-2")
    bad_token = _tamper(issued.token)

    with caplog.at_level(logging.WARNING):
        with pytest.raises(TokenValidationError):
            manager.decode_token(bad_token)

    assert any(record.message == "Auth token rejected" for record in caplog.records)


def test_access_token_logging_success_and_failure(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        token = create_access_token(
            "user-3",
            secret="secret-key",
            roles=["ops"],
        )

    assert any(record.message == "Created access token" for record in caplog.records)

    caplog.clear()

    with caplog.at_level(logging.INFO):
        decode_access_token(token, secret="secret-key")

    assert any(record.message == "Validated access token" for record in caplog.records)

    caplog.clear()

    with caplog.at_level(logging.WARNING):
        with pytest.raises(InvalidTokenError):
            decode_access_token(_tamper(token), secret="secret-key")

    assert any(record.message == "Access token rejected" for record in caplog.records)
