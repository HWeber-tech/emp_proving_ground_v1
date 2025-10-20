"""Minimal JWT-style token helpers with explicit role embedding."""

from __future__ import annotations

import base64
import binascii
import json
import logging
import hmac
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence


logger = logging.getLogger(__name__)

__all__ = [
    "AuthTokenError",
    "InvalidTokenError",
    "ExpiredTokenError",
    "create_access_token",
    "create_auth_token",
    "decode_access_token",
    "decode_auth_token",
]


class AuthTokenError(RuntimeError):
    """Base error raised for token encoding or decoding problems."""


class InvalidTokenError(AuthTokenError):
    """Raised when the token signature or structure is invalid."""


class ExpiredTokenError(AuthTokenError):
    """Raised when a token is decoded after its expiry."""


def _b64url_encode(payload: bytes) -> str:
    return base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")


def _b64url_decode(payload: str) -> bytes:
    padding = "=" * (-len(payload) % 4)
    return base64.urlsafe_b64decode(payload + padding)


def _sign(message: bytes, secret: bytes) -> bytes:
    return hmac.new(secret, message, hashlib.sha256).digest()


def _normalise_roles(roles: Iterable[str] | None) -> list[str]:
    seen: dict[str, None] = {}
    if not roles:
        return []
    for role in roles:
        text = str(role).strip()
        if not text:
            continue
        seen.setdefault(text, None)
    return list(seen.keys())


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def create_access_token(
    subject: str,
    *,
    secret: str | bytes,
    roles: Iterable[str] | None = None,
    expires_in: timedelta | None = None,
    issued_at: datetime | None = None,
    audience: str | None = None,
    extra_claims: Mapping[str, Any] | None = None,
) -> str:
    """Create an HMAC-SHA256 signed token embedding supplied roles."""

    if issued_at is None:
        issued_at = _utc_now()
    if issued_at.tzinfo is None:
        issued_at = issued_at.replace(tzinfo=timezone.utc)
    iat = int(issued_at.timestamp())

    payload: dict[str, Any] = {"sub": str(subject), "iat": iat}
    if audience is not None:
        payload["aud"] = str(audience)

    normalised_roles = _normalise_roles(roles)
    payload["roles"] = normalised_roles

    if expires_in is not None:
        if expires_in.total_seconds() <= 0:
            raise ValueError("expires_in must be positive")
        expiry = issued_at + expires_in
        payload["exp"] = int(expiry.timestamp())

    if extra_claims:
        for key, value in extra_claims.items():
            if key in payload:
                raise ValueError(f"claim '{key}' would overwrite a reserved field")
            payload[key] = value

    header = {"typ": "JWT", "alg": "HS256"}

    encoded_header = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    encoded_payload = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))

    signing_input = f"{encoded_header}.{encoded_payload}".encode("ascii")
    secret_bytes = secret.encode("utf-8") if isinstance(secret, str) else bytes(secret)
    signature = _b64url_encode(_sign(signing_input, secret_bytes))
    token = f"{encoded_header}.{encoded_payload}.{signature}"

    logger.info(
        "Created access token",
        extra={
            "auth.subject": payload["sub"],
            "auth.roles_count": len(normalised_roles),
            "auth.audience": payload.get("aud"),
            "auth.expires_at": expiry.isoformat() if expires_in is not None else None,
        },
    )
    return token


def decode_access_token(
    token: str,
    *,
    secret: str | bytes,
    verify_expiry: bool = True,
    expected_audience: str | None = None,
) -> Mapping[str, Any]:
    """Decode and validate a token previously produced by ``create_access_token``."""

    fingerprint = _token_fingerprint(token)
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise InvalidTokenError("token must have three segments")

        header_part, payload_part, signature_part = parts
        signing_input = f"{header_part}.{payload_part}".encode("ascii")
        secret_bytes = secret.encode("utf-8") if isinstance(secret, str) else bytes(secret)
        expected_signature = _sign(signing_input, secret_bytes)
        try:
            provided_signature = _b64url_decode(signature_part)
        except (ValueError, binascii.Error) as exc:
            raise InvalidTokenError("token signature is not valid base64") from exc

        if not hmac.compare_digest(expected_signature, provided_signature):
            raise InvalidTokenError("token signature mismatch")

        try:
            payload_bytes = _b64url_decode(payload_part)
            payload = json.loads(payload_bytes)
        except Exception as exc:  # pragma: no cover - defensive guard
            raise InvalidTokenError("token payload could not be decoded") from exc

        if expected_audience is not None:
            audience = payload.get("aud")
            if audience != expected_audience:
                raise InvalidTokenError("unexpected token audience")

        if verify_expiry and "exp" in payload:
            now = int(_utc_now().timestamp())
            try:
                expiry = int(payload["exp"])
            except (TypeError, ValueError) as exc:
                raise InvalidTokenError("invalid exp claim") from exc
            if now >= expiry:
                raise ExpiredTokenError("token has expired")

        roles_claim = payload.get("roles")
        if roles_claim is None:
            payload["roles"] = []
        elif isinstance(roles_claim, Sequence) and not isinstance(roles_claim, (str, bytes)):
            payload["roles"] = [str(role) for role in roles_claim]
        else:
            payload["roles"] = [str(roles_claim)]

    except ExpiredTokenError:
        logger.warning(
            "Access token rejected",
            extra={
                "auth.token_hash": fingerprint,
                "auth.reason": "expired",
                "auth.expected_audience": expected_audience,
            },
        )
        raise
    except InvalidTokenError as exc:
        logger.warning(
            "Access token rejected",
            extra={
                "auth.token_hash": fingerprint,
                "auth.reason": exc.__class__.__name__,
                "auth.expected_audience": expected_audience,
            },
        )
        raise
    except AuthTokenError:
        logger.warning(
            "Access token rejected",
            extra={
                "auth.token_hash": fingerprint,
                "auth.reason": "auth_token_error",
                "auth.expected_audience": expected_audience,
            },
        )
        raise
    except Exception:
        logger.exception(
            "Access token validation errored",
            extra={
                "auth.token_hash": fingerprint,
                "auth.expected_audience": expected_audience,
            },
        )
        raise
    else:
        logger.info(
            "Validated access token",
            extra={
                "auth.token_hash": fingerprint,
                "auth.subject": payload.get("sub"),
                "auth.roles_count": len(payload.get("roles", [])),
                "auth.audience": payload.get("aud"),
            },
        )
        return payload


# Backwards compatible aliases
create_auth_token = create_access_token
decode_auth_token = decode_access_token


def _token_fingerprint(token: str) -> str:
    """Return a truncated token fingerprint suitable for logging."""

    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return digest[:16]
