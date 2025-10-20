"""Simple HMAC-signed user token management with explicit expirations."""

from __future__ import annotations

import base64
import hashlib
import logging
import hmac
import json
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Mapping


__all__ = [
    "IssuedToken",
    "TokenExpired",
    "TokenManager",
    "TokenManagerError",
    "TokenRevoked",
    "TokenValidationError",
]


def _b64urlsafe_encode(payload: bytes) -> str:
    """Encode *payload* using base64-url without padding."""

    return base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")


def _b64urlsafe_decode(segment: str) -> bytes:
    """Decode a base64-url segment, adding padding when necessary."""

    padding = "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(f"{segment}{padding}".encode("ascii"))


class TokenManagerError(RuntimeError):
    """Base class for token manager exceptions."""


class TokenValidationError(TokenManagerError):
    """Raised when a token signature or structure is invalid."""


class TokenExpired(TokenValidationError):
    """Raised when a token has expired."""


class TokenRevoked(TokenValidationError):
    """Raised when a token was explicitly revoked."""


@dataclass(frozen=True, slots=True)
class IssuedToken:
    """Materialised token metadata returned by :class:`TokenManager`."""

    token: str
    user_id: str
    issued_at: datetime
    expires_at: datetime
    claims: Mapping[str, Any]


def _normalise_positive_duration(value: timedelta | int | float, *, field: str) -> timedelta:
    """Coerce durations expressed as ``timedelta`` or seconds into ``timedelta``."""

    if isinstance(value, timedelta):
        duration = value
    else:
        duration = timedelta(seconds=float(value))
    if duration <= timedelta(0):
        raise ValueError(f"{field} must be positive")
    return duration


def _normalise_non_negative_duration(value: timedelta | int | float, *, field: str) -> timedelta:
    """Coerce non-negative durations; used for validation leeway."""

    if isinstance(value, timedelta):
        duration = value
    else:
        duration = timedelta(seconds=float(value))
    if duration < timedelta(0):
        raise ValueError(f"{field} must be greater or equal to zero")
    return duration


def _coerce_epoch_seconds(value: Any, *, claim: str, required: bool) -> datetime | None:
    """Coerce JWT epoch claims into timezone-aware :class:`datetime` objects."""

    if value is None:
        if required:
            raise TokenValidationError(f"Token missing '{claim}' claim")
        return None
    try:
        timestamp = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
        raise TokenValidationError(f"Token claim '{claim}' is not a valid epoch timestamp") from exc
    return datetime.fromtimestamp(timestamp, tz=UTC)


class TokenManager:
    """Create, sign, and validate tokens with enforced expiration times."""

    def __init__(
        self,
        secret_key: str,
        *,
        default_ttl: timedelta | int | float = timedelta(hours=1),
    ) -> None:
        if not secret_key or not secret_key.strip():
            raise ValueError("secret_key must be a non-empty string")

        self._secret_key = secret_key.encode("utf-8")
        self._default_ttl = _normalise_positive_duration(default_ttl, field="default_ttl")
        self._revoked_tokens: set[str] = set()
        self._issued_tokens: dict[str, IssuedToken] = {}

    # ------------------------------------------------------------------
    # Token lifecycle helpers
    # ------------------------------------------------------------------
    def issue_token(
        self,
        user_id: str,
        *,
        ttl: timedelta | int | float | None = None,
        claims: Mapping[str, Any] | None = None,
    ) -> IssuedToken:
        """Issue a new token for *user_id* with an explicit expiration."""

        if not user_id or not str(user_id).strip():
            raise ValueError("user_id must be provided")

        duration = self._default_ttl if ttl is None else _normalise_positive_duration(ttl, field="ttl")

        issued_at = datetime.now(tz=UTC)
        expires_at = issued_at + duration

        payload: dict[str, Any] = {
            "sub": str(user_id),
            "iat": int(issued_at.timestamp()),
            "exp": int(expires_at.timestamp()),
            "jti": secrets.token_hex(16),
        }

        if claims:
            for key, value in claims.items():
                if key in {"sub", "iat", "exp", "jti"}:
                    raise ValueError(f"Claim '{key}' is reserved and cannot be overridden")
                payload[key] = value

        token = self._encode(payload)
        issued = IssuedToken(token=token, user_id=str(user_id), issued_at=issued_at, expires_at=expires_at, claims=dict(payload))

        self._issued_tokens[token] = issued
        self._revoked_tokens.discard(token)
        logger.info(
            "Issued auth token",
            extra={
                "auth.user_id": issued.user_id,
                "auth.jti": issued.claims.get("jti"),
                "auth.expires_at": issued.expires_at.isoformat(),
                "auth.claim_keys": sorted(key for key in issued.claims.keys() if key not in {"sub", "iat", "exp", "jti"}),
            },
        )
        return issued

    def revoke_token(self, token: str) -> None:
        """Explicitly revoke a token so future validations fail."""

        metadata = self._issued_tokens.get(token)
        fingerprint = _token_fingerprint(token)
        self._revoked_tokens.add(token)
        self._issued_tokens.pop(token, None)
        logger.info(
            "Revoked auth token",
            extra={
                "auth.token_hash": fingerprint,
                "auth.user_id": metadata.user_id if metadata else None,
                "auth.jti": metadata.claims.get("jti") if metadata else None,
            },
        )

    def is_revoked(self, token: str) -> bool:
        """Return ``True`` when *token* has been revoked."""

        return token in self._revoked_tokens

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def decode_token(
        self,
        token: str,
        *,
        verify_expiration: bool = True,
        leeway: timedelta | int | float = 0,
    ) -> IssuedToken:
        """Validate *token* and return its metadata.

        When ``verify_expiration`` is ``False`` the ``exp`` claim is still
        parsed but not enforced. ``leeway`` allows a small non-negative grace
        period for clock skew during validation.
        """

        fingerprint = _token_fingerprint(token)
        if self.is_revoked(token):
            logger.warning(
                "Auth token rejected",
                extra={
                    "auth.token_hash": fingerprint,
                    "auth.reason": "revoked",
                },
            )
            raise TokenRevoked("Token has been revoked")

        try:
            payload = self._decode(token)
        except TokenValidationError as exc:
            logger.warning(
                "Auth token rejected",
                extra={
                    "auth.token_hash": fingerprint,
                    "auth.reason": exc.__class__.__name__,
                },
            )
            raise

        expires_at = _coerce_epoch_seconds(payload.get("exp"), claim="exp", required=True)
        if expires_at is None:  # pragma: no cover - defensive fallback
            message = "Token missing 'exp' claim"
            logger.warning(
                "Auth token rejected",
                extra={
                    "auth.token_hash": fingerprint,
                    "auth.reason": "missing_exp",
                },
            )
            raise TokenValidationError(message)

        leeway_duration = _normalise_non_negative_duration(leeway, field="leeway")

        if verify_expiration:
            now = datetime.now(tz=UTC)
            if expires_at is not None and now - leeway_duration >= expires_at:
                logger.warning(
                    "Auth token rejected",
                    extra={
                        "auth.token_hash": fingerprint,
                        "auth.reason": "expired",
                    },
                )
                raise TokenExpired("Token has expired")

        issued_at = _coerce_epoch_seconds(payload.get("iat"), claim="iat", required=False) or expires_at

        user_id_raw = payload.get("sub")
        if user_id_raw is None:
            logger.warning(
                "Auth token rejected",
                extra={
                    "auth.token_hash": fingerprint,
                    "auth.reason": "missing_sub",
                },
            )
            raise TokenValidationError("Token missing 'sub' claim")

        issued = IssuedToken(
            token=token,
            user_id=str(user_id_raw),
            issued_at=issued_at,
            expires_at=expires_at,
            claims=dict(payload),
        )
        self._issued_tokens.setdefault(token, issued)
        logger.info(
            "Validated auth token",
            extra={
                "auth.token_hash": fingerprint,
                "auth.user_id": issued.user_id,
                "auth.jti": issued.claims.get("jti"),
                "auth.expires_at": issued.expires_at.isoformat() if issued.expires_at else None,
            },
        )
        return issued

    def is_token_valid(self, token: str, *, leeway: timedelta | int | float = 0) -> bool:
        """Return ``True`` when *token* validates successfully."""

        try:
            self.decode_token(token, verify_expiration=True, leeway=leeway)
        except TokenValidationError:
            return False
        return True

    def clear_expired(self, *, now: datetime | None = None) -> int:
        """Remove cached metadata for expired tokens and return the count."""

        moment = now or datetime.now(tz=UTC)
        expired_tokens = [token for token, issued in self._issued_tokens.items() if issued.expires_at <= moment]
        for token in expired_tokens:
            self._issued_tokens.pop(token, None)
            self._revoked_tokens.discard(token)
        return len(expired_tokens)

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _encode(self, payload: Mapping[str, Any]) -> str:
        header = {"alg": "HS256", "typ": "JWT"}
        try:
            header_segment = _b64urlsafe_encode(
                json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
            )
            payload_segment = _b64urlsafe_encode(
                json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
            )
        except (TypeError, ValueError) as exc:  # pragma: no cover - surfaced to caller
            raise ValueError("Token payload contains non-serialisable data") from exc

        signing_input = f"{header_segment}.{payload_segment}".encode("ascii")
        signature = hmac.new(self._secret_key, signing_input, hashlib.sha256).digest()
        signature_segment = _b64urlsafe_encode(signature)
        return f"{header_segment}.{payload_segment}.{signature_segment}"

    def _decode(self, token: str) -> Mapping[str, Any]:
        try:
            header_segment, payload_segment, signature_segment = token.split(".")
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise TokenValidationError("Token is not in header.payload.signature format") from exc

        signing_input = f"{header_segment}.{payload_segment}".encode("ascii")
        expected_signature = hmac.new(self._secret_key, signing_input, hashlib.sha256).digest()
        provided_signature = _b64urlsafe_decode(signature_segment)

        if not hmac.compare_digest(expected_signature, provided_signature):
            raise TokenValidationError("Token signature mismatch")

        try:
            payload_json = _b64urlsafe_decode(payload_segment)
            payload = json.loads(payload_json)
        except (json.JSONDecodeError, ValueError) as exc:  # pragma: no cover - defensive guard
            raise TokenValidationError("Token payload is not valid JSON") from exc

        return payload
logger = logging.getLogger(__name__)


def _token_fingerprint(token: str) -> str:
    """Return a stable truncated fingerprint for logging without leaking tokens."""

    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return digest[:16]
