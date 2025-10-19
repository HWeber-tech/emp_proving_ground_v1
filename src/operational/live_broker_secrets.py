"""Helper utilities for managing live broker credentials across environments.

This module normalises sandbox/production credential mappings, surfaces
rotation metadata, and emits redacted summaries that downstream components can
log without leaking secrets.  It is intentionally isolated from broker
implementations so operational tooling can reason about credential hygiene
without importing heavy trading dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping


_MASK = "***"


def _normalise_key(key: object) -> str:
    return str(key).strip().upper().replace("-", "_")


def _normalise_mapping(mapping: Mapping[str, object] | None) -> dict[str, str]:
    normalised: dict[str, str] = {}
    if not mapping:
        return normalised
    for key, value in mapping.items():
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        normalised[_normalise_key(key)] = text
    return normalised


def _mask_identifier(value: str, keep: int = 4) -> str:
    text = str(value)
    if not text:
        return text
    if len(text) <= keep:
        return "*" * len(text)
    hidden = "*" * (len(text) - keep)
    return f"{hidden}{text[-keep:]}"


def _parse_timestamp(raw: str | None) -> datetime | None:
    if not raw:
        return None
    text = raw.strip()
    if not text:
        return None
    if text.isdigit():
        try:
            return datetime.fromtimestamp(int(text), tz=timezone.utc)
        except (OverflowError, ValueError):
            return None
    candidate = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _first_value(mapping: Mapping[str, str], keys: Iterable[str]) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if value:
            return value
    return None


def _has_any_payload(*values: object) -> bool:
    return any(value not in (None, "", (), [], {}) for value in values)


@dataclass(frozen=True)
class BrokerSessionCredentials:
    """Credentials for a single broker session (price or trade)."""

    sender_comp_id: str
    username: str
    password: str

    def is_complete(self) -> bool:
        return all((self.sender_comp_id, self.username, self.password))

    def as_dict(self) -> dict[str, str]:
        return {
            "sender_comp_id": self.sender_comp_id,
            "username": self.username,
            "password": self.password,
        }

    def masked(self) -> dict[str, str]:
        return {
            "sender_comp_id": _mask_identifier(self.sender_comp_id, keep=4),
            "username": _mask_identifier(self.username, keep=4),
            "password": _MASK if self.password else "",
        }


@dataclass(frozen=True)
class BrokerCredentialProfile:
    """Credential bundle for a specific broker environment profile."""

    name: str
    price: BrokerSessionCredentials | None = None
    trade: BrokerSessionCredentials | None = None
    rotated_at: datetime | None = None
    expires_at: datetime | None = None
    secret_reference: str | None = None
    source: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def is_complete(self) -> bool:
        return self.price is not None and self.trade is not None

    def age_days(self, now: datetime | None = None) -> float | None:
        if self.rotated_at is None:
            return None
        moment = now or datetime.now(timezone.utc)
        delta = moment - self.rotated_at
        return max(delta.total_seconds() / 86400.0, 0.0)

    def describe(self, *, masked: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "complete": self.is_complete(),
        }
        if self.price is not None:
            payload["price"] = self.price.masked() if masked else self.price.as_dict()
        if self.trade is not None:
            payload["trade"] = self.trade.masked() if masked else self.trade.as_dict()
        if self.rotated_at is not None:
            payload["rotated_at"] = self.rotated_at.isoformat()
        if self.expires_at is not None:
            payload["expires_at"] = self.expires_at.isoformat()
        if self.secret_reference:
            payload["secret_reference"] = self.secret_reference
        if self.source:
            payload["source"] = self.source
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


_ENVIRONMENT_ALIASES: Mapping[str, tuple[str, ...]] = {
    "SANDBOX": ("SANDBOX", "DEMO", "STAGING"),
    "PROD": ("PROD", "PRODUCTION", "LIVE"),
}

_SESSION_SUFFIXES: Mapping[str, tuple[str, str, str]] = {
    "PRICE": ("PRICE_SENDER_COMP_ID", "PRICE_USERNAME", "PRICE_PASSWORD"),
    "TRADE": ("TRADE_SENDER_COMP_ID", "TRADE_USERNAME", "TRADE_PASSWORD"),
}


def _classify_environment(environment: str | object | None) -> str:
    if environment is None:
        return "SANDBOX"
    text = str(environment).strip().lower()
    if text in {alias.lower() for alias in _ENVIRONMENT_ALIASES["PROD"]}:
        return "PROD"
    return "SANDBOX"


def _session_from_prefix(
    mapping: Mapping[str, str],
    prefix: str,
    session: str,
) -> BrokerSessionCredentials | None:
    suffixes = _SESSION_SUFFIXES[session]
    sender = mapping.get(f"{prefix}{suffixes[0]}")
    username = mapping.get(f"{prefix}{suffixes[1]}")
    password = mapping.get(f"{prefix}{suffixes[2]}")
    if sender and username and password:
        return BrokerSessionCredentials(sender, username, password)
    return None


def _extract_profile(
    mapping: Mapping[str, str],
    *,
    name: str,
    prefixes: Iterable[str],
    fallback_prefixes: Iterable[str] = (),
) -> BrokerCredentialProfile | None:
    for prefix in list(prefixes) + list(fallback_prefixes):
        price = _session_from_prefix(mapping, prefix, "PRICE")
        trade = _session_from_prefix(mapping, prefix, "TRADE")
        rotated_at = _parse_timestamp(mapping.get(f"{prefix}ROTATED_AT"))
        expires_at = _parse_timestamp(mapping.get(f"{prefix}EXPIRES_AT"))
        secret_ref = _first_value(
            mapping,
            (
                f"{prefix}SECRET_NAME",
                f"{prefix}SECRET_PATH",
                f"{prefix}SECRET_REF",
            ),
        )

        metadata_prefix = f"{prefix}METADATA_"
        metadata: dict[str, Any] = {
            key[len(metadata_prefix) :].lower(): value
            for key, value in mapping.items()
            if key.startswith(metadata_prefix)
        }

        if _has_any_payload(price, trade, rotated_at, expires_at, secret_ref, metadata):
            return BrokerCredentialProfile(
                name=name.lower(),
                price=price,
                trade=trade,
                rotated_at=rotated_at,
                expires_at=expires_at,
                secret_reference=secret_ref,
                source=prefix.rstrip("_"),
                metadata=metadata,
            )
    return None


@dataclass(frozen=True)
class LiveBrokerSecrets:
    """Resolved credential bundle across sandbox/production environments."""

    environment: str
    profiles: Mapping[str, BrokerCredentialProfile]
    active_key: str | None = None

    @property
    def active_profile(self) -> BrokerCredentialProfile | None:
        if self.active_key is None:
            return None
        return self.profiles.get(self.active_key)

    @property
    def healthy(self) -> bool:
        profile = self.active_profile
        return profile.is_complete() if profile else False

    def credential_age_days(self, now: datetime | None = None) -> float | None:
        profile = self.active_profile
        if profile is None:
            return None
        return profile.age_days(now)

    def describe(self, *, masked: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "environment": self.environment,
            "active": self.active_key.lower() if self.active_key else None,
            "healthy": self.healthy,
        }
        details: dict[str, Any] = {}
        for key, profile in self.profiles.items():
            details[key.lower()] = profile.describe(masked=masked)
        payload["profiles"] = details
        return payload


def load_live_broker_secrets(
    mapping: Mapping[str, object] | None,
    *,
    environment: str | object | None,
    fallback: Mapping[str, object] | None = None,
) -> LiveBrokerSecrets:
    """Return resolved live broker credentials for the requested environment."""

    normalised: dict[str, str] = {}
    normalised.update(_normalise_mapping(fallback))
    normalised.update(_normalise_mapping(mapping))

    active_key = _classify_environment(environment)

    profiles: dict[str, BrokerCredentialProfile] = {}
    for key, aliases in _ENVIRONMENT_ALIASES.items():
        prefixes = [f"LIVE_BROKER_{alias.upper()}_" for alias in (key, *aliases)]
        fallback_prefixes: list[str] = []
        if key == "SANDBOX":
            fallback_prefixes.extend([
                "FIX_",
                "BROKER_",
            ])
        elif key == "PROD":
            fallback_prefixes.extend([
                "FIX_PROD_",
                "FIX_LIVE_",
                "BROKER_PROD_",
            ])

        profile = _extract_profile(
            normalised,
            name=key,
            prefixes=prefixes,
            fallback_prefixes=fallback_prefixes,
        )
        if profile is None:
            profile = BrokerCredentialProfile(name=key.lower())
        profiles[key] = profile

    return LiveBrokerSecrets(
        environment=str(environment or "sandbox"),
        profiles=profiles,
        active_key=active_key,
    )


__all__ = [
    "BrokerSessionCredentials",
    "BrokerCredentialProfile",
    "LiveBrokerSecrets",
    "load_live_broker_secrets",
]
