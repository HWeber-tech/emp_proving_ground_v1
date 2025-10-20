"""Helper utilities for managing live broker credentials across environments.

This module normalises sandbox/production credential mappings, surfaces
rotation metadata, and emits redacted summaries that downstream components can
log without leaking secrets.  It is intentionally isolated from broker
implementations so operational tooling can reason about credential hygiene
without importing heavy trading dependencies.
"""

from __future__ import annotations

import base64
import json
import logging
from importlib import import_module
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, MutableMapping


from src.operations.secrets_manager import resolve_secret_reference


logger = logging.getLogger(__name__)

_MASK = "***"
logger = logging.getLogger(__name__)


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


def _normalise_secret_payload(payload: object) -> dict[str, str]:
    if isinstance(payload, Mapping):
        return {
            str(key): str(value)
            for key, value in payload.items()
            if value not in (None, "")
        }
    if isinstance(payload, bytes):
        return _normalise_secret_payload(payload.decode("utf-8", errors="ignore"))
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            env_pairs: dict[str, str] = {}
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                key, _, value = stripped.partition("=")
                key = key.strip()
                if key:
                    env_pairs[key] = value.strip()
            if env_pairs:
                return env_pairs
            return {"value": text}
        return _normalise_secret_payload(parsed)
    return {}


def _select_secret_payload(
    payload: Mapping[str, object] | None,
    *,
    selector: str | None,
    environment: str | None,
) -> Mapping[str, object] | None:
    current: object | None = payload
    if current is None:
        return None
    if selector:
        parts = [part for part in selector.replace("/", ".").split(".") if part]
        for part in parts:
            if isinstance(current, Mapping):
                current = current.get(part)
            else:
                current = None
                break
    if environment and isinstance(current, Mapping):
        env_key = environment.strip()
        if env_key:
            value = current.get(env_key)
            if isinstance(value, Mapping):
                current = value
    return current if isinstance(current, Mapping) else None


def _canonical_provider_name(provider: str) -> str:
    normalised = provider.strip().lower().replace("-", "_")
    if normalised in {
        "aws",
        "aws_secrets_manager",
        "awssecretsmanager",
        "aws_secretmanager",
        "secretsmanager",
        "secretmanager",
    }:
        return "aws"
    if normalised in {
        "vault",
        "hashicorp_vault",
        "hashicorpvault",
    }:
        return "vault"
    return normalised


def _load_from_aws_secrets_manager(mapping: Mapping[str, str]) -> dict[str, str]:
    secret_id = (
        mapping.get("LIVE_BROKER_SECRET_ARN")
        or mapping.get("LIVE_BROKER_SECRET_ID")
        or mapping.get("LIVE_BROKER_SECRET_NAME")
    )
    if not secret_id:
        return {}

    try:
        boto3 = import_module("boto3")
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        logger.debug("boto3 is not available; skipping AWS secrets lookup")
        return {}

    session_kwargs: MutableMapping[str, object] = {}
    profile_name = mapping.get("LIVE_BROKER_AWS_PROFILE") or mapping.get("AWS_PROFILE")
    if profile_name:
        session_kwargs["profile_name"] = profile_name

    region_name = (
        mapping.get("LIVE_BROKER_SECRET_REGION")
        or mapping.get("AWS_REGION")
        or mapping.get("AWS_DEFAULT_REGION")
    )

    client = None
    session = getattr(boto3, "session", None)
    if session is not None and hasattr(session, "Session"):
        try:
            client = session.Session(**session_kwargs).client(
                "secretsmanager", region_name=region_name
            )
        except Exception:  # pragma: no cover - defensive
            logger.debug("Failed to establish boto3 session", exc_info=True)
            client = None
    if client is None:
        client_factory = getattr(boto3, "client", None)
        if callable(client_factory):
            client = client_factory("secretsmanager", region_name=region_name)
    if client is None:
        return {}

    try:
        response = client.get_secret_value(SecretId=secret_id)
    except Exception:  # pragma: no cover - runtime dependency failure
        logger.debug("Failed to retrieve secret %s from AWS Secrets Manager", secret_id, exc_info=True)
        return {}

    secret_string = response.get("SecretString")
    if secret_string is None:
        secret_binary = response.get("SecretBinary")
        if secret_binary is None:
            return {}
        if isinstance(secret_binary, str):
            secret_bytes = base64.b64decode(secret_binary.encode("utf-8"))
        else:
            secret_bytes = base64.b64decode(secret_binary)
        secret_string = secret_bytes.decode("utf-8", errors="ignore")

    parsed_payload: object | None
    try:
        parsed_payload = json.loads(secret_string)
    except json.JSONDecodeError:
        parsed_payload = None

    payload = _normalise_secret_payload(parsed_payload if parsed_payload is not None else secret_string)

    selector = mapping.get("LIVE_BROKER_SECRET_FIELD") or mapping.get("LIVE_BROKER_SECRET_JSON_PATH")
    environment = mapping.get("LIVE_BROKER_SECRET_ENVIRONMENT")
    nested = _select_secret_payload(
        parsed_payload if isinstance(parsed_payload, Mapping) else None,
        selector=selector,
        environment=environment,
    )
    if nested:
        payload = _normalise_secret_payload(nested)

    if secret_id and "LIVE_BROKER_SECRET_REFERENCE" not in payload:
        payload["LIVE_BROKER_SECRET_REFERENCE"] = secret_id
    return payload


def _load_from_vault(mapping: Mapping[str, str]) -> dict[str, str]:
    path = mapping.get("LIVE_BROKER_VAULT_PATH") or mapping.get("LIVE_BROKER_SECRET_PATH")
    if not path:
        return {}

    try:
        hvac = import_module("hvac")
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        logger.debug("hvac is not available; skipping Vault secrets lookup")
        return {}

    client_kwargs: MutableMapping[str, object] = {}
    url = mapping.get("LIVE_BROKER_VAULT_ADDR") or mapping.get("VAULT_ADDR")
    if url:
        client_kwargs["url"] = url
    token = mapping.get("LIVE_BROKER_VAULT_TOKEN") or mapping.get("VAULT_TOKEN")
    if token:
        client_kwargs["token"] = token

    try:
        client = hvac.Client(**client_kwargs)
    except Exception:  # pragma: no cover - runtime dependency failure
        logger.debug("Failed to initialise Vault client", exc_info=True)
        return {}

    mount_point = mapping.get("LIVE_BROKER_VAULT_MOUNT") or "secret"
    version_raw = mapping.get("LIVE_BROKER_VAULT_VERSION")
    kwargs: MutableMapping[str, object] = {"path": path, "mount_point": mount_point}
    if version_raw is not None:
        try:
            kwargs["version"] = int(str(version_raw).strip())
        except ValueError:
            logger.debug("Invalid Vault secret version %r", version_raw)

    try:
        response = client.secrets.kv.v2.read_secret_version(**kwargs)
    except Exception:  # pragma: no cover - runtime dependency failure
        logger.debug("Failed to read secret %s from Vault", path, exc_info=True)
        return {}

    data = response.get("data") if isinstance(response, Mapping) else None
    if isinstance(data, Mapping):
        if isinstance(data.get("data"), Mapping):
            payload_raw = data["data"]
        else:
            payload_raw = data
    else:
        payload_raw = {}

    selector = mapping.get("LIVE_BROKER_SECRET_FIELD") or mapping.get("LIVE_BROKER_SECRET_JSON_PATH")
    environment = mapping.get("LIVE_BROKER_SECRET_ENVIRONMENT")
    nested = _select_secret_payload(payload_raw, selector=selector, environment=environment)
    selected_payload = payload_raw if nested is None else nested

    payload = _normalise_secret_payload(selected_payload)
    if path and "LIVE_BROKER_SECRET_REFERENCE" not in payload:
        payload["LIVE_BROKER_SECRET_REFERENCE"] = path
    return payload


def _load_secret_manager_payload(mapping: Mapping[str, str]) -> dict[str, str]:
    provider_raw = _first_value(
        mapping,
        (
            "LIVE_BROKER_SECRET_MANAGER",
            "LIVE_BROKER_SECRET_MANAGER_PROVIDER",
            "LIVE_BROKER_SECRET_PROVIDER",
        ),
    )
    if not provider_raw:
        return {}

    provider = _canonical_provider_name(provider_raw)
    loader = {
        "aws": _load_from_aws_secrets_manager,
        "vault": _load_from_vault,
    }.get(provider)
    if loader is None:
        logger.debug("Unsupported secrets manager provider %s", provider_raw)
        return {}

    try:
        payload = loader(mapping)
    except Exception:  # pragma: no cover - defensive guard around optional integrations
        logger.debug("Secrets manager lookup failed for provider %s", provider, exc_info=True)
        return {}
    return payload


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
    secret_loader: Callable[[str], Mapping[str, str] | None] | None = None,
) -> BrokerCredentialProfile | None:
    for prefix in list(prefixes) + list(fallback_prefixes):
        local_mapping: MutableMapping[str, str] = dict(mapping)
        secret_ref = _first_value(
            local_mapping,
            (
                f"{prefix}SECRET_NAME",
                f"{prefix}SECRET_PATH",
                f"{prefix}SECRET_REF",
            ),
        )

        if secret_loader is not None and secret_ref:
            try:
                secret_payload = secret_loader(secret_ref)
            except Exception:  # pragma: no cover - defensive guard to keep optional
                logger.debug(
                    "Secret loader failed for reference %s", secret_ref, exc_info=True
                )
                secret_payload = None
            if secret_payload:
                for key, value in secret_payload.items():
                    if not value:
                        continue
                    if key.startswith("LIVE_BROKER_"):
                        local_mapping.setdefault(key, value)
                    else:
                        local_mapping.setdefault(f"{prefix}{key}", value)

        price = _session_from_prefix(local_mapping, prefix, "PRICE")
        trade = _session_from_prefix(local_mapping, prefix, "TRADE")
        rotated_at = _parse_timestamp(local_mapping.get(f"{prefix}ROTATED_AT"))
        expires_at = _parse_timestamp(local_mapping.get(f"{prefix}EXPIRES_AT"))

        metadata_prefix = f"{prefix}METADATA_"
        metadata: dict[str, Any] = {
            key[len(metadata_prefix) :].lower(): value
            for key, value in local_mapping.items()
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
    secret_loader: Callable[[str], Mapping[str, str] | None] | None = None,
) -> LiveBrokerSecrets:
    """Return resolved live broker credentials for the requested environment."""

    fallback_mapping = _normalise_mapping(fallback)
    explicit_mapping = _normalise_mapping(mapping)

    lookup_mapping: dict[str, str] = dict(fallback_mapping)
    lookup_mapping.update(explicit_mapping)

    secret_manager_payload = _load_secret_manager_payload(lookup_mapping)

    normalised: dict[str, str] = {}
    normalised.update(fallback_mapping)
    if secret_manager_payload:
        normalised.update(_normalise_mapping(secret_manager_payload))
    normalised.update(explicit_mapping)

    active_key = _classify_environment(environment)

    profiles: dict[str, BrokerCredentialProfile] = {}

    default_loader = secret_loader
    if default_loader is None:

        def _default_loader(reference: str) -> Mapping[str, str] | None:
            try:
                return resolve_secret_reference(reference)
            except Exception:  # pragma: no cover - diagnostics only
                logger.debug(
                    "Failed to resolve secret reference %s", reference, exc_info=True
                )
                return None

        default_loader = _default_loader

    secret_cache: dict[str, Mapping[str, str] | None] = {}

    def _load_secret(reference: str) -> Mapping[str, str] | None:
        if reference in secret_cache:
            return secret_cache[reference]
        payload = default_loader(reference) if default_loader is not None else None
        if payload is not None:
            normalised_payload = _normalise_mapping(payload)
        else:
            normalised_payload = None
        secret_cache[reference] = normalised_payload
        return normalised_payload

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
            secret_loader=_load_secret,
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
