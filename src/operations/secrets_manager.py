"""Lightweight secret manager integration helpers.

This module provides an optional integration layer for resolving secret
payloads from external secret managers such as AWS Secrets Manager and HashiCorp
Vault.  The helpers are designed so that deployments can opt-in to managed
secrets without introducing hard dependencies on the provider SDKs; if the
packages are unavailable at runtime the resolution gracefully returns ``None``.

The entrypoint is :func:`resolve_secret_reference`, which accepts an opaque
reference string (for example ``aws://prod/emp/trading`` or
``vault://kv/data/emp/trading?field=broker``) and returns a normalised mapping
of upper-case keys to string values.  Nested structures in the underlying
secret payload are flattened using ``_`` separators so downstream consumers can
reuse the existing environment variable parsing logic.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Mapping, MutableMapping, Sequence
from urllib.parse import parse_qs, urlparse


logger = logging.getLogger(__name__)


class SecretManagerError(RuntimeError):
    """Raised when a secret manager integration encounters a non-recoverable error."""


ProviderOptions = MutableMapping[str, str]


def resolve_secret_reference(
    reference: str,
    *,
    provider_hint: str | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, str] | None:
    """Resolve a secret reference against supported secret managers.

    Parameters
    ----------
    reference:
        Opaque reference describing the secret location.  Supports schemes such
        as ``aws://`` and ``vault://`` as well as raw identifiers (for example an
        AWS ARN).  When the scheme is omitted a provider hint or environment
        configuration must indicate the backend.
    provider_hint:
        Optional override for the provider (``"aws"`` or ``"vault"``).  When
        omitted the function falls back to ``EMP_SECRETS_MANAGER_PROVIDER`` or
        ``SECRETS_MANAGER_PROVIDER`` in the supplied environment mapping.  The
        value is case-insensitive and may include common aliases (e.g.
        ``aws-secretsmanager``).
    env:
        Environment mapping used to configure provider specific options such as
        regions, endpoints, or Vault connection parameters.  Defaults to
        ``os.environ``.

    Returns
    -------
    dict[str, str] | None
        A mapping of upper-case keys to string values when the secret is
        resolved successfully, otherwise ``None`` when the reference cannot be
        processed.  Consumers should treat a ``None`` result as "secret not
        available" and continue with fallback configuration.
    """

    reference = reference.strip()
    if not reference:
        return None

    environment = env or os.environ
    provider_hint = _normalise_provider(provider_hint or environment.get("EMP_SECRETS_MANAGER_PROVIDER") or environment.get("SECRETS_MANAGER_PROVIDER"))

    provider, identifier, options = _parse_reference(reference, provider_hint)
    if provider is None or not identifier:
        logger.debug("Secret reference %r ignored: provider could not be determined", reference)
        return None

    if provider == "aws":
        region = options.get("region")
        if not region:
            region = environment.get("EMP_SECRETS_MANAGER_REGION")
        if not region:
            region = environment.get("SECRETS_MANAGER_REGION")
        if not region:
            region = environment.get("AWS_REGION") or environment.get("AWS_DEFAULT_REGION")

        endpoint_url = options.get("endpoint_url") or environment.get("EMP_SECRETS_MANAGER_ENDPOINT_URL")
        profile_name = options.get("profile") or environment.get("AWS_PROFILE")

        try:
            payload = _resolve_aws_secret(
                identifier,
                region_name=region,
                endpoint_url=endpoint_url,
                profile_name=profile_name,
            )
        except SecretManagerError:
            raise
        except Exception:  # pragma: no cover - defensive diagnostics
            logger.exception("Failed to resolve AWS secret %s", identifier)
            return None
        return payload

    if provider == "vault":
        mount_point = options.get("mount") or options.get("mount_point")
        if not mount_point:
            mount_point = environment.get("EMP_SECRETS_MANAGER_VAULT_MOUNT") or "secret"
        namespace = options.get("namespace") or environment.get("VAULT_NAMESPACE")
        version = options.get("version")
        field = options.get("field") or options.get("fragment")
        addr = environment.get("VAULT_ADDR")
        token = environment.get("VAULT_TOKEN")
        if not addr or not token:
            logger.warning(
                "Vault address/token not configured; unable to resolve %s", identifier
            )
            return None

        verify: bool | str = True
        cacert = environment.get("VAULT_CACERT")
        if cacert:
            verify = cacert
        skip_verify = environment.get("VAULT_SKIP_VERIFY")
        if skip_verify and skip_verify.strip().lower() in {"1", "true", "yes"}:
            verify = False

        try:
            payload = _resolve_vault_secret(
                identifier,
                mount_point=mount_point,
                token=token,
                url=addr,
                namespace=namespace,
                version=version,
                field=field,
                verify=verify,
            )
        except SecretManagerError:
            raise
        except Exception:  # pragma: no cover - defensive diagnostics
            logger.exception("Failed to resolve Vault secret %s", identifier)
            return None
        return payload

    logger.debug("Secret reference %r ignored: unsupported provider %r", reference, provider)
    return None


def _parse_reference(
    reference: str,
    provider_hint: str | None,
) -> tuple[str | None, str, ProviderOptions]:
    parsed = urlparse(reference)
    options: ProviderOptions = {}

    if parsed.scheme:
        provider = _normalise_provider(parsed.scheme)
        identifier = _combine_reference_parts(parsed.netloc, parsed.path)
        if parsed.query:
            for key, values in parse_qs(parsed.query, keep_blank_values=False).items():
                if not values:
                    continue
                options[key.strip().lower()] = values[-1]
        if parsed.fragment:
            options.setdefault("fragment", parsed.fragment)
        return provider, identifier, options

    lowered = reference.lower()
    if lowered.startswith("arn:aws:secretsmanager:"):
        return "aws", reference, options
    if lowered.startswith("vault:"):
        provider = "vault"
        identifier = reference.split(":", 1)[1]
        return provider, identifier, options

    return _normalise_provider(provider_hint), reference, options


def _combine_reference_parts(netloc: str, path: str) -> str:
    if not netloc:
        return path.lstrip("/")
    if not path or path == "/":
        return netloc
    combined = f"{netloc}{path}"
    return combined.lstrip("/")


def _resolve_aws_secret(
    secret_id: str,
    *,
    region_name: str | None,
    endpoint_url: str | None,
    profile_name: str | None,
) -> dict[str, str] | None:
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - optional dependency
        logger.warning("boto3 is required to resolve AWS secrets: %s", exc)
        return None

    session_kwargs: dict[str, str] = {}
    if profile_name:
        session_kwargs["profile_name"] = profile_name
    if region_name:
        session_kwargs["region_name"] = region_name

    try:
        session = boto3.session.Session(**session_kwargs)
    except Exception as exc:  # pragma: no cover - boto3 diagnostics
        raise SecretManagerError(f"Failed to initialise boto3 session: {exc!s}") from exc

    client_kwargs: dict[str, str] = {}
    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url
    try:
        client = session.client("secretsmanager", **client_kwargs)
    except Exception as exc:  # pragma: no cover - boto3 diagnostics
        raise SecretManagerError(f"Failed to create secretsmanager client: {exc!s}") from exc

    try:
        response = client.get_secret_value(SecretId=secret_id)
    except Exception as exc:  # pragma: no cover - boto3 diagnostics
        raise SecretManagerError(f"Failed to fetch secret {secret_id!r}: {exc!s}") from exc

    if not isinstance(response, Mapping):  # pragma: no cover - defensive guard
        return None

    secret_string = response.get("SecretString")
    if isinstance(secret_string, str):
        return _normalise_secret_payload(secret_string)

    secret_binary = response.get("SecretBinary")
    if secret_binary is not None:
        try:
            if isinstance(secret_binary, str):
                decoded = base64.b64decode(secret_binary)
            else:
                decoded = base64.b64decode(bytes(secret_binary))
            text = decoded.decode("utf-8")
        except Exception as exc:  # pragma: no cover - guard for unexpected payload
            raise SecretManagerError(f"Failed to decode binary secret payload: {exc!s}") from exc
        return _normalise_secret_payload(text)

    return None


def _resolve_vault_secret(
    identifier: str,
    *,
    mount_point: str,
    token: str,
    url: str,
    namespace: str | None,
    version: str | None,
    field: str | None,
    verify: bool | str,
) -> dict[str, str] | None:
    try:
        import hvac
    except ImportError as exc:  # pragma: no cover - optional dependency
        logger.warning("hvac is required to resolve Vault secrets: %s", exc)
        return None

    try:
        client = hvac.Client(url=url, token=token, namespace=namespace, verify=verify)
    except Exception as exc:  # pragma: no cover - hvac diagnostics
        raise SecretManagerError(f"Failed to initialise Vault client: {exc!s}") from exc

    secret_kwargs: dict[str, object] = {
        "path": identifier.lstrip("/"),
        "mount_point": mount_point,
    }
    if version:
        try:
            secret_kwargs["version"] = int(version)
        except ValueError:  # pragma: no cover - invalid query parameter
            logger.debug("Ignoring non-integer Vault version %r", version)

    try:
        response = client.secrets.kv.v2.read_secret_version(**secret_kwargs)
    except AttributeError as exc:  # pragma: no cover - unsupported KV engine
        raise SecretManagerError(
            "Vault KV v2 support is required for managed secrets"
        ) from exc
    except Exception as exc:  # pragma: no cover - hvac diagnostics
        raise SecretManagerError(f"Failed to fetch Vault secret {identifier!r}: {exc!s}") from exc

    data = response.get("data") if isinstance(response, Mapping) else None
    if isinstance(data, Mapping):
        payload = data.get("data")
        if not isinstance(payload, Mapping):
            payload = data
    else:
        payload = None

    if not isinstance(payload, Mapping):
        return None

    normalised = _normalise_secret_payload(payload)
    if field:
        field_key = _normalise_key(field)
        filtered = {
            key: value
            for key, value in normalised.items()
            if key == field_key or key.startswith(f"{field_key}_")
        }
        if not filtered:
            raw_value = payload.get(field)
            if raw_value not in (None, ""):
                filtered[field_key] = str(raw_value)
        normalised = filtered

    return normalised or None


def _normalise_secret_payload(payload: Mapping[str, object] | str | bytes) -> dict[str, str]:
    if isinstance(payload, Mapping):
        flattened: dict[str, str] = {}
        for key, value in payload.items():
            _flatten_secret_payload((_normalise_key(key),), value, flattened)
        return flattened

    if isinstance(payload, bytes):
        try:
            payload = payload.decode("utf-8")
        except UnicodeDecodeError:
            logger.debug("Secret payload is not valid UTF-8; skipping")
            return {}

    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            mapping = _parse_key_value_payload(text)
            if mapping:
                return mapping
            return {"VALUE": text}
        if isinstance(parsed, Mapping):
            return _normalise_secret_payload(parsed)
        if isinstance(parsed, Sequence):
            flattened: dict[str, str] = {}
            for index, item in enumerate(parsed):
                _flatten_secret_payload((f"ITEM_{index}",), item, flattened)
            return flattened
        return {"VALUE": text}

    return {}


def _flatten_secret_payload(prefix: Sequence[str], value: object, output: MutableMapping[str, str]) -> None:
    if value is None:
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            new_prefix = tuple(part for part in prefix if part)
            new_prefix = new_prefix + (_normalise_key(key),)
            _flatten_secret_payload(new_prefix, item, output)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            new_prefix = tuple(part for part in prefix if part)
            new_prefix = new_prefix + (f"ITEM_{index}",)
            _flatten_secret_payload(new_prefix, item, output)
        return

    text = str(value).strip()
    if not text:
        return
    key = "_".join(part for part in prefix if part)
    output[key] = text


def _parse_key_value_payload(text: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        normalised_key = _normalise_key(key)
        normalised_value = value.strip()
        if normalised_value:
            mapping[normalised_key] = normalised_value
    return mapping


def _normalise_key(key: str | None) -> str:
    if key is None:
        return ""
    return str(key).strip().upper().replace("-", "_")


def _normalise_provider(provider: str | None) -> str | None:
    if provider is None:
        return None
    text = provider.strip().lower()
    if not text:
        return None
    aliases = {
        "aws": "aws",
        "aws-sm": "aws",
        "awssecretsmanager": "aws",
        "aws-secretsmanager": "aws",
        "aws+secretsmanager": "aws",
        "awssecrets": "aws",
        "vault": "vault",
        "vault-kv": "vault",
        "vault+kv": "vault",
    }
    return aliases.get(text, text)


__all__ = ["resolve_secret_reference", "SecretManagerError"]
