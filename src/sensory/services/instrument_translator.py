"""Instrument translation service.

The translator bridges *external* identifiers (broker symbols, Bloomberg
tickers, etc.) to a universal representation used inside EMP.  The module is
intentionally defensive – the production system is expected to ingest
thousands of instruments from multiple venues with wildly inconsistent naming
conventions.  The implementation therefore focuses on:

* Deterministic translation from aliases to a canonical identifier.
* Bidirectional lookups so we can emit identifiers back to upstream venues.
* Zero-footgun ergonomics – conflicting aliases raise explicit errors and
  normalisation is well documented.

While the roadmap calls for translating “1000+ instruments”, the service is
designed to scale to that requirement by loading configuration from disk and
exposing runtime mutation hooks for dynamic sources.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from decimal import Decimal
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

from src.core.instrument import Instrument
from src.data_foundation.reference.reference_data_loader import (
    InstrumentDefinition,
    ReferenceDataLoader,
)

AliasMap = Mapping[str, Mapping[str, Sequence[str]]]


def _normalise_alias(text: str) -> str:
    """Return a normalised alias key used for conflict-free lookups."""

    stripped = "".join(ch for ch in text.upper() if ch.isalnum())
    return stripped


def _normalise_namespace(name: str) -> str:
    return name.strip().lower()


@dataclass(frozen=True, slots=True)
class UniversalInstrument:
    """Canonical representation of a tradeable instrument."""

    universal_symbol: str
    canonical_symbol: str
    asset_class: str | None
    venue: str | None
    margin_currency: str
    contract_size: Decimal
    pip_decimal_places: int
    long_swap_rate: Decimal | None = None
    short_swap_rate: Decimal | None = None
    swap_time: time | None = None

    def to_dict(self) -> dict[str, object]:
        """Convert the object to a serialisable mapping."""

        return {
            "universal_symbol": self.universal_symbol,
            "canonical_symbol": self.canonical_symbol,
            "asset_class": self.asset_class,
            "venue": self.venue,
            "margin_currency": self.margin_currency,
            "contract_size": str(self.contract_size),
            "pip_decimal_places": self.pip_decimal_places,
            "long_swap_rate": str(self.long_swap_rate)
            if self.long_swap_rate is not None
            else None,
            "short_swap_rate": str(self.short_swap_rate)
            if self.short_swap_rate is not None
            else None,
            "swap_time": self.swap_time.isoformat() if self.swap_time is not None else None,
        }


class AliasConflictError(RuntimeError):
    """Raised when the same alias is mapped to multiple instruments."""


class UnknownInstrumentError(KeyError):
    """Raised when an alias cannot be translated to a universal instrument."""


class InstrumentTranslator:
    """Translate between broker identifiers and the universal instrument model."""

    DEFAULT_ALIAS_FILE = (
        Path(__file__).resolve().parents[3] / "config" / "system" / "instrument_aliases.json"
    )

    def __init__(
        self,
        reference_loader: ReferenceDataLoader | None = None,
        *,
        alias_path: str | Path | None = None,
        alias_source: AliasMap | None = None,
    ) -> None:
        self._reference_loader = reference_loader or ReferenceDataLoader()
        self._definitions: Mapping[str, InstrumentDefinition] = (
            self._reference_loader.load_instruments()
        )
        self._universal: Dict[str, UniversalInstrument] = {
            symbol: self._build_universal(symbol, definition)
            for symbol, definition in self._definitions.items()
        }

        raw_aliases = alias_source or self._load_alias_config(alias_path)
        self._aliases: Dict[str, Dict[str, tuple[str, ...]]] = {
            symbol: {
                _normalise_namespace(namespace): tuple(str(alias) for alias in aliases)
                for namespace, aliases in mapping.items()
            }
            for symbol, mapping in raw_aliases.items()
        }

        # Ensure the canonical symbol itself is always an alias.
        for symbol in self._universal:
            canonical_aliases = self._aliases.setdefault(symbol, {})
            canonical = canonical_aliases.setdefault("canonical", tuple())
            if symbol not in canonical:
                canonical_aliases["canonical"] = canonical + (symbol,)

        self._alias_lookup: Dict[tuple[str | None, str], str] = {}
        self._reverse_lookup: Dict[tuple[str, str], str] = {}
        self._build_alias_indices()

    # ------------------------------------------------------------------
    def _build_universal(
        self, symbol: str, definition: InstrumentDefinition
    ) -> UniversalInstrument:
        asset = definition.asset_class.upper() if definition.asset_class else "UNKNOWN"
        venue = definition.venue.upper() if definition.venue else "OTC"
        universal_symbol = f"{asset}::{venue}::{symbol.upper()}"
        return UniversalInstrument(
            universal_symbol=universal_symbol,
            canonical_symbol=symbol,
            asset_class=definition.asset_class,
            venue=definition.venue,
            margin_currency=definition.margin_currency,
            contract_size=definition.contract_size,
            pip_decimal_places=definition.pip_decimal_places,
            long_swap_rate=definition.long_swap_rate,
            short_swap_rate=definition.short_swap_rate,
            swap_time=definition.swap_time,
        )

    def _load_alias_config(self, alias_path: str | Path | None) -> AliasMap:
        path = Path(alias_path) if alias_path is not None else self.DEFAULT_ALIAS_FILE
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise TypeError("Instrument alias config must be a mapping")

        normalised: Dict[str, Dict[str, Sequence[str]]] = {}
        for symbol, payload in data.items():
            if not isinstance(payload, Mapping):
                continue
            raw_aliases = payload.get("aliases")
            if not isinstance(raw_aliases, Mapping):
                continue
            namespace_map: Dict[str, Sequence[str]] = {}
            for namespace, aliases in raw_aliases.items():
                if isinstance(aliases, Sequence) and not isinstance(aliases, (str, bytes)):
                    namespace_map[str(namespace)] = [str(alias) for alias in aliases]
                elif aliases:
                    namespace_map[str(namespace)] = [str(aliases)]
            if namespace_map:
                normalised[str(symbol).upper()] = namespace_map
        return normalised

    def _build_alias_indices(self) -> None:
        for symbol, namespace_map in self._aliases.items():
            universal = self._universal.get(symbol)
            if universal is None:
                continue
            for namespace, aliases in namespace_map.items():
                normalised_namespace = _normalise_namespace(namespace)
                for alias in aliases:
                    key = (normalised_namespace, _normalise_alias(alias))
                    self._register_alias_key(key, symbol)
                    # Namespace-agnostic lookup fallback.
                    fallback_key = (None, _normalise_alias(alias))
                    self._register_alias_key(fallback_key, symbol)
                    self._reverse_lookup.setdefault((symbol, normalised_namespace), alias)

    def _register_alias_key(self, key: tuple[str | None, str], symbol: str) -> None:
        existing = self._alias_lookup.get(key)
        if existing is not None and existing != symbol:
            raise AliasConflictError(
                f"Alias {_describe_key(key)} already mapped to {existing}; cannot assign to {symbol}"
            )
        self._alias_lookup[key] = symbol

    # ------------------------------------------------------------------
    def translate(self, identifier: str, *, namespace: str | None = None) -> UniversalInstrument:
        """Translate an external identifier into a :class:`UniversalInstrument`."""

        symbol = self._resolve_symbol(identifier, namespace=namespace)
        universal = self._universal.get(symbol)
        if universal is None:
            raise UnknownInstrumentError(identifier)
        return universal

    def _resolve_symbol(self, identifier: str, *, namespace: str | None) -> str:
        normalised_identifier = _normalise_alias(identifier)
        if not normalised_identifier:
            raise UnknownInstrumentError(identifier)

        if namespace is not None:
            key = (_normalise_namespace(namespace), normalised_identifier)
            symbol = self._alias_lookup.get(key)
            if symbol is not None:
                return symbol

        key = (None, normalised_identifier)
        symbol = self._alias_lookup.get(key)
        if symbol is None:
            raise UnknownInstrumentError(identifier)
        return symbol

    def reverse_translate(self, universal_symbol: str, namespace: str) -> str | None:
        """Return the preferred alias for ``universal_symbol`` in ``namespace``."""

        normalised_namespace = _normalise_namespace(namespace)
        symbol = self._get_symbol_from_universal(universal_symbol)
        return self._reverse_lookup.get((symbol, normalised_namespace))

    def aliases_for(self, universal_symbol: str) -> Mapping[str, tuple[str, ...]]:
        """Return the alias mapping for ``universal_symbol``."""

        symbol = self._get_symbol_from_universal(universal_symbol)
        mapping = self._aliases.get(symbol, {})
        return {namespace: aliases for namespace, aliases in mapping.items()}

    def register_alias(
        self, universal_symbol: str, namespace: str, aliases: Iterable[str]
    ) -> None:
        """Register additional aliases at runtime."""

        symbol = self._get_symbol_from_universal(universal_symbol)
        namespace_key = _normalise_namespace(namespace)
        entries = self._aliases.setdefault(symbol, {})
        existing = list(entries.get(namespace_key, tuple()))
        for alias in aliases:
            if not alias:
                continue
            normalised = _normalise_alias(alias)
            key = (namespace_key, normalised)
            self._register_alias_key(key, symbol)
            fallback_key = (None, normalised)
            self._register_alias_key(fallback_key, symbol)
            if str(alias) not in existing:
                existing.append(str(alias))
        if existing:
            entries[namespace_key] = tuple(existing)
            self._reverse_lookup.setdefault((symbol, namespace_key), existing[0])

    def to_core_instrument(self, universal_symbol: str) -> Instrument:
        """Convert a universal instrument into the :class:`Instrument` model."""

        symbol = self._get_symbol_from_universal(universal_symbol)
        definition = self._definitions[symbol]
        instrument_type = self._infer_instrument_type(definition)
        pip_value = 10 ** (-definition.pip_decimal_places)
        tick_size = pip_value
        contract_size = float(definition.contract_size)

        base_currency = definition.margin_currency
        quote_currency = definition.margin_currency
        if definition.asset_class == "fx_spot" and len(definition.symbol) >= 6:
            base_currency = definition.symbol[:3]
            quote_currency = definition.symbol[3:6]

        return Instrument(
            symbol=definition.symbol,
            name=f"{definition.symbol} {instrument_type.title()}",
            instrument_type=instrument_type,
            base_currency=base_currency,
            quote_currency=quote_currency,
            pip_value=pip_value,
            contract_size=contract_size,
            min_lot_size=1.0,
            max_lot_size=contract_size,
            tick_size=tick_size,
        )

    def _infer_instrument_type(self, definition: InstrumentDefinition) -> str:
        mapping = {
            "fx_spot": "forex",
            "fx_fut": "futures",
            "equity": "equity",
        }
        return mapping.get(definition.asset_class or "", "unclassified")

    def _get_symbol_from_universal(self, universal_symbol: str) -> str:
        if universal_symbol in self._universal:
            return universal_symbol
        for symbol, universal in self._universal.items():
            if universal.universal_symbol == universal_symbol:
                return symbol
        raise UnknownInstrumentError(universal_symbol)


def _describe_key(key: tuple[str | None, str]) -> str:
    namespace, alias = key
    if namespace is None:
        return alias
    return f"{namespace}:{alias}"


__all__ = ["InstrumentTranslator", "UniversalInstrument", "AliasConflictError", "UnknownInstrumentError"]
