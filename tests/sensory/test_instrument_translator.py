from __future__ import annotations

import pytest

from src.sensory.services.instrument_translator import (
    AliasConflictError,
    InstrumentTranslator,
    UnknownInstrumentError,
)


@pytest.fixture()
def translator() -> InstrumentTranslator:
    alias_map = {
        "EURUSD": {
            "ctrader": ["EURUSD", "1"],
            "bloomberg": ["EURUSD CURNCY"],
        },
        "6E_DEC24": {
            "cme": ["6E DEC24", "6EZ4"],
        },
    }
    return InstrumentTranslator(alias_source=alias_map)


def test_translate_known_alias_returns_universal(translator: InstrumentTranslator) -> None:
    instrument = translator.translate("EURUSD", namespace="ctrader")
    assert instrument.canonical_symbol == "EURUSD"
    assert instrument.universal_symbol.startswith("FX_SPOT::")


def test_translate_unknown_alias_raises(translator: InstrumentTranslator) -> None:
    with pytest.raises(UnknownInstrumentError):
        translator.translate("UNKNOWN")


def test_reverse_translation_uses_first_alias(translator: InstrumentTranslator) -> None:
    instrument = translator.translate("EURUSD")
    alias = translator.reverse_translate(instrument.universal_symbol, "ctrader")
    assert alias == "EURUSD"


def test_register_alias_allows_dynamic_extension(translator: InstrumentTranslator) -> None:
    instrument = translator.translate("6E DEC24", namespace="cme")
    translator.register_alias(instrument.universal_symbol, "bloomberg", ["6EZ4 COMB"])
    translated = translator.translate("6EZ4 COMB", namespace="bloomberg")
    assert translated.universal_symbol == instrument.universal_symbol


def test_register_alias_detects_conflicts(translator: InstrumentTranslator) -> None:
    instrument = translator.translate("EURUSD")
    translator.register_alias(instrument.universal_symbol, "alt", ["FX1"])
    with pytest.raises(AliasConflictError):
        other = translator.translate("6E DEC24", namespace="cme")
        translator.register_alias(other.universal_symbol, "alt", ["FX1"])


def test_to_core_instrument_uses_reference_definition(translator: InstrumentTranslator) -> None:
    instrument = translator.translate("EURUSD")
    core = translator.to_core_instrument(instrument.universal_symbol)
    assert core.symbol == "EURUSD"
    assert core.instrument_type == "forex"
    assert core.base_currency == "EUR"
    assert core.quote_currency == "USD"
