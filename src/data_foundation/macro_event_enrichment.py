"""Helpers for enriching macro and fundamental event payloads.

The roadmap requires macro releases and fundamental catalysts to be timestamped
and causally linked to the assets they influence.  This module centralises the
lightweight heuristics used across ingest and persistence layers so both paths
classify events consistently.
"""

from __future__ import annotations

from typing import Mapping, Sequence

__all__ = ["enrich_macro_event_payload"]

_EVENT_CLASSIFIERS: tuple[tuple[str, tuple[str, tuple[str, ...]]], ...] = (
    ("CPI", ("macro", ("inflation",))),
    ("CONSUMER PRICE", ("macro", ("inflation",))),
    ("NFP", ("macro", ("employment",))),
    ("NONFARM", ("macro", ("employment",))),
    ("PAYROLL", ("macro", ("employment",))),
    ("FOMC", ("macro", ("monetary_policy",))),
    ("FEDERAL OPEN", ("macro", ("monetary_policy",))),
    ("ECB", ("macro", ("monetary_policy",))),
    ("BOE", ("macro", ("monetary_policy",))),
    ("BOJ", ("macro", ("monetary_policy",))),
    ("EARNING", ("fundamental", ("corporate_earnings",))),
    ("DIVIDEND", ("fundamental", ("capital_distribution",))),
    ("ETF", ("fundamental", ("portfolio_flows",))),
    ("REBAL", ("fundamental", ("portfolio_flows",))),
)

_SYMBOL_KEYS: tuple[str, ...] = (
    "related_symbols",
    "symbols",
    "tickers",
    "assets",
    "instrument",
    "instruments",
    "security",
    "securities",
    "symbol",
    "ticker",
)


def enrich_macro_event_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Return an enriched copy of an event payload with causal metadata."""

    record = dict(payload)

    event_name = str(record.get("event_name") or record.get("event") or "").strip()
    if event_name:
        record["event_name"] = event_name
        record.setdefault("event", event_name)

    calendar = str(record.get("calendar") or "").strip()
    category = str(record.get("category") or "").strip().lower() or None
    causal_links = _normalise_text_sequence(record.get("causal_links"))

    summary = f"{event_name} {calendar}".upper()
    if not category or not causal_links:
        inferred_category, inferred_links = _classify_event(summary)
        if not category and inferred_category:
            category = inferred_category
        if not causal_links and inferred_links:
            causal_links = list(inferred_links)

    record["category"] = category
    record["causal_links"] = causal_links

    related_symbols = _collect_related_symbols(record)
    record["related_symbols"] = related_symbols

    for key in _SYMBOL_KEYS:
        if key != "related_symbols":
            record.pop(key, None)

    return record


def _classify_event(summary_text: str) -> tuple[str | None, tuple[str, ...]]:
    for pattern, (category, links) in _EVENT_CLASSIFIERS:
        if pattern in summary_text:
            return category, links
    return None, tuple()


def _normalise_text_sequence(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        candidate = [text]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        candidate = [str(item).strip() for item in value]
    else:
        text = str(value).strip()
        candidate = [text] if text else []

    cleaned: list[str] = []
    for item in candidate:
        token = item.strip()
        if not token:
            continue
        cleaned.append(token)

    seen: dict[str, None] = {}
    for token in cleaned:
        if token not in seen:
            seen[token] = None
    return list(seen)


def _collect_related_symbols(record: Mapping[str, object]) -> list[str]:
    symbols: list[str] = []
    for key in _SYMBOL_KEYS:
        value = record.get(key)
        if value is None:
            continue
        symbols.extend(_normalise_symbol_values(value))

    seen: dict[str, None] = {}
    for symbol in symbols:
        upper = symbol.upper()
        if upper and upper not in seen:
            seen[upper] = None
    return list(seen)


def _normalise_symbol_values(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        return [text]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        tokens: list[str] = []
        for item in value:
            tokens.extend(_normalise_symbol_values(item))
        return tokens
    text = str(value).strip()
    return [text] if text else []
