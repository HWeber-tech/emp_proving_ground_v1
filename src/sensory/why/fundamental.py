from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

__all__ = [
    "FundamentalSnapshot",
    "FundamentalMetrics",
    "normalise_fundamental_snapshot",
    "compute_fundamental_metrics",
    "score_fundamentals",
]


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
    elif isinstance(value, str):
        try:
            numeric = float(value.strip())
        except ValueError:
            return None
    else:
        try:
            numeric = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
    if not isfinite(numeric):
        return None
    return numeric


def _as_mapping(payload: object) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        return payload
    to_dict = getattr(payload, "to_dict", None)
    if callable(to_dict):
        try:
            candidate = to_dict()
        except Exception:  # pragma: no cover - defensive
            candidate = None
        if isinstance(candidate, Mapping):
            return candidate
    return {}


_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "price": ("price", "close", "last_price"),
    "eps": ("eps", "earnings_per_share", "eps_ttm", "diluted_eps"),
    "book_value_per_share": (
        "book_value_per_share",
        "book_value_ps",
        "bvps",
        "book_value",
    ),
    "dividend_yield": ("dividend_yield", "forward_dividend_yield", "div_yield"),
    "free_cash_flow": ("free_cash_flow", "fcf", "free_cash_flow_ttm"),
    "free_cash_flow_per_share": (
        "free_cash_flow_per_share",
        "fcf_per_share",
        "free_cash_flow_ps",
    ),
    "shares_outstanding": (
        "shares_outstanding",
        "shares_out",
        "shares_basic",
        "shares_float",
    ),
    "growth_rate": (
        "growth_rate",
        "long_term_growth",
        "revenue_growth",
        "earnings_growth",
    ),
    "discount_rate": ("discount_rate", "wacc", "required_return"),
    "quality_score": ("quality_score", "fundamental_quality", "composite_quality"),
    "net_income": ("net_income", "net_income_ttm"),
    "revenue": ("revenue", "revenue_ttm", "sales"),
    "revenue_growth": ("revenue_growth", "sales_growth"),
}


@dataclass(slots=True)
class FundamentalSnapshot:
    price: float | None = None
    eps: float | None = None
    book_value_per_share: float | None = None
    dividend_yield: float | None = None
    free_cash_flow: float | None = None
    free_cash_flow_per_share: float | None = None
    shares_outstanding: float | None = None
    growth_rate: float | None = None
    discount_rate: float | None = None
    quality_score: float | None = None
    net_income: float | None = None
    revenue: float | None = None
    revenue_growth: float | None = None

    def as_dict(self) -> dict[str, float]:
        payload: dict[str, float] = {}
        for field_name in self.__slots__:
            value = getattr(self, field_name)
            if value is None:
                continue
            payload[field_name] = float(value)
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: object,
        *,
        fallback_price: float | None = None,
    ) -> FundamentalSnapshot | None:
        mapping = _as_mapping(payload)
        if not mapping and fallback_price is None:
            return None

        data: dict[str, float | None] = {}
        for field, aliases in _FIELD_ALIASES.items():
            for alias in aliases:
                if alias in mapping:
                    numeric = _coerce_float(mapping[alias])
                    if numeric is not None:
                        data[field] = numeric
                        break

        if "price" not in data and fallback_price is not None:
            data["price"] = fallback_price

        if not any(value is not None for value in data.values()):
            return None

        return cls(**data)


@dataclass(slots=True)
class FundamentalMetrics:
    pe_ratio: float | None = None
    earnings_yield: float | None = None
    pb_ratio: float | None = None
    dividend_yield: float | None = None
    fcf_yield: float | None = None
    intrinsic_value: float | None = None
    valuation_gap: float | None = None
    growth_rate: float | None = None
    discount_rate: float | None = None
    quality_score: float | None = None
    fcf_per_share: float | None = None

    def as_dict(self) -> dict[str, float]:
        payload: dict[str, float] = {}
        for field_name in self.__slots__:
            value = getattr(self, field_name)
            if value is None:
                continue
            payload[field_name] = float(value)
        return payload


def normalise_fundamental_snapshot(
    payload: object,
    *,
    fallback_price: float | None = None,
) -> FundamentalSnapshot | None:
    if isinstance(payload, FundamentalSnapshot):
        return payload
    return FundamentalSnapshot.from_payload(payload, fallback_price=fallback_price)


def compute_fundamental_metrics(
    snapshot: FundamentalSnapshot,
) -> FundamentalMetrics:
    price = snapshot.price or 0.0
    eps = snapshot.eps or 0.0
    book_value = snapshot.book_value_per_share or 0.0
    fcf = snapshot.free_cash_flow
    fcf_per_share = snapshot.free_cash_flow_per_share
    shares = snapshot.shares_outstanding

    pe_ratio: float | None = None
    earnings_yield: float | None = None
    if price > 0 and eps != 0:
        pe_ratio = price / eps if eps > 0 else None
        earnings_yield = eps / price

    pb_ratio: float | None = None
    if price > 0 and book_value > 0:
        pb_ratio = price / book_value

    if fcf_per_share is None and fcf is not None and shares:
        if shares > 0:
            fcf_per_share = fcf / shares
    if fcf_per_share is not None and price > 0:
        fcf_yield: float | None = fcf_per_share / price
    else:
        fcf_yield = None

    intrinsic_value: float | None = None
    valuation_gap: float | None = None
    if (
        fcf_per_share is not None
        and snapshot.discount_rate is not None
        and snapshot.growth_rate is not None
        and snapshot.discount_rate > snapshot.growth_rate
    ):
        intrinsic_value = fcf_per_share * (1.0 + snapshot.growth_rate) / (
            snapshot.discount_rate - snapshot.growth_rate
        )
        if price > 0:
            valuation_gap = (intrinsic_value - price) / price

    dividend_yield = snapshot.dividend_yield

    return FundamentalMetrics(
        pe_ratio=pe_ratio,
        earnings_yield=earnings_yield,
        pb_ratio=pb_ratio,
        dividend_yield=dividend_yield,
        fcf_yield=fcf_yield,
        intrinsic_value=intrinsic_value,
        valuation_gap=valuation_gap,
        growth_rate=snapshot.growth_rate,
        discount_rate=snapshot.discount_rate,
        quality_score=snapshot.quality_score,
        fcf_per_share=fcf_per_share,
    )


def _clamp(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def score_fundamentals(metrics: FundamentalMetrics) -> tuple[float, float]:
    components: list[tuple[float, float]] = []  # (score, weight)

    pe = metrics.pe_ratio
    if pe is not None:
        if pe <= 0:
            components.append((-0.6, 1.2))
        else:
            score = _clamp((20.0 - min(pe, 60.0)) / 40.0)
            components.append((score, 1.0))

    earnings_yield = metrics.earnings_yield
    if earnings_yield is not None and earnings_yield > 0:
        score = _clamp((earnings_yield - 0.05) / 0.08)
        components.append((score, 0.8))

    pb_ratio = metrics.pb_ratio
    if pb_ratio is not None and pb_ratio > 0:
        score = _clamp((2.5 - min(pb_ratio, 6.0)) / 3.5)
        components.append((score, 0.6))

    dividend_yield = metrics.dividend_yield
    if dividend_yield is not None and dividend_yield >= 0:
        score = _clamp((dividend_yield - 0.02) / 0.05)
        components.append((score, 0.4))

    fcf_yield = metrics.fcf_yield
    if fcf_yield is not None:
        score = _clamp((fcf_yield - 0.04) / 0.08)
        components.append((score, 0.9))

    valuation_gap = metrics.valuation_gap
    if valuation_gap is not None:
        score = _clamp(valuation_gap / 0.4)
        components.append((score, 0.9))

    growth_rate = metrics.growth_rate
    if growth_rate is not None:
        score = _clamp(growth_rate / 0.18)
        components.append((score, 0.7))

    if not components:
        return 0.0, 0.0

    total_weight = sum(weight for _, weight in components)
    aggregated = sum(score * weight for score, weight in components) / total_weight

    base_confidence = _clamp(0.35 + 0.12 * len(components), 0.0, 1.0)
    quality = metrics.quality_score
    if quality is not None:
        base_confidence = 0.6 * base_confidence + 0.4 * _clamp(quality, 0.0, 1.0)

    return float(aggregated), float(base_confidence)
