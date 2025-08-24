from __future__ import annotations


class EconomicDataProvider:
    def analyze_currency_strength(self, df: object) -> float:
        """Analyze relative currency strength (typed no-op)."""
        return 0.0

    def get_economic_calendar(self) -> list[dict[str, object]]:
        """Return upcoming economic events (typed no-op)."""
        return []

    def get_central_bank_policies(self) -> dict[str, object]:
        """Return central bank policy summaries (typed no-op)."""
        return {}

    def get_geopolitical_events(self) -> list[dict[str, object]]:
        """Return geopolitical events (typed no-op)."""
        return []


class FundamentalAnalyzer:
    def analyze_economic_momentum(self, df: object) -> dict[str, float]:
        """Return economic momentum metrics (typed no-op)."""
        return {"momentum_score": 0.0}

    def analyze_risk_sentiment(self, df: object) -> dict[str, float]:
        """Return risk sentiment metrics (typed no-op)."""
        return {"risk_sentiment": 0.0}

    def analyze_yield_differentials(self, df: object) -> dict[str, float]:
        """Return yield differential metrics (typed no-op)."""
        return {}

    def analyze_central_bank_divergence(self, df: object) -> dict[str, float]:
        """Return divergence metrics (typed no-op)."""
        return {}

    def analyze_economic_calendar_impact(self, df: object) -> dict[str, float]:
        """Return calendar impact metrics (typed no-op)."""
        return {}


__all__ = ["EconomicDataProvider", "FundamentalAnalyzer"]
