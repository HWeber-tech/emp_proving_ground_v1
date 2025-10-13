#!/usr/bin/env python3
"""
Niche Detection System
======================

Identifies and segments market conditions into distinct niches where
specialized predator strategies can excel.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, TypedDict, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.core.types import JSONArray, JSONObject


logger = logging.getLogger(__name__)


@dataclass
class MarketNiche:
    """Represents a detected market niche."""

    niche_id: str
    regime_type: str
    volatility_range: tuple[float, float]
    volume_range: tuple[float, float]
    trend_strength: float
    duration: int
    opportunity_score: float
    risk_level: str
    preferred_species: list[str]


class NicheRow(TypedDict):
    regime_type: str
    volatility_range: tuple[float, float]
    volume_range: tuple[float, float]
    trend_strength: float
    duration: int
    preferred_species: list[str]


class OHLCVRecord(TypedDict):
    close: float
    high: float
    low: float
    volume: int


class ScalerLike(Protocol):
    """Minimal interface for scalers like sklearn.preprocessing.StandardScaler."""

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]: ...


class ClustererLike(Protocol):
    """Minimal interface for clusterers like sklearn.cluster.KMeans."""

    def fit_predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]: ...


class _FallbackScaler:
    """Fallback normaliser when scikit-learn is unavailable."""

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        array = np.asarray(X, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError("Expected 2D array for scaling")
        mean = np.mean(array, axis=0)
        std = np.std(array, axis=0)
        std = np.where(std == 0.0, 1.0, std)
        return cast(NDArray[np.float64], (array - mean) / std)


class _FallbackKMeans:
    """Simple KMeans-style clusterer used as a compatibility shim."""

    def __init__(
        self,
        n_clusters: int,
        random_state: int | None = None,
        max_iter: int = 20,
    ) -> None:
        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive")
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.cluster_centers_: NDArray[np.float64] | None = None

    def fit_predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        data = np.asarray(X, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError("Expected 2D array for clustering")
        n_samples = data.shape[0]
        if n_samples == 0:
            return np.zeros(0, dtype=np.int64)

        rng = np.random.default_rng(self.random_state)
        seeds = min(self.n_clusters, n_samples)
        indices = rng.choice(n_samples, size=seeds, replace=False)
        centers = np.copy(data[indices])
        extra_count = self.n_clusters - seeds
        if extra_count > 0:
            extra_vectors = np.vstack(
                [data[rng.integers(0, n_samples)] for _ in range(extra_count)]
            )
            centers = np.concatenate([centers, extra_vectors], axis=0)

        labels = np.zeros(n_samples, dtype=np.int64)
        for _ in range(self.max_iter):
            distances = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
            new_labels = distances.argmin(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for idx in range(self.n_clusters):
                members = data[labels == idx]
                if len(members) == 0:
                    centers[idx] = data[rng.integers(0, n_samples)]
                else:
                    centers[idx] = members.mean(axis=0)

        self.cluster_centers_ = centers
        return labels


class NicheDetector:
    """Advanced market niche detection system."""

    def __init__(self) -> None:
        self.scaler: ScalerLike
        self.clusterer: ClustererLike

        try:
            cluster_mod = importlib.import_module("sklearn.cluster")
            preprocessing_mod = importlib.import_module("sklearn.preprocessing")
        except ImportError:
            logger.warning(
                "scikit-learn unavailable; using fallback niche clustering stack"
            )
            self.scaler = _FallbackScaler()
            self.clusterer = _FallbackKMeans(n_clusters=5, random_state=42)
        else:
            KMeans = getattr(cluster_mod, "KMeans")
            StandardScaler = getattr(preprocessing_mod, "StandardScaler")

            self.scaler = cast(ScalerLike, StandardScaler())
            self.clusterer = cast(ClustererLike, KMeans(n_clusters=5, random_state=42))
        self.niche_history: list[MarketNiche] = []

    async def detect_niches(self, market_data: JSONObject) -> dict[str, MarketNiche]:
        """Detect and segment market into different niches."""
        raw = market_data.get("data", [])
        if not isinstance(raw, list):
            return {}

        records = cast(list[OHLCVRecord], raw)
        df = pd.DataFrame(records)
        if len(df) < 50:
            return {}

        # Calculate market features
        features = self._calculate_market_features(df)

        # Detect regimes
        regimes = self._detect_regimes(features)

        # Identify niches within regimes
        niches = self._identify_niches(features, regimes)

        # Score opportunities
        scored_niches = self._score_opportunities(niches)

        return scored_niches

    def _calculate_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive market features."""
        features = pd.DataFrame()

        # Price features
        features["returns"] = df["close"].astype(float).pct_change()
        features["volatility"] = features["returns"].rolling(20).std() * np.sqrt(252.0)
        features["trend_strength"] = self._calculate_trend_strength(df)

        # Volume features
        features["volume_ma"] = df["volume"].rolling(20).mean()
        features["volume_ratio"] = df["volume"] / features["volume_ma"]
        features["volume_volatility"] = df["volume"].pct_change().rolling(20).std()

        # Technical indicators
        features["rsi"] = self._calculate_rsi(df["close"].astype(float))
        features["atr"] = self._calculate_atr(df)
        features["momentum"] = self._calculate_momentum(df)

        # Market microstructure
        features["spread"] = (df["high"].astype(float) - df["low"].astype(float)) / df[
            "close"
        ].astype(float)
        features["efficiency"] = df["close"].diff().abs().astype(float) / (
            df["high"].astype(float) - df["low"].astype(float)
        )

        return features.dropna()

    def _calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series[float]:
        """Calculate trend strength using linear regression."""

        def rolling_trend(window: pd.Series[float]) -> float:
            if len(window) < 10:
                return 0.0
            arr = np.asarray(window, dtype=float)
            x: NDArray[np.float64] = np.arange(len(arr), dtype=float)
            coef = np.polyfit(x, arr, 1)
            slope = float(coef[0])
            std_val = float(np.std(arr))
            return slope / std_val if std_val > 0.0 else 0.0

        result = (
            df["close"]
            .astype(float)
            .rolling(20)
            .apply(lambda x: rolling_trend(x[-10:]) if len(x) >= 10 else 0.0)
        )
        return result

    def _calculate_rsi(self, prices: pd.Series[float], period: int = 14) -> pd.Series[float]:
        """Calculate RSI indicator."""
        prices_f = prices.astype(float)
        delta = prices_f.diff()
        gain = delta.where(delta > 0.0, 0.0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0.0, 0.0)).rolling(window=period).mean()
        rs = gain / loss.replace(0.0, np.nan)
        result = 100.0 - (100.0 / (1.0 + rs))
        return result

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series[float]:
        """Calculate Average True Range."""
        high_low = (df["high"] - df["low"]).astype(float)
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        result = true_range.rolling(period).mean()
        return result

    def _calculate_momentum(self, df: pd.DataFrame) -> pd.Series[float]:
        """Calculate momentum indicator."""
        result = df["close"].astype(float).pct_change(10)
        return result

    def _detect_regimes(self, features: pd.DataFrame) -> pd.Series[str]:
        """Detect market regimes using clustering."""
        # Select key features for clustering
        cluster_features = features[["volatility", "trend_strength", "volume_ratio"]].dropna()

        if len(cluster_features) < 10:
            fallback = pd.Series(["neutral"] * len(features), index=features.index, dtype="string")
            return cast(pd.Series[str], fallback.astype(str))

        # Standardize features
        scaled_features = self.scaler.fit_transform(cluster_features.to_numpy(dtype=float))

        # Cluster market states
        clusters = self.clusterer.fit_predict(scaled_features)

        # Map clusters to regimes
        regime_map = {
            0: "trending_bull",
            1: "trending_bear",
            2: "ranging",
            3: "volatile",
            4: "quiet",
        }

        regimes = pd.Series(
            [regime_map.get(int(c), "neutral") for c in clusters],
            index=cluster_features.index,
            dtype="string",
        )
        regimes = regimes.astype(str)
        regimes = cast(pd.Series[str], regimes)

        # Extend to full length
        full_regimes_raw: pd.Series[str] = cast(
            pd.Series[str],
            pd.Series(["neutral"] * len(features), index=features.index, dtype="string").astype(
                str
            ),
        )
        # Assign detected regimes by index to avoid pandas Series.update typing issues
        full_regimes_raw.loc[regimes.index] = regimes.astype(str)
        full_regimes = cast(pd.Series[str], full_regimes_raw.astype(str))

        return full_regimes

    def _identify_niches(self, features: pd.DataFrame, regimes: pd.Series[str]) -> list[NicheRow]:
        """Identify specific niches within market regimes."""
        niches: list[NicheRow] = []

        for regime in regimes.unique():
            regime_mask = regimes == regime
            regime_data = features[regime_mask]

            if len(regime_data) < 10:
                continue

            # Calculate regime characteristics
            volatility_range = (
                regime_data["volatility"].quantile(0.25),
                regime_data["volatility"].quantile(0.75),
            )
            volume_range = (
                regime_data["volume_ratio"].quantile(0.25),
                regime_data["volume_ratio"].quantile(0.75),
            )
            trend_strength = float(regime_data["trend_strength"].mean())

            # Determine preferred species for this niche
            preferred_species = self._determine_preferred_species(regime, volatility_range)

            niche: NicheRow = {
                "regime_type": regime,
                "volatility_range": volatility_range,
                "volume_range": volume_range,
                "trend_strength": trend_strength,
                "duration": len(regime_data),
                "preferred_species": preferred_species,
            }

            niches.append(niche)

        return niches

    def _determine_preferred_species(
        self, regime: str, volatility_range: tuple[float, float]
    ) -> list[str]:
        """Determine which species are best suited for this regime."""
        preferred: list[str] = []

        _vol_low, vol_high = volatility_range

        if regime == "trending_bull":
            preferred = ["stalker", "alpha", "pack_hunter"]
        elif regime == "trending_bear":
            preferred = ["stalker", "pack_hunter"]
        elif regime == "ranging":
            if vol_high < 0.02:
                preferred = ["ambusher", "scavenger"]
            else:
                preferred = ["pack_hunter", "scavenger"]
        elif regime == "volatile":
            if vol_high > 0.05:
                preferred = ["scavenger", "alpha"]
            else:
                preferred = ["pack_hunter", "ambusher"]
        elif regime == "quiet":
            preferred = ["ambusher", "scavenger"]

        return preferred

    def _score_opportunities(self, niches: list[NicheRow]) -> dict[str, MarketNiche]:
        """Score niches based on opportunity potential."""
        scored_niches: dict[str, MarketNiche] = {}

        for i, niche_data in enumerate(niches):
            # Calculate opportunity score
            volatility_score = min(niche_data["volatility_range"][1] * 10.0, 1.0)
            volume_score = min(niche_data["volume_range"][1] * 0.5, 1.0)
            trend_score = abs(niche_data["trend_strength"]) * 2.0

            opportunity_score = (volatility_score + volume_score + trend_score) / 3.0

            # Determine risk level
            if niche_data["volatility_range"][1] > 0.05:
                risk_level = "high"
            elif niche_data["volatility_range"][1] > 0.02:
                risk_level = "medium"
            else:
                risk_level = "low"

            niche = MarketNiche(
                niche_id=f"niche_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                regime_type=niche_data["regime_type"],
                volatility_range=niche_data["volatility_range"],
                volume_range=niche_data["volume_range"],
                trend_strength=niche_data["trend_strength"],
                duration=niche_data["duration"],
                opportunity_score=opportunity_score,
                risk_level=risk_level,
                preferred_species=niche_data["preferred_species"],
            )

            scored_niches[niche.niche_id] = niche

        return scored_niches

    async def get_current_regime(self, market_data: JSONObject) -> str:
        """Get current market regime classification."""
        niches = await self.detect_niches(market_data)

        if not niches:
            return "neutral"

        # Return the most recent/active niche
        return list(niches.values())[0].regime_type

    async def get_species_recommendations(self, market_data: JSONObject) -> list[str]:
        """Get species recommendations for current market conditions."""
        niches = await self.detect_niches(market_data)

        if not niches:
            return ["pack_hunter"]  # Default fallback

        # Get recommendations from highest-scoring niche
        best_niche = max(niches.values(), key=lambda x: x.opportunity_score)
        return best_niche.preferred_species

    def get_niche_history(self) -> list[MarketNiche]:
        """Get historical niche data."""
        return self.niche_history


# Example usage
async def test_niche_detection() -> None:
    """Test the niche detection system."""

    # Generate test market data
    np.random.seed(42)

    # Create trending market
    trend = np.linspace(1.0, 1.1, 100)
    noise = np.random.normal(0, 0.001, 100)
    prices = trend + noise

    # Build synthetic OHLCV records for the detector
    records: list[OHLCVRecord] = []
    for i in range(100):
        close = float(prices[i])
        high = float(close + 0.001)
        low = float(close - 0.001)
        volume = int(1000 + np.random.randint(0, 500))
        records.append({"close": close, "high": high, "low": low, "volume": volume})

    market_data: JSONObject = {"data": cast(JSONArray, records)}

    detector = NicheDetector()
    niches = await detector.detect_niches(market_data)

    print(f"Detected {len(niches)} niches:")
    for niche_id, niche in niches.items():
        print(
            f"  {niche_id} -> {niche.regime_type}: score={niche.opportunity_score:.2f}, species={niche.preferred_species}"
        )


if __name__ == "__main__":
    asyncio.run(test_niche_detection())
