"""Hidden Markov Model based market regime detection utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from hmmlearn import hmm


FeatureArray = np.ndarray


@dataclass(frozen=True)
class RegimeStatistics:
    """Container describing the most recent regime assessment."""

    regime: int
    regime_name: str
    confidence: float
    probabilities: Dict[str, float]


class RegimeDetector:
    """Detects market regimes using a Gaussian Hidden Markov Model."""

    def __init__(
        self,
        n_regimes: int = 5,
        lookback_days: int = 252,
        feature_window: int = 20,
        n_iter: int = 100,
        random_state: int | None = 42,
    ) -> None:
        self.n_regimes = n_regimes
        self.lookback_days = lookback_days
        self.feature_window = feature_window
        self.n_iter = n_iter
        self.random_state = random_state

        self.model: hmm.GaussianHMM | None = None
        self.is_trained = False
        self.regime_names = {
            0: "Bull",
            1: "Bear",
            2: "Transitional",
            3: "High Volatility",
            4: "Low Volatility",
        }

    def extract_features(self, prices: Iterable[float], volumes: Iterable[float]) -> FeatureArray:
        price_arr, volume_arr = self._prepare_inputs(prices, volumes)
        df = pd.DataFrame({"price": price_arr, "volume": volume_arr})

        log_prices = np.log(df["price"])  # type: ignore[pd-unknown-array]
        returns = log_prices.diff()
        volatility = returns.rolling(window=self.feature_window, min_periods=self.feature_window).std()
        mean_return = returns.rolling(window=self.feature_window, min_periods=self.feature_window).mean()
        avg_volume = df["volume"].rolling(window=self.feature_window, min_periods=self.feature_window).mean()
        volume_ratio = df["volume"] / (avg_volume + 1e-8)
        momentum = df["price"].pct_change(periods=self.feature_window)

        feature_frame = pd.DataFrame(
            {
                "returns": returns,
                "volatility": volatility,
                "mean_return": mean_return,
                "volume_ratio": volume_ratio,
                "momentum": momentum,
            }
        ).dropna()

        if feature_frame.empty:
            raise ValueError("Not enough data to extract features; increase the lookback window.")

        features = np.nan_to_num(feature_frame.to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        normalized_features = (features - mean) / std
        return normalized_features

    def train(self, prices: Iterable[float], volumes: Iterable[float]) -> None:
        features = self.extract_features(prices, volumes)
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=self.random_state,
            min_covar=1e-3,
        )
        self.model.fit(features)
        self.is_trained = True

    def predict_regime(self, prices: Iterable[float], volumes: Iterable[float]) -> int:
        model = self._ensure_model()
        features = self.extract_features(prices, volumes)
        regime_sequence = model.predict(features)
        return int(regime_sequence[-1])

    def predict_regime_probabilities(self, prices: Iterable[float], volumes: Iterable[float]) -> np.ndarray:
        model = self._ensure_model()
        features = self.extract_features(prices, volumes)
        regime_probabilities = model.predict_proba(features)
        return regime_probabilities[-1]

    def get_regime_statistics(self, prices: Iterable[float], volumes: Iterable[float]) -> RegimeStatistics:
        regime = self.predict_regime(prices, volumes)
        probabilities = self.predict_regime_probabilities(prices, volumes)
        confidence = float(np.max(probabilities))
        probability_map = {
            self.regime_names.get(index, str(index)): float(prob)
            for index, prob in enumerate(probabilities)
        }
        regime_name = self.regime_names.get(regime, str(regime))
        return RegimeStatistics(
            regime=regime,
            regime_name=regime_name,
            confidence=confidence,
            probabilities=probability_map,
        )

    def _prepare_inputs(
        self, prices: Iterable[float], volumes: Iterable[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        price_arr = np.asarray(list(prices), dtype=float)
        volume_arr = np.asarray(list(volumes), dtype=float)

        if price_arr.ndim != 1 or volume_arr.ndim != 1:
            raise ValueError("Prices and volumes must be one-dimensional sequences.")
        if price_arr.size != volume_arr.size:
            raise ValueError("Prices and volumes must have the same length.")
        if price_arr.size < max(self.lookback_days, self.feature_window + 1):
            raise ValueError("Insufficient data points for regime detection.")

        price_arr = price_arr[-self.lookback_days :]
        volume_arr = volume_arr[-self.lookback_days :]
        if np.any(price_arr <= 0):
            raise ValueError("Prices must be strictly positive to compute log returns.")

        return price_arr, volume_arr

    def _ensure_model(self) -> hmm.GaussianHMM:
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained")
        return self.model


__all__ = ["RegimeDetector", "RegimeStatistics"]
