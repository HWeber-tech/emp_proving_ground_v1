"""Feature encoding for TRM strategy batches."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean, pstdev
from typing import Iterable, Mapping

from .types import DecisionDiaryEntry, StrategyEncoding, StrategyStats

_FEATURE_NAMES = (
    "count_log",
    "mean_pnl_scaled",
    "pnl_std_scaled",
    "risk_rate",
    "win_rate",
    "loss_rate",
    "volatility_mean",
    "spread_mean_pips",
    "belief_confidence_mean",
    "pnl_trend_scaled",
    "drawdown_ratio",
)


class RIMEncoder:
    """Encodes decision diary slices into model-ready feature vectors."""

    def encode(self, entries: Iterable[DecisionDiaryEntry]) -> list[StrategyEncoding]:
        strategies: dict[str, list[DecisionDiaryEntry]] = defaultdict(list)
        for entry in entries:
            strategies[entry.strategy_id].append(entry)

        encodings: list[StrategyEncoding] = []
        for strategy_id, strategy_entries in strategies.items():
            features = self._build_feature_map(strategy_entries)
            stats = self._build_stats(strategy_entries, features)
            audit_hashes = tuple(entry.input_hash for entry in strategy_entries if entry.input_hash)[:5]
            encodings.append(
                StrategyEncoding(
                    strategy_id=strategy_id,
                    features=features,
                    stats=stats,
                    audit_entry_hashes=audit_hashes,
                )
            )
        return encodings

    def _build_feature_map(self, entries: list[DecisionDiaryEntry]) -> Mapping[str, float]:
        count = max(len(entries), 1)
        pnl_values = [entry.pnl for entry in entries]
        mean_pnl = mean(pnl_values)
        pnl_std = pstdev(pnl_values) if len(pnl_values) > 1 else 0.0
        risk_counts = sum(len(entry.risk_flags) for entry in entries)
        win_count = sum(1 for entry in entries if "win" in entry.outcome_labels)
        loss_count = sum(1 for entry in entries if "loss" in entry.outcome_labels)
        volatility_values = [entry.features_digest.get("volatility", 0.0) for entry in entries]
        spread_values = [entry.features_digest.get("spread", 0.0) for entry in entries]
        belief_values = [entry.belief_confidence for entry in entries if entry.belief_confidence is not None]
        pnl_trend = _pnl_trend(entries)
        min_pnl = min(pnl_values)
        max_pnl = max(pnl_values)
        drawdown_ratio = abs(min_pnl) / (abs(max_pnl) + 1e-6) if max_pnl != 0 else 1.0

        return {
            "count_log": math.log1p(count),
            "mean_pnl_scaled": mean_pnl / 1000.0,
            "pnl_std_scaled": pnl_std / 1000.0,
            "risk_rate": risk_counts / count,
            "win_rate": win_count / count,
            "loss_rate": loss_count / count,
            "volatility_mean": mean(volatility_values) if volatility_values else 0.0,
            "spread_mean_pips": mean(spread_values) * 10000 if spread_values else 0.0,
            "belief_confidence_mean": mean(belief_values) if belief_values else 0.5,
            "pnl_trend_scaled": pnl_trend / 1000.0,
            "drawdown_ratio": drawdown_ratio,
        }

    @staticmethod
    def _build_stats(entries: list[DecisionDiaryEntry], features: Mapping[str, float]) -> StrategyStats:
        count = max(len(entries), 1)
        pnl_values = [entry.pnl for entry in entries]
        mean_pnl = mean(pnl_values)
        pnl_std = pstdev(pnl_values) if len(pnl_values) > 1 else 0.0
        risk_counts = sum(len(entry.risk_flags) for entry in entries)
        win_count = sum(1 for entry in entries if "win" in entry.outcome_labels)
        loss_count = sum(1 for entry in entries if "loss" in entry.outcome_labels)
        min_pnl = min(pnl_values)
        max_pnl = max(pnl_values)
        volatility_mean = features.get("volatility_mean", 0.0)
        spread_mean = features.get("spread_mean_pips", 0.0)
        belief_confidence = features.get("belief_confidence_mean", 0.0)
        pnl_trend = features.get("pnl_trend_scaled", 0.0) * 1000.0
        drawdown_ratio = features.get("drawdown_ratio", 0.0)
        return StrategyStats(
            entry_count=count,
            mean_pnl=mean_pnl,
            pnl_std=pnl_std,
            risk_rate=risk_counts / count,
            win_rate=win_count / count,
            loss_rate=loss_count / count,
            volatility_mean=volatility_mean,
            spread_mean=spread_mean,
            belief_confidence_mean=belief_confidence,
            pnl_trend=pnl_trend,
            drawdown_ratio=drawdown_ratio,
        )


def _pnl_trend(entries: list[DecisionDiaryEntry]) -> float:
    if len(entries) < 2:
        return 0.0
    midpoint = len(entries) // 2
    first_half = entries[:midpoint]
    second_half = entries[midpoint:]
    if not first_half or not second_half:
        return 0.0
    mean_first = mean(entry.pnl for entry in first_half)
    mean_second = mean(entry.pnl for entry in second_half)
    return mean_second - mean_first


__all__ = ["RIMEncoder", "_FEATURE_NAMES"]
