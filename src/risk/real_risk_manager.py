from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict


@dataclass
class RealRiskConfig:
    """Configuration for the :class:`RealRiskManager` implementation.

    The configuration mirrors the simplified knobs exposed by ``RiskManagerImpl``
    and keeps defaults intentionally conservative so that the manager can be
    instantiated without bespoke wiring during tests.

    ``max_drawdown`` is retained for backwards compatibility but callers should
    prefer ``max_total_exposure`` when providing new configuration payloads.
    """

    max_position_risk: float = 0.02
    """Maximum allowed risk as a fraction of equity for any single position."""

    max_total_exposure: float = 0.25
    """Maximum aggregate exposure tolerated before flagging elevated risk."""

    max_drawdown: float = 0.25
    """Legacy alias for ``max_total_exposure`` used by older call sites."""

    max_leverage: float = 10.0
    """Maximum tolerated gross leverage relative to equity."""

    equity: float = 10000.0
    """Baseline account equity used when computing risk budgets."""

    def __post_init__(self) -> None:
        """Normalise configuration values for downstream calculations."""

        if self.max_total_exposure <= 0 and self.max_drawdown > 0:
            self.max_total_exposure = float(self.max_drawdown)
        if self.max_total_exposure <= 0:
            self.max_total_exposure = 0.25
        if self.max_position_risk <= 0:
            self.max_position_risk = 0.02
        if self.max_leverage <= 0:
            self.max_leverage = 10.0
        if self.equity < 0:
            self.equity = 0.0


class RealRiskManager:
    """Concrete portfolio risk assessor used by :class:`RiskManagerImpl`.

    The implementation keeps track of account equity and evaluates incoming
    position dictionaries against three simple guardrails. The inputs are
    risk-weighted exposures (for example, position notional multiplied by the
    configured stop-loss fraction) so the resulting score reflects utilisation
    of the risk budget rather than raw position size.

    * per-position exposure relative to ``max_position_risk``
    * aggregate exposure relative to ``max_drawdown``
    * gross leverage relative to current equity

    The final risk score is the maximum of those ratios. A score ``> 1``
    indicates that at least one guardrail has been breached.
    """

    def __init__(self, config: RealRiskConfig) -> None:
        self.config = config
        self.equity: float = max(float(config.equity), 0.0)
        self._last_snapshot: Dict[str, float] = {
            "total_exposure": 0.0,
            "max_exposure": 0.0,
            "position_ratio": 0.0,
            "total_ratio": 0.0,
            "gross_leverage": 0.0,
            "leverage_ratio": 0.0,
            "risk_score": 0.0,
        }

    def update_equity(self, equity: float | Decimal) -> None:
        """Update the account equity used when computing risk budgets."""

        try:
            new_equity = float(equity)
        except (TypeError, ValueError):
            return

        self.equity = max(new_equity, 0.0)
        self.config.equity = self.equity

    def assess_risk(self, positions: Mapping[str, float]) -> float:
        """Return a scalar risk score for the supplied positions.

        Args:
            positions: Mapping of symbol to risk-weighted exposure (e.g. notional
                multiplied by stop-loss percentage).

        Returns:
            Maximum utilization of the configured risk budgets. ``0.0`` denotes
            no risk, while values ``> 1`` indicate that at least one constraint
            is currently violated.
        """

        exposures: list[float] = []
        for raw_size in positions.values():
            try:
                size = float(raw_size)
            except (TypeError, ValueError):
                continue

            if not math.isfinite(size):
                continue

            exposures.append(abs(size))

        if not exposures:
            self._last_snapshot = {
                "total_exposure": 0.0,
                "max_exposure": 0.0,
                "position_ratio": 0.0,
                "total_ratio": 0.0,
                "gross_leverage": 0.0,
                "leverage_ratio": 0.0,
                "risk_score": 0.0,
            }
            return 0.0

        total_exposure = float(sum(exposures))
        max_exposure = float(max(exposures))
        equity = float(self.equity)

        per_position_budget = self._resolve_budget(
            self.config.max_position_risk, equity, max_exposure
        )
        total_budget = self._resolve_budget(self.config.max_total_exposure, equity, total_exposure)

        position_ratio = max_exposure / per_position_budget if per_position_budget else 0.0
        total_ratio = total_exposure / total_budget if total_budget else 0.0
        gross_leverage = total_exposure / equity if equity > 0 else total_exposure

        leverage_limit = float(self.config.max_leverage)
        leverage_ratio = gross_leverage / leverage_limit if leverage_limit > 0 else gross_leverage

        risk_score = float(max(position_ratio, total_ratio, leverage_ratio))

        self._last_snapshot = {
            "total_exposure": total_exposure,
            "max_exposure": max_exposure,
            "position_ratio": position_ratio,
            "total_ratio": total_ratio,
            "gross_leverage": gross_leverage,
            "leverage_ratio": leverage_ratio,
            "risk_score": risk_score,
        }

        return risk_score

    @staticmethod
    def _resolve_budget(percent: float, equity: float, fallback: float) -> float:
        """Compute a positive budget based on the provided percentage and equity."""

        try:
            pct = float(percent)
        except (TypeError, ValueError):
            pct = 0.0

        candidate = pct * equity
        if candidate > 0:
            return candidate

        if equity > 0:
            return equity

        if fallback > 0:
            return fallback

        return 1.0

    @property
    def last_snapshot(self) -> Dict[str, float]:
        """Return a copy of the most recent risk assessment snapshot."""

        return dict(self._last_snapshot)


__all__ = ["RealRiskConfig", "RealRiskManager"]
