#!/usr/bin/env python3

from __future__ import annotations

from typing import List
import asyncio

from src.sensory.signals import SensorSignal, IntegratedSignal


class BayesianSignalIntegrator:
    """Simple weighted aggregator of sensor signals.

    Maps aggregated strength to direction and averages confidence.
    """

    def integrate_sync(self, signals: List[SensorSignal]) -> IntegratedSignal:
        if not signals:
            return IntegratedSignal(direction='neutral', strength=0.0, confidence=0.0, contributing=[])

        total_weight = sum(max(0.0, min(1.0, s.confidence)) for s in signals)
        if total_weight == 0:
            avg_conf = 0.0
            strength = 0.0
        else:
            strength = sum(s.strength * max(0.0, min(1.0, s.confidence)) for s in signals) / total_weight
            avg_conf = sum(max(0.0, min(1.0, s.confidence)) for s in signals) / len(signals)

        direction = 'neutral'
        if strength > 0.1:
            direction = 'bullish'
        elif strength < -0.1:
            direction = 'bearish'

        return IntegratedSignal(
            direction=direction,
            strength=float(max(-1.0, min(1.0, strength))),
            confidence=float(max(0.0, min(1.0, avg_conf))),
            contributing=[s.name for s in signals],
        )

    async def integrate(self, signals: List[SensorSignal]) -> IntegratedSignal:
        # Async wrapper for event-loop compatibility
        return self.integrate_sync(signals)


