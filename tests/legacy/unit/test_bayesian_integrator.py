#!/usr/bin/env python3

from src.sensory.integrate.bayesian_integrator import BayesianSignalIntegrator
from src.sensory.signals import SensorSignal


def test_bayesian_integrator_weighting():
    integrator = BayesianSignalIntegrator()
    signals = [
        SensorSignal(name='WHY', strength=0.6, confidence=0.8),
        SensorSignal(name='HOW', strength=0.2, confidence=0.5),
        SensorSignal(name='WHAT', strength=-0.1, confidence=0.4),
    ]
    out = integrator.integrate_sync(signals)
    assert out.direction in {'bullish', 'neutral', 'bearish'}
    assert 0.0 <= out.confidence <= 1.0
    # Weighted average should be positive given weights
    assert out.strength > 0.0
    assert 'WHY' in out.contributing


