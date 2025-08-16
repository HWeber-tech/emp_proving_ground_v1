# WHAT Pattern Orchestrator and Scientific Stack Policy

Goal: provide a stable API surface for pattern synthesis in the WHAT dimension and enforce optimal, non-degraded scientific stack (NumPy/Pandas/SciPy) at runtime.

## Summary

- Pattern synthesis logic is implemented in the existing engine at:
  - src/sensory/organs/dimensions/pattern_engine.py

- A thin coordination façade has been introduced to expose a minimal and stable API:
  - src/sensory/what/patterns/orchestrator.py

- WHAT callers can use either:
  - src/sensory/what/what_sensor.py (synchronous sensor using the orchestrator; skips orchestration when already inside an event loop), or
  - src/sensory/organs/dimensions/what_organ.py (engine-level analysis that best-effort runs orchestration and returns a composite payload).

- Strict dependency policy (no fallbacks on critical algorithms):
  - Peak detection is strictly SciPy-backed:
    - src/sensory/what/features/swing_analysis.py
  - Runtime fail-fast integrity check is provided at:
    - src/system/requirements_check.py

## Runtime enforcement

- Application entrypoint (main runtime) validates the scientific stack at startup:
  - main.py

- Backtest report utility validates before execution:
  - scripts/backtest_report.py

If required libraries are missing or below the minimum versions, the system stops with a clear diagnostic and detected versions, avoiding degraded or “best-effort” modes.

## Version policy for scientific stack

Pinned via PEP 508 markers in requirements.txt (Python 3.11 and 3.13+):

- NumPy
- Pandas
- SciPy

Pip installs appropriate wheels automatically for the active interpreter.

## Orchestrator usage pattern (recommendation)

Use the orchestrator directly when you need pattern synthesis as a self-contained step:

- from src.sensory.what.patterns.orchestrator import PatternOrchestrator
- orchestrator = PatternOrchestrator()
- patterns = await orchestrator.analyze(df)  # returns dict payload with:
  - fractal_patterns
  - harmonic_patterns
  - volume_profile
  - price_action_dna
  - pattern_strength
  - confidence_score

For synchronous call sites, use src/sensory/what/what_sensor.py which:
- Computes a simple breakout signal for baseline
- Attempts orchestration when not nested within an existing event loop
- Produces a SensorSignal(signal_type="WHAT", value={pattern_strength, pattern_details}, confidence=...)

## Testing

New targeted tests:
- tests/current/test_swing_analysis_peaks.py validates peak detection semantics across simple scenarios.

All CI gates (Ruff, Mypy, Import-Linter, policy, tests/coverage) are expected to remain green.

## Migration plan

- Prefer calling the orchestrator (or WhatSensor) rather than importing internal engine components directly.
- Avoid creating new algorithmic fallbacks; treat missing/mismatched scientific libs as a hard error.
- If you need async orchestration from an existing event loop, wire PatternOrchestrator within your async flow instead of calling from a synchronous wrapper.