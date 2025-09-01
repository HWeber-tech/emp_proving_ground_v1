from __future__ import annotations

from typing import Any, Dict, List

class PatternSynthesis:
    fractal_patterns: List[Dict[str, Any]]
    harmonic_patterns: List[Dict[str, Any]]
    volume_profile: Dict[str, Any]
    price_action_dna: Dict[str, Any]
    pattern_strength: float
    confidence_score: float

class PatternSynthesisEngine:
    def __init__(self) -> None: ...
    async def synthesize_patterns(self, market_data: Any) -> PatternSynthesis: ...