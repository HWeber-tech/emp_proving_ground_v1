#!/usr/bin/env python3
"""
Canonical AdversarialTrainer for unifying training interfaces across modules.

This class provides a minimal, stable interface used by both intelligence and thinking layers:
- train_generator(generator, survival_results, target_failure_rate=0.3) -> bool
- train_discriminator(strategy_population, synthetic_scenarios, survival_results) -> list

It intentionally avoids domain-specific logic to minimize coupling and enable structural unification.
"""

from __future__ import annotations

from typing import Any, List


class AdversarialTrainer:
    """
    Canonical trainer with tolerant signatures to accommodate different call sites.
    """

    def __init__(self) -> None:
        # A soft state that callers may read/adjust
        self.current_difficulty: float = 0.1

    async def train_generator(
        self,
        generator: Any,
        survival_results: List[Any],
        target_failure_rate: float = 0.3,
    ) -> bool:
        """
        Adjust generator difficulty to approximate a target failure rate, or delegate when available.

        Behavior:
        - If generator provides train_generator(survival_results, target_failure_rate) return its result
        - Else adjust local current_difficulty and return True to indicate an update occurred
        """
        # Delegate when supported by the generator (as seen in thinking/adversarial/market_gan.py)
        if hasattr(generator, "train_generator"):
            try:
                result = await generator.train_generator(survival_results, target_failure_rate)
                return bool(result)
            except TypeError:
                # Fallback to difficulty tuning if signature mismatch
                pass
            except Exception:
                # Swallow to avoid breaking training loops during structural migration
                pass

        # Heuristic difficulty tuning
        try:
            total = len(survival_results)
            failed = 0
            # survival_results may be dicts or objects with survived/survival_rate; handle defensively
            for r in survival_results:
                survived = None
                if isinstance(r, dict):
                    survived = r.get("survived")
                    if survived is None and "survival_rate" in r:
                        survived = r["survival_rate"] > 0
                else:
                    # Object with attributes?
                    survived = getattr(r, "survived", None)
                    if survived is None:
                        sr = getattr(r, "survival_rate", None)
                        if sr is not None:
                            survived = sr > 0

                if survived is False:
                    failed += 1

            failure_rate = (failed / total) if total else 0.0
            if failure_rate < target_failure_rate:
                self.current_difficulty = min(1.0, self.current_difficulty + 0.1)
            elif failure_rate > target_failure_rate:
                self.current_difficulty = max(0.1, self.current_difficulty - 0.05)
        except Exception:
            # Non-fatal; keep loop going
            pass

        return True

    async def train_discriminator(
        self,
        strategy_population: List[Any],
        synthetic_scenarios: List[Any],
        survival_results: List[Any],
    ) -> List[Any]:
        """
        Improve strategies using survival_results feedback.

        Behavior:
        - Tolerant: if strategy objects are dicts, annotate a hint based on survival
        - Returns a list of improved strategies (shallow-copied)
        """
        improved: List[Any] = []
        try:
            for idx, strategy in enumerate(strategy_population):
                sr = survival_results[idx] if idx < len(survival_results) else None
                survived = None
                if isinstance(sr, dict):
                    survived = sr.get("survived")
                    if survived is None and "survival_rate" in sr:
                        survived = sr["survival_rate"] > 0
                else:
                    survived = getattr(sr, "survived", None)
                    if survived is None:
                        rate = getattr(sr, "survival_rate", None)
                        if rate is not None:
                            survived = rate > 0

                # Shallow copy dict-like strategies to add hints
                if isinstance(strategy, dict):
                    s2 = dict(strategy)
                    if not survived:
                        # Suggest generic improvements (non-breaking)
                        rm = s2.setdefault("risk_management", {})
                        rm["max_drawdown"] = max(0.02, float(rm.get("max_drawdown", 0.05)) * 0.9)
                        af = s2.setdefault("adaptation_features", {})
                        af["dynamic_risk"] = True
                        af["regime_detection"] = True
                    improved.append(s2)
                else:
                    # Leave non-dict strategies unchanged
                    improved.append(strategy)
        except Exception:
            # Non-fatal; return population unchanged on error
            return strategy_population

        return improved