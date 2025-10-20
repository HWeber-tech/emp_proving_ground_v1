"""
Red Team AI System
Dedicated AI system to attack and improve strategies.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import logging
import uuid
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, cast

import numpy as np
from src.core.state_store import StateStore
from src.thinking.models.normalizers import normalize_attack_report
from src.thinking.models.types import AttackReportTD

logger = logging.getLogger(__name__)

_MISSING = object()


def _safe_getattr(obj: object, key: str) -> object:
    """Return attribute ``key`` or ``_MISSING`` while logging unexpected errors."""

    try:
        return getattr(obj, key)
    except AttributeError:
        return _MISSING
    except Exception as exc:  # pragma: no cover - defensive logging path
        logger.debug(
            "Failed to read attribute '%s' from %s: %s",
            key,
            type(obj).__name__,
            exc,
            exc_info=exc,
        )
        return _MISSING


def _to_mapping(obj: object) -> dict[str, object]:
    """
    Best-effort conversion to a plain dict without raising.
    Order of attempts:
    - obj.dict() if available and mapping-like
    - dict(obj) if mapping-like
    - minimal attribute projection fallback
    """
    if hasattr(obj, "dict"):
        try:
            candidate = obj.dict()
        except (AttributeError, TypeError, ValueError) as exc:
            logger.warning(
                "Failed to materialize mapping via dict() on %s: %s",
                type(obj).__name__,
                exc,
                exc_info=exc,
            )
        except Exception as exc:  # pragma: no cover - unexpected failure
            logger.warning(
                "Unexpected error calling dict() on %s: %s",
                type(obj).__name__,
                exc,
                exc_info=exc,
            )
        else:
            if isinstance(candidate, Mapping):
                return dict(candidate)
            logger.warning(
                "Object %s.dict() returned non-mapping %s",
                type(obj).__name__,
                type(candidate).__name__,
            )
    if isinstance(obj, dict):
        return obj
    # Fallback: gather common attributes if present
    out: dict[str, object] = {}
    for key in (
        "strategy_id",
        "timestamp",
        "behavior_profile",
        "risk_factors",
        "performance_patterns",
        "metadata",
    ):
        value = _safe_getattr(obj, key)
        if value is not _MISSING:
            out[key] = value
    return out


class StrategyAnalyzer:
    """Deep analysis of strategy behavior patterns."""

    def __init__(self) -> None:
        self.analysis_metrics = [
            "volatility_sensitivity",
            "trend_following_strength",
            "mean_reversion_tendency",
            "risk_tolerance",
            "position_sizing_behavior",
        ]

    async def analyze_behavior(
        self, target_strategy: str, test_scenarios: List[dict[str, object]]
    ) -> object:
        """Analyze strategy behavior patterns."""
        try:
            # Simulate strategy behavior across scenarios
            behavior_data = []

            for scenario in test_scenarios:
                behavior = await self._simulate_strategy_behavior(target_strategy, scenario)
                behavior_data.append(behavior)

            # Calculate behavior metrics
            metrics = self._calculate_behavior_metrics(behavior_data)

            # Create behavior profile
            analysis = {
                "strategy_id": target_strategy,
                "timestamp": datetime.utcnow(),
                "behavior_profile": metrics,
                "risk_factors": self._identify_risk_factors(metrics),
                "performance_patterns": self._identify_performance_patterns(behavior_data),
                "metadata": {"analysis_type": "comprehensive"},
            }

            logger.debug(f"Analyzed strategy {target_strategy}: {len(metrics)} metrics calculated")

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing strategy behavior: {e}")
            return {
                "strategy_id": target_strategy,
                "timestamp": datetime.utcnow(),
                "behavior_profile": {},
                "risk_factors": [],
                "performance_patterns": [],
                "metadata": {"error": str(e)},
            }

    async def _simulate_strategy_behavior(
        self, strategy_id: str, scenario: dict[str, object]
    ) -> dict[str, object]:
        """Simulate strategy behavior in a scenario."""
        try:
            # This would be enhanced with actual strategy simulation
            return {
                "volatility_sensitivity": np.random.normal(0.5, 0.2),
                "trend_following_strength": np.random.normal(0.7, 0.15),
                "mean_reversion_tendency": np.random.normal(0.3, 0.1),
                "risk_tolerance": np.random.normal(0.6, 0.2),
                "position_sizing_behavior": np.random.normal(0.4, 0.1),
            }

        except Exception as e:
            logger.error(f"Error simulating strategy behavior: {e}")
            return {}

    def _calculate_behavior_metrics(
        self, behavior_data: List[dict[str, object]]
    ) -> Dict[str, float]:
        """Calculate aggregate behavior metrics."""
        try:
            if not behavior_data:
                return {}

            metrics: Dict[str, float] = {}
            for metric in self.analysis_metrics:
                values = [
                    (float(v) if isinstance((v := b.get(metric, 0)), (int, float, str)) else 0.0)
                    for b in behavior_data
                ]
                metrics[metric] = float(np.mean(values))

            return metrics

        except Exception as e:
            logger.error(f"Error calculating behavior metrics: {e}")
            return {}

    def _identify_risk_factors(self, metrics: Dict[str, float]) -> List[str]:
        """Identify risk factors from behavior metrics."""
        try:
            risk_factors = []

            if metrics.get("volatility_sensitivity", 0) > 0.8:
                risk_factors.append("high_volatility_sensitivity")

            if metrics.get("risk_tolerance", 0) > 0.9:
                risk_factors.append("excessive_risk_taking")

            if metrics.get("trend_following_strength", 0) < 0.3:
                risk_factors.append("weak_trend_following")

            return risk_factors

        except Exception as e:
            logger.error(f"Error identifying risk factors: {e}")
            return []

    def _identify_performance_patterns(self, behavior_data: List[dict[str, object]]) -> List[str]:
        """Identify performance patterns from behavior data."""
        try:
            patterns = []

            # Simple pattern detection
            if len(behavior_data) > 5:
                patterns.append("sufficient_data")

            return patterns

        except Exception as e:
            logger.error(f"Error identifying performance patterns: {e}")
            return []


class WeaknessDetector:
    """Identifies potential weaknesses in strategies."""

    def __init__(self) -> None:
        self.known_vulnerabilities = [
            "volatility_spike_vulnerability",
            "trend_reversal_blindness",
            "mean_reversion_trap",
            "overfitting_to_historical_data",
            "position_sizing_errors",
            "stop_loss_clustering",
        ]

    async def find_weaknesses(
        self, behavior_profile: dict[str, object], known_vulnerabilities: List[str]
    ) -> List[str]:
        """Find potential weaknesses in strategy."""
        try:
            weaknesses = []

            # Check against known vulnerabilities
            for vulnerability in known_vulnerabilities:
                if self._check_vulnerability(behavior_profile, vulnerability):
                    weaknesses.append(vulnerability)

            # Check for new weaknesses
            new_weaknesses = self._detect_new_weaknesses(behavior_profile)
            weaknesses.extend(new_weaknesses)

            logger.debug(f"Found {len(weaknesses)} weaknesses: {', '.join(weaknesses)}")

            return weaknesses

        except Exception as e:
            logger.error(f"Error finding weaknesses: {e}")
            return []

    def _check_vulnerability(self, behavior_profile: dict[str, object], vulnerability: str) -> bool:
        """Check if strategy has specific vulnerability."""
        try:
            # Simple vulnerability checks
            def _f(key: str) -> float:
                v = behavior_profile.get(key, 0)
                return float(v) if isinstance(v, (int, float, str)) else 0.0

            if vulnerability == "volatility_spike_vulnerability":
                return _f("volatility_sensitivity") > 0.8
            elif vulnerability == "trend_reversal_blindness":
                return _f("trend_following_strength") > 0.9
            elif vulnerability == "mean_reversion_trap":
                return _f("mean_reversion_tendency") > 0.8

            return False

        except Exception as e:
            logger.error(f"Error checking vulnerability: {e}")
            return False

    def _detect_new_weaknesses(self, behavior_profile: dict[str, object]) -> List[str]:
        """Detect new weaknesses not in known list."""
        try:
            new_weaknesses = []

            # Check for extreme values
            for metric, value in behavior_profile.items():
                v = float(value) if isinstance(value, (int, float, str)) else 0.0
                if abs(v) > 2.0:  # Extreme value threshold
                    new_weaknesses.append(f"extreme_{metric}")

            return new_weaknesses

        except Exception as e:
            logger.error(f"Error detecting new weaknesses: {e}")
            return []


class AttackGenerator:
    """Generates targeted attacks for discovered weaknesses."""

    def __init__(self) -> None:
        self.attack_templates = {
            "volatility_spike_vulnerability": {
                "attack_type": "volatility_spike",
                "intensity": "high",
                "duration": "short",
            },
            "trend_reversal_blindness": {
                "attack_type": "trend_reversal",
                "intensity": "medium",
                "duration": "medium",
            },
            "mean_reversion_trap": {
                "attack_type": "false_mean_reversion",
                "intensity": "medium",
                "duration": "long",
            },
        }

    async def create_attack(self, weakness: str, target_strategy: str) -> object:
        """Create a targeted attack for a weakness."""
        try:
            # Get attack template
            template = self.attack_templates.get(
                weakness, {"attack_type": "generic", "intensity": "low", "duration": "short"}
            )

            # Generate attack parameters
            attack_params = self._generate_attack_parameters(
                cast(dict[str, object], template), target_strategy
            )

            attack = {
                "attack_id": str(uuid.uuid4()),
                "strategy_id": target_strategy,
                "weakness_targeted": weakness,
                "attack_type": template["attack_type"],
                "parameters": attack_params,
                "expected_impact": self._calculate_expected_impact(weakness),
                "timestamp": datetime.utcnow(),
            }

            logger.debug(
                f"Created attack {attack.get('attack_id', 'unknown')} "
                f"targeting {weakness} in {target_strategy}"
            )

            return attack

        except Exception as e:
            logger.error(f"Error creating attack: {e}")
            return {
                "attack_id": str(uuid.uuid4()),
                "strategy_id": target_strategy,
                "weakness_targeted": weakness,
                "attack_type": "error",
                "parameters": {},
                "expected_impact": 0.0,
                "timestamp": datetime.utcnow(),
            }

    def _generate_attack_parameters(
        self, template: dict[str, object], target_strategy: str
    ) -> dict[str, object]:
        """Generate attack parameters."""
        try:
            return {
                "intensity": template["intensity"],
                "duration": template["duration"],
                "target_strategy": target_strategy,
                "attack_vector": template["attack_type"],
            }

        except Exception as e:
            logger.error(f"Error generating attack parameters: {e}")
            return {}

    def _calculate_expected_impact(self, weakness: str) -> float:
        """Calculate expected impact of attack."""
        try:
            # Simple impact calculation
            impact_map = {
                "volatility_spike_vulnerability": 0.8,
                "trend_reversal_blindness": 0.6,
                "mean_reversion_trap": 0.7,
                "overfitting_to_historical_data": 0.9,
                "position_sizing_errors": 0.5,
                "stop_loss_clustering": 0.4,
            }

            return impact_map.get(weakness, 0.3)

        except Exception as e:
            logger.error(f"Error calculating expected impact: {e}")
            return 0.3


class ExploitDeveloper:
    """Develops exploits for discovered weaknesses."""

    def __init__(self) -> None:
        self.exploit_templates = {
            "volatility_spike_vulnerability": {
                "exploit_type": "volatility_manipulation",
                "severity": "high",
                "complexity": "medium",
            },
            "trend_reversal_blindness": {
                "exploit_type": "trend_deception",
                "severity": "medium",
                "complexity": "high",
            },
            "mean_reversion_trap": {
                "exploit_type": "false_reversion",
                "severity": "medium",
                "complexity": "medium",
            },
        }

    async def develop_exploits(self, weaknesses: List[str], target_strategy: str) -> list[object]:
        """Develop exploits for discovered weaknesses."""
        try:
            exploits = []

            for weakness in weaknesses:
                exploit = await self._create_exploit(weakness, target_strategy)
                if exploit:
                    exploits.append(exploit)

            logger.debug(f"Developed {len(exploits)} exploits for strategy {target_strategy}")

            return exploits

        except Exception as e:
            logger.error(f"Error developing exploits: {e}")
            return []

    async def _create_exploit(self, weakness: str, target_strategy: str) -> Optional[Any]:
        """Create a specific exploit for a weakness."""
        try:
            template = self.exploit_templates.get(
                weakness, {"exploit_type": "generic", "severity": "low", "complexity": "low"}
            )

            exploit = {
                "exploit_id": str(uuid.uuid4()),
                "strategy_id": target_strategy,
                "weakness_exploited": weakness,
                "exploit_type": template["exploit_type"],
                "severity": template["severity"],
                "complexity": template["complexity"],
                "parameters": self._generate_exploit_parameters(weakness),
                "timestamp": datetime.utcnow(),
            }

            return exploit

        except Exception as e:
            logger.error(f"Error creating exploit: {e}")
            return None

    def _generate_exploit_parameters(self, weakness: str) -> dict[str, object]:
        """Generate exploit parameters."""
        try:
            return {
                "weakness": weakness,
                "attack_vector": f"exploit_{weakness}",
                "severity_level": "high",
                "complexity_level": "medium",
            }

        except Exception as e:
            logger.error(f"Error generating exploit parameters: {e}")
            return {}


class RedTeamAI:
    """
    Dedicated AI system to attack and improve strategies.

    Features:
    - Deep strategy behavior analysis
    - Weakness identification and exploitation
    - Attack generation and execution
    - Strategy improvement recommendations
    """

    def __init__(self, state_store: StateStore) -> None:
        self.state_store = state_store
        self.strategy_analyzer = StrategyAnalyzer()
        self.weakness_detector = WeaknessDetector()
        self.attack_generator = AttackGenerator()
        self.exploit_developer = ExploitDeveloper()

        self._attack_history_key = "emp:red_team_attacks"
        self._exploit_history_key = "emp:red_team_exploits"
        self._agent_registry_key = "emp:red_team_agents"

    async def initialize(self) -> bool:
        return True

    async def stop(self) -> bool:
        return True

    async def attack_strategy(self, target_strategy: str) -> dict[str, object]:
        """
        Execute a comprehensive attack on a strategy.

        Args:
            target_strategy: ID of the strategy to attack

        Returns:
            Comprehensive attack report
        """
        try:
            # Step 1: Deep behavior analysis
            test_scenarios = await self._generate_test_scenarios()
            behavior_profile = await self.strategy_analyzer.analyze_behavior(
                target_strategy, test_scenarios
            )

            # Step 2: Identify weaknesses
            known_vulnerabilities = await self._get_known_vulnerabilities()
            profile_map = (
                behavior_profile.get("behavior_profile", {})
                if isinstance(behavior_profile, dict)
                else getattr(behavior_profile, "behavior_profile", {})
            )
            weaknesses = await self.weakness_detector.find_weaknesses(
                profile_map, known_vulnerabilities
            )

            agent_map = await self._load_persistent_agents(weaknesses) if weaknesses else {}

            # Step 3: Generate attacks
            attacks = []
            for weakness in weaknesses:
                attack = await self.attack_generator.create_attack(weakness, target_strategy)
                if agent_map:
                    agent = agent_map.get(weakness)
                    if agent is not None:
                        attack = self._apply_agent_specialization(agent, attack)
                attacks.append(attack)

            # Step 4: Develop exploits
            exploits = await self.exploit_developer.develop_exploits(weaknesses, target_strategy)

            # Step 5: Execute attacks
            attack_results: List[AttackReportTD] = []
            for attack in attacks:
                result = await self._execute_attack(target_strategy, attack)
                attack_results.append(result)

            if agent_map:
                await self._update_persistent_agents(agent_map, attacks, attack_results)

            # Step 6: Store results
            await self._store_attack_results(
                target_strategy, behavior_profile, weaknesses, attacks, exploits, attack_results
            )

            # Step 7: Generate improvement recommendations
            recommendations = await self._generate_improvements(weaknesses, attack_results)

            report: Dict[str, object] = {
                "strategy_id": target_strategy,
                "behavior_analysis": _to_mapping(behavior_profile),
                "weaknesses_found": weaknesses,
                "attacks_generated": [normalize_attack_report(a) for a in attacks],
                "exploits_developed": [_to_mapping(e) for e in exploits],
                "attack_results": attack_results,
                "improvements": recommendations,
                "timestamp": datetime.utcnow().isoformat(),
            }

            if agent_map:
                report["persistent_agents"] = [
                    {
                        "agent_id": str(agent.get("agent_id", "")),
                        "weakness": str(agent.get("weakness", "")),
                        "skill_level": str(agent.get("skill_level", "")),
                        "attack_count": int(agent.get("attack_count", 0)),
                        "success_rate": float(agent.get("success_rate", 0.0)),
                    }
                    for _, agent in sorted(agent_map.items(), key=lambda item: item[0])
                ]

            logger.info(
                f"Red team attack complete on {target_strategy}: "
                f"{len(weaknesses)} weaknesses found, "
                f"{len(attacks)} attacks generated"
            )

            return report

        except Exception as e:
            logger.error(f"Error in red team attack: {e}")
            return {
                "strategy_id": target_strategy,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    async def _generate_test_scenarios(self) -> List[dict[str, object]]:
        """Generate test scenarios for analysis."""
        try:
            # This would be enhanced with actual scenario generation
            return [
                {"volatility": 0.02, "trend": "up", "duration": 30},
                {"volatility": 0.05, "trend": "down", "duration": 15},
                {"volatility": 0.01, "trend": "sideways", "duration": 45},
            ]
        except Exception as e:
            logger.error(f"Error generating test scenarios: {e}")
            return []

    async def _get_known_vulnerabilities(self) -> List[str]:
        """Get list of known vulnerabilities."""
        try:
            return [
                "volatility_spike_vulnerability",
                "trend_reversal_blindness",
                "mean_reversion_trap",
                "overfitting_to_historical_data",
                "position_sizing_errors",
                "stop_loss_clustering",
            ]
        except Exception as e:
            logger.error(f"Error getting known vulnerabilities: {e}")
            return []

    async def _execute_attack(self, target_strategy: str, attack: object) -> AttackReportTD:
        """Execute an attack against a strategy."""
        try:
            # Simulate attack execution
            exp = getattr(
                attack,
                "expected_impact",
                (attack.get("expected_impact", 0.0) if isinstance(attack, dict) else 0.0),
            )
            try:
                success_probability = float(exp)
            except Exception:
                success_probability = 0.0
            actual_success = np.random.random() < success_probability

            attack_id = getattr(
                attack, "attack_id", (attack.get("attack_id") if isinstance(attack, dict) else None)
            ) or str(uuid.uuid4())

            if isinstance(attack, Mapping):
                agent_id = attack.get("assigned_agent_id")
                specialization_level = attack.get("specialization_level")
                params = attack.get("parameters") if isinstance(attack.get("parameters"), Mapping) else None
            else:
                agent_id = getattr(attack, "assigned_agent_id", None)
                specialization_level = getattr(attack, "specialization_level", None)
                params = getattr(attack, "parameters", None)

            intensity: object | None = None
            if isinstance(params, Mapping):
                intensity = params.get("intensity")

            return {
                "attack_id": attack_id,
                "strategy_id": target_strategy,
                "success": actual_success,
                "impact": success_probability if actual_success else 0.0,
                "timestamp": datetime.utcnow().isoformat(),
                "assigned_agent_id": agent_id,
                "specialization_level": specialization_level,
                "intensity": intensity,
            }

        except Exception as e:
            logger.error(f"Error executing attack: {e}")
            aid = (
                getattr(
                    attack,
                    "attack_id",
                    (attack.get("attack_id") if isinstance(attack, dict) else None),
                )
                or "unknown"
            )
            return {
                "attack_id": aid,
                "strategy_id": target_strategy,
                "success": False,
                "impact": 0.0,
                "error": str(e),
            }

    async def _load_persistent_agents(
        self, weaknesses: Sequence[str]
    ) -> dict[str, dict[str, object]]:
        """Load or initialize persistent adversaries for each weakness."""
        agents: dict[str, dict[str, object]] = {}
        seen: set[str] = set()
        for weakness in weaknesses:
            if not isinstance(weakness, str) or not weakness:
                continue
            if weakness in seen:
                continue
            seen.add(weakness)

            key = f"{self._agent_registry_key}:{weakness}"
            raw_agent = await self.state_store.get(key)
            agent: dict[str, object]
            is_new = False
            if raw_agent:
                try:
                    decoded = json.loads(raw_agent)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Invalid persistent red-team agent payload for %s: %s",
                        key,
                        exc,
                        exc_info=exc,
                    )
                    decoded = None
                if isinstance(decoded, dict):
                    agent = decoded
                else:
                    agent = self._create_agent_profile(weakness)
                    is_new = True
            else:
                agent = self._create_agent_profile(weakness)
                is_new = True

            agent.setdefault("weakness", weakness)
            agent.setdefault("attack_count", 0)
            agent.setdefault("success_count", 0)
            agent.setdefault("success_rate", 0.0)
            agent.setdefault("known_tactics", [])
            agent.setdefault("agent_id", str(uuid.uuid4()))
            agent.setdefault("created_at", datetime.utcnow().isoformat())
            agent.setdefault("skill_level", self._classify_agent_skill(agent))
            if is_new:
                agent["updated_at"] = agent["created_at"]
                await self._persist_agent_profile(agent)
            agents[weakness] = agent

        return agents

    def _create_agent_profile(self, weakness: str) -> dict[str, object]:
        """Create a default agent profile for a weakness."""
        now = datetime.utcnow().isoformat()
        return {
            "agent_id": str(uuid.uuid4()),
            "weakness": weakness,
            "attack_count": 0,
            "success_count": 0,
            "success_rate": 0.0,
            "skill_level": "novice",
            "known_tactics": [],
            "preferred_intensity": "medium",
            "created_at": now,
            "updated_at": now,
        }

    def _forecast_agent_skill(self, agent: Mapping[str, object]) -> str:
        """Estimate agent skill for the upcoming engagement."""
        try:
            attack_count = int(agent.get("attack_count", 0))
        except Exception:
            attack_count = 0
        try:
            success_rate = float(agent.get("success_rate", 0.0))
        except Exception:
            success_rate = 0.0

        if attack_count >= 4 and success_rate >= 0.6:
            return "elite"
        if attack_count >= 1 and success_rate >= 0.4:
            return "seasoned"
        return "novice"

    def _apply_agent_specialization(
        self, agent: dict[str, object], attack: object
    ) -> dict[str, object]:
        """Tailor an attack to leverage a persistent agent's expertise."""

        payload: dict[str, object]
        if isinstance(attack, Mapping):
            payload = dict(attack)
        else:
            payload = _to_mapping(attack)

        parameters = payload.get("parameters")
        if isinstance(parameters, Mapping):
            params = dict(parameters)
        else:
            params = {}

        skill = self._forecast_agent_skill(agent)
        base_intensity = str(params.get("intensity", "medium"))
        calibrated_intensity = self._calibrate_intensity(base_intensity, skill)

        params["intensity"] = calibrated_intensity
        params["assigned_agent_id"] = agent.get("agent_id")
        params["specialization_level"] = skill
        params.setdefault("attack_vector", payload.get("attack_type", params.get("attack_vector")))
        params["focus_weakness"] = agent.get("weakness")

        payload["parameters"] = params
        payload["assigned_agent_id"] = agent.get("agent_id")
        payload["specialization_level"] = skill
        payload.setdefault("weakness_targeted", agent.get("weakness"))

        agent["skill_level"] = skill
        agent["preferred_intensity"] = calibrated_intensity

        return payload

    def _calibrate_intensity(self, current: str, skill: str) -> str:
        """Raise attack intensity based on agent skill while preserving intent."""
        order = ["low", "medium", "high", "extreme"]
        try:
            current_idx = order.index(current)
        except ValueError:
            current_idx = 1

        target_idx = {
            "novice": 1,
            "seasoned": 2,
            "elite": 3,
        }.get(skill, 1)

        final_idx = max(current_idx, target_idx)
        return order[final_idx]

    def _classify_agent_skill(self, agent: Mapping[str, object]) -> str:
        """Classify an agent's long-term proficiency."""
        try:
            attack_count = int(agent.get("attack_count", 0))
        except Exception:
            attack_count = 0
        try:
            success_rate = float(agent.get("success_rate", 0.0))
        except Exception:
            success_rate = 0.0

        if attack_count >= 5 and success_rate >= 0.6:
            return "elite"
        if attack_count >= 1 and success_rate >= 0.5:
            return "seasoned"
        return "novice"

    async def _persist_agent_profile(self, agent: Mapping[str, object]) -> None:
        """Persist agent metadata with a generous TTL."""
        weakness = agent.get("weakness")
        if not isinstance(weakness, str) or not weakness:
            return

        try:
            payload = json.dumps(agent, sort_keys=True, separators=(",", ":"), default=str)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "Failed to encode persistent red-team agent for %s: %s",
                weakness,
                exc,
                exc_info=exc,
            )
            return

        await self.state_store.set(
            f"{self._agent_registry_key}:{weakness}",
            payload,
            expire=86400 * 90,
        )

    async def _update_persistent_agents(
        self,
        agent_map: Mapping[str, dict[str, object]],
        attacks: Sequence[object],
        attack_results: Sequence[Mapping[str, object]],
    ) -> None:
        """Update persistent agent records after executing attacks."""
        if not agent_map:
            return

        attack_lookup: dict[str, Mapping[str, object]] = {}
        for attack in attacks:
            if isinstance(attack, Mapping):
                payload = attack
            else:
                payload = _to_mapping(attack)
            aid = payload.get("attack_id")
            if isinstance(aid, str) and aid:
                attack_lookup[aid] = payload

        for result in attack_results:
            if not isinstance(result, Mapping):
                continue
            attack_id = result.get("attack_id")
            if not isinstance(attack_id, str) or attack_id not in attack_lookup:
                continue

            attack_payload = attack_lookup[attack_id]
            weakness = attack_payload.get("weakness_targeted")
            if not isinstance(weakness, str):
                continue

            agent = agent_map.get(weakness)
            if agent is None:
                continue

            success = bool(result.get("success", False))
            total_attacks = int(agent.get("attack_count", 0)) + 1
            success_count = int(agent.get("success_count", 0)) + (1 if success else 0)
            success_rate = float(success_count / total_attacks) if total_attacks else 0.0

            agent["attack_count"] = total_attacks
            agent["success_count"] = success_count
            agent["success_rate"] = success_rate
            agent["last_attack"] = datetime.utcnow().isoformat()

            params = attack_payload.get("parameters")
            if isinstance(params, Mapping):
                maybe_intensity = params.get("intensity")
                if isinstance(maybe_intensity, str):
                    agent["preferred_intensity"] = maybe_intensity

            tactic = attack_payload.get("attack_type")
            if not isinstance(tactic, str):
                if isinstance(params, Mapping):
                    maybe = params.get("attack_vector")
                    if isinstance(maybe, str):
                        tactic = maybe
            if isinstance(tactic, str):
                known_raw = agent.get("known_tactics", [])
                known = list(known_raw) if isinstance(known_raw, list) else []
                if tactic not in known:
                    known.append(tactic)
                    if len(known) > 5:
                        known = known[-5:]
                agent["known_tactics"] = known

            agent["skill_level"] = self._classify_agent_skill(agent)
            agent.setdefault("agent_id", str(uuid.uuid4()))
            agent.setdefault("created_at", datetime.utcnow().isoformat())
            agent["updated_at"] = datetime.utcnow().isoformat()

            try:
                await self._persist_agent_profile(agent)
            except Exception as exc:  # pragma: no cover - defensive persistence guard
                logger.error(
                    "Failed to persist red-team agent %s: %s",
                    agent.get("agent_id"),
                    exc,
                    exc_info=exc,
                )

    async def _store_attack_results(
        self,
        strategy_id: str,
        behavior_profile: object,
        weaknesses: List[str],
        attacks: list[object],
        exploits: list[object],
        attack_results: List[AttackReportTD],
    ) -> None:
        """Store attack results for analysis."""
        try:
            # Store attack history
            attack_record = {
                "strategy_id": strategy_id,
                "timestamp": datetime.utcnow().isoformat(),
                "weaknesses": weaknesses,
                "attacks_count": len(attacks),
                "exploits_count": len(exploits),
                "successful_attacks": sum(
                    1 for r in attack_results if isinstance(r, dict) and r.get("success", False)
                ),
            }

            key = f"{self._attack_history_key}:{strategy_id}:{datetime.utcnow().date()}"
            payload = json.dumps(attack_record, sort_keys=True, separators=(",", ":"))
            await self.state_store.set(key, payload, expire=86400 * 30)  # 30 days

            # Store exploit history
            for exploit in exploits:
                eid = (
                    getattr(
                        exploit,
                        "exploit_id",
                        (exploit.get("exploit_id") if isinstance(exploit, dict) else None),
                    )
                    or "unknown"
                )
                key = f"{self._exploit_history_key}:{strategy_id}:{eid}"
                payload = json.dumps(_to_mapping(exploit), sort_keys=True, separators=(",", ":"))
                await self.state_store.set(key, payload, expire=86400 * 30)  # 30 days

        except Exception as e:
            logger.error(f"Error storing attack results: {e}")

    async def _generate_improvements(
        self, weaknesses: List[str], attack_results: Sequence[Mapping[str, object]]
    ) -> List[str]:
        """Generate improvement recommendations."""
        try:
            recommendations = []

            for weakness in weaknesses:
                if weakness == "volatility_spike_vulnerability":
                    recommendations.append("Implement volatility filtering")
                elif weakness == "trend_reversal_blindness":
                    recommendations.append("Add trend reversal detection")
                elif weakness == "mean_reversion_trap":
                    recommendations.append("Improve mean reversion validation")
                elif weakness == "overfitting_to_historical_data":
                    recommendations.append("Increase out-of-sample testing")
                elif weakness == "position_sizing_errors":
                    recommendations.append("Implement dynamic position sizing")
                elif weakness == "stop_loss_clustering":
                    recommendations.append("Use adaptive stop losses")

            return recommendations

        except Exception as e:
            logger.error(f"Error generating improvements: {e}")
            return []

    async def get_red_team_stats(self) -> dict[str, object]:
        """Get Red Team AI statistics."""
        try:
            keys = await self.state_store.keys(f"{self._attack_history_key}:*")
            agent_keys = await self.state_store.keys(f"{self._agent_registry_key}:*")

            total_attacks = 0
            successful_attacks = 0
            total_weaknesses = 0

            for key in keys:
                data = await self.state_store.get(key)
                if data:
                    try:
                        record = json.loads(data)
                    except json.JSONDecodeError as exc:
                        logger.warning("Discarding invalid red-team payload for %s", key, exc_info=exc)
                        continue

                    if not isinstance(record, dict):
                        logger.warning(
                            "Red-team payload must be a JSON object for %s; got %s",
                            key,
                            type(record),
                        )
                        continue

                    total_attacks += int(record.get("attacks_count", 0))
                    successful_attacks += int(record.get("successful_attacks", 0))
                    weaknesses = record.get("weaknesses", [])
                    if isinstance(weaknesses, list):
                        total_weaknesses += len(weaknesses)

            skill_distribution = {"novice": 0, "seasoned": 0, "elite": 0}
            for agent_key in agent_keys:
                payload = await self.state_store.get(agent_key)
                if not payload:
                    continue
                try:
                    agent_record = json.loads(payload)
                except json.JSONDecodeError as exc:
                    logger.debug(
                        "Skipping malformed agent record %s: %s",
                        agent_key,
                        exc,
                        exc_info=exc,
                    )
                    continue
                if not isinstance(agent_record, dict):
                    continue
                skill = agent_record.get("skill_level", "novice")
                if isinstance(skill, str) and skill in skill_distribution:
                    skill_distribution[skill] += 1

            return {
                "total_strategies_attacked": len(keys),
                "total_attacks": total_attacks,
                "successful_attacks": successful_attacks,
                "total_weaknesses_found": total_weaknesses,
                "success_rate": successful_attacks / total_attacks if total_attacks > 0 else 0,
                "dedicated_agents": len(agent_keys),
                "agent_skill_distribution": skill_distribution,
                "last_attack": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting red team stats: {e}")
            return {
                "total_strategies_attacked": 0,
                "total_attacks": 0,
                "successful_attacks": 0,
                "total_weaknesses_found": 0,
                "success_rate": 0,
                "dedicated_agents": 0,
                "agent_skill_distribution": {"novice": 0, "seasoned": 0, "elite": 0},
                "last_attack": None,
            }
