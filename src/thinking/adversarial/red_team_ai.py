"""
Red Team AI System
Dedicated AI system to attack and improve strategies.
"""

import logging
import uuid
from ast import literal_eval
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from src.core.events import AttackResult, ExploitResult, StrategyAnalysis  # legacy
except Exception:  # pragma: no cover
    StrategyAnalysis = AttackResult = ExploitResult = object  # type: ignore
from src.core.state_store import StateStore

logger = logging.getLogger(__name__)


class StrategyAnalyzer:
    """Deep analysis of strategy behavior patterns."""
    
    def __init__(self):
        self.analysis_metrics = [
            'volatility_sensitivity',
            'trend_following_strength',
            'mean_reversion_tendency',
            'risk_tolerance',
            'position_sizing_behavior'
        ]
    
    async def analyze_behavior(
        self,
        target_strategy: str,
        test_scenarios: List[Dict[str, Any]]
    ) -> StrategyAnalysis:
        """Analyze strategy behavior patterns."""
        try:
            # Simulate strategy behavior across scenarios
            behavior_data = []
            
            for scenario in test_scenarios:
                behavior = await self._simulate_strategy_behavior(
                    target_strategy,
                    scenario
                )
                behavior_data.append(behavior)
            
            # Calculate behavior metrics
            metrics = self._calculate_behavior_metrics(behavior_data)
            
            # Create behavior profile
            analysis = StrategyAnalysis(
                strategy_id=target_strategy,
                timestamp=datetime.utcnow(),
                behavior_profile=metrics,
                risk_factors=self._identify_risk_factors(metrics),
                performance_patterns=self._identify_performance_patterns(behavior_data),
                metadata={'analysis_type': 'comprehensive'}
            )
            
            logger.debug(
                f"Analyzed strategy {target_strategy}: "
                f"{len(metrics)} metrics calculated"
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing strategy behavior: {e}")
            return StrategyAnalysis(
                strategy_id=target_strategy,
                timestamp=datetime.utcnow(),
                behavior_profile={},
                risk_factors=[],
                performance_patterns=[],
                metadata={'error': str(e)}
            )
    
    async def _simulate_strategy_behavior(
        self,
        strategy_id: str,
        scenario: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate strategy behavior in a scenario."""
        try:
            # This would be enhanced with actual strategy simulation
            return {
                'volatility_sensitivity': np.random.normal(0.5, 0.2),
                'trend_following_strength': np.random.normal(0.7, 0.15),
                'mean_reversion_tendency': np.random.normal(0.3, 0.1),
                'risk_tolerance': np.random.normal(0.6, 0.2),
                'position_sizing_behavior': np.random.normal(0.4, 0.1)
            }
            
        except Exception as e:
            logger.error(f"Error simulating strategy behavior: {e}")
            return {}
    
    def _calculate_behavior_metrics(
        self,
        behavior_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate aggregate behavior metrics."""
        try:
            if not behavior_data:
                return {}
            
            metrics = {}
            for metric in self.analysis_metrics:
                values = [b.get(metric, 0) for b in behavior_data]
                metrics[metric] = np.mean(values)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating behavior metrics: {e}")
            return {}
    
    def _identify_risk_factors(
        self,
        metrics: Dict[str, float]
    ) -> List[str]:
        """Identify risk factors from behavior metrics."""
        try:
            risk_factors = []
            
            if metrics.get('volatility_sensitivity', 0) > 0.8:
                risk_factors.append('high_volatility_sensitivity')
            
            if metrics.get('risk_tolerance', 0) > 0.9:
                risk_factors.append('excessive_risk_taking')
            
            if metrics.get('trend_following_strength', 0) < 0.3:
                risk_factors.append('weak_trend_following')
            
            return risk_factors
            
        except Exception as e:
            logger.error(f"Error identifying risk factors: {e}")
            return []
    
    def _identify_performance_patterns(
        self,
        behavior_data: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify performance patterns from behavior data."""
        try:
            patterns = []
            
            # Simple pattern detection
            if len(behavior_data) > 5:
                patterns.append('sufficient_data')
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error identifying performance patterns: {e}")
            return []


class WeaknessDetector:
    """Identifies potential weaknesses in strategies."""
    
    def __init__(self):
        self.known_vulnerabilities = [
            'volatility_spike_vulnerability',
            'trend_reversal_blindness',
            'mean_reversion_trap',
            'overfitting_to_historical_data',
            'position_sizing_errors',
            'stop_loss_clustering'
        ]
    
    async def find_weaknesses(
        self,
        behavior_profile: Dict[str, Any],
        known_vulnerabilities: List[str]
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
            
            logger.debug(
                f"Found {len(weaknesses)} weaknesses: "
                f"{', '.join(weaknesses)}"
            )
            
            return weaknesses
            
        except Exception as e:
            logger.error(f"Error finding weaknesses: {e}")
            return []
    
    def _check_vulnerability(
        self,
        behavior_profile: Dict[str, Any],
        vulnerability: str
    ) -> bool:
        """Check if strategy has specific vulnerability."""
        try:
            # Simple vulnerability checks
            if vulnerability == 'volatility_spike_vulnerability':
                return behavior_profile.get('volatility_sensitivity', 0) > 0.8
            
            elif vulnerability == 'trend_reversal_blindness':
                return behavior_profile.get('trend_following_strength', 0) > 0.9
            
            elif vulnerability == 'mean_reversion_trap':
                return behavior_profile.get('mean_reversion_tendency', 0) > 0.8
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking vulnerability: {e}")
            return False
    
    def _detect_new_weaknesses(
        self,
        behavior_profile: Dict[str, Any]
    ) -> List[str]:
        """Detect new weaknesses not in known list."""
        try:
            new_weaknesses = []
            
            # Check for extreme values
            for metric, value in behavior_profile.items():
                if abs(value) > 2.0:  # Extreme value threshold
                    new_weaknesses.append(f"extreme_{metric}")
            
            return new_weaknesses
            
        except Exception as e:
            logger.error(f"Error detecting new weaknesses: {e}")
            return []


class AttackGenerator:
    """Generates targeted attacks for discovered weaknesses."""
    
    def __init__(self):
        self.attack_templates = {
            'volatility_spike_vulnerability': {
                'attack_type': 'volatility_spike',
                'intensity': 'high',
                'duration': 'short'
            },
            'trend_reversal_blindness': {
                'attack_type': 'trend_reversal',
                'intensity': 'medium',
                'duration': 'medium'
            },
            'mean_reversion_trap': {
                'attack_type': 'false_mean_reversion',
                'intensity': 'medium',
                'duration': 'long'
            }
        }
    
    async def create_attack(
        self,
        weakness: str,
        target_strategy: str
    ) -> AttackResult:
        """Create a targeted attack for a weakness."""
        try:
            # Get attack template
            template = self.attack_templates.get(weakness, {
                'attack_type': 'generic',
                'intensity': 'low',
                'duration': 'short'
            })
            
            # Generate attack parameters
            attack_params = self._generate_attack_parameters(
                template,
                target_strategy
            )
            
            attack = AttackResult(
                attack_id=str(uuid.uuid4()),
                strategy_id=target_strategy,
                weakness_targeted=weakness,
                attack_type=template['attack_type'],
                parameters=attack_params,
                expected_impact=self._calculate_expected_impact(weakness),
                timestamp=datetime.utcnow()
            )
            
            logger.debug(
                f"Created attack {attack.attack_id} "
                f"targeting {weakness} in {target_strategy}"
            )
            
            return attack
            
        except Exception as e:
            logger.error(f"Error creating attack: {e}")
            return AttackResult(
                attack_id=str(uuid.uuid4()),
                strategy_id=target_strategy,
                weakness_targeted=weakness,
                attack_type='error',
                parameters={},
                expected_impact=0.0,
                timestamp=datetime.utcnow()
            )
    
    def _generate_attack_parameters(
        self,
        template: Dict[str, Any],
        target_strategy: str
    ) -> Dict[str, Any]:
        """Generate attack parameters."""
        try:
            return {
                'intensity': template['intensity'],
                'duration': template['duration'],
                'target_strategy': target_strategy,
                'attack_vector': template['attack_type']
            }
            
        except Exception as e:
            logger.error(f"Error generating attack parameters: {e}")
            return {}
    
    def _calculate_expected_impact(self, weakness: str) -> float:
        """Calculate expected impact of attack."""
        try:
            # Simple impact calculation
            impact_map = {
                'volatility_spike_vulnerability': 0.8,
                'trend_reversal_blindness': 0.6,
                'mean_reversion_trap': 0.7,
                'overfitting_to_historical_data': 0.9,
                'position_sizing_errors': 0.5,
                'stop_loss_clustering': 0.4
            }
            
            return impact_map.get(weakness, 0.3)
            
        except Exception as e:
            logger.error(f"Error calculating expected impact: {e}")
            return 0.3


class ExploitDeveloper:
    """Develops exploits for discovered weaknesses."""
    
    def __init__(self):
        self.exploit_templates = {
            'volatility_spike_vulnerability': {
                'exploit_type': 'volatility_manipulation',
                'severity': 'high',
                'complexity': 'medium'
            },
            'trend_reversal_blindness': {
                'exploit_type': 'trend_deception',
                'severity': 'medium',
                'complexity': 'high'
            },
            'mean_reversion_trap': {
                'exploit_type': 'false_reversion',
                'severity': 'medium',
                'complexity': 'medium'
            }
        }
    
    async def develop_exploits(
        self,
        weaknesses: List[str],
        target_strategy: str
    ) -> List[ExploitResult]:
        """Develop exploits for discovered weaknesses."""
        try:
            exploits = []
            
            for weakness in weaknesses:
                exploit = await self._create_exploit(weakness, target_strategy)
                if exploit:
                    exploits.append(exploit)
            
            logger.debug(
                f"Developed {len(exploits)} exploits "
                f"for strategy {target_strategy}"
            )
            
            return exploits
            
        except Exception as e:
            logger.error(f"Error developing exploits: {e}")
            return []
    
    async def _create_exploit(
        self,
        weakness: str,
        target_strategy: str
    ) -> Optional[ExploitResult]:
        """Create a specific exploit for a weakness."""
        try:
            template = self.exploit_templates.get(weakness, {
                'exploit_type': 'generic',
                'severity': 'low',
                'complexity': 'low'
            })
            
            exploit = ExploitResult(
                exploit_id=str(uuid.uuid4()),
                strategy_id=target_strategy,
                weakness_exploited=weakness,
                exploit_type=template['exploit_type'],
                severity=template['severity'],
                complexity=template['complexity'],
                parameters=self._generate_exploit_parameters(weakness),
                timestamp=datetime.utcnow()
            )
            
            return exploit
            
        except Exception as e:
            logger.error(f"Error creating exploit: {e}")
            return None
    
    def _generate_exploit_parameters(self, weakness: str) -> Dict[str, Any]:
        """Generate exploit parameters."""
        try:
            return {
                'weakness': weakness,
                'attack_vector': f"exploit_{weakness}",
                'severity_level': 'high',
                'complexity_level': 'medium'
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
    
    def __init__(self, state_store: StateStore):
        self.state_store = state_store
        self.strategy_analyzer = StrategyAnalyzer()
        self.weakness_detector = WeaknessDetector()
        self.attack_generator = AttackGenerator()
        self.exploit_developer = ExploitDeveloper()
        
        self._attack_history_key = "emp:red_team_attacks"
        self._exploit_history_key = "emp:red_team_exploits"
        
    async def initialize(self) -> bool:
        return True

    async def stop(self) -> bool:
        return True
        

    async def attack_strategy(
        self,
        target_strategy: str
    ) -> Dict[str, Any]:
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
                target_strategy,
                test_scenarios
            )
            
            # Step 2: Identify weaknesses
            known_vulnerabilities = await self._get_known_vulnerabilities()
            weaknesses = await self.weakness_detector.find_weaknesses(
                behavior_profile.behavior_profile,
                known_vulnerabilities
            )
            
            # Step 3: Generate attacks
            attacks = []
            for weakness in weaknesses:
                attack = await self.attack_generator.create_attack(
                    weakness,
                    target_strategy
                )
                attacks.append(attack)
            
            # Step 4: Develop exploits
            exploits = await self.exploit_developer.develop_exploits(
                weaknesses,
                target_strategy
            )
            
            # Step 5: Execute attacks
            attack_results = []
            for attack in attacks:
                result = await self._execute_attack(target_strategy, attack)
                attack_results.append(result)
            
            # Step 6: Store results
            await self._store_attack_results(
                target_strategy,
                behavior_profile,
                weaknesses,
                attacks,
                exploits,
                attack_results
            )
            
            # Step 7: Generate improvement recommendations
            recommendations = await self._generate_improvements(
                weaknesses,
                attack_results
            )
            
            report = {
                'strategy_id': target_strategy,
                'behavior_analysis': behavior_profile.dict(),
                'weaknesses_found': weaknesses,
                'attacks_generated': [a.dict() for a in attacks],
                'exploits_developed': [e.dict() for e in exploits],
                'attack_results': attack_results,
                'improvements': recommendations,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info(
                f"Red team attack complete on {target_strategy}: "
                f"{len(weaknesses)} weaknesses found, "
                f"{len(attacks)} attacks generated"
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error in red team attack: {e}")
            return {
                'strategy_id': target_strategy,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def _generate_test_scenarios(self) -> List[Dict[str, Any]]:
        """Generate test scenarios for analysis."""
        try:
            # This would be enhanced with actual scenario generation
            return [
                {'volatility': 0.02, 'trend': 'up', 'duration': 30},
                {'volatility': 0.05, 'trend': 'down', 'duration': 15},
                {'volatility': 0.01, 'trend': 'sideways', 'duration': 45}
            ]
        except Exception as e:
            logger.error(f"Error generating test scenarios: {e}")
            return []
    
    async def _get_known_vulnerabilities(self) -> List[str]:
        """Get list of known vulnerabilities."""
        try:
            return [
                'volatility_spike_vulnerability',
                'trend_reversal_blindness',
                'mean_reversion_trap',
                'overfitting_to_historical_data',
                'position_sizing_errors',
                'stop_loss_clustering'
            ]
        except Exception as e:
            logger.error(f"Error getting known vulnerabilities: {e}")
            return []
    
    async def _execute_attack(
        self,
        target_strategy: str,
        attack: AttackResult
    ) -> Dict[str, Any]:
        """Execute an attack against a strategy."""
        try:
            # Simulate attack execution
            success_probability = attack.expected_impact
            actual_success = np.random.random() < success_probability
            
            return {
                'attack_id': attack.attack_id,
                'strategy_id': target_strategy,
                'success': actual_success,
                'impact': attack.expected_impact if actual_success else 0.0,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error executing attack: {e}")
            return {
                'attack_id': attack.attack_id,
                'strategy_id': target_strategy,
                'success': False,
                'impact': 0.0,
                'error': str(e)
            }
    
    async def _store_attack_results(
        self,
        strategy_id: str,
        behavior_profile: StrategyAnalysis,
        weaknesses: List[str],
        attacks: List[AttackResult],
        exploits: List[ExploitResult],
        attack_results: List[Dict[str, Any]]
    ) -> None:
        """Store attack results for analysis."""
        try:
            # Store attack history
            attack_record = {
                'strategy_id': strategy_id,
                'timestamp': datetime.utcnow().isoformat(),
                'weaknesses': weaknesses,
                'attacks_count': len(attacks),
                'exploits_count': len(exploits),
                'successful_attacks': len([r for r in attack_results if r.get('success', False)])
            }
            
            key = f"{self._attack_history_key}:{strategy_id}:{datetime.utcnow().date()}"
            await self.state_store.set(
                key,
                str(attack_record),
                expire=86400 * 30  # 30 days
            )
            
            # Store exploit history
            for exploit in exploits:
                key = f"{self._exploit_history_key}:{strategy_id}:{exploit.exploit_id}"
                await self.state_store.set(
                    key,
                    str(exploit.dict()),
                    expire=86400 * 30  # 30 days
                )
                
        except Exception as e:
            logger.error(f"Error storing attack results: {e}")
    
    async def _generate_improvements(
        self,
        weaknesses: List[str],
        attack_results: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate improvement recommendations."""
        try:
            recommendations = []
            
            for weakness in weaknesses:
                if weakness == 'volatility_spike_vulnerability':
                    recommendations.append('Implement volatility filtering')
                elif weakness == 'trend_reversal_blindness':
                    recommendations.append('Add trend reversal detection')
                elif weakness == 'mean_reversion_trap':
                    recommendations.append('Improve mean reversion validation')
                elif weakness == 'overfitting_to_historical_data':
                    recommendations.append('Increase out-of-sample testing')
                elif weakness == 'position_sizing_errors':
                    recommendations.append('Implement dynamic position sizing')
                elif weakness == 'stop_loss_clustering':
                    recommendations.append('Use adaptive stop losses')
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating improvements: {e}")
            return []
    
    async def get_red_team_stats(self) -> Dict[str, Any]:
        """Get Red Team AI statistics."""
        try:
            keys = await self.state_store.keys(f"{self._attack_history_key}:*")
            
            total_attacks = 0
            successful_attacks = 0
            total_weaknesses = 0
            
            for key in keys:
                data = await self.state_store.get(key)
                if data:
                    # Bandit B307: replaced eval with safe parsing
                    try:
                        record = literal_eval(data)
                    except (ValueError, SyntaxError):
                        record = {}
                    total_attacks += record.get('attacks_count', 0)
                    successful_attacks += record.get('successful_attacks', 0)
                    total_weaknesses += len(record.get('weaknesses', []))
            
            return {
                'total_strategies_attacked': len(keys),
                'total_attacks': total_attacks,
                'successful_attacks': successful_attacks,
                'total_weaknesses_found': total_weaknesses,
                'success_rate': successful_attacks / total_attacks if total_attacks > 0 else 0,
                'last_attack': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting red team stats: {e}")
            return {
                'total_strategies_attacked': 0,
                'total_attacks': 0,
                'successful_attacks': 0,
                'total_weaknesses_found': 0,
                'success_rate': 0,
                'last_attack': None
            }
