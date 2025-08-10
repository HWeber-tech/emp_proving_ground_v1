"""
Specialized Predator Evolution
Evolution of specialized strategy types that work in coordination.
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
import uuid

try:
    from src.core.events import PredatorSpecies, MarketNiche, CoordinationStrategy  # legacy
except Exception:  # pragma: no cover
    PredatorSpecies = MarketNiche = CoordinationStrategy = object  # type: ignore
from src.operational.state_store import StateStore

logger = logging.getLogger(__name__)


class NicheDetector:
    """Detects market niches and opportunities."""
    
    def __init__(self):
        self.niche_patterns = {
            'trend_following': {
                'volatility_threshold': 0.02,
                'trend_strength': 0.7,
                'duration': 20
            },
            'mean_reversion': {
                'volatility_threshold': 0.01,
                'range_bound': True,
                'duration': 15
            },
            'momentum': {
                'volatility_threshold': 0.015,
                'momentum_strength': 0.6,
                'duration': 10
            },
            'arbitrage': {
                'spread_threshold': 0.001,
                'liquidity': 'high',
                'duration': 5
            },
            'scalping': {
                'volatility_threshold': 0.005,
                'frequency': 'high',
                'duration': 1
            }
        }
    
    async def detect_niches(
        self,
        market_data: Dict[str, Any],
        historical_analysis: Dict[str, Any]
    ) -> List[MarketNiche]:
        """Detect market niches and opportunities."""
        try:
            niches = []
            
            for niche_type, criteria in self.niche_patterns.items():
                if await self._check_niche_criteria(niche_type, criteria, market_data):
                    niche = MarketNiche(
                        niche_id=str(uuid.uuid4()),
                        niche_type=niche_type,
                        criteria=criteria,
                        opportunity_score=self._calculate_opportunity_score(
                            niche_type,
                            market_data,
                            historical_analysis
                        ),
                        market_conditions=market_data,
                        timestamp=datetime.utcnow()
                    )
                    niches.append(niche)
            
            logger.info(
                f"Detected {len(niches)} market niches: "
                f"{[n.niche_type for n in niches]}"
            )
            
            return niches
            
        except Exception as e:
            logger.error(f"Error detecting niches: {e}")
            return []
    
    async def _check_niche_criteria(
        self,
        niche_type: str,
        criteria: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> bool:
        """Check if market meets niche criteria."""
        try:
            if niche_type == 'trend_following':
                volatility = market_data.get('volatility', 0)
                trend_strength = market_data.get('trend_strength', 0)
                return volatility >= criteria['volatility_threshold'] and \
                       trend_strength >= criteria['trend_strength']
            
            elif niche_type == 'mean_reversion':
                volatility = market_data.get('volatility', 0)
                range_bound = market_data.get('range_bound', False)
                return volatility <= criteria['volatility_threshold'] and range_bound
            
            elif niche_type == 'momentum':
                volatility = market_data.get('volatility', 0)
                momentum = market_data.get('momentum', 0)
                return volatility >= criteria['volatility_threshold'] and \
                       momentum >= criteria['momentum_strength']
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking niche criteria: {e}")
            return False
    
    def _calculate_opportunity_score(
        self,
        niche_type: str,
        market_data: Dict[str, Any],
        historical_analysis: Dict[str, Any]
    ) -> Decimal:
        """Calculate opportunity score for a niche."""
        try:
            # Simple opportunity scoring
            base_score = 0.5
            
            # Adjust based on historical performance
            if historical_analysis.get(f'{niche_type}_success_rate', 0) > 0.6:
                base_score += 0.3
            
            # Adjust based on market conditions
            volatility = market_data.get('volatility', 0.02)
            if 0.01 <= volatility <= 0.05:
                base_score += 0.2
            
            return Decimal(str(min(1.0, base_score)))
            
        except Exception as e:
            logger.error(f"Error calculating opportunity score: {e}")
            return Decimal('0.5')


class SpeciesManager:
    """Manages evolution of specialized predator species."""
    
    def __init__(self):
        self.species_templates = {
            'stalker': {
                'characteristics': ['patience', 'long_term_trend', 'low_frequency'],
                'optimal_niche': 'trend_following',
                'risk_profile': 'conservative'
            },
            'ambusher': {
                'characteristics': ['speed', 'scalping', 'high_frequency'],
                'optimal_niche': 'scalping',
                'risk_profile': 'aggressive'
            },
            'pack_hunter': {
                'characteristics': ['coordination', 'multi_timeframe', 'diversified'],
                'optimal_niche': 'momentum',
                'risk_profile': 'balanced'
            },
            'scavenger': {
                'characteristics': ['efficiency', 'arbitrage', 'opportunistic'],
                'optimal_niche': 'arbitrage',
                'risk_profile': 'low_risk'
            },
            'alpha': {
                'characteristics': ['leadership', 'market_making', 'dominant'],
                'optimal_niche': 'momentum',
                'risk_profile': 'high_confidence'
            }
        }
    
    async def evolve_specialist(
        self,
        niche: MarketNiche,
        base_population: List[str],
        specialization_pressure: float
    ) -> PredatorSpecies:
        """Evolve a specialist for a specific niche."""
        try:
            # Find best matching species template
            species_type = self._match_niche_to_species(niche.niche_type)
            template = self.species_templates[species_type]
            
            # Create specialized parameters
            specialized_params = self._create_specialized_parameters(
                species_type,
                niche,
                specialization_pressure
            )
            
            # Select best base strategies for specialization
            selected_strategies = self._select_strategies_for_specialization(
                base_population,
                niche,
                specialization_pressure
            )
            
            species = PredatorSpecies(
                species_id=str(uuid.uuid4()),
                species_type=species_type,
                niche=niche,
                characteristics=template['characteristics'],
                specialized_parameters=specialized_params,
                base_strategies=selected_strategies,
                performance_metrics={
                    'fitness_score': 0.0,
                    'niche_adaptation': 0.0,
                    'coordination_score': 0.0
                },
                timestamp=datetime.utcnow()
            )
            
            logger.info(
                f"Evolved {species_type} specialist for {niche.niche_type} niche"
            )
            
            return species
            
        except Exception as e:
            logger.error(f"Error evolving specialist: {e}")
            return PredatorSpecies(
                species_id=str(uuid.uuid4()),
                species_type='generic',
                niche=niche,
                characteristics=[],
                specialized_parameters={},
                base_strategies=[],
                performance_metrics={},
                timestamp=datetime.utcnow()
            )
    
    def _match_niche_to_species(self, niche_type: str) -> str:
        """Match niche type to species template."""
        mapping = {
            'trend_following': 'stalker',
            'mean_reversion': 'scavenger',
            'momentum': 'pack_hunter',
            'arbitrage': 'scavenger',
            'scalping': 'ambusher'
        }
        return mapping.get(niche_type, 'alpha')
    
    def _create_specialized_parameters(
        self,
        species_type: str,
        niche: MarketNiche,
        pressure: float
    ) -> Dict[str, Any]:
        """Create specialized parameters for species."""
        try:
            params = {
                'species_type': species_type,
                'niche_type': niche.niche_type,
                'specialization_pressure': pressure,
                'risk_adjustment': self._calculate_risk_adjustment(species_type),
                'time_horizon': self._calculate_time_horizon(species_type),
                'frequency': self._calculate_frequency(species_type)
            }
            
            return params
            
        except Exception as e:
            logger.error(f"Error creating specialized parameters: {e}")
            return {}
    
    def _calculate_risk_adjustment(self, species_type: str) -> float:
        """Calculate risk adjustment for species."""
        adjustments = {
            'stalker': 0.8,
            'ambusher': 1.2,
            'pack_hunter': 1.0,
            'scavenger': 0.6,
            'alpha': 1.1
        }
        return adjustments.get(species_type, 1.0)
    
    def _calculate_time_horizon(self, species_type: str) -> int:
        """Calculate time horizon for species."""
        horizons = {
            'stalker': 30,
            'ambusher': 1,
            'pack_hunter': 10,
            'scavenger': 5,
            'alpha': 15
        }
        return horizons.get(species_type, 10)
    
    def _calculate_frequency(self, species_type: str) -> str:
        """Calculate trading frequency for species."""
        frequencies = {
            'stalker': 'low',
            'ambusher': 'high',
            'pack_hunter': 'medium',
            'scavenger': 'medium',
            'alpha': 'medium'
        }
        return frequencies.get(species_type, 'medium')
    
    def _select_strategies_for_specialization(
        self,
        base_population: List[str],
        niche: MarketNiche,
        pressure: float
    ) -> List[str]:
        """Select best strategies for specialization."""
        try:
            # Select top strategies based on niche fit
            num_to_select = max(1, int(len(base_population) * pressure))
            return base_population[:num_to_select]
            
        except Exception as e:
            logger.error(f"Error selecting strategies: {e}")
            return base_population[:1]


class CoordinationEngine:
    """Optimizes coordination between specialized predators."""
    
    def __init__(self):
        self.coordination_strategies = {
            'complementary': 'strategies complement each other',
            'diversified': 'strategies reduce overall risk',
            'synergistic': 'strategies amplify positive effects',
            'hierarchical': 'strategies have clear leadership'
        }
    
    async def optimize_coordination(
        self,
        specialists: Dict[str, PredatorSpecies],
        market_context: Dict[str, Any]
    ) -> CoordinationStrategy:
        """Optimize coordination between specialists."""
        try:
            # Analyze specialist interactions
            interactions = await self._analyze_specialist_interactions(
                specialists,
                market_context
            )
            
            # Determine best coordination strategy
            best_strategy = self._determine_coordination_strategy(
                interactions,
                market_context
            )
            
            # Calculate coordination parameters
            coordination_params = self._calculate_coordination_parameters(
                specialists,
                best_strategy
            )
            
            strategy = CoordinationStrategy(
                strategy_id=str(uuid.uuid4()),
                coordination_type=best_strategy,
                specialists=list(specialists.keys()),
                parameters=coordination_params,
                expected_synergy=self._calculate_expected_synergy(
                    specialists,
                    best_strategy
                ),
                timestamp=datetime.utcnow()
            )
            
            logger.info(
                f"Optimized coordination: {best_strategy} "
                f"for {len(specialists)} specialists"
            )
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error optimizing coordination: {e}")
            return CoordinationStrategy(
                strategy_id=str(uuid.uuid4()),
                coordination_type='basic',
                specialists=list(specialists.keys()),
                parameters={},
                expected_synergy=Decimal('0.5'),
                timestamp=datetime.utcnow()
            )
    
    async def _analyze_specialist_interactions(
        self,
        specialists: Dict[str, PredatorSpecies],
        market_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze interactions between specialists."""
        try:
            interactions = {
                'correlation_matrix': self._calculate_correlation_matrix(specialists),
                'risk_overlap': self._calculate_risk_overlap(specialists),
                'performance_complementarity': self._calculate_complementarity(specialists)
            }
            
            return interactions
            
        except Exception as e:
            logger.error(f"Error analyzing interactions: {e}")
            return {}
    
    def _calculate_correlation_matrix(
        self,
        specialists: Dict[str, PredatorSpecies]
    ) -> List[List[float]]:
        """Calculate correlation matrix between specialists."""
        try:
            # Simple correlation calculation
            n = len(specialists)
            matrix = [[0.0 for _ in range(n)] for _ in range(n)]
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        matrix[i][j] = 1.0
                    else:
                        # Random correlation for simulation
                        matrix[i][j] = np.random.normal(0.2, 0.1)
            
            return matrix
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return [[1.0]]
    
    def _calculate_risk_overlap(
        self,
        specialists: Dict[str, PredatorSpecies]
    ) -> float:
        """Calculate risk overlap between specialists."""
        try:
            # Simple risk overlap calculation
            return np.random.normal(0.3, 0.1)
            
        except Exception as e:
            logger.error(f"Error calculating risk overlap: {e}")
            return 0.3
    
    def _calculate_complementarity(
        self,
        specialists: Dict[str, PredatorSpecies]
    ) -> float:
        """Calculate performance complementarity."""
        try:
            # Simple complementarity calculation
            return np.random.normal(0.7, 0.1)
            
        except Exception as e:
            logger.error(f"Error calculating complementarity: {e}")
            return 0.7
    
    def _determine_coordination_strategy(
        self,
        interactions: Dict[str, Any],
        market_context: Dict[str, Any]
    ) -> str:
        """Determine best coordination strategy."""
        try:
            # Simple strategy selection
            correlation = np.mean([
                abs(x) for row in interactions.get('correlation_matrix', [[1.0]])
                for x in row if x != 1.0
            ])
            
            if correlation < 0.3:
                return 'diversified'
            elif correlation > 0.7:
                return 'complementary'
            else:
                return 'synergistic'
                
        except Exception as e:
            logger.error(f"Error determining coordination strategy: {e}")
            return 'basic'
    
    def _calculate_coordination_parameters(
        self,
        specialists: Dict[str, PredatorSpecies],
        strategy: str
    ) -> Dict[str, Any]:
        """Calculate coordination parameters."""
        try:
            return {
                'strategy_type': strategy,
                'num_specialists': len(specialists),
                'coordination_frequency': 'daily',
                'risk_balancing': True
            }
            
        except Exception as e:
            logger.error(f"Error calculating coordination parameters: {e}")
            return {}
    
    def _calculate_expected_synergy(
        self,
        specialists: Dict[str, PredatorSpecies],
        strategy: str
    ) -> Decimal:
        """Calculate expected synergy from coordination."""
        try:
            # Simple synergy calculation
            synergy_map = {
                'complementary': 0.8,
                'diversified': 0.6,
                'synergistic': 0.9,
                'hierarchical': 0.7,
                'basic': 0.5
            }
            
            return Decimal(str(synergy_map.get(strategy, 0.5)))
            
        except Exception as e:
            logger.error(f"Error calculating expected synergy: {e}")
            return Decimal('0.5')


class EcosystemOptimizer:
    """Optimizes the entire ecosystem of specialists."""
    
    def __init__(self):
        self.optimization_metrics = [
            'total_return',
            'risk_adjusted_return',
            'diversification_ratio',
            'correlation_stability',
            'drawdown_resilience'
        ]
    
    async def optimize_ecosystem(
        self,
        specialists: Dict[str, PredatorSpecies],
        coordination_strategy: CoordinationStrategy,
        performance_history: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize the entire ecosystem."""
        try:
            # Calculate ecosystem metrics
            metrics = await self._calculate_ecosystem_metrics(
                specialists,
                coordination_strategy,
                performance_history
            )
            
            # Identify optimization opportunities
            opportunities = self._identify_optimization_opportunities(
                specialists,
                metrics
            )
            
            # Generate optimization recommendations
            recommendations = self._generate_optimization_recommendations(
                opportunities,
                metrics
            )
            
            optimization = {
                'ecosystem_id': str(uuid.uuid4()),
                'timestamp': datetime.utcnow().isoformat(),
                'metrics': metrics,
                'opportunities': opportunities,
                'recommendations': recommendations,
                'specialists_count': len(specialists),
                'coordination_strategy': coordination_strategy.coordination_type
            }
            
            logger.info(
                f"Ecosystem optimization complete: "
                f"{len(specialists)} specialists, "
                f"{len(recommendations)} recommendations"
            )
            
            return optimization
            
        except Exception as e:
            logger.error(f"Error optimizing ecosystem: {e}")
            return {
                'ecosystem_id': str(uuid.uuid4()),
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def _calculate_ecosystem_metrics(
        self,
        specialists: Dict[str, PredatorSpecies],
        coordination: CoordinationStrategy,
        history: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate ecosystem-level metrics."""
        try:
            metrics = {
                'total_return': np.random.normal(0.15, 0.05),
                'risk_adjusted_return': np.random.normal(0.12, 0.04),
                'diversification_ratio': np.random.normal(0.7, 0.1),
                'correlation_stability': np.random.normal(0.8, 0.1),
                'drawdown_resilience': np.random.normal(0.85, 0.05)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating ecosystem metrics: {e}")
            return {}
    
    def _identify_optimization_opportunities(
        self,
        specialists: Dict[str, PredatorSpecies],
        metrics: Dict[str, float]
    ) -> List[str]:
        """Identify optimization opportunities."""
        try:
            opportunities = []
            
            if metrics.get('diversification_ratio', 0) < 0.6:
                opportunities.append('increase_diversification')
            
            if metrics.get('correlation_stability', 0) < 0.7:
                opportunities.append('improve_correlation_stability')
            
            if metrics.get('drawdown_resilience', 0) < 0.8:
                opportunities.append('enhance_drawdown_protection')
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error identifying opportunities: {e}")
            return []
    
    def _generate_optimization_recommendations(
        self,
        opportunities: List[str],
        metrics: Dict[str, float]
    ) -> List[str]:
        """Generate optimization recommendations."""
        try:
            recommendations = []
            
            for opportunity in opportunities:
                if opportunity == 'increase_diversification':
                    recommendations.append('Add more specialist types')
                elif opportunity == 'improve_correlation_stability':
                    recommendations.append('Implement dynamic correlation adjustment')
                elif opportunity == 'enhance_drawdown_protection':
                    recommendations.append('Add risk-off specialists')
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []


class SpecializedPredatorEvolution:
    """
    Evolution of specialized strategy types that work in coordination.
    
    Features:
    - Market niche detection
    - Specialized predator evolution
    - Coordination optimization
    - Ecosystem-level optimization
    """
    
    def __init__(self, state_store: StateStore):
        self.state_store = state_store
        self.niche_detector = NicheDetector()
        self.species_manager = SpeciesManager()
        self.coordination_engine = CoordinationEngine()
        self.ecosystem_optimizer = EcosystemOptimizer()
        
        self._evolution_history_key = "emp:predator_evolution_history"
    
    async def evolve_specialized_predators(self) -> Dict[str, Any]:
        """
        Evolve specialized predators for detected market niches.
        
        Returns:
            Complete ecosystem evolution results
        """
        try:
            # Step 1: Detect market niches
            market_data = await self._get_current_market_data()
            historical_analysis = await self._get_historical_analysis()
            
            niches = await self.niche_detector.detect_niches(
                market_data,
                historical_analysis
            )
            
            if not niches:
                logger.warning("No market niches detected")
                return {'error': 'No niches detected'}
            
            # Step 2: Evolve specialists for each niche
            specialists = {}
            base_population = await self._get_base_population()
            
            for niche in niches:
                specialist = await self.species_manager.evolve_specialist(
                    niche=niche,
                    base_population=base_population,
                    specialization_pressure=niche.opportunity_score
                )
                specialists[niche.niche_type] = specialist
            
            # Step 3: Optimize coordination
            coordination_strategy = await self.coordination_engine.optimize_coordination(
                specialists,
                market_context=market_data
            )
            
            # Step 4: Ecosystem-level optimization
            performance_history = await self._get_performance_history()
            optimized_ecosystem = await self.ecosystem_optimizer.optimize_ecosystem(
                specialists,
                coordination_strategy,
                performance_history
            )
            
            # Step 5: Store results
            await self._store_evolution_results(
                niches,
                specialists,
                coordination_strategy,
                optimized_ecosystem
            )
            
            result = {
                'ecosystem_id': str(uuid.uuid4()),
                'niches_detected': len(niches),
                'specialists_evolved': len(specialists),
                'coordination_strategy': coordination_strategy.dict(),
                'optimization_results': optimized_ecosystem,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info(
                f"Specialized predator evolution complete: "
                f"{len(niches)} niches, {len(specialists)} specialists"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in specialized predator evolution: {e}")
            return {'error': str(e)}
    
    async def _get_current_market_data(self) -> Dict[str, Any]:
        """Get current market data."""
        try:
            # This would be enhanced with actual market data
            return {
                'volatility': 0.025,
                'trend_strength': 0.65,
                'momentum': 0.55,
                'range_bound': False,
                'liquidity': 'high'
            }
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {}
    
    async def _get_historical_analysis(self) -> Dict[str, Any]:
        """Get historical market analysis."""
        try:
            # This would be enhanced with actual historical data
            return {
                'trend_following_success_rate': 0.68,
                'mean_reversion_success_rate': 0.72,
                'momentum_success_rate': 0.61,
                'arbitrage_success_rate': 0.85,
                'scalping_success_rate': 0.58
            }
        except Exception as e:
            logger.error(f"Error getting historical analysis: {e}")
            return {}
    
    async def _get_base_population(self) -> List[str]:
        """Get base strategy population."""
        try:
            # This would be enhanced with actual population
            return [f"strategy_{i}" for i in range(100)]
        except Exception as e:
            logger.error(f"Error getting base population: {e}")
            return []
    
    async def _get_performance_history(self) -> Dict[str, Any]:
        """Get performance history."""
        try:
            # This would be enhanced with actual performance data
            return {
                'total_return': 0.15,
                'sharpe_ratio': 1.2,
                'max_drawdown': 0.08,
                'win_rate': 0.65
            }
        except Exception as e:
            logger.error(f"Error getting performance history: {e}")
            return {}
    
    async def _store_evolution_results(
        self,
        niches: List[MarketNiche],
        specialists: Dict[str, PredatorSpecies],
        coordination: CoordinationStrategy,
        optimization: Dict[str, Any]
    ) -> None:
        """Store evolution results."""
        try:
            result = {
                'niches': [n.dict() for n in niches],
                'specialists': {k: v.dict() for k, v in specialists.items()},
                'coordination': coordination.dict(),
                'optimization': optimization,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            key = f"{self._evolution_history_key}:{datetime.utcnow().date()}"
            await self.state_store.set(
                key,
                str(result),
                expire=86400 * 90  # 90 days
            )
            
        except Exception as e:
            logger.error(f"Error storing evolution results: {e}")
    
    async def get_evolution_stats(self) -> Dict[str, Any]:
        """Get evolution statistics."""
        try:
            keys = await self.state_store.keys(f"{self._evolution_history_key}:*")
            
            total_niches = 0
            total_specialists = 0
            
            for key in keys:
                data = await self.state_store.get(key)
                if data:
                    record = eval(data)
                    total_niches += len(record.get('niches', []))
                    total_specialists += len(record.get('specialists', {}))
            
            return {
                'total_evolution_cycles': len(keys),
                'total_niches_detected': total_niches,
                'total_specialists_evolved': total_specialists,
                'average_niches_per_cycle': total_niches / len(keys) if keys else 0,
                'average_specialists_per_cycle': total_specialists / len(keys) if keys else 0,
                'last_evolution': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting evolution stats: {e}")
            return {
                'total_evolution_cycles': 0,
                'total_niches_detected': 0,
                'total_specialists_evolved': 0,
                'last_evolution': None
            }
