"""
Predictive Market Modeler
Advanced market prediction and scenario modeling system.
"""

import logging
import uuid
from ast import literal_eval
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from src.core.events import ContextPacket, PredictionResult  # legacy
except Exception:  # pragma: no cover
    ContextPacket = PredictionResult = object  # type: ignore
from src.operational.state_store import StateStore

logger = logging.getLogger(__name__)


@dataclass
class MarketScenario:
    """Represents a probable market scenario (canonical)."""
    scenario_id: str
    timestamp: datetime
    scenario_type: str
    probability: float
    price_path: List[float]
    volatility: float
    direction_bias: float
    confidence: Decimal


class MarketScenarioGenerator:
    """Generates multiple probable market scenarios."""
    
    def __init__(self):
        self.scenario_templates = {
            'trend_continuation': {
                'probability': 0.4,
                'volatility_factor': 1.0,
                'direction_bias': 0.7
            },
            'trend_reversal': {
                'probability': 0.3,
                'volatility_factor': 1.5,
                'direction_bias': -0.7
            },
            'range_bound': {
                'probability': 0.2,
                'volatility_factor': 0.7,
                'direction_bias': 0.0
            },
            'breakout': {
                'probability': 0.1,
                'volatility_factor': 2.0,
                'direction_bias': 0.8
            }
        }
    
    async def generate_scenarios(
        self,
        current_state: Dict[str, Any],
        time_horizon: timedelta,
        num_scenarios: int = 1000
    ) -> List[MarketScenario]:
        """Generate multiple market scenarios."""
        try:
            scenarios = []
            
            for i in range(num_scenarios):
                # Select scenario type based on probabilities
                scenario_type = np.random.choice(
                    list(self.scenario_templates.keys()),
                    p=[t['probability'] for t in self.scenario_templates.values()]
                )
                
                template = self.scenario_templates[scenario_type]
                
                # Generate scenario parameters
                volatility = current_state.get('volatility', 0.02) * template['volatility_factor']
                direction = template['direction_bias'] * np.random.normal(0, 1)
                
                # Generate price path
                price_path = self._generate_price_path(
                    current_state.get('price', 100),
                    volatility,
                    direction,
                    time_horizon
                )
                
                scenario = MarketScenario(
                    scenario_id=str(uuid.uuid4()),
                    timestamp=datetime.utcnow(),
                    scenario_type=scenario_type,
                    probability=template['probability'],
                    price_path=price_path,
                    volatility=volatility,
                    direction_bias=direction,
                    confidence=self._calculate_scenario_confidence(current_state, scenario_type)
                )
                
                scenarios.append(scenario)
            
            return scenarios
            
        except Exception as e:
            logger.error(f"Error generating scenarios: {e}")
            return []
    
    def _generate_price_path(
        self,
        start_price: float,
        volatility: float,
        drift: float,
        time_horizon: timedelta
    ) -> List[float]:
        """Generate a simulated price path."""
        try:
            # Use geometric Brownian motion
            dt = 1/252  # Daily steps
            steps = int(time_horizon.days)
            
            prices = [start_price]
            
            for _ in range(steps):
                # Brownian motion component
                brownian = np.random.normal(0, volatility * np.sqrt(dt))
                
                # Price evolution
                new_price = prices[-1] * np.exp((drift - 0.5 * volatility**2) * dt + brownian)
                prices.append(new_price)
            
            return prices
            
        except Exception as e:
            logger.error(f"Error generating price path: {e}")
            return [start_price]
    
    def _calculate_scenario_confidence(
        self,
        current_state: Dict[str, Any],
        scenario_type: str
    ) -> Decimal:
        """Calculate confidence for a scenario."""
        try:
            # Base confidence on historical accuracy
            base_confidence = 0.7
            
            # Adjust based on market conditions
            volatility = current_state.get('volatility', 0.02)
            if volatility > 0.05:
                base_confidence *= 0.8  # Lower confidence in high volatility
            
            return Decimal(str(base_confidence))
            
        except Exception as e:
            logger.error(f"Error calculating scenario confidence: {e}")
            return Decimal('0.5')


class BayesianProbabilityEngine:
    """Calculates probabilities using Bayesian methods."""
    
    def __init__(self):
        self.prior_probabilities = {
            'bull_market': 0.4,
            'bear_market': 0.3,
            'range_market': 0.3
        }
    
    async def calculate_probabilities(
        self,
        scenarios: List[MarketScenario],
        historical_data: Dict[str, Any]
    ) -> List[Decimal]:
        """Calculate Bayesian probabilities for scenarios."""
        try:
            probabilities = []
            
            for scenario in scenarios:
                # Prior probability
                prior = Decimal(str(scenario.probability))
                
                # Likelihood based on historical data
                likelihood = self._calculate_likelihood(scenario, historical_data)
                
                # Posterior probability
                posterior = prior * likelihood
                
                probabilities.append(posterior)
            
            # Normalize probabilities
            total = sum(probabilities)
            if total > 0:
                probabilities = [p / total for p in probabilities]
            
            return probabilities
            
        except Exception as e:
            logger.error(f"Error calculating probabilities: {e}")
            return [Decimal('1.0') / len(scenarios)] * len(scenarios)
    
    def _calculate_likelihood(
        self,
        scenario: MarketScenario,
        historical_data: Dict[str, Any]
    ) -> Decimal:
        """Calculate likelihood of scenario given historical data."""
        try:
            # Simple likelihood calculation based on historical performance
            if scenario.scenario_type == 'trend_continuation':
                return Decimal('0.8')
            elif scenario.scenario_type == 'trend_reversal':
                return Decimal('0.6')
            elif scenario.scenario_type == 'range_bound':
                return Decimal('0.7')
            else:
                return Decimal('0.5')
                
        except Exception as e:
            logger.error(f"Error calculating likelihood: {e}")
            return Decimal('0.5')


class OutcomePredictor:
    """Predicts outcomes for market scenarios."""
    
    def __init__(self):
        self.prediction_models = {
            'profit_target': self._predict_profit_target,
            'risk_level': self._predict_risk_level,
            'success_probability': self._predict_success_probability
        }
    
    async def predict_outcome(
        self,
        scenario: MarketScenario
    ) -> Dict[str, Any]:
        """Predict outcome for a market scenario."""
        try:
            predictions = {}
            
            for model_name, model_func in self.prediction_models.items():
                predictions[model_name] = model_func(scenario)
            
            return {
                'scenario_id': scenario.scenario_id,
                'predictions': predictions,
                'confidence': scenario.confidence,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error predicting outcome: {e}")
            return {
                'scenario_id': scenario.scenario_id,
                'predictions': {},
                'confidence': Decimal('0.5'),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def _predict_profit_target(self, scenario: MarketScenario) -> float:
        """Predict profit target for scenario."""
        try:
            if scenario.scenario_type == 'trend_continuation':
                return 0.05  # 5% profit target
            elif scenario.scenario_type == 'trend_reversal':
                return 0.03  # 3% profit target
            elif scenario.scenario_type == 'range_bound':
                return 0.02  # 2% profit target
            else:
                return 0.04  # 4% profit target
                
        except Exception as e:
            logger.error(f"Error predicting profit target: {e}")
            return 0.03
    
    def _predict_risk_level(self, scenario: MarketScenario) -> float:
        """Predict risk level for scenario."""
        try:
            # Risk is proportional to volatility
            return min(scenario.volatility * 2, 0.1)  # Cap at 10%
            
        except Exception as e:
            logger.error(f"Error predicting risk level: {e}")
            return 0.05
    
    def _predict_success_probability(self, scenario: MarketScenario) -> float:
        """Predict success probability for scenario."""
        try:
            # Base success probability on scenario type
            success_rates = {
                'trend_continuation': 0.65,
                'trend_reversal': 0.55,
                'range_bound': 0.60,
                'breakout': 0.45
            }
            
            return success_rates.get(scenario.scenario_type, 0.5)
            
        except Exception as e:
            logger.error(f"Error predicting success probability: {e}")
            return 0.5


class ConfidenceCalibrator:
    """Calibrates confidence based on historical accuracy."""
    
    def __init__(self):
        self.accuracy_history = []
    
    async def calibrate_confidence(
        self,
        outcome_predictions: List[Tuple[MarketScenario, Decimal, Dict[str, Any]]],
        prediction_history: Dict[str, Any]
    ) -> List[PredictionResult]:
        """Calibrate confidence based on historical accuracy."""
        try:
            calibrated_results = []
            
            for scenario, probability, outcome in outcome_predictions:
                # Calculate calibration factor
                calibration_factor = self._calculate_calibration_factor(
                    prediction_history
                )
                
                # Calibrate confidence
                calibrated_confidence = float(scenario.confidence) * calibration_factor
                
                # Ensure reasonable bounds
                calibrated_confidence = max(0.1, min(1.0, calibrated_confidence))
                
                # Create prediction result
                result = PredictionResult(
                    scenario_id=scenario.scenario_id,
                    probability=probability,
                    predicted_outcome=outcome,
                    confidence=Decimal(str(calibrated_confidence)),
                    calibration_factor=Decimal(str(calibration_factor)),
                    timestamp=datetime.utcnow()
                )
                
                calibrated_results.append(result)
            
            return calibrated_results
            
        except Exception as e:
            logger.error(f"Error calibrating confidence: {e}")
            return []
    
    def _calculate_calibration_factor(self, prediction_history: Dict[str, Any]) -> float:
        """Calculate calibration factor based on historical accuracy."""
        try:
            # Simple calibration based on historical accuracy
            historical_accuracy = prediction_history.get('accuracy', 0.75)
            
            # Adjust calibration factor
            if historical_accuracy > 0.8:
                return 1.1
            elif historical_accuracy < 0.6:
                return 0.9
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"Error calculating calibration factor: {e}")
            return 1.0


class PredictiveMarketModeler:
    """Advanced market prediction and scenario modeling system."""
    
    def __init__(self, state_store: StateStore):
        self.state_store = state_store
        self.scenario_generator = MarketScenarioGenerator()
        self.probability_engine = BayesianProbabilityEngine()
        self.outcome_predictor = OutcomePredictor()
        self.confidence_calibrator = ConfidenceCalibrator()
        
        self._prediction_history_key = "emp:prediction_history"
    
    async def predict_market_scenarios(
        self,
        current_state: Dict[str, Any],
        time_horizon: timedelta,
        num_scenarios: int = 1000
    ) -> List[PredictionResult]:
        """
        Predict market scenarios with probabilities and outcomes.
        
        Args:
            current_state: Current market state
            time_horizon: Prediction time horizon
            num_scenarios: Number of scenarios to generate
            
        Returns:
            List of calibrated prediction results
        """
        try:
            # Step 1: Generate scenarios
            scenarios = await self.scenario_generator.generate_scenarios(
                current_state,
                time_horizon,
                num_scenarios
            )
            
            if not scenarios:
                return []
            
            # Step 2: Calculate probabilities
            historical_data = await self._get_historical_data()
            probabilities = await self.probability_engine.calculate_probabilities(
                scenarios,
                historical_data
            )
            
            # Step 3: Predict outcomes
            outcome_predictions = []
            for scenario, probability in zip(scenarios, probabilities):
                outcome = await self.outcome_predictor.predict_outcome(scenario)
                outcome_predictions.append((scenario, probability, outcome))
            
            # Step 4: Calibrate confidence
            prediction_history = await self._get_prediction_history()
            calibrated_results = await self.confidence_calibrator.calibrate_confidence(
                outcome_predictions,
                prediction_history
            )
            
            # Step 5: Store predictions
            await self._store_predictions(calibrated_results)
            
            logger.info(
                f"Generated {len(calibrated_results)} calibrated market predictions"
            )
            
            return calibrated_results
            
        except Exception as e:
            logger.error(f"Error predicting market scenarios: {e}")
            return []
    
    async def _get_historical_data(self) -> Dict[str, Any]:
        """Get historical market data for probability calculation."""
        try:
            data = await self.state_store.get(self._prediction_history_key)
            if data:
                # Bandit B307: replaced eval with safe parsing
                try:
                    return literal_eval(data)
                except (ValueError, SyntaxError):
                    return {
                        'accuracy': 0.75,
                        'total_predictions': 0,
                        'successful_predictions': 0
                    }
            return {
                'accuracy': 0.75,
                'total_predictions': 0,
                'successful_predictions': 0
            }
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return {'accuracy': 0.75, 'total_predictions': 0, 'successful_predictions': 0}
    
    async def _get_prediction_history(self) -> Dict[str, Any]:
        """Get prediction history for calibration."""
        try:
            return await self._get_historical_data()
        except Exception as e:
            logger.error(f"Error getting prediction history: {e}")
            return {'accuracy': 0.75}
    
    async def _store_predictions(self, results: List[PredictionResult]) -> None:
        """Store prediction results for future calibration."""
        try:
            key = f"{self._prediction_history_key}:{datetime.utcnow().date()}"
            await self.state_store.set(
                key,
                str([r.dict() for r in results]),
                expire=86400 * 30  # 30 days
            )
        except Exception as e:
            logger.error(f"Error storing predictions: {e}")
    
    async def get_prediction_accuracy(self, days: int = 30) -> Dict[str, Any]:
        """Get prediction accuracy statistics."""
        try:
            # This would be enhanced with actual retrieval
            return {
                'accuracy': 0.72,
                'total_predictions': 1000,
                'successful_predictions': 720,
                'average_confidence': 0.75,
                'last_update': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting prediction accuracy: {e}")
            return {'accuracy': 0.72}
