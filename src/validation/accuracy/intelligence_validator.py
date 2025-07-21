#!/usr/bin/env python3
"""
Intelligence Accuracy Validator
Comprehensive accuracy validation for Phase 2 intelligence systems
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class AccuracyResult:
    """Result of accuracy validation"""
    test_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sample_size: int
    confidence_interval: Tuple[float, float]
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ValidationDataset:
    """Validation dataset for accuracy testing"""
    name: str
    data: List[Dict[str, Any]]
    expected_outputs: List[Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntelligenceValidator:
    """Comprehensive intelligence accuracy validation system"""
    
    def __init__(self):
        self.results: List[AccuracyResult] = []
        self.validation_datasets: Dict[str, ValidationDataset] = {}
        
    def create_anomaly_detection_dataset(self) -> ValidationDataset:
        """Create validation dataset for anomaly detection"""
        
        # Generate synthetic anomaly detection data
        np.random.seed(42)
        
        normal_data = []
        anomaly_data = []
        
        # Normal market data
        for i in range(800):
            normal_data.append({
                'price': 100 + np.random.normal(0, 2),
                'volume': 1000 + np.random.normal(0, 100),
                'volatility': 0.15 + np.random.normal(0, 0.05),
                'expected_anomaly': False
            })
        
        # Anomalous market data
        for i in range(200):
            anomaly_data.append({
                'price': 100 + np.random.normal(0, 10),
                'volume': 1000 + np.random.normal(0, 500),
                'volatility': 0.15 + np.random.normal(0, 0.2),
                'expected_anomaly': True
            })
        
        all_data = normal_data + anomaly_data
        np.random.shuffle(all_data)
        
        return ValidationDataset(
            name="anomaly_detection",
            data=[{k: v for k, v in d.items() if k != 'expected_anomaly'} 
                  for d in all_data],
            expected_outputs=[d['expected_anomaly'] for d in all_data],
            metadata={
                'normal_samples': 800,
                'anomaly_samples': 200,
                'total_samples': 1000,
                'anomaly_rate': 0.2
            }
        )
    
    def create_regime_classification_dataset(self) -> ValidationDataset:
        """Create validation dataset for regime classification"""
        
        # Generate synthetic regime data
        np.random.seed(42)
        
        regimes = ['TRENDING_UP', 'TRENDING_DOWN', 'RANGING', 'VOLATILE', 'CRISIS']
        data = []
        expected_outputs = []
        
        for regime in regimes:
            for _ in range(200):
                if regime == 'TRENDING_UP':
                    data.append({
                        'price_trend': 0.8 + np.random.normal(0, 0.1),
                        'volatility': 0.15 + np.random.normal(0, 0.05),
                        'volume_trend': 0.7 + np.random.normal(0, 0.2)
                    })
                elif regime == 'TRENDING_DOWN':
                    data.append({
                        'price_trend': -0.8 + np.random.normal(0, 0.1),
                        'volatility': 0.15 + np.random.normal(0, 0.05),
                        'volume_trend': 0.7 + np.random.normal(0, 0.2)
                    })
                elif regime == 'RANGING':
                    data.append({
                        'price_trend': 0.0 + np.random.normal(0, 0.1),
                        'volatility': 0.1 + np.random.normal(0, 0.02),
                        'volume_trend': 0.0 + np.random.normal(0, 0.1)
                    })
                elif regime == 'VOLATILE':
                    data.append({
                        'price_trend': 0.0 + np.random.normal(0, 0.3),
                        'volatility': 0.4 + np.random.normal(0, 0.1),
                        'volume_trend': 0.5 + np.random.normal(0, 0.3)
                    })
                elif regime == 'CRISIS':
                    data.append({
                        'price_trend': -0.5 + np.random.normal(0, 0.4),
                        'volatility': 0.8 + np.random.normal(0, 0.2),
                        'volume_trend': 2.0 + np.random.normal(0, 1.0)
                    })
                
                expected_outputs.append(regime)
        
        return ValidationDataset(
            name="regime_classification",
            data=data,
            expected_outputs=expected_outputs,
            metadata={
                'regimes': regimes,
                'samples_per_regime': 200,
                'total_samples': 1000
            }
        )
    
    def create_fitness_evaluation_dataset(self) -> ValidationDataset:
        """Create validation dataset for fitness evaluation"""
        
        # Generate synthetic fitness data
        np.random.seed(42)
        
        data = []
        expected_outputs = []
        
        for _ in range(1000):
            # Generate realistic performance metrics
            sharpe_ratio = np.random.normal(1.8, 0.5)
            max_drawdown = np.random.normal(0.02, 0.01)
            win_rate = np.random.normal(0.65, 0.1)
            profit_factor = np.random.normal(1.8, 0.5)
            
            data.append({
                'sharpe_ratio': max(0, sharpe_ratio),
                'max_drawdown': max(0, min(0.1, max_drawdown)),
                'win_rate': max(0, min(1, win_rate)),
                'profit_factor': max(0, profit_factor),
                'total_return': np.random.normal(0.15, 0.1)
            })
            
            # Expected fitness score (0-1)
            fitness_score = min(1.0, (sharpe_ratio / 3.0) * 0.4 + 
                              (1 - max_drawdown / 0.05) * 0.3 + 
                              win_rate * 0.3)
            expected_outputs.append(fitness_score)
        
        return ValidationDataset(
            name="fitness_evaluation",
            data=data,
            expected_outputs=expected_outputs,
            metadata={
                'samples': 1000,
                'metric_ranges': {
                    'sharpe_ratio': '0-3',
                    'max_drawdown': '0-0.1',
                    'win_rate': '0-1',
                    'profit_factor': '0-3'
                }
            }
        )
    
    async def validate_anomaly_detection(
        self,
        detector_function: Callable,
        min_accuracy: float = 0.90
    ) -> AccuracyResult:
        """Validate anomaly detection accuracy"""
        
        logger.info("Validating anomaly detection accuracy")
        
        dataset = self.create_anomaly_detection_dataset()
        
        # Run predictions
        predictions = []
        for data_point in dataset.data:
            prediction = await detector_function(data_point)
            predictions.append(prediction)
        
        # Calculate metrics
        true_positives = sum(1 for p, e in zip(predictions, dataset.expected_outputs) 
                           if p and e)
        true_negatives = sum(1 for p, e in zip(predictions, dataset.expected_outputs) 
                           if not p and not e)
        false_positives = sum(1 for p, e in zip(predictions, dataset.expected_outputs) 
                            if p and not e)
        false_negatives = sum(1 for p, e in zip(predictions, dataset.expected_outputs) 
                            if not p and e)
        
        total = len(dataset.expected_outputs)
        accuracy = (true_positives + true_negatives) / total
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(
            dataset.expected_outputs, 
            predictions
        )
        
        passed = accuracy >= min_accuracy
        
        result = AccuracyResult(
            test_name="anomaly_detection_accuracy",
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            sample_size=len(dataset.data),
            confidence_interval=confidence_interval,
            passed=passed,
            details={
                'true_positives': true_positives,
                'true_negatives': true_negatives,
                'false_positives': false_positives,
                'false_negatives': false_negatives
            }
        )
        
        self.results.append(result)
        return result
    
    async def validate_regime_classification(
        self,
        classifier_function: Callable,
        min_accuracy: float = 0.85
    ) -> AccuracyResult:
        """Validate regime classification accuracy"""
        
        logger.info("Validating regime classification accuracy")
        
        dataset = self.create_regime_classification_dataset()
        
        # Run predictions
        predictions = []
        for data_point in dataset.data:
            prediction = await classifier_function(data_point)
            predictions.append(prediction)
        
        # Calculate metrics
        correct = sum(1 for p, e in zip(predictions, dataset.expected_outputs) if p == e)
        accuracy = correct / len(dataset.expected_outputs)
        
        # Calculate per-class metrics
        class_metrics = {}
        for regime in set(dataset.expected_outputs):
            true_pos = sum(1 for p, e in zip(predictions, dataset.expected_outputs) 
                         if p == regime and e == regime)
            false_pos = sum(1 for p, e in zip(predictions, dataset.expected_outputs) 
                          if p == regime and e != regime)
            false_neg = sum(1 for p, e in zip(predictions, dataset.expected_outputs) 
                          if p != regime and e == regime)
            
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            recall = true_pos / (true_pos
