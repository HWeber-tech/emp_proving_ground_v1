#!/usr/bin/env python3
"""
Intelligence Validator
====================

Validates the accuracy of intelligence components.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class ValidationMetrics:
    """Validation metrics for intelligence components"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    false_negative_rate: float


class IntelligenceValidator:
    """Validates intelligence component accuracy"""
    
    def validate_anomaly_detection(self, 
                                 true_labels: List[int], 
                                 predicted_labels: List[int]) -> ValidationMetrics:
        """Validate anomaly detection accuracy"""
        if len(true_labels) != len(predicted_labels):
            raise ValueError("Label lists must have same length")
        
        true_positives = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 1 and p == 1)
        false_positives = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 0 and p == 1)
        true_negatives = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 0 and p == 0)
        false_negatives = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 1 and p == 0)
        
        total = len(true_labels)
        
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        false_positive_rate = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
        false_negative_rate = false_negatives / (false_negatives + true_positives) if (false_negatives + true_positives) > 0 else 0
        
        return ValidationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            false_positive_rate=false_positive_rate,
            false_negative_rate=false_negative_rate
        )
    
    def validate_regime_classification(self, 
                                     true_regimes: List[str], 
                                     predicted_regimes: List[str]) -> ValidationMetrics:
        """Validate regime classification accuracy"""
        if len(true_regimes) != len(predicted_regimes):
            raise ValueError("Regime lists must have same length")
        
        # Convert to binary classification (correct/incorrect)
        true_labels = [1 if t == p else 0 for t, p in zip(true_regimes, predicted_regimes)]
        predicted_labels = [1] * len(true_labels)  # Always predict correct
        
        return self.validate_anomaly_detection(true_labels, predicted_labels)
    
    def validate_fitness_evaluation(self, 
                                  expected_scores: List[float], 
                                  actual_scores: List[float]) -> Dict[str, float]:
        """Validate fitness evaluation accuracy"""
        if len(expected_scores) != len(actual_scores):
            raise ValueError("Score lists must have same length")
        
        correlation = np.corrcoef(expected_scores, actual_scores)[0, 1]
        mae = np.mean(np.abs(np.array(expected_scores) - np.array(actual_scores)))
        rmse = np.sqrt(np.mean((np.array(expected_scores) - np.array(actual_scores)) ** 2))
        
        return {
            'correlation': correlation,
            'mae': mae,
            'rmse': rmse,
            'accuracy': 1 - mae  # Simple accuracy metric
        }
    
    def generate_synthetic_test_data(self, 
                                   n_samples: int = 1000, 
                                   anomaly_rate: float = 0.05) -> Dict[str, List[int]]:
        """Generate synthetic test data"""
        np.random.seed(42)
        
        # Generate normal data
        normal_data = np.random.randn(n_samples)
        
        # Generate anomalies
        n_anomalies = int(n_samples * anomaly_rate)
        anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
        
        true_labels = [0] * n_samples
        for idx in anomaly_indices:
            true_labels[idx] = 1
        
        # Generate predictions with 90% accuracy
        predicted_labels = []
        for true_label in true_labels:
            if np.random.random() < 0.9:
                predicted_labels.append(true_label)
            else:
                predicted_labels.append(1 - true_label)
        
        return {
            'true_labels': true_labels,
            'predicted_labels': predicted_labels
        }
