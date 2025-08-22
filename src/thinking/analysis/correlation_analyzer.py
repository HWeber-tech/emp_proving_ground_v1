"""
EMP Correlation Analyzer v1.1

Analyzes correlations between different market signals and assets.
Provides correlation insights for portfolio management and risk assessment.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from src.core.interfaces import AnalysisResult, SensorySignal, ThinkingPattern
except Exception:  # pragma: no cover
    ThinkingPattern = SensorySignal = AnalysisResult = object  # type: ignore
from src.core.exceptions import ThinkingException

logger = logging.getLogger(__name__)


class CorrelationAnalyzer(ThinkingPattern):
    """Analyzes correlations between market signals and assets."""
    
    def __init__(self, config: Optional[dict[str, object]] = None):
        self.config = config or {}
        self.lookback_periods = self.config.get('lookback_periods', 50)
        self.min_correlation_threshold = self.config.get('min_correlation_threshold', 0.3)
        self._signal_history: Dict[str, List[float]] = {}
        
    def analyze(self, signals: List[SensorySignal]) -> AnalysisResult:
        """Analyze correlations between signals."""
        try:
            # Update signal history
            self._update_signal_history(signals)
            
            # Calculate correlations
            correlation_matrix = self._calculate_correlation_matrix()
            
            # Identify significant correlations
            significant_correlations = self._identify_significant_correlations(correlation_matrix)
            
            # Analyze correlation clusters
            correlation_clusters = self._analyze_correlation_clusters(correlation_matrix)
            
            # Create analysis result
            return AnalysisResult(
                timestamp=datetime.now(),
                analysis_type="correlation_analysis",
                result={
                    'correlation_matrix': correlation_matrix,
                    'significant_correlations': significant_correlations,
                    'correlation_clusters': correlation_clusters,
                    'diversification_score': self._calculate_diversification_score(correlation_matrix),
                    'risk_concentration': self._assess_risk_concentration(correlation_matrix)
                },
                confidence=self._calculate_correlation_confidence(correlation_matrix),
                metadata={
                    'signal_count': len(signals),
                    'signal_types': list(self._signal_history.keys()),
                    'analysis_method': 'pearson_correlation'
                }
            )
            
        except Exception as e:
            raise ThinkingException(f"Error in correlation analysis: {e}")
            
    def learn(self, feedback: dict[str, object]) -> bool:
        """Learn from feedback to improve correlation analysis."""
        try:
            # Extract learning data from feedback
            if 'correlation_accuracy' in feedback:
                accuracy = feedback['correlation_accuracy']
                # Adjust correlation threshold based on accuracy
                if accuracy < 0.5:
                    self.min_correlation_threshold *= 0.9  # Lower threshold
                elif accuracy > 0.8:
                    self.min_correlation_threshold *= 1.1  # Raise threshold
                    
            logger.info("Correlation analyzer learned from feedback")
            return True
            
        except Exception as e:
            logger.error(f"Error in correlation analyzer learning: {e}")
            return False
            
    def _update_signal_history(self, signals: List[SensorySignal]):
        """Update the signal history."""
        for signal in signals:
            signal_type = signal.signal_type
            
            if signal_type not in self._signal_history:
                self._signal_history[signal_type] = []
                
            self._signal_history[signal_type].append(signal.value)
            
            # Keep only recent signals
            if len(self._signal_history[signal_type]) > self.lookback_periods:
                self._signal_history[signal_type] = self._signal_history[signal_type][-self.lookback_periods:]
                
    def _calculate_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """Calculate correlation matrix between all signal types."""
        signal_types = list(self._signal_history.keys())
        correlation_matrix = {}
        
        for i, signal_type1 in enumerate(signal_types):
            correlation_matrix[signal_type1] = {}
            
            for j, signal_type2 in enumerate(signal_types):
                if i == j:
                    correlation_matrix[signal_type1][signal_type2] = 1.0
                else:
                    correlation = self._calculate_correlation(
                        self._signal_history[signal_type1],
                        self._signal_history[signal_type2]
                    )
                    correlation_matrix[signal_type1][signal_type2] = correlation
                    
        return correlation_matrix
        
    def _calculate_correlation(self, series1: List[float], series2: List[float]) -> float:
        """Calculate correlation between two signal series."""
        if len(series1) != len(series2) or len(series1) < 2:
            return 0.0
            
        try:
            # Convert to numpy arrays
            arr1 = np.array(series1)
            arr2 = np.array(series2)
            
            # Calculate Pearson correlation
            correlation = np.corrcoef(arr1, arr2)[0, 1]
            
            # Handle NaN values
            if np.isnan(correlation):
                return 0.0
                
            return float(correlation)
            
        except Exception as e:
            logger.warning(f"Error calculating correlation: {e}")
            return 0.0
            
    def _identify_significant_correlations(self, correlation_matrix: Dict[str, Dict[str, float]]) -> List[dict[str, object]]:
        """Identify significant correlations above threshold."""
        significant_correlations = []
        signal_types = list(correlation_matrix.keys())
        
        for i, signal_type1 in enumerate(signal_types):
            for j, signal_type2 in enumerate(signal_types):
                if i < j:  # Avoid duplicate pairs
                    correlation = correlation_matrix[signal_type1][signal_type2]
                    
                    if abs(correlation) >= self.min_correlation_threshold:
                        significant_correlations.append({
                            'signal_type1': signal_type1,
                            'signal_type2': signal_type2,
                            'correlation': correlation,
                            'strength': 'strong' if abs(correlation) > 0.7 else 'moderate',
                            'direction': 'positive' if correlation > 0 else 'negative'
                        })
                        
        return significant_correlations
        
    def _analyze_correlation_clusters(self, correlation_matrix: Dict[str, Dict[str, float]]) -> List[dict[str, object]]:
        """Analyze clusters of highly correlated signals."""
        signal_types = list(correlation_matrix.keys())
        clusters = []
        processed = set()
        
        for signal_type in signal_types:
            if signal_type in processed:
                continue
                
            # Find cluster for this signal type
            cluster = self._find_correlation_cluster(signal_type, correlation_matrix, processed)
            
            if len(cluster) > 1:  # Only include clusters with multiple signals
                clusters.append({
                    'cluster_id': f"cluster_{len(clusters)}",
                    'signals': cluster,
                    'size': len(cluster),
                    'avg_correlation': self._calculate_cluster_avg_correlation(cluster, correlation_matrix)
                })
                
        return clusters
        
    def _find_correlation_cluster(self, signal_type: str, 
                                correlation_matrix: Dict[str, Dict[str, float]],
                                processed: set) -> List[str]:
        """Find all signals highly correlated with the given signal type."""
        cluster = [signal_type]
        processed.add(signal_type)
        
        for other_signal in correlation_matrix.keys():
            if other_signal not in processed:
                correlation = correlation_matrix[signal_type][other_signal]
                if abs(correlation) > 0.8:  # High correlation threshold for clustering
                    cluster.append(other_signal)
                    processed.add(other_signal)
                    
        return cluster
        
    def _calculate_cluster_avg_correlation(self, cluster: List[str],
                                         correlation_matrix: Dict[str, Dict[str, float]]) -> float:
        """Calculate average correlation within a cluster."""
        if len(cluster) < 2:
            return 1.0
            
        correlations = []
        for i, signal1 in enumerate(cluster):
            for j, signal2 in enumerate(cluster):
                if i < j:
                    correlations.append(correlation_matrix[signal1][signal2])
                    
        return np.mean(correlations) if correlations else 1.0
        
    def _calculate_diversification_score(self, correlation_matrix: Dict[str, Dict[str, float]]) -> float:
        """Calculate diversification score based on correlations."""
        signal_types = list(correlation_matrix.keys())
        
        if len(signal_types) < 2:
            return 1.0  # Perfect diversification for single signal
            
        # Calculate average absolute correlation
        correlations = []
        for i, signal1 in enumerate(signal_types):
            for j, signal2 in enumerate(signal_types):
                if i < j:
                    correlations.append(abs(correlation_matrix[signal1][signal2]))
                    
        avg_correlation = np.mean(correlations) if correlations else 0.0
        
        # Diversification score: lower correlation = higher diversification
        diversification_score = 1.0 - avg_correlation
        return max(0.0, min(1.0, diversification_score))
        
    def _assess_risk_concentration(self, correlation_matrix: Dict[str, Dict[str, float]]) -> dict[str, object]:
        """Assess risk concentration based on correlations."""
        signal_types = list(correlation_matrix.keys())
        
        if len(signal_types) < 2:
            return {
                'risk_level': 'low',
                'concentration_score': 0.0,
                'warnings': []
            }
            
        # Calculate risk concentration metrics
        high_correlations = 0
        total_pairs = 0
        warnings = []
        
        for i, signal1 in enumerate(signal_types):
            for j, signal2 in enumerate(signal_types):
                if i < j:
                    total_pairs += 1
                    correlation = correlation_matrix[signal1][signal2]
                    
                    if abs(correlation) > 0.8:
                        high_correlations += 1
                        warnings.append(f"High correlation between {signal1} and {signal2}: {correlation:.3f}")
                        
        concentration_score = high_correlations / total_pairs if total_pairs > 0 else 0.0
        
        # Classify risk level
        if concentration_score > 0.5:
            risk_level = 'high'
        elif concentration_score > 0.2:
            risk_level = 'medium'
        else:
            risk_level = 'low'
            
        return {
            'risk_level': risk_level,
            'concentration_score': concentration_score,
            'high_correlation_pairs': high_correlations,
            'total_pairs': total_pairs,
            'warnings': warnings
        }
        
    def _calculate_correlation_confidence(self, correlation_matrix: Dict[str, Dict[str, float]]) -> float:
        """Calculate confidence in correlation analysis."""
        signal_types = list(correlation_matrix.keys())
        
        if len(signal_types) < 2:
            return 0.0
            
        # Confidence based on data sufficiency
        min_history_length = min(len(self._signal_history.get(st, [])) for st in signal_types)
        
        if min_history_length < 10:
            return 0.3
        elif min_history_length < 30:
            return 0.6
        else:
            return 0.9 
