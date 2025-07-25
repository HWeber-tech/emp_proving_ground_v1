"""
ANOMALY Dimension - Manipulation Detection System

This module implements the ANOMALY dimension of the sensory cortex,
responsible for detecting market manipulation, unusual patterns,
and anomalous behaviors that deviate from normal market dynamics.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats


@dataclass
class AnomalyMetrics:
    """Metrics for anomaly detection."""
    anomaly_score: float
    confidence: float
    severity: str
    type: str
    timestamp: datetime


class AnomalyDimension:
    """
    Manipulation Detection System for market anomalies.
    
    This class implements the ANOMALY dimension functionality for detecting
    market manipulation, unusual trading patterns, and behaviors that
    deviate significantly from normal market dynamics.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the ANOMALY dimension."""
        self.config = config or {}
        self.anomaly_threshold = self.config.get('anomaly_threshold', 2.5)
        self.sensitivity = self.config.get('sensitivity', 0.8)
        self.detected_anomalies = []
        self.baseline_stats = {}
        
    def detect_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect anomalies in market data.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Dictionary containing anomaly detection results
        """
        anomalies = []
        
        # Statistical anomaly detection
        price_anomalies = self._detect_price_anomalies(data)
        volume_anomalies = self._detect_volume_anomalies(data)
        volatility_anomalies = self._detect_volatility_anomalies(data)
        
        anomalies.extend(price_anomalies)
        anomalies.extend(volume_anomalies)
        anomalies.extend(volatility_anomalies)
        
        return {
            'anomalies': anomalies,
            'summary': {
                'total_detected': len(anomalies),
                'high_severity': len([a for a in anomalies if a['severity'] == 'high']),
                'medium_severity': len([a for a in anomalies if a['severity'] == 'medium']),
                'low_severity': len([a for a in anomalies if a['severity'] == 'low'])
            },
            'metadata': {
                'analysis_time': datetime.now(),
                'data_points': len(data),
                'threshold_used': self.anomaly_threshold
            }
        }
        
    def _detect_price_anomalies(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect price-based anomalies."""
        if 'close' not in data.columns:
            return []
            
        prices = data['close'].values
        anomalies = []
        
        # Z-score based anomaly detection
        z_scores = np.abs(stats.zscore(prices))
        
        for i, z_score in enumerate(z_scores):
            if z_score > self.anomaly_threshold:
                anomalies.append({
                    'type': 'price_anomaly',
                    'severity': self._calculate_severity(z_score),
                    'score': float(z_score),
                    'value': float(prices[i]),
                    'timestamp': data.index[i] if hasattr(data.index, '__getitem__') else datetime.now(),
                    'description': f'Price anomaly detected (z-score: {z_score:.2f})'
                })
                
        return anomalies
        
    def _detect_volume_anomalies(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect volume-based anomalies."""
        if 'volume' not in data.columns:
            return []
            
        volumes = data['volume'].values
        anomalies = []
        
        # Volume spike detection
        mean_volume = np.mean(volumes)
        std_volume = np.std(volumes)
        
        for i, volume in enumerate(volumes):
            if std_volume > 0:
                z_score = abs(volume - mean_volume) / std_volume
                if z_score > self.anomaly_threshold:
                    anomalies.append({
                        'type': 'volume_spike',
                        'severity': self._calculate_severity(z_score),
                        'score': float(z_score),
                        'value': float(volume),
                        'timestamp': data.index[i] if hasattr(data.index, '__getitem__') else datetime.now(),
                        'description': f'Volume spike detected (z-score: {z_score:.2f})'
                    })
                    
        return anomalies
        
    def _detect_volatility_anomalies(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect volatility-based anomalies."""
        if len(data) < 2:
            return []
            
        # Calculate rolling volatility
        if 'close' in data.columns:
            returns = data['close'].pct_change().dropna()
            rolling_vol = returns.rolling(window=20).std()
            
            anomalies = []
            for i, vol in enumerate(rolling_vol):
                if not np.isnan(vol) and vol > returns.std() * 2:
                    anomalies.append({
                        'type': 'volatility_anomaly',
                        'severity': 'medium',
                        'score': float(vol),
                        'value': float(vol),
                        'timestamp': data.index[i] if hasattr(data.index, '__getitem__') else datetime.now(),
                        'description': f'Volatility anomaly detected'
                    })
                    
            return anomalies
            
        return []
        
    def _calculate_severity(self, score: float) -> str:
        """Calculate severity level based on anomaly score."""
        if score > 4.0:
            return 'high'
        elif score > 2.5:
            return 'medium'
        else:
            return 'low'
            
    def set_baseline(self, data: pd.DataFrame) -> None:
        """Set baseline statistics for anomaly detection."""
        self.baseline_stats = {
            'price_mean': data['close'].mean() if 'close' in data.columns else 0,
            'price_std': data['close'].std() if 'close' in data.columns else 1,
            'volume_mean': data['volume'].mean() if 'volume' in data.columns else 0,
            'volume_std': data['volume'].std() if 'volume' in data.columns else 1
        }
        
    def get_recent_anomalies(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get anomalies from the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            anomaly for anomaly in self.detected_anomalies
            if anomaly.get('timestamp', datetime.min) > cutoff_time
        ]
        
    def clear_anomalies(self) -> None:
        """Clear all detected anomalies."""
        self.detected_anomalies.clear()
        
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the ANOMALY dimension."""
        return {
            'total_anomalies': len(self.detected_anomalies),
            'recent_anomalies': len(self.get_recent_anomalies(24)),
            'baseline_set': len(self.baseline_stats) > 0,
            'health': 'healthy',
            'last_scan': datetime.now()
        }
