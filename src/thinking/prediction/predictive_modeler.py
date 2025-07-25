#!/usr/bin/env python3
"""
Predictive Market Modeler - Epic 3: The Predictor
Production-ready inference service that loads trained models and makes real-time forecasts.

This service:
1. Loads trained models from MLflow
2. Performs real-time feature engineering
3. Makes predictions with probability outputs
4. Handles async inference
"""

import pandas as pd
import numpy as np
import torch
import mlflow
import mlflow.pytorch
from pathlib import Path
import logging
from typing import Dict, Optional, List
import asyncio
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PredictiveMarketModeler:
    """Production-ready market forecasting service."""
    
    def __init__(self, model_run_id: str, mlflow_uri: str = "http://localhost:5000"):
        """
        Initialize the predictive modeler.
        
        Args:
            model_run_id: MLflow run ID for the trained model
            mlflow_uri: MLflow tracking server URI
        """
        self.model_run_id = model_run_id
        self.mlflow_uri = mlflow_uri
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        
        # Load model and artifacts
        self._load_model()
        
    def _load_model(self):
        """Load the trained model and preprocessing artifacts from MLflow."""
        logger.info(f"Loading model from MLflow run: {self.model_run_id}")
        
        try:
            # Set MLflow tracking URI
            mlflow.set_tracking_uri(self.mlflow_uri)
            
            # Load model
            model_uri = f"runs:/{self.model_run_id}/model"
            self.model = mlflow.pytorch.load_model(model_uri)
            self.model.eval()
            
            # Load preprocessing artifacts
            client = mlflow.tracking.MlflowClient()
            run = client.get_run(self.model_run_id)
            
            # Load feature names
            features_path = client.download_artifacts(self.model_run_id, "features.txt")
            with open(features_path, 'r') as f:
                self.feature_names = [line.strip() for line in f]
            
            # Load scaler and encoder (saved as artifacts)
            try:
                scaler_path = client.download_artifacts(self.model_run_id, "scaler.pkl")
                import joblib
                self.scaler = joblib.load(scaler_path)
            except:
                logger.warning("Scaler not found, will use identity scaling")
                self.scaler = None
                
            try:
                encoder_path = client.download_artifacts(self.model_run_id, "label_encoder.pkl")
                self.label_encoder = joblib.load(encoder_path)
            except:
                logger.warning("Label encoder not found, using default mapping")
                self.label_encoder = None
                
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for inference (same as training)."""
        features_df = df.copy()
        
        # Price-based features
        features_df['returns'] = features_df['close'].pct_change()
        features_df['log_returns'] = np.log(features_df['close'] / features_df['close'].shift(1))
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            features_df[f'sma_{period}'] = features_df['close'].rolling(window=period).mean()
            features_df[f'ema_{period}'] = features_df['close'].ewm(span=period, adjust=False).mean()
        
        # RSI calculation
        delta = features_df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features_df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD calculation
        ema_12 = features_df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = features_df['close'].ewm(span=26, adjust=False).mean()
        features_df['macd'] = ema_12 - ema_26
        features_df['macd_signal'] = features_df['macd'].ewm(span=9, adjust=False).mean()
        features_df['macd_histogram'] = features_df['macd'] - features_df['macd_signal']
        
        # Bollinger Bands
        sma_20 = features_df['close'].rolling(window=20).mean()
        std_20 = features_df['close'].rolling(window=20).std()
        features_df['bb_upper'] = sma_20 + (std_20 * 2)
        features_df['bb_lower'] = sma_20 - (std_20 * 2)
        features_df['bb_middle'] = sma_20
        features_df['bb_width'] = features_df['bb_upper'] - features_df['bb_lower']
        features_df['bb_position'] = (features_df['close'] - features_df['bb_lower']) / features_df['bb_width']
        
        # ATR (Average True Range)
        high_low = features_df['high'] - features_df['low']
        high_close = np.abs(features_df['high'] - features_df['close'].shift(1))
        low_close = np.abs(features_df['low'] - features_df['close'].shift(1))
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        features_df['atr_14'] = true_range.rolling(window=14).mean()
        
        # Volume indicators
        features_df['volume_sma_20'] = features_df['volume'].rolling(window=20).mean()
        features_df['volume_ratio'] = features_df['volume'] / features_df['volume_sma_20']
        
        # Price position indicators
        features_df['price_sma_ratio'] = features_df['close'] / features_df['sma_20']
        features_df['price_ema_ratio'] = features_df['close'] / features_df['ema_20']
        
        # Volatility
        features_df['volatility_20'] = features_df['returns'].rolling(window=20).std()
        
        return features_df
    
    def _prepare_features(self, recent_market_data: pd.DataFrame) -> np.ndarray:
        """Prepare features for inference from recent market data."""
        # Ensure we have OHLC data
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in recent_market_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Calculate technical indicators
        features_df = self._calculate_technical_indicators(recent_market_data)
        
        # Select only the features used in training
        if self.feature_names:
            features_df = features_df[self.feature_names]
        
        # Handle NaN values
        features_df = features_df.fillna(0)
        
        # Scale features
        features = features_df.values
        if self.scaler:
            features = self.scaler.transform(features)
        
        return features
    
    def _create_sequences(self, features: np.ndarray, sequence_length: int = 20) -> torch.Tensor:
        """Create sequences for LSTM inference."""
        if len(features) < sequence_length:
            raise ValueError(f"Insufficient data: need {sequence_length} timesteps, got {len(features)}")
        
        # Take the last sequence_length timesteps
        sequence = features[-sequence_length:]
        return torch.FloatTensor(sequence).unsqueeze(0)  # Add batch dimension
    
    async def forecast(self, recent_market_data: pd.DataFrame) -> Dict[str, float]:
        """
        Generate market forecast from recent market data.
        
        Args:
            recent_market_data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with predicted probabilities: {"prob_up": float, "prob_down": float, "prob_flat": float}
        """
        logger.info("Generating market forecast...")
        
        try:
            # Prepare features
            features = self._prepare_features(recent_market_data)
            
            # Create sequences
            sequence = self._create_sequences(features)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(sequence)
                probabilities = torch.softmax(outputs, dim=1)
                
            # Convert to probabilities
            probs = probabilities.numpy()[0]
            
            # Map to class names
            class_names = ['UP', 'DOWN', 'FLAT']
            if self.label_encoder and len(self.label_encoder.classes_) == 3:
                class_names = self.label_encoder.classes_
            
            # Create result dictionary
            result = {
                "prob_up": float(probs[0]) if class_names[0] == 'UP' else float(probs[1] if class_names[1] == 'UP' else probs[2]),
                "prob_down": float(probs[0]) if class_names[0] == 'DOWN' else float(probs[1] if class_names[1] == 'DOWN' else probs[2]),
                "prob_flat": float(probs[0]) if class_names[0] == 'FLAT' else float(probs[1] if class_names[1] == 'FLAT' else probs[2]),
                "model_version": self.model_run_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Ensure probabilities sum to 1.0
            total = sum([result["prob_up"], result["prob_down"], result["prob_flat"]])
            if abs(total - 1.0) > 0.001:
                logger.warning(f"Probabilities don't sum to 1.0: {total}")
                # Normalize
                for key in ["prob_up", "prob_down", "prob_flat"]:
                    result[key] /= total
            
            logger.info(f"Forecast generated: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the loaded model."""
        return {
            "model_run_id": self.model_run_id,
            "mlflow_uri": self.mlflow_uri,
            "feature_count": len(self.feature_names) if self.feature_names else 0,
            "classes": list(self.label_encoder.classes_) if self.label_encoder else ['UP', 'DOWN', 'FLAT'],
            "loaded_at": datetime.utcnow().isoformat()
        }

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_predictive_modeler():
        """Test the predictive modeler with sample data."""
        
        # This would normally use a real model_run_id from training
        model_run_id = "test_run_id"
        
        try:
            # Initialize modeler
            modeler = PredictiveMarketModeler(model_run_id)
            
            # Create sample market data
            sample_data = pd.DataFrame({
                'open': [1.1000, 1.1005, 1.1010, 1.1008, 1.1012],
                'high': [1.1005, 1.1010, 1.1015, 1.1013, 1.1017],
                'low': [1.0995, 1.1000, 1.1005, 1.1003, 1.1007],
                'close': [1.1002, 1.1007, 1.1012, 1.1010, 1.1014],
                'volume': [1000, 1200, 1100, 1300, 1250]
            })
            
            # Generate forecast
            forecast = await modeler.forecast(sample_data)
            print("Forecast:", forecast)
            
        except Exception as e:
            print(f"Test failed: {e}")
    
    # Run test
    asyncio.run(test_predictive_modeler())
