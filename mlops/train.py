#!/usr/bin/env python3
"""
Model Training Script - Epic 2: The Predictive Brain
Trains, validates, and registers a robust baseline LSTM model using rolling-origin cross-validation.

This script:
1. Loads the training dataset from Epic 1
2. Implements rolling-origin cross-validation (walk-forward validation)
3. Trains an LSTM model as baseline
4. Logs experiments to MLflow
5. Registers the trained model
"""

import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MarketDataset(Dataset):
    """PyTorch Dataset for market data with sequences."""

    def __init__(self, features: np.ndarray, targets: np.ndarray, sequence_length: int = 20):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.features) - self.sequence_length

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.features[idx : idx + self.sequence_length]),
            torch.LongTensor([self.targets[idx + self.sequence_length - 1]]),
        )


class LSTMModel(nn.Module):
    """LSTM model for market prediction."""

    def __init__(
        self, input_size: int, hidden_size: int = 64, num_layers: int = 2, num_classes: int = 3
    ):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


class ModelTrainer:
    """Handles model training with rolling-origin cross-validation."""

    def __init__(self, data_path: str = "data/training/v1_training_dataset.parquet"):
        self.data_path = Path(data_path)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def load_data(self) -> pd.DataFrame:
        """Load the training dataset."""
        logger.info(f"Loading training data from {self.data_path}")
        df = pd.read_parquet(self.data_path)
        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features and targets for training."""
        # Separate features and target
        feature_cols = [col for col in df.columns if col != "target"]
        X = df[feature_cols].values
        y = df["target"].values

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y_encoded, feature_cols

    def rolling_origin_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        train_size: int = 500,
        test_size: int = 100,
    ) -> List[Dict]:
        """Implement rolling-origin cross-validation (walk-forward validation)."""
        logger.info("Setting up rolling-origin cross-validation...")

        results = []
        start_idx = 0

        for fold in range(n_splits):
            # Define train/test indices
            train_end = start_idx + train_size
            test_end = train_end + test_size

            if test_end > len(X):
                logger.warning("Not enough data for full validation, adjusting...")
                test_end = len(X)
                train_end = max(train_size, test_end - test_size)

            train_idx = slice(start_idx, train_end)
            test_idx = slice(train_end, test_end)

            logger.info(
                f"Fold {fold + 1}: Train {train_idx.start}-{train_idx.stop}, Test {test_idx.start}-{test_idx.stop}"
            )

            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            results.append(
                {
                    "fold": fold + 1,
                    "X_train": X_train,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_test": y_test,
                    "train_start": train_idx.start,
                    "train_end": train_idx.stop,
                    "test_start": test_idx.start,
                    "test_end": test_idx.stop,
                }
            )

            start_idx += test_size

        return results

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        fold: int = 1,
    ) -> Dict[str, float]:
        """Train LSTM model for one fold."""
        logger.info(f"Training model for fold {fold}...")

        # Create datasets
        sequence_length = 20
        train_dataset = MarketDataset(X_train, y_train, sequence_length)
        test_dataset = MarketDataset(X_test, y_test, sequence_length)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Initialize model
        model = LSTMModel(input_size=X_train.shape[1])
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        epochs = 50
        best_accuracy = 0

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            correct = 0
            total = 0

            for batch_features, batch_targets in train_loader:
                optimizer.zero_grad()

                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets.squeeze())
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_targets.size(0)
                correct += (predicted == batch_targets.squeeze()).sum().item()

            train_accuracy = 100 * correct / total

            # Validation
            model.eval()
            test_correct = 0
            test_total = 0
            test_predictions = []
            test_targets = []

            with torch.no_grad():
                for batch_features, batch_targets in test_loader:
                    outputs = model(batch_features)
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += batch_targets.size(0)
                    test_correct += (predicted == batch_targets.squeeze()).sum().item()

                    test_predictions.extend(predicted.numpy())
                    test_targets.extend(batch_targets.squeeze().numpy())

            test_accuracy = 100 * test_correct / test_total

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy

            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}: Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%"
                )

        # Calculate final metrics
        test_f1 = f1_score(test_targets, test_predictions, average="weighted")

        return {
            "accuracy": test_accuracy,
            "f1_score": test_f1,
            "best_accuracy": best_accuracy,
            "model": model,
        }

    def run_training(self, experiment_name: str = "market_prediction"):
        """Run complete training pipeline."""
        logger.info("Starting model training...")

        # Set up MLflow
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():
            # Load and prepare data
            df = self.load_data()
            X, y, feature_names = self.prepare_features(df)

            # Log parameters
            mlflow.log_param("dataset_size", len(df))
            mlflow.log_param("features", len(feature_names))
            mlflow.log_param("classes", len(np.unique(y)))
            mlflow.log_param("model_type", "LSTM")
            mlflow.log_param("sequence_length", 20)

            # Rolling-origin cross-validation
            cv_results = self.rolling_origin_validation(X, y)

            fold_metrics = []
            best_model = None
            best_accuracy = 0

            for fold_data in cv_results:
                fold_result = self.train_model(
                    fold_data["X_train"],
                    fold_data["y_train"],
                    fold_data["X_test"],
                    fold_data["y_test"],
                    feature_names,
                    fold_data["fold"],
                )

                fold_metrics.append(
                    {
                        "fold": fold_data["fold"],
                        "accuracy": fold_result["accuracy"],
                        "f1_score": fold_result["f1_score"],
                    }
                )

                # Log fold metrics
                mlflow.log_metric(f"fold_{fold_data['fold']}_accuracy", fold_result["accuracy"])
                mlflow.log_metric(f"fold_{fold_data['fold']}_f1", fold_result["f1_score"])

                if fold_result["accuracy"] > best_accuracy:
                    best_accuracy = fold_result["accuracy"]
                    best_model = fold_result["model"]

            # Calculate average metrics
            avg_accuracy = np.mean([m["accuracy"] for m in fold_metrics])
            avg_f1 = np.mean([m["f1_score"] for m in fold_metrics])

            # Log final metrics
            mlflow.log_metric("avg_accuracy", avg_accuracy)
            mlflow.log_metric("avg_f1_score", avg_f1)

            # Save model
            if best_model is not None:
                mlflow.pytorch.log_model(best_model, "model")
                logger.info(f"Model saved to MLflow with avg accuracy: {avg_accuracy:.2f}%")

            # Save feature names
            with open("features.txt", "w") as f:
                for feat in feature_names:
                    f.write(f"{feat}\n")
            mlflow.log_artifact("features.txt")

            print("\n=== Training Results ===")
            print(f"Average Accuracy: {avg_accuracy:.2f}%")
            print(f"Average F1 Score: {avg_f1:.4f}")
            print(f"Best Fold Accuracy: {best_accuracy:.2f}%")
            print(f"\nModel registered in MLflow experiment: {experiment_name}")

            return {
                "avg_accuracy": avg_accuracy,
                "avg_f1_score": avg_f1,
                "best_accuracy": best_accuracy,
                "fold_metrics": fold_metrics,
            }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train market prediction model")
    parser.add_argument("--experiment", default="market_prediction", help="MLflow experiment name")
    parser.add_argument(
        "--data-path",
        default="data/training/v1_training_dataset.parquet",
        help="Path to training data",
    )

    args = parser.parse_args()

    trainer = ModelTrainer(args.data_path)
    results = trainer.run_training(args.experiment)

    print("\nTraining completed successfully!")
    return results


if __name__ == "__main__":
    main()
