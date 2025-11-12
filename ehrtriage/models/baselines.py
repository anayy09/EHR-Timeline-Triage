"""
Baseline models for risk prediction.

Implements Logistic Regression as the primary baseline model.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ehrtriage.config import get_default_model_config


class LogisticBaseline:
    """Logistic Regression baseline model."""

    def __init__(self, **kwargs):
        """
        Initialize logistic regression model.

        Args:
            **kwargs: Parameters for LogisticRegression
        """
        default_config = get_default_model_config("logistic")
        config = {**default_config, **kwargs}

        self.model = LogisticRegression(**config)
        self.scaler = StandardScaler()
        self.feature_names = None
        self.config = config

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_names: Optional[list] = None,
    ) -> "LogisticBaseline":
        """
        Fit the model.

        Args:
            X: Feature matrix
            y: Labels
            feature_names: Optional feature names

        Returns:
            Self
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        elif feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Fit model
        self.model.fit(X_scaled, y)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities.

        Args:
            X: Feature matrix

        Returns:
            Array of probabilities [N, 2]
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Feature matrix

        Returns:
            Array of predictions [N]
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance based on coefficients.

        Returns:
            DataFrame with features and their coefficients
        """
        if self.feature_names is None:
            raise ValueError("Model not fitted yet")

        coefficients = self.model.coef_[0]

        importance_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "coefficient": coefficients,
                "abs_coefficient": np.abs(coefficients),
            }
        )

        importance_df = importance_df.sort_values("abs_coefficient", ascending=False)

        return importance_df

    def get_top_features(self, k: int = 10) -> pd.DataFrame:
        """
        Get top k most important features.

        Args:
            k: Number of features to return

        Returns:
            DataFrame with top features
        """
        importance_df = self.get_feature_importance()
        return importance_df.head(k)

    def save(self, output_dir: Path, model_name: str = "logistic") -> None:
        """
        Save model to directory.

        Args:
            output_dir: Output directory
            model_name: Name for the model files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        joblib.dump(self.model, output_dir / f"{model_name}_model.joblib")
        joblib.dump(self.scaler, output_dir / f"{model_name}_scaler.joblib")

        # Save metadata
        metadata = {
            "model_type": "logistic_regression",
            "feature_names": self.feature_names,
            "n_features": len(self.feature_names) if self.feature_names else 0,
            "config": self.config,
        }

        with open(output_dir / f"{model_name}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved logistic model to {output_dir}")

    @classmethod
    def load(cls, input_dir: Path, model_name: str = "logistic") -> "LogisticBaseline":
        """
        Load model from directory.

        Args:
            input_dir: Input directory
            model_name: Name of the model files

        Returns:
            Loaded model
        """
        input_dir = Path(input_dir)

        # Load metadata
        with open(input_dir / f"{model_name}_metadata.json", "r") as f:
            metadata = json.load(f)

        # Create instance
        instance = cls(**metadata["config"])
        instance.model = joblib.load(input_dir / f"{model_name}_model.joblib")
        instance.scaler = joblib.load(input_dir / f"{model_name}_scaler.joblib")
        instance.feature_names = metadata["feature_names"]

        return instance


def train_logistic_model(
    train_X: pd.DataFrame,
    train_y: np.ndarray,
    val_X: Optional[pd.DataFrame] = None,
    val_y: Optional[np.ndarray] = None,
    **kwargs,
) -> Tuple[LogisticBaseline, Dict]:
    """
    Train a logistic regression model.

    Args:
        train_X: Training features
        train_y: Training labels
        val_X: Validation features (optional)
        val_y: Validation labels (optional)
        **kwargs: Additional model parameters

    Returns:
        Tuple of (trained model, metrics dict)
    """
    from ehrtriage.evaluation.metrics import compute_classification_metrics

    print("Training logistic regression model...")

    # Train model
    model = LogisticBaseline(**kwargs)
    model.fit(train_X, train_y)

    # Compute training metrics
    train_probs = model.predict_proba(train_X)[:, 1]
    train_metrics = compute_classification_metrics(train_y, train_probs)

    print(f"Training metrics:")
    print(f"  - AUROC: {train_metrics['auroc']:.4f}")
    print(f"  - AUPRC: {train_metrics['auprc']:.4f}")

    metrics = {"train": train_metrics}

    # Validation metrics
    if val_X is not None and val_y is not None:
        val_probs = model.predict_proba(val_X)[:, 1]
        val_metrics = compute_classification_metrics(val_y, val_probs)

        print(f"Validation metrics:")
        print(f"  - AUROC: {val_metrics['auroc']:.4f}")
        print(f"  - AUPRC: {val_metrics['auprc']:.4f}")

        metrics["val"] = val_metrics

    return model, metrics
