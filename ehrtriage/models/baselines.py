"""
Baseline models for risk prediction.

Implements enhanced Logistic Regression as the primary baseline model.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from ehrtriage.config import get_default_model_config


class LogisticBaseline:
    """Logistic Regression baseline model with preprocessing and calibration."""

    def __init__(self, **kwargs):
        """
        Initialize logistic regression model.

        Args:
            **kwargs: Parameters for LogisticRegression
        """
        default_config = get_default_model_config("logistic")
        config = {**default_config, **kwargs}

        self.model = LogisticRegression(**config)
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.selector = VarianceThreshold(threshold=0.0)
        self.feature_names = None
        self.selected_feature_names = None
        self.calibrator: Optional[CalibratedClassifierCV] = None
        self.calibration_method = "isotonic"
        self.decision_threshold = 0.5
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
        X_array = self._prepare_features(X, fit=True, feature_names=feature_names)

        # Reset calibration when refitting
        self.calibrator = None
        self.decision_threshold = 0.5

        # Fit model
        self.model.fit(X_array, y)

        return self

    def calibrate(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        method: Optional[str] = None,
    ) -> None:
        """
        Fit a probability calibrator using held-out data.

        Args:
            X: Validation features
            y: Validation labels
            method: Calibration method ('sigmoid' or 'isotonic')
        """
        if len(np.unique(y)) < 2:
            # Not enough signal to calibrate
            return

        calibration_method = method or self.calibration_method
        X_array = self._prepare_features(X, fit=False)
        self.calibrator = CalibratedClassifierCV(
            estimator=self.model,
            method=calibration_method,
            cv="prefit",
        )
        self.calibrator.fit(X_array, y)
        self.calibration_method = calibration_method

    def set_decision_threshold(self, threshold: float) -> None:
        """Override the classification decision threshold."""
        self.decision_threshold = float(np.clip(threshold, 0.0, 1.0))

    def _prepare_features(
        self,
        X: pd.DataFrame,
        fit: bool = False,
        feature_names: Optional[list] = None,
    ) -> np.ndarray:
        """Apply preprocessing chain (impute -> scale -> variance filter)."""
        if isinstance(X, pd.DataFrame):
            data = X.values
            cols = list(X.columns)
        else:
            data = np.asarray(X)
            if feature_names is not None:
                cols = feature_names
            elif self.feature_names is not None:
                cols = self.feature_names
            else:
                cols = [f"feature_{i}" for i in range(data.shape[1])]

        if fit or self.feature_names is None:
            self.feature_names = feature_names or cols
        elif len(cols) != len(self.feature_names):
            raise ValueError(
                "Feature dimension mismatch between training and inference inputs"
            )

        if fit:
            data = self.imputer.fit_transform(data)
            data = self.scaler.fit_transform(data)
            data = self.selector.fit_transform(data)

            support_mask = self.selector.get_support()
            self.selected_feature_names = [
                name for name, keep in zip(self.feature_names, support_mask) if keep
            ]
            if len(self.selected_feature_names) == 0:
                raise ValueError("VarianceThreshold removed all features")
        else:
            data = self.imputer.transform(data)
            data = self.scaler.transform(data)
            data = self.selector.transform(data)

        return data

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities.

        Args:
            X: Feature matrix

        Returns:
            Array of probabilities [N, 2]
        """
        X_array = self._prepare_features(X, fit=False)

        if self.calibrator is not None:
            return self.calibrator.predict_proba(X_array)
        return self.model.predict_proba(X_array)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Feature matrix

        Returns:
            Array of predictions [N]
        """
        probs = self.predict_proba(X)[:, 1]
        threshold = getattr(self, "decision_threshold", 0.5)
        return (probs >= threshold).astype(int)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance based on coefficients.

        Returns:
            DataFrame with features and their coefficients
        """
        if self.feature_names is None:
            raise ValueError("Model not fitted yet")

        coefficients = self.model.coef_[0]
        feature_list = self.selected_feature_names or self.feature_names

        importance_df = pd.DataFrame(
            {
                "feature": feature_list,
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
        joblib.dump(self.imputer, output_dir / f"{model_name}_imputer.joblib")
        joblib.dump(self.scaler, output_dir / f"{model_name}_scaler.joblib")
        joblib.dump(self.selector, output_dir / f"{model_name}_selector.joblib")
        if self.calibrator is not None:
            joblib.dump(self.calibrator, output_dir / f"{model_name}_calibrator.joblib")

        # Save metadata
        metadata = {
            "model_type": "logistic_regression",
            "feature_names": self.feature_names,
            "selected_feature_names": self.selected_feature_names,
            "n_features": len(self.feature_names) if self.feature_names else 0,
            "config": self.config,
            "decision_threshold": self.decision_threshold,
            "calibration_method": self.calibration_method,
            "calibrated": self.calibrator is not None,
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
        instance.feature_names = metadata["feature_names"]
        instance.selected_feature_names = metadata.get("selected_feature_names")

        # Load model components
        instance.model = joblib.load(input_dir / f"{model_name}_model.joblib")
        instance.imputer = joblib.load(input_dir / f"{model_name}_imputer.joblib")
        instance.scaler = joblib.load(input_dir / f"{model_name}_scaler.joblib")
        instance.selector = joblib.load(input_dir / f"{model_name}_selector.joblib")
        instance.decision_threshold = metadata.get("decision_threshold", 0.5)
        instance.calibration_method = metadata.get("calibration_method", "isotonic")

        calibrator_path = input_dir / f"{model_name}_calibrator.joblib"
        if metadata.get("calibrated") and calibrator_path.exists():
            instance.calibrator = joblib.load(calibrator_path)

        return instance


def find_optimal_threshold(
    y_true: np.ndarray, probs: np.ndarray, metric: str = "f1"
) -> Tuple[float, float]:
    """
    Search for the decision threshold that maximizes a metric.

    Args:
        y_true: True labels
        probs: Predicted probabilities
        metric: Metric to optimize (currently supports 'f1')

    Returns:
        Tuple of (best_threshold, best_score)
    """
    if metric != "f1":
        raise ValueError(f"Unsupported metric for threshold search: {metric}")

    thresholds = np.linspace(0.1, 0.9, 33)
    best_threshold = 0.5
    best_score = -np.inf

    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        score = f1_score(y_true, preds)
        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


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

    # Validation metrics / calibration
    if val_X is not None and val_y is not None:
        # Calibrate probabilities if possible
        model.calibrate(val_X, val_y)
        val_probs = model.predict_proba(val_X)[:, 1]
        val_metrics = compute_classification_metrics(val_y, val_probs)

        best_threshold, best_score = find_optimal_threshold(val_y, val_probs)
        model.set_decision_threshold(best_threshold)
        val_metrics["optimal_threshold"] = best_threshold
        val_metrics["optimal_f1"] = best_score

        print(f"Validation metrics:")
        print(f"  - AUROC: {val_metrics['auroc']:.4f}")
        print(f"  - AUPRC: {val_metrics['auprc']:.4f}")
        print(f"  - Optimal F1: {best_score:.4f} @ threshold={best_threshold:.2f}")

        metrics["val"] = val_metrics

    return model, metrics
