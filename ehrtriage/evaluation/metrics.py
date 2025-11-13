"""
Evaluation metrics for risk prediction models.

Includes AUROC, AUPRC, Brier score, calibration, etc.
"""

from typing import Dict, Tuple

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)


def compute_classification_metrics(
    y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True labels [N]
        y_pred_proba: Predicted probabilities [N]
        threshold: Decision threshold for binary classification

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # AUROC
    try:
        metrics["auroc"] = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        metrics["auroc"] = 0.5

    # AUPRC
    try:
        metrics["auprc"] = average_precision_score(y_true, y_pred_proba)
    except ValueError:
        metrics["auprc"] = 0.0

    # Brier score (lower is better)
    metrics["brier_score"] = brier_score_loss(y_true, y_pred_proba)

    # Binary classification metrics
    y_pred = (y_pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.shape == (1, 1):
        # Only one class present
        unique_class = np.unique(y_true)[0]
        if unique_class == 0:
            # All negative
            tn = cm[0, 0]
            fp = 0
            fn = 0
            tp = 0
        else:
            # All positive
            tn = 0
            fp = 0
            fn = 0
            tp = cm[0, 0]
    else:
        tn, fp, fn, tp = cm.ravel()

    metrics["accuracy"] = (tp + tn) / (tp + tn + fp + fn)
    metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics["ppv"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Precision
    metrics["npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    # F1 score
    if metrics["ppv"] + metrics["sensitivity"] > 0:
        metrics["f1_score"] = (
            2 * metrics["ppv"] * metrics["sensitivity"]
        ) / (metrics["ppv"] + metrics["sensitivity"])
    else:
        metrics["f1_score"] = 0.0

    return metrics


def compute_calibration_metrics(
    y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10
) -> Dict[str, any]:
    """
    Compute calibration metrics.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins for calibration curve

    Returns:
        Dictionary with calibration data
    """
    prob_true, prob_pred = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins, strategy="uniform"
    )

    # Expected Calibration Error (ECE)
    bin_counts, bin_edges = np.histogram(y_pred_proba, bins=n_bins, range=(0, 1))
    bin_indices = np.digitize(y_pred_proba, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_accuracy = np.mean(y_true[mask])
            bin_confidence = np.mean(y_pred_proba[mask])
            bin_weight = np.sum(mask) / len(y_true)
            ece += bin_weight * np.abs(bin_accuracy - bin_confidence)

    return {
        "prob_true": prob_true,
        "prob_pred": prob_pred,
        "ece": ece,
    }


def get_roc_curve_data(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict:
    """
    Get ROC curve data.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities

    Returns:
        Dictionary with FPR, TPR, thresholds
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)

    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auroc": roc_auc_score(y_true, y_pred_proba),
    }


def get_pr_curve_data(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict:
    """
    Get Precision-Recall curve data.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities

    Returns:
        Dictionary with precision, recall, thresholds
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

    return {
        "precision": precision,
        "recall": recall,
        "thresholds": thresholds,
        "auprc": average_precision_score(y_true, y_pred_proba),
    }


def compare_models(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
) -> Dict[str, Dict]:
    """
    Compare multiple models on the same test set.

    Args:
        y_true: True labels
        predictions: Dictionary mapping model names to predicted probabilities

    Returns:
        Dictionary mapping model names to their metrics
    """
    results = {}

    for model_name, y_pred_proba in predictions.items():
        metrics = compute_classification_metrics(y_true, y_pred_proba)
        results[model_name] = metrics

    return results


def print_metrics(metrics: Dict[str, float], title: str = "Metrics") -> None:
    """
    Print metrics in a formatted way.

    Args:
        metrics: Dictionary of metrics
        title: Title for the metrics
    """
    print(f"\n{title}")
    print("=" * 50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:20s}: {value:.4f}")
        else:
            print(f"  {key:20s}: {value}")
    print("=" * 50)


def print_comparison(results: Dict[str, Dict], metric_keys: list = None) -> None:
    """
    Print comparison of multiple models.

    Args:
        results: Dictionary of model results
        metric_keys: Keys to print (if None, print all)
    """
    if metric_keys is None:
        # Get all metric keys from first model
        metric_keys = list(next(iter(results.values())).keys())

    print("\nModel Comparison")
    print("=" * 80)

    # Header
    header = f"{'Metric':<20s}"
    for model_name in results.keys():
        header += f"{model_name:<15s}"
    print(header)
    print("-" * 80)

    # Metrics
    for metric_key in metric_keys:
        row = f"{metric_key:<20s}"
        for model_name, metrics in results.items():
            value = metrics.get(metric_key, 0.0)
            if isinstance(value, float):
                row += f"{value:<15.4f}"
            else:
                row += f"{str(value):<15s}"
        print(row)

    print("=" * 80)
