"""
Visualization functions for model evaluation.

Creates ROC curves, PR curves, calibration plots, etc.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ehrtriage.evaluation.metrics import (
    get_roc_curve_data,
    get_pr_curve_data,
    compute_calibration_metrics,
)


# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


def plot_roc_curves(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    title: str = "ROC Curves",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot ROC curves for multiple models.

    Args:
        y_true: True labels
        predictions: Dict mapping model names to predicted probabilities
        title: Plot title
        save_path: Path to save figure (optional)

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    for model_name, y_pred_proba in predictions.items():
        roc_data = get_roc_curve_data(y_true, y_pred_proba)
        ax.plot(
            roc_data["fpr"],
            roc_data["tpr"],
            label=f"{model_name} (AUROC={roc_data['auroc']:.3f})",
            linewidth=2,
        )

    # Diagonal line
    ax.plot([0, 1], [0, 1], "k--", label="Random", linewidth=1)

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved ROC curve to {save_path}")

    return fig


def plot_pr_curves(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    title: str = "Precision-Recall Curves",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot Precision-Recall curves for multiple models.

    Args:
        y_true: True labels
        predictions: Dict mapping model names to predicted probabilities
        title: Plot title
        save_path: Path to save figure (optional)

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    baseline_rate = np.mean(y_true)

    for model_name, y_pred_proba in predictions.items():
        pr_data = get_pr_curve_data(y_true, y_pred_proba)
        ax.plot(
            pr_data["recall"],
            pr_data["precision"],
            label=f"{model_name} (AUPRC={pr_data['auprc']:.3f})",
            linewidth=2,
        )

    # Baseline
    ax.axhline(
        y=baseline_rate,
        color="k",
        linestyle="--",
        label=f"Baseline (rate={baseline_rate:.3f})",
        linewidth=1,
    )

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved PR curve to {save_path}")

    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    n_bins: int = 10,
    title: str = "Calibration Curves",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot calibration curves (reliability diagrams).

    Args:
        y_true: True labels
        predictions: Dict mapping model names to predicted probabilities
        n_bins: Number of bins
        title: Plot title
        save_path: Path to save figure (optional)

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    for model_name, y_pred_proba in predictions.items():
        cal_data = compute_calibration_metrics(y_true, y_pred_proba, n_bins)
        ax.plot(
            cal_data["prob_pred"],
            cal_data["prob_true"],
            marker="o",
            label=f"{model_name} (ECE={cal_data['ece']:.3f})",
            linewidth=2,
            markersize=8,
        )

    # Perfect calibration
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=1)

    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved calibration curve to {save_path}")

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels (not probabilities)
        labels: Class labels
        title: Plot title
        save_path: Path to save figure (optional)

    Returns:
        matplotlib Figure
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    if labels is None:
        labels = ["Negative", "Positive"]

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={"label": "Count"},
    )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved confusion matrix to {save_path}")

    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training History",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot training history (loss and metrics over epochs).

    Args:
        history: Dictionary with training history
        title: Plot title
        save_path: Path to save figure (optional)

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    if "train_loss" in history:
        axes[0].plot(history["train_loss"], label="Train Loss", linewidth=2)
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="Val Loss", linewidth=2)

    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training and Validation Loss", fontsize=12, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Metrics
    if "val_auroc" in history:
        axes[1].plot(history["val_auroc"], label="AUROC", linewidth=2)
    if "val_auprc" in history:
        axes[1].plot(history["val_auprc"], label="AUPRC", linewidth=2)

    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Score", fontsize=12)
    axes[1].set_title("Validation Metrics", fontsize=12, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved training history to {save_path}")

    return fig


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    top_k: int = 20,
    title: str = "Feature Importance",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot feature importance.

    Args:
        feature_names: List of feature names
        importances: Feature importance values
        top_k: Number of top features to show
        title: Plot title
        save_path: Path to save figure (optional)

    Returns:
        matplotlib Figure
    """
    # Sort by absolute importance
    indices = np.argsort(np.abs(importances))[::-1][:top_k]

    fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.3)))

    y_pos = np.arange(len(indices))
    colors = ["green" if importances[i] > 0 else "red" for i in indices]

    ax.barh(y_pos, importances[indices], color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved feature importance to {save_path}")

    return fig


def create_evaluation_report(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    output_dir: Path,
    task_name: str = "prediction",
) -> None:
    """
    Create a complete evaluation report with all plots.

    Args:
        y_true: True labels
        predictions: Dict mapping model names to predicted probabilities
        output_dir: Directory to save plots
        task_name: Name of the prediction task
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating evaluation report for {task_name}...")

    # ROC curves
    plot_roc_curves(
        y_true,
        predictions,
        title=f"ROC Curves - {task_name}",
        save_path=output_dir / f"{task_name}_roc.png",
    )
    plt.close()

    # PR curves
    plot_pr_curves(
        y_true,
        predictions,
        title=f"Precision-Recall Curves - {task_name}",
        save_path=output_dir / f"{task_name}_pr.png",
    )
    plt.close()

    # Calibration
    plot_calibration_curve(
        y_true,
        predictions,
        title=f"Calibration Curves - {task_name}",
        save_path=output_dir / f"{task_name}_calibration.png",
    )
    plt.close()

    print(f"Evaluation report saved to {output_dir}")
