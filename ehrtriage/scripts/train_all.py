"""
Complete training pipeline for all models.

This script:
1. Loads or generates synthetic data
2. Builds cohorts for both tasks
3. Engineers features (snapshot and sequence)
4. Trains baseline and sequence models
5. Evaluates and saves results

Usage:
    python -m ehrtriage.scripts.train_all
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ehrtriage.config import (
    DATA_DIR,
    MODELS_DIR,
    get_default_model_config,
    get_feature_config,
)
from ehrtriage.cohort import (
    build_readmission_cohort,
    build_icu_mortality_cohort,
    split_cohort,
    save_cohort,
)
from ehrtriage.features import (
    build_snapshot_features,
    load_features,
    normalize_features,
    save_features,
)
from ehrtriage.sequence_builder import (
    SequenceDataset,
    build_sequence_features,
    load_sequence_features,
)
from ehrtriage.models.baselines import train_logistic_model
from ehrtriage.models.sequence import (
    GRURiskModel,
    TransformerRiskModel,
    train_sequence_model,
    save_sequence_model,
)
from ehrtriage.evaluation.metrics import compute_classification_metrics, print_comparison
from ehrtriage.evaluation.plots import create_evaluation_report
from ehrtriage.synthetic_data import generate_synthetic_data


def load_or_generate_data():
    """Load existing synthetic data or generate new."""
    synthetic_dir = DATA_DIR / "synthetic"

    # Check if data exists
    if (synthetic_dir / "admissions.parquet").exists():
        print("Loading existing synthetic data...")
        admissions = pd.read_parquet(synthetic_dir / "admissions.parquet")
        patients = pd.read_parquet(synthetic_dir / "patients.parquet")
        icustays = pd.read_parquet(synthetic_dir / "icustays.parquet")
        events = pd.read_parquet(synthetic_dir / "events.parquet")
    else:
        print("Generating new synthetic data...")
        data = generate_synthetic_data()
        admissions = data["admissions"]
        patients = data["patients"]
        icustays = data["icustays"]
        events = data["events"]

    return admissions, patients, icustays, events

def _snapshot_features_path(task_name: str, split_name: str) -> Path:
    """Return the cache path for snapshot features."""

    return DATA_DIR / "processed" / f"{task_name}_snapshot_{split_name}.parquet"


def _load_or_build_snapshot_features(
    task_name: str,
    split_name: str,
    cohort_split: pd.DataFrame,
    events_df: pd.DataFrame,
    patients_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Build snapshot features and cache them on disk for reuse."""

    cache_path = _snapshot_features_path(task_name, split_name)
    if cache_path.exists():
        print(
            f"Loading cached snapshot features for {task_name} [{split_name}] from {cache_path}"
        )
        return load_features(cache_path)

    print(f"Building snapshot features for {task_name} [{split_name}]...")
    features = build_snapshot_features(events_df, cohort_split, patients_df, task_name, config)
    save_features(features, str(cache_path))
    return features


def _load_or_build_sequence_features(
    task_name: str,
    split_name: str,
    cohort_split: pd.DataFrame,
    events_df: pd.DataFrame,
    patients_df: pd.DataFrame,
    config: dict,
) -> dict:
    """Build sequence features and cache them to disk for reuse."""

    output_dir = DATA_DIR / "processed" / f"{task_name}_sequence_{split_name}"
    required_files = [
        output_dir / "sequences.npy",
        output_dir / "masks.npy",
        output_dir / "labels.npy",
        output_dir / "static_features.npy",
    ]

    if all(path.exists() for path in required_files):
        print(
            f"Loading cached sequence features for {task_name} [{split_name}] from {output_dir}"
        )
        return load_sequence_features(output_dir)

    print(f"Building sequence features for {task_name} [{split_name}]...")
    return build_sequence_features(
        events_df,
        cohort_split,
        patients_df,
        task_name,
        config,
        output_dir,
    )

def train_task(
    task_name: str,
    cohort_df: pd.DataFrame,
    events_df: pd.DataFrame,
    patients_df: pd.DataFrame,
    config: dict,
):
    """
    Train all models for a single task.

    Args:
        task_name: Name of the task
        cohort_df: Cohort DataFrame
        events_df: Events DataFrame
        patients_df: Patients DataFrame
        config: Feature configuration
    """
    print(f"\n{'='*60}")
    print(f"Training models for: {task_name}")
    print(f"{'='*60}\n")

    # Split cohort
    label_col = "readmit_30d" if task_name == "readmission" else "mortality_label"
    train_cohort, val_cohort, test_cohort = split_cohort(cohort_df, stratify_col=label_col)

    print(f"Split sizes:")
    print(f"  Train: {len(train_cohort)}")
    print(f"  Val: {len(val_cohort)}")
    print(f"  Test: {len(test_cohort)}")

    # Output directory
    output_dir = MODELS_DIR / "artifacts" / task_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ===== Baseline Model (Logistic Regression) =====
    print(f"\n{'='*60}")
    print("Training Baseline Model (Logistic Regression)")
    print(f"{'='*60}\n")

    # Build snapshot features
    print("Building snapshot features...")
    # Build or load snapshot features for each split
    train_features = _load_or_build_snapshot_features(
        task_name, "train", train_cohort, events_df, patients_df, config
    )
    val_features = _load_or_build_snapshot_features(
        task_name, "val", val_cohort, events_df, patients_df, config
    )
    test_features = _load_or_build_snapshot_features(
        task_name, "test", test_cohort, events_df, patients_df, config
    )

    # Normalize
    train_features, val_features, test_features = normalize_features(
        train_features, val_features, test_features
    )

    # Separate features and labels
    exclude_cols = ["subject_id", "hadm_id", "stay_id", "label"]
    feature_cols = [c for c in train_features.columns if c not in exclude_cols]

    train_X = train_features[feature_cols]
    train_y = train_features["label"].values
    val_X = val_features[feature_cols]
    val_y = val_features["label"].values
    test_X = test_features[feature_cols]
    test_y = test_features["label"].values

    # Train
    logistic_model, logistic_metrics = train_logistic_model(
        train_X, train_y, val_X, val_y
    )

    # Test evaluation
    test_probs = logistic_model.predict_proba(test_X)[:, 1]
    test_metrics = compute_classification_metrics(test_y, test_probs)

    print(f"\nLogistic Regression Test Metrics:")
    print(f"  AUROC: {test_metrics['auroc']:.4f}")
    print(f"  AUPRC: {test_metrics['auprc']:.4f}")
    print(f"  Brier: {test_metrics['brier_score']:.4f}")

    # Save model
    logistic_model.save(output_dir, "logistic")

    # Save metrics
    with open(output_dir / "logistic_metrics.json", "w") as f:
        json.dump(
            {
                "train": logistic_metrics.get("train", {}),
                "val": logistic_metrics.get("val", {}),
                "test": test_metrics,
            },
            f,
            indent=2,
        )

    # ===== Sequence Models =====
    print(f"\n{'='*60}")
    print("Training Sequence Models")
    print(f"{'='*60}\n")

    # Build sequence features
    print("Building sequence features...")
    train_seq_dir = DATA_DIR / "processed" / f"{task_name}_sequence_train"
    train_seq_data = _load_or_build_sequence_features(
        task_name, "train", train_cohort, events_df, patients_df, config
    )

    val_seq_dir = DATA_DIR / "processed" / f"{task_name}_sequence_val"
    val_seq_data = _load_or_build_sequence_features(
        task_name, "val", val_cohort, events_df, patients_df, config
    )

    test_seq_dir = DATA_DIR / "processed" / f"{task_name}_sequence_test"
    test_seq_data = _load_or_build_sequence_features(
        task_name, "test", test_cohort, events_df, patients_df, config
    )

    # Create datasets
    train_dataset = SequenceDataset(
        train_seq_data["sequences"],
        train_seq_data["masks"],
        train_seq_data["labels"],
        train_seq_data["static_features"],
    )

    val_dataset = SequenceDataset(
        val_seq_data["sequences"],
        val_seq_data["masks"],
        val_seq_data["labels"],
        val_seq_data["static_features"],
    )

    test_dataset = SequenceDataset(
        test_seq_data["sequences"],
        test_seq_data["masks"],
        test_seq_data["labels"],
        test_seq_data["static_features"],
    )

    # Get dimensions
    input_dim = train_seq_data["sequences"].shape[2]
    static_dim = train_seq_data["static_features"].shape[1]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Data loaders
    gru_config = get_default_model_config("gru")
    batch_size = gru_config.get("batch_size", 32)
    num_workers = min(4, (os.cpu_count() or 1))
    pin_memory = device == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    # Train GRU model
    print("\nTraining GRU model...")
    gru_model = GRURiskModel(
        input_dim=input_dim,
        static_dim=static_dim,
        hidden_dim=gru_config.get("hidden_dim", 64),
        num_layers=gru_config.get("num_layers", 2),
        dropout=gru_config.get("dropout", 0.3),
        bidirectional=gru_config.get("bidirectional", True),
    )

    gru_model.to(device)

    gru_model, gru_history = train_sequence_model(
        gru_model,
        train_loader,
        val_loader,
        num_epochs=gru_config.get("epochs", 50),
        learning_rate=gru_config.get("learning_rate", 0.001),
        weight_decay=gru_config.get("weight_decay", 0.0001),
        early_stopping_patience=gru_config.get("early_stopping_patience", 10),
        device=device,
    )

    # Save GRU model
    gru_config = {
        "input_dim": input_dim,
        "static_dim": static_dim,
        "hidden_dim": gru_config.get("hidden_dim", 64),
        "num_layers": gru_config.get("num_layers", 2),
        "dropout": gru_config.get("dropout", 0.3),
        "bidirectional": gru_config.get("bidirectional", True),
    }
    save_sequence_model(gru_model, output_dir, "gru", gru_config, gru_history)

    # Test GRU
    gru_model.eval()
    gru_test_probs = []
    gru_test_labels = []

    with torch.no_grad():
        for batch in test_loader:
            sequence = batch["sequence"].to(device)
            static = batch["static"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["label"].to(device)

            logits = gru_model(sequence, static, mask).squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()

            gru_test_probs.extend(probs)
            gru_test_labels.extend(labels.cpu().numpy())

    gru_test_metrics = compute_classification_metrics(
        np.array(gru_test_labels), np.array(gru_test_probs)
    )

    print(f"\nGRU Test Metrics:")
    print(f"  AUROC: {gru_test_metrics['auroc']:.4f}")
    print(f"  AUPRC: {gru_test_metrics['auprc']:.4f}")

    # Save GRU metrics
    with open(output_dir / "gru_metrics.json", "w") as f:
        json.dump({"test": gru_test_metrics}, f, indent=2)

    # ===== Comparison and Evaluation Report =====
    print(f"\n{'='*60}")
    print("Model Comparison")
    print(f"{'='*60}\n")

    all_predictions = {
        "Logistic Regression": test_probs,
        "GRU": np.array(gru_test_probs),
    }

    from ehrtriage.evaluation.metrics import compare_models

    comparison = compare_models(test_y, all_predictions)
    print_comparison(comparison, ["auroc", "auprc", "brier_score", "f1_score"])

    # Generate evaluation plots
    plot_dir = output_dir / "plots"
    create_evaluation_report(test_y, all_predictions, plot_dir, task_name)

    print(f"\nModel training complete for {task_name}!")
    print(f"Models saved to: {output_dir}")


def main():
    """Main training pipeline."""
    print("="*60)
    print("EHR Timeline Triage - Model Training Pipeline")
    print("="*60)

    # Load configuration
    config = get_feature_config()

    # Load or generate data
    admissions, patients, icustays, events = load_or_generate_data()

    # ===== Task 1: Readmission Prediction =====
    print("\n" + "="*60)
    print("Building Readmission Cohort")
    print("="*60)

    readmission_cohort = build_readmission_cohort(admissions)
    save_cohort(readmission_cohort, DATA_DIR / "processed" / "cohort_readmission.parquet")

    train_task("readmission", readmission_cohort, events, patients, config)

    # ===== Task 2: ICU Mortality Prediction =====
    print("\n" + "="*60)
    print("Building ICU Mortality Cohort")
    print("="*60)

    icu_mortality_cohort = build_icu_mortality_cohort(icustays, admissions)
    save_cohort(icu_mortality_cohort, DATA_DIR / "processed" / "cohort_icu_mortality.parquet")

    train_task("icu_mortality", icu_mortality_cohort, events, patients, config)

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nModels and results saved to: {MODELS_DIR / 'artifacts'}")
    print("\nNext steps:")
    print("  1. Start the API: uvicorn api.app:app --reload")
    print("  2. View documentation: http://localhost:8000/docs")
    print("  3. Make predictions using the API endpoints")


if __name__ == "__main__":
    main()
