"""
Sequence feature builder for temporal models (GRU, Transformer).

Converts raw events into time-binned sequences suitable for RNN/Transformer models.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from ehrtriage.config import get_feature_config
from ehrtriage.features import extract_time_window


def create_time_bins(
    start_time: pd.Timestamp, end_time: pd.Timestamp, bin_hours: int
) -> List[pd.Timestamp]:
    """
    Create time bins between start and end time.

    Args:
        start_time: Window start
        end_time: Window end
        bin_hours: Hours per bin

    Returns:
        List of bin start timestamps
    """
    bins = []
    current = start_time

    while current < end_time:
        bins.append(current)
        current += pd.Timedelta(hours=bin_hours)

    return bins


def aggregate_events_in_bin(
    events_df: pd.DataFrame, bin_start: pd.Timestamp, bin_end: pd.Timestamp, config: Dict
) -> Dict[str, float]:
    """
    Aggregate all events within a single time bin.

    Args:
        events_df: Events DataFrame
        bin_start: Bin start time
        bin_end: Bin end time
        config: Feature configuration

    Returns:
        Dictionary of feature values for this bin
    """
    bin_events = events_df[
        (events_df["time"] >= bin_start) & (events_df["time"] < bin_end)
    ].copy()

    features = {}
    feature_config = config["features"]

    # Vitals - use mean in bin
    vitals = bin_events[bin_events["type"] == "vital"]
    for vital_name in feature_config["vitals"].keys():
        vital_values = vitals[vitals["code"] == vital_name]["value"]
        if len(vital_values) > 0:
            features[f"vital_{vital_name}"] = float(vital_values.mean())
        else:
            features[f"vital_{vital_name}"] = 0.0

    # Labs - use most recent in bin
    labs = bin_events[bin_events["type"] == "lab"]
    for lab_name in feature_config["labs"].keys():
        lab_values = labs[labs["code"] == lab_name]["value"]
        if len(lab_values) > 0:
            features[f"lab_{lab_name}"] = float(lab_values.iloc[-1])
        else:
            features[f"lab_{lab_name}"] = 0.0

    # Medications - binary presence
    meds = bin_events[bin_events["type"] == "medication"]
    for med_name in feature_config["medications"]:
        med_given = len(meds[meds["code"] == med_name]) > 0
        features[f"med_{med_name}"] = 1.0 if med_given else 0.0

    return features


def build_sequence_for_stay(
    events_df: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    subject_id: str,
    hadm_id: Optional[str],
    stay_id: Optional[str],
    config: Dict,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build time-binned sequence for a single stay.

    Args:
        events_df: All events
        start_time: Sequence start
        end_time: Sequence end
        subject_id: Patient ID
        hadm_id: Admission ID
        stay_id: ICU stay ID
        config: Configuration

    Returns:
        Tuple of (sequence array [T, D], mask array [T], feature names)
    """
    bin_hours = config["time_bin_hours"]
    max_length = config["sequence"]["max_length"]

    # Get events for this stay
    window_events = extract_time_window(
        events_df, start_time, end_time, subject_id, hadm_id, stay_id
    )

    # Create time bins
    time_bins = create_time_bins(start_time, end_time, bin_hours)

    # Aggregate features for each bin
    sequence_data = []
    feature_names = None

    for i in range(len(time_bins)):
        bin_start = time_bins[i]
        bin_end = time_bins[i + 1] if i + 1 < len(time_bins) else end_time

        bin_features = aggregate_events_in_bin(window_events, bin_start, bin_end, config)

        if feature_names is None:
            feature_names = sorted(bin_features.keys())

        # Create feature vector for this bin
        bin_vector = [bin_features.get(name, 0.0) for name in feature_names]
        sequence_data.append(bin_vector)

    # Convert to array
    sequence = np.array(sequence_data, dtype=np.float32)

    # Pad or truncate to max_length
    actual_length = len(sequence)
    mask = np.ones(min(actual_length, max_length), dtype=np.float32)

    if actual_length < max_length:
        # Pad
        padding = np.zeros((max_length - actual_length, sequence.shape[1]), dtype=np.float32)
        sequence = np.vstack([sequence, padding])
        mask = np.concatenate([mask, np.zeros(max_length - actual_length, dtype=np.float32)])
    elif actual_length > max_length:
        # Truncate (take most recent bins)
        sequence = sequence[-max_length:]
        mask = mask[-max_length:]

    return sequence, mask, feature_names


def build_sequence_features(
    events_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    patients_df: pd.DataFrame,
    task: str = "readmission",
    config: Optional[Dict] = None,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Build sequence features for all stays in cohort.

    Args:
        events_df: All clinical events
        cohort_df: Cohort DataFrame
        patients_df: Patient demographics
        task: Task type
        config: Feature configuration
        output_dir: Directory to save sequences (optional)

    Returns:
        Dictionary with sequences, masks, labels, and metadata
    """
    if config is None:
        config = get_feature_config()

    lookback_hours = config["tasks"][task]["lookback_hours"]

    sequences = []
    masks = []
    labels = []
    static_features = []
    feature_names = None

    for idx, row in cohort_df.iterrows():
        if task == "readmission":
            subject_id = row["subject_id"]
            hadm_id = row["hadm_id"]
            stay_id = None
            end_time = row["discharge_time"]
            start_time = end_time - pd.Timedelta(hours=lookback_hours)
            label = row["readmit_30d"]
        elif task == "icu_mortality":
            subject_id = row["subject_id"]
            hadm_id = row.get("hadm_id")
            stay_id = row["stay_id"]
            start_time = row["icu_intime"]
            end_time = start_time + pd.Timedelta(hours=lookback_hours)
            label = row["mortality_label"]
        else:
            raise ValueError(f"Unknown task: {task}")

        # Build sequence
        seq, mask, feat_names = build_sequence_for_stay(
            events_df, start_time, end_time, subject_id, hadm_id, stay_id, config
        )

        if feature_names is None:
            feature_names = feat_names

        sequences.append(seq)
        masks.append(mask)
        labels.append(label)

        # Static features
        patient = patients_df[patients_df["subject_id"] == subject_id].iloc[0]
        static = {
            "age": 2024 - patient["birth_year"],
            "sex_M": 1.0 if patient["sex"] == "M" else 0.0,
            "sex_F": 1.0 if patient["sex"] == "F" else 0.0,
        }

        for comorbidity in config["features"]["static"]:
            if comorbidity.startswith("comorbidity_"):
                static[comorbidity] = float(patient.get(comorbidity, False))

        static_features.append(static)

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(cohort_df)} sequences")

    # Convert to arrays
    sequences = np.array(sequences, dtype=np.float32)  # [N, T, D]
    masks = np.array(masks, dtype=np.float32)  # [N, T]
    labels = np.array(labels, dtype=np.int64)  # [N]
    static_features = pd.DataFrame(static_features).values.astype(np.float32)  # [N, S]

    print(f"\nBuilt sequence features for {task}:")
    print(f"  - Shape: {sequences.shape}")
    print(f"  - Features per timestep: {len(feature_names)}")
    print(f"  - Static features: {static_features.shape[1]}")
    print(f"  - Positive class: {labels.sum()} ({labels.mean():.1%})")

    result = {
        "sequences": sequences,
        "masks": masks,
        "labels": labels,
        "static_features": static_features,
        "feature_names": feature_names,
        "static_feature_names": list(pd.DataFrame(static_features).columns),
    }

    # Save if output directory provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / "sequences.npy", sequences)
        np.save(output_dir / "masks.npy", masks)
        np.save(output_dir / "labels.npy", labels)
        np.save(output_dir / "static_features.npy", static_features)

        import json
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(
                {
                    "feature_names": feature_names,
                    "static_feature_names": list(pd.DataFrame(static_features).columns),
                    "n_samples": len(labels),
                    "sequence_length": sequences.shape[1],
                    "n_features": sequences.shape[2],
                },
                f,
                indent=2,
            )

        print(f"Saved sequence features to {output_dir}")

    return result


def load_sequence_features(input_dir: Path) -> Dict:
    """
    Load sequence features from directory.

    Args:
        input_dir: Directory containing saved sequences

    Returns:
        Dictionary with sequences, masks, labels, and metadata
    """
    input_dir = Path(input_dir)

    sequences = np.load(input_dir / "sequences.npy")
    masks = np.load(input_dir / "masks.npy")
    labels = np.load(input_dir / "labels.npy")
    static_features = np.load(input_dir / "static_features.npy")

    import json
    with open(input_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    return {
        "sequences": sequences,
        "masks": masks,
        "labels": labels,
        "static_features": static_features,
        "feature_names": metadata["feature_names"],
        "static_feature_names": metadata["static_feature_names"],
    }


class SequenceDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for sequence data."""

    def __init__(
        self,
        sequences: np.ndarray,
        masks: np.ndarray,
        labels: np.ndarray,
        static_features: np.ndarray,
    ):
        self.sequences = torch.from_numpy(sequences)
        self.masks = torch.from_numpy(masks)
        self.labels = torch.from_numpy(labels)
        self.static_features = torch.from_numpy(static_features)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "sequence": self.sequences[idx],
            "mask": self.masks[idx],
            "static": self.static_features[idx],
            "label": self.labels[idx],
        }
