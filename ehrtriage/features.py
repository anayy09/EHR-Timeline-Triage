"""
Feature engineering for EHR timelines.

Converts raw events into aggregated snapshot features for traditional ML models.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from ehrtriage.config import get_feature_config


def extract_time_window(
    events_df: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    subject_id: Optional[str] = None,
    hadm_id: Optional[str] = None,
    stay_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Extract events within a time window for a specific stay.

    Args:
        events_df: All events
        start_time: Window start
        end_time: Window end
        subject_id: Patient ID (optional filter)
        hadm_id: Admission ID (optional filter)
        stay_id: ICU stay ID (optional filter)

    Returns:
        Filtered events DataFrame
    """
    mask = (events_df["time"] >= start_time) & (events_df["time"] <= end_time)

    if subject_id is not None:
        mask &= events_df["subject_id"] == subject_id
    if hadm_id is not None:
        mask &= events_df["hadm_id"] == hadm_id
    if stay_id is not None:
        mask &= events_df["stay_id"] == stay_id

    return events_df[mask].copy()


def aggregate_vital_features(events_df: pd.DataFrame, vital_codes: List[str]) -> Dict[str, float]:
    """
    Aggregate vital signs into statistical features.

    Args:
        events_df: Events in the window
        vital_codes: List of vital sign codes to process

    Returns:
        Dictionary of aggregated features
    """
    features = {}

    vitals = events_df[events_df["type"] == "vital"]

    for code in vital_codes:
        code_events = vitals[vitals["code"] == code]

        if len(code_events) > 0:
            values = code_events["value"].values
            features[f"{code}_mean"] = np.mean(values)
            features[f"{code}_min"] = np.min(values)
            features[f"{code}_max"] = np.max(values)
            features[f"{code}_std"] = np.std(values) if len(values) > 1 else 0.0
            features[f"{code}_count"] = len(values)
        else:
            features[f"{code}_mean"] = np.nan
            features[f"{code}_min"] = np.nan
            features[f"{code}_max"] = np.nan
            features[f"{code}_std"] = 0.0
            features[f"{code}_count"] = 0

    return features


def aggregate_lab_features(events_df: pd.DataFrame, lab_codes: List[str]) -> Dict[str, float]:
    """
    Aggregate lab values into statistical features.

    Args:
        events_df: Events in the window
        lab_codes: List of lab codes to process

    Returns:
        Dictionary of aggregated features
    """
    features = {}

    labs = events_df[events_df["type"] == "lab"]

    for code in lab_codes:
        code_events = labs[labs["code"] == code]

        if len(code_events) > 0:
            values = code_events["value"].values
            features[f"{code}_mean"] = np.mean(values)
            features[f"{code}_min"] = np.min(values)
            features[f"{code}_max"] = np.max(values)
            features[f"{code}_last"] = values[-1]  # Most recent value
            features[f"{code}_count"] = len(values)
        else:
            features[f"{code}_mean"] = np.nan
            features[f"{code}_min"] = np.nan
            features[f"{code}_max"] = np.nan
            features[f"{code}_last"] = np.nan
            features[f"{code}_count"] = 0

    return features


def aggregate_medication_features(
    events_df: pd.DataFrame, med_codes: List[str]
) -> Dict[str, float]:
    """
    Aggregate medication administration into features.

    Args:
        events_df: Events in the window
        med_codes: List of medication codes

    Returns:
        Dictionary of medication flags and counts
    """
    features = {}

    meds = events_df[events_df["type"] == "medication"]

    for code in med_codes:
        code_events = meds[meds["code"] == code]
        features[f"med_{code}_given"] = 1.0 if len(code_events) > 0 else 0.0
        features[f"med_{code}_count"] = len(code_events)

    return features


def build_snapshot_features(
    events_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    patients_df: pd.DataFrame,
    task: str = "readmission",
    config: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Build snapshot features for each stay in the cohort.

    Args:
        events_df: All clinical events
        cohort_df: Cohort DataFrame with time windows
        patients_df: Patient demographics and comorbidities
        task: Task type ('readmission' or 'icu_mortality')
        config: Feature configuration

    Returns:
        DataFrame with one row per stay and aggregated features
    """
    if config is None:
        config = get_feature_config()

    feature_config = config["features"]
    vital_codes = list(feature_config["vitals"].keys())
    lab_codes = list(feature_config["labs"].keys())
    med_codes = feature_config["medications"]

    # Determine time window based on task
    lookback_hours = config["tasks"][task]["lookback_hours"]

    feature_rows = []

    for _, row in cohort_df.iterrows():
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

        # Extract events in window
        window_events = extract_time_window(
            events_df, start_time, end_time, subject_id, hadm_id, stay_id
        )

        # Aggregate features
        features = {}
        features["subject_id"] = subject_id
        features["hadm_id"] = hadm_id if hadm_id else ""
        features["stay_id"] = stay_id if stay_id else ""

        # Vital signs
        features.update(aggregate_vital_features(window_events, vital_codes))

        # Labs
        features.update(aggregate_lab_features(window_events, lab_codes))

        # Medications
        features.update(aggregate_medication_features(window_events, med_codes))

        # Static features from patients
        patient = patients_df[patients_df["subject_id"] == subject_id].iloc[0]
        features["age"] = 2024 - patient["birth_year"]
        features["sex_M"] = 1.0 if patient["sex"] == "M" else 0.0
        features["sex_F"] = 1.0 if patient["sex"] == "F" else 0.0

        for comorbidity in feature_config["static"]:
            if comorbidity.startswith("comorbidity_"):
                features[comorbidity] = float(patient.get(comorbidity, False))

        # Label
        features["label"] = label

        feature_rows.append(features)

    feature_df = pd.DataFrame(feature_rows)

    print(f"Built snapshot features for {task}: {feature_df.shape}")
    print(f"  - Positive class: {feature_df['label'].sum()} ({feature_df['label'].mean():.1%})")

    return feature_df


def normalize_features(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    method: str = "standard",
    exclude_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, ...]:
    """
    Normalize features using training set statistics.

    Args:
        train_df: Training features
        val_df: Validation features (optional)
        test_df: Test features (optional)
        method: Normalization method ('standard', 'minmax', 'robust')
        exclude_cols: Columns to exclude from normalization

    Returns:
        Tuple of normalized DataFrames
    """
    if exclude_cols is None:
        exclude_cols = ["subject_id", "hadm_id", "stay_id", "label"]

    # Get numeric columns to normalize
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_norm = [c for c in numeric_cols if c not in exclude_cols]

    # Select scaler
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Fit on training data
    train_df = train_df.copy()
    train_df[cols_to_norm] = scaler.fit_transform(train_df[cols_to_norm].fillna(0))

    results = [train_df]

    # Transform validation and test
    if val_df is not None:
        val_df = val_df.copy()
        val_df[cols_to_norm] = scaler.transform(val_df[cols_to_norm].fillna(0))
        results.append(val_df)

    if test_df is not None:
        test_df = test_df.copy()
        test_df[cols_to_norm] = scaler.transform(test_df[cols_to_norm].fillna(0))
        results.append(test_df)

    return tuple(results) if len(results) > 1 else results[0]


def save_features(feature_df: pd.DataFrame, output_path: str) -> None:
    """Save features to parquet file."""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    feature_df.to_parquet(output_path, index=False)
    print(f"Saved features to {output_path}")


def load_features(input_path: str) -> pd.DataFrame:
    """Load features from parquet file."""
    return pd.read_parquet(input_path)
