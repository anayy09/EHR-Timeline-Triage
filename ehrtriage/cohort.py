"""
Cohort building for prediction tasks.

This module defines cohorts for:
1. 30-day readmission prediction
2. 48-hour ICU mortality prediction
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import numpy as np


def build_readmission_cohort(
    admissions_df: pd.DataFrame,
    min_age: int = 18,
    min_los_hours: float = 24.0,
) -> pd.DataFrame:
    """
    Build cohort for 30-day readmission prediction.

    Args:
        admissions_df: DataFrame with columns:
            - subject_id: Patient identifier
            - hadm_id: Admission identifier
            - admittime: Admission timestamp
            - dischtime: Discharge timestamp
            - deathtime: Death timestamp (optional)
        min_age: Minimum patient age for inclusion
        min_los_hours: Minimum length of stay in hours

    Returns:
        DataFrame with columns:
            - subject_id
            - hadm_id (index admission)
            - index_admit_time
            - discharge_time
            - readmit_30d (0/1)
            - days_to_readmit (if readmitted)
            - los_hours
    """
    # Ensure datetime columns
    for col in ["admittime", "dischtime"]:
        if col in admissions_df.columns:
            admissions_df[col] = pd.to_datetime(admissions_df[col])

    if "deathtime" in admissions_df.columns:
        admissions_df["deathtime"] = pd.to_datetime(admissions_df["deathtime"])

    # Sort by patient and admission time
    admissions_df = admissions_df.sort_values(["subject_id", "admittime"])

    # Calculate length of stay
    admissions_df["los_hours"] = (
        admissions_df["dischtime"] - admissions_df["admittime"]
    ).dt.total_seconds() / 3600

    # Filter by minimum LOS
    admissions_df = admissions_df[admissions_df["los_hours"] >= min_los_hours].copy()

    # Exclude admissions that end in death
    if "deathtime" in admissions_df.columns:
        admissions_df = admissions_df[admissions_df["deathtime"].isna()].copy()

    # For each admission, check if there's a readmission within 30 days
    cohort_rows = []

    for subject_id, group in admissions_df.groupby("subject_id"):
        admissions = group.sort_values("admittime").reset_index(drop=True)

        for idx, row in admissions.iterrows():
            # Look for next admission
            future_admissions = admissions[
                admissions["admittime"] > row["dischtime"]
            ]

            if len(future_admissions) > 0:
                next_admission = future_admissions.iloc[0]
                days_to_readmit = (
                    next_admission["admittime"] - row["dischtime"]
                ).total_seconds() / (24 * 3600)

                readmit_30d = 1 if days_to_readmit <= 30 else 0
            else:
                days_to_readmit = None
                readmit_30d = 0

            cohort_rows.append(
                {
                    "subject_id": subject_id,
                    "hadm_id": row["hadm_id"],
                    "index_admit_time": row["admittime"],
                    "discharge_time": row["dischtime"],
                    "readmit_30d": readmit_30d,
                    "days_to_readmit": days_to_readmit,
                    "los_hours": row["los_hours"],
                }
            )

    cohort_df = pd.DataFrame(cohort_rows)

    print(f"Readmission cohort: {len(cohort_df)} admissions")
    print(f"  - Readmitted: {cohort_df['readmit_30d'].sum()} "
          f"({cohort_df['readmit_30d'].mean():.1%})")

    return cohort_df


def build_icu_mortality_cohort(
    icustays_df: pd.DataFrame,
    admissions_df: Optional[pd.DataFrame] = None,
    min_icu_hours: float = 48.0,
) -> pd.DataFrame:
    """
    Build cohort for 48-hour ICU mortality prediction.

    Args:
        icustays_df: DataFrame with columns:
            - stay_id: ICU stay identifier
            - subject_id: Patient identifier
            - hadm_id: Hospital admission identifier
            - intime: ICU admission time
            - outtime: ICU discharge time
        admissions_df: Optional admissions data with deathtime
        min_icu_hours: Minimum ICU stay length for inclusion

    Returns:
        DataFrame with columns:
            - stay_id
            - subject_id
            - hadm_id
            - icu_intime
            - prediction_time_48h (intime + 48 hours)
            - mortality_label (0/1)
            - actual_death_time (if died)
            - icu_los_hours
    """
    # Ensure datetime columns
    icustays_df["intime"] = pd.to_datetime(icustays_df["intime"])
    icustays_df["outtime"] = pd.to_datetime(icustays_df["outtime"])

    # Calculate ICU length of stay
    icustays_df["icu_los_hours"] = (
        icustays_df["outtime"] - icustays_df["intime"]
    ).dt.total_seconds() / 3600

    # Filter: must have at least 48 hours of observed data
    icustays_df = icustays_df[icustays_df["icu_los_hours"] >= min_icu_hours].copy()

    # Calculate prediction time (48 hours after ICU admission)
    icustays_df["prediction_time_48h"] = icustays_df["intime"] + pd.Timedelta(
        hours=48
    )

    # Determine mortality label
    if admissions_df is not None and "deathtime" in admissions_df.columns:
        admissions_df["deathtime"] = pd.to_datetime(admissions_df["deathtime"])

        # Merge to get death information
        icustays_df = icustays_df.merge(
            admissions_df[["hadm_id", "deathtime"]],
            on="hadm_id",
            how="left",
        )

        # Label = 1 if patient died after the 48-hour mark
        icustays_df["mortality_label"] = (
            (~icustays_df["deathtime"].isna())
            & (icustays_df["deathtime"] > icustays_df["prediction_time_48h"])
        ).astype(int)

        icustays_df["actual_death_time"] = icustays_df["deathtime"]
    else:
        # No death information available - use synthetic or placeholder
        icustays_df["mortality_label"] = 0
        icustays_df["actual_death_time"] = None

    cohort_df = icustays_df[
        [
            "stay_id",
            "subject_id",
            "hadm_id",
            "intime",
            "prediction_time_48h",
            "mortality_label",
            "actual_death_time",
            "icu_los_hours",
        ]
    ].rename(columns={"intime": "icu_intime"})

    print(f"ICU mortality cohort: {len(cohort_df)} ICU stays")
    print(f"  - Mortality: {cohort_df['mortality_label'].sum()} "
          f"({cohort_df['mortality_label'].mean():.1%})")

    return cohort_df


def split_cohort(
    cohort_df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split cohort into train, validation, and test sets.

    Args:
        cohort_df: Cohort DataFrame
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        random_state: Random seed

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    n = len(cohort_df)
    indices = np.arange(n)
    np.random.seed(random_state)
    np.random.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    train_df = cohort_df.iloc[train_idx].reset_index(drop=True)
    val_df = cohort_df.iloc[val_idx].reset_index(drop=True)
    test_df = cohort_df.iloc[test_idx].reset_index(drop=True)

    return train_df, val_df, test_df


def load_cohort(cohort_path: Path) -> pd.DataFrame:
    """Load cohort from parquet file."""
    return pd.read_parquet(cohort_path)


def save_cohort(cohort_df: pd.DataFrame, cohort_path: Path) -> None:
    """Save cohort to parquet file."""
    cohort_path.parent.mkdir(parents=True, exist_ok=True)
    cohort_df.to_parquet(cohort_path, index=False)
    print(f"Saved cohort to {cohort_path}")
