"""Tests for cohort building."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ehrtriage.cohort import (
    build_readmission_cohort,
    build_icu_mortality_cohort,
    split_cohort,
)


def test_readmission_cohort_basic():
    """Test basic readmission cohort building."""
    # Create synthetic admissions
    admissions = pd.DataFrame([
        {
            "subject_id": "P001",
            "hadm_id": "H001",
            "admittime": datetime(2024, 1, 1),
            "dischtime": datetime(2024, 1, 5),
            "deathtime": None,
        },
        {
            "subject_id": "P001",
            "hadm_id": "H002",
            "admittime": datetime(2024, 1, 15),  # 10 days later - readmission
            "dischtime": datetime(2024, 1, 20),
            "deathtime": None,
        },
        {
            "subject_id": "P002",
            "hadm_id": "H003",
            "admittime": datetime(2024, 1, 1),
            "dischtime": datetime(2024, 1, 10),
            "deathtime": None,
        },
    ])

    cohort = build_readmission_cohort(admissions, min_los_hours=24)

    assert len(cohort) >= 2
    assert "readmit_30d" in cohort.columns
    assert cohort["readmit_30d"].dtype in [np.int64, np.int32]


def test_icu_mortality_cohort_basic():
    """Test basic ICU mortality cohort building."""
    # Create synthetic ICU stays
    icustays = pd.DataFrame([
        {
            "stay_id": "ICU001",
            "subject_id": "P001",
            "hadm_id": "H001",
            "intime": datetime(2024, 1, 1, 10, 0),
            "outtime": datetime(2024, 1, 5, 10, 0),
        },
        {
            "stay_id": "ICU002",
            "subject_id": "P002",
            "hadm_id": "H002",
            "intime": datetime(2024, 1, 2, 10, 0),
            "outtime": datetime(2024, 1, 4, 10, 0),
        },
    ])

    admissions = pd.DataFrame([
        {
            "hadm_id": "H001",
            "deathtime": None,
        },
        {
            "hadm_id": "H002",
            "deathtime": datetime(2024, 1, 3, 15, 0),  # Death after 48h
        },
    ])

    cohort = build_icu_mortality_cohort(icustays, admissions, min_icu_hours=48)

    assert len(cohort) >= 1
    assert "mortality_label" in cohort.columns
    assert cohort["mortality_label"].dtype in [np.int64, np.int32]


def test_cohort_split():
    """Test cohort splitting."""
    cohort = pd.DataFrame({
        "subject_id": [f"P{i:03d}" for i in range(100)],
        "label": np.random.randint(0, 2, 100),
    })

    train, val, test = split_cohort(cohort, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15
    assert len(train) + len(val) + len(test) == len(cohort)
