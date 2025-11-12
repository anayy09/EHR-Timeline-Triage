"""Tests for feature engineering."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ehrtriage.features import (
    extract_time_window,
    aggregate_vital_features,
    aggregate_lab_features,
    normalize_features,
)


def test_extract_time_window():
    """Test event extraction within time window."""
    events = pd.DataFrame({
        "subject_id": ["P001"] * 5,
        "hadm_id": ["H001"] * 5,
        "stay_id": [None] * 5,
        "time": [
            datetime(2024, 1, 1, i, 0) for i in range(5)
        ],
        "type": ["vital"] * 5,
        "code": ["heart_rate"] * 5,
        "value": [80, 85, 90, 95, 100],
    })

    start_time = datetime(2024, 1, 1, 1, 0)
    end_time = datetime(2024, 1, 1, 3, 0)

    window_events = extract_time_window(
        events, start_time, end_time, subject_id="P001"
    )

    assert len(window_events) == 2  # Hours 1 and 2
    assert window_events["value"].tolist() == [85, 90]


def test_aggregate_vital_features():
    """Test vital sign aggregation."""
    events = pd.DataFrame({
        "type": ["vital"] * 4,
        "code": ["heart_rate", "heart_rate", "sbp", "sbp"],
        "value": [80, 90, 120, 130],
    })

    features = aggregate_vital_features(events, ["heart_rate", "sbp"])

    assert "heart_rate_mean" in features
    assert "heart_rate_min" in features
    assert "heart_rate_max" in features
    assert features["heart_rate_mean"] == 85.0
    assert features["heart_rate_min"] == 80.0
    assert features["heart_rate_max"] == 90.0


def test_normalize_features():
    """Test feature normalization."""
    train_df = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
        "label": [0, 1, 0, 1, 0],
    })

    test_df = pd.DataFrame({
        "feature1": [3.0, 4.0],
        "feature2": [30.0, 40.0],
        "label": [0, 1],
    })

    train_norm, test_norm = normalize_features(
        train_df, test_df=test_df, method="standard"
    )

    # Check that normalization happened
    assert train_norm["feature1"].mean() < 0.1  # Should be close to 0
    assert abs(train_norm["feature1"].std() - 1.0) < 0.1  # Should be close to 1

    # Labels should be unchanged
    assert train_norm["label"].tolist() == train_df["label"].tolist()
