"""Tests for models."""

import numpy as np
import pandas as pd
import torch

from ehrtriage.models.baselines import LogisticBaseline, find_optimal_threshold
from ehrtriage.models.sequence import GRURiskModel, TransformerRiskModel


def test_logistic_model_fit_predict():
    """Test logistic regression model."""
    # Create synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 10

    X = pd.DataFrame(np.random.randn(n_samples, n_features))
    y = np.random.randint(0, 2, n_samples)

    # Train model
    model = LogisticBaseline()
    model.fit(X, y)

    # Predict
    probs = model.predict_proba(X)
    preds = model.predict(X)

    # Calibrate and re-predict
    model.calibrate(X, y, method="sigmoid")
    probs_cal = model.predict_proba(X)
    threshold, score = find_optimal_threshold(y, probs_cal[:, 1])
    model.set_decision_threshold(threshold)
    preds_cal = model.predict(X)

    assert probs.shape == (n_samples, 2)
    assert preds.shape == (n_samples,)
    assert np.all((probs >= 0) & (probs <= 1))
    assert np.all((preds == 0) | (preds == 1))
    assert np.all((probs_cal >= 0) & (probs_cal <= 1))
    assert 0.0 <= threshold <= 1.0
    assert score <= 1.0
    assert preds_cal.shape == (n_samples,)
    assert np.all((preds_cal == 0) | (preds_cal == 1))


def test_gru_model_forward():
    """Test GRU model forward pass."""
    batch_size = 4
    seq_len = 10
    input_dim = 8
    static_dim = 5

    model = GRURiskModel(
        input_dim=input_dim,
        static_dim=static_dim,
        hidden_dim=16,
        num_layers=1,
    )

    # Create dummy input
    sequence = torch.randn(batch_size, seq_len, input_dim)
    static = torch.randn(batch_size, static_dim)
    mask = torch.ones(batch_size, seq_len)

    # Forward pass
    logits = model(sequence, static, mask)

    assert logits.shape == (batch_size, 1)
    assert not torch.isnan(logits).any()


def test_transformer_model_forward():
    """Test Transformer model forward pass."""
    batch_size = 4
    seq_len = 10
    input_dim = 8
    static_dim = 5

    model = TransformerRiskModel(
        input_dim=input_dim,
        static_dim=static_dim,
        d_model=16,
        nhead=2,
        num_layers=2,
    )

    # Create dummy input
    sequence = torch.randn(batch_size, seq_len, input_dim)
    static = torch.randn(batch_size, static_dim)
    mask = torch.ones(batch_size, seq_len)

    # Forward pass
    logits = model(sequence, static, mask)

    assert logits.shape == (batch_size, 1)
    assert not torch.isnan(logits).any()
