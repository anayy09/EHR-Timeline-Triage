# Models Documentation

## Overview

This document describes all prediction models implemented in EHR Timeline Triage, their architectures, training procedures, and performance characteristics.

## Model Comparison Summary

| Model | Type | Input | Strengths | Weaknesses |
|-------|------|-------|-----------|------------|
| Logistic Regression | Baseline | Snapshot features | Fast, interpretable, good baseline | Ignores temporal patterns |
| GRU | Sequence | Time-binned sequences | Captures temporal dependencies | More complex, needs more data |
| Transformer | Sequence | Time-binned sequences | Attention mechanism, parallelizable | High capacity, can overfit |

## 1. Logistic Regression Baseline

### Architecture

**Type**: Generalized Linear Model

**Implementation**: `ehrtriage/models/baselines.py::LogisticBaseline`

**Input**: Snapshot features [D] where D ≈ 100-150
- Aggregated vitals, labs, medications
- Static demographics and comorbidities

**Output**: Risk probability [0, 1]

**Parameters**:
- `C`: Inverse regularization strength (default: 1.0)
- `penalty`: L2 regularization
- `solver`: lbfgs
- `class_weight`: balanced (handles class imbalance)

### Training

**Optimizer**: L-BFGS

**Loss**: Cross-entropy (binary classification)

**Regularization**: L2 penalty

**Training Time**: Seconds to minutes

**Hyperparameters**:
```python
{
    "C": 1.0,
    "penalty": "l2",
    "max_iter": 1000,
    "class_weight": "balanced"
}
```

### Interpretability

**Method**: Linear coefficients

For each feature:
```
contribution = coefficient × feature_value
```

**Top Features**: Ranked by absolute coefficient magnitude

**Advantages**:
- Direct feature importance
- Sign indicates direction (increases/decreases risk)
- Easy to explain to clinicians

### Performance

**Expected Metrics** (synthetic data):
- AUROC: 0.65-0.75
- AUPRC: 0.25-0.40 (depends on prevalence)
- Brier Score: 0.15-0.20

**Use Cases**:
- Strong baseline
- Feature selection
- Deployment where interpretability is critical

## 2. GRU (Gated Recurrent Unit)

### Architecture

**Type**: Recurrent Neural Network

**Implementation**: `ehrtriage/models/sequence.py::GRURiskModel`

**Structure**:
```
Input Sequence [T, D]
    ↓
Bidirectional GRU [hidden_dim × 2]
    ↓
Last Hidden States (concatenated)
    ↓
Concatenate with Static Features
    ↓
Dense Layer [hidden_dim] + ReLU + Dropout
    ↓
Dense Layer [1] → Risk Logit
    ↓
Sigmoid → Risk Probability
```

**Parameters**:
- `input_dim`: Features per timestep (~30-40)
- `static_dim`: Static features (~10)
- `hidden_dim`: 64 (default)
- `num_layers`: 2
- `dropout`: 0.3
- `bidirectional`: True

**Total Parameters**: ~100K-200K

### Training

**Optimizer**: Adam
- Learning rate: 0.001
- Weight decay: 0.0001

**Loss**: Binary cross-entropy with logits

**Batch Size**: 32

**Epochs**: 50 (with early stopping)

**Early Stopping**:
- Patience: 10 epochs
- Metric: Validation AUROC
- Restores best model

**Training Time**: 5-30 minutes (depends on data size and GPU)

**Data Augmentation**: None (future: time warping, masking)

### Interpretability

**Method**: Gradient-based temporal attribution

**Process**:
1. Compute gradient of output w.r.t. input sequence
2. Aggregate gradient magnitude across features per timestep
3. Normalize by mask (valid timesteps only)

**Output**: Importance score for each time bin

**Top Timesteps**: Identify 3-5 most influential periods

**Advantages**:
- Temporal localization
- Identifies critical time windows

**Limitations**:
- Gradients can be noisy
- Requires backpropagation

### Performance

**Expected Metrics** (synthetic data):
- AUROC: 0.68-0.78 (slight improvement over logistic)
- AUPRC: 0.28-0.42
- Better calibration on temporal patterns

**Advantages**:
- Captures short-term temporal dependencies
- Efficient training
- Handles variable-length sequences

**Limitations**:
- Vanishing gradients on very long sequences
- Sequential processing (slower inference than Transformer)

## 3. Transformer

### Architecture

**Type**: Attention-based Neural Network

**Implementation**: `ehrtriage/models/sequence.py::TransformerRiskModel`

**Structure**:
```
Input Sequence [T, D]
    ↓
Linear Projection [d_model]
    ↓
Positional Encoding
    ↓
Transformer Encoder Layers × num_layers
  (Multi-Head Self-Attention + FFN)
    ↓
Mean Pooling (masked)
    ↓
Concatenate with Static Features
    ↓
Dense Layer [d_model] + ReLU + Dropout
    ↓
Dense Layer [1] → Risk Logit
    ↓
Sigmoid → Risk Probability
```

**Parameters**:
- `input_dim`: Features per timestep
- `static_dim`: Static features
- `d_model`: 64 (default)
- `nhead`: 4 (attention heads)
- `num_layers`: 3 (transformer layers)
- `dim_feedforward`: 256
- `dropout`: 0.3

**Total Parameters**: ~150K-300K

### Training

**Optimizer**: Adam
- Learning rate: 0.0005 (lower than GRU)
- Weight decay: 0.0001

**Loss**: Binary cross-entropy with logits

**Batch Size**: 32

**Epochs**: 50 (with early stopping)

**Training Time**: 10-60 minutes

**Special Considerations**:
- Attention mask for padding
- Positional encoding crucial for temporal order

### Interpretability

**Method**: Attention weights (future enhancement)

Currently uses gradient-based attribution similar to GRU.

**Future**: Extract and visualize attention scores
- Shows which timesteps attend to which
- Multi-head attention provides multiple views

### Performance

**Expected Metrics** (synthetic data):
- AUROC: 0.70-0.80 (best performance)
- AUPRC: 0.30-0.45
- Good calibration

**Advantages**:
- Parallel processing (faster than GRU on GPU)
- Long-range dependencies
- Attention mechanism provides interpretability

**Limitations**:
- More parameters (risk of overfitting)
- Requires more data
- Computational cost

## Model Selection Guidelines

### Use Logistic Regression when:
- Interpretability is paramount
- Limited training data (<500 examples)
- Fast inference required
- Establishing baseline

### Use GRU when:
- Temporal patterns expected
- Moderate data available (500-5000 examples)
- Sequential dependencies matter
- CPU deployment

### Use Transformer when:
- Large dataset available (>5000 examples)
- Complex temporal patterns
- GPU available
- Best performance needed

## Evaluation Metrics

### Primary Metrics

**AUROC** (Area Under ROC Curve):
- Threshold-independent
- Measures discrimination
- Target: >0.70

**AUPRC** (Area Under Precision-Recall Curve):
- Better for imbalanced datasets
- Clinically relevant
- Target: >2× baseline rate

### Secondary Metrics

**Brier Score**: Calibration measure (lower is better)

**Expected Calibration Error (ECE)**: Calibration accuracy

**Sensitivity/Specificity**: At operating point (e.g., 0.5 threshold)

### Comparison Results

After running `train_all.py`, you'll see output like:

```
Model Comparison
================================================================================
Metric              Logistic         GRU              Transformer     
--------------------------------------------------------------------------------
auroc               0.7123           0.7445           0.7589          
auprc               0.3234           0.3567           0.3789          
brier_score         0.1823           0.1745           0.1701          
f1_score            0.4123           0.4467           0.4589          
================================================================================
```

## Model Outputs

### Prediction Response

All models return:
```python
{
    "risk_score": 0.75,        # Probability [0,1]
    "risk_label": "high",      # Categorical: low/medium/high
    "explanation": "...",       # Natural language explanation
    "contributing_events": [   # Top contributing factors
        {
            "time": "Hour 40-44",
            "type": "lab",
            "code": "lactate",
            "value": 3.5,
            "contribution_score": 0.15
        },
        ...
    ]
}
```

## Future Enhancements

### Additional Models
- Cox Proportional Hazards (survival analysis)
- Ensemble methods (stacking logistic + GRU)
- Multi-task learning (joint prediction)

### Architecture Improvements
- Temporal convolutional networks
- Graph neural networks (for patient trajectory graphs)
- Pre-trained embeddings from large EHR datasets

### Training Enhancements
- Active learning for data efficiency
- Uncertainty quantification
- Adversarial robustness
