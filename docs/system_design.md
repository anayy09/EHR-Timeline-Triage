# System Design

## Overview

EHR Timeline Triage is a complete system for temporal risk prediction from Electronic Health Record data. The system processes longitudinal clinical events and generates risk predictions with interpretable explanations.

## Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (React)                      │
│         - Timeline Visualization                         │
│         - Risk Dashboard                                 │
│         - Model Comparison UI                            │
└────────────────┬────────────────────────────────────────┘
                 │ HTTP/REST
┌────────────────▼────────────────────────────────────────┐
│                  FastAPI Backend                         │
│         - Prediction Endpoints                           │
│         - Model Serving                                  │
│         - Feature Engineering                            │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│               Core Python Package (ehrtriage)            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Cohort     │  │  Features    │  │   Models     │  │
│  │   Builder    │  │  Engineer    │  │   - Logistic │  │
│  └──────────────┘  └──────────────┘  │   - GRU      │  │
│                                       │   - Transformer│ │
│  ┌──────────────┐  ┌──────────────┐  └──────────────┘  │
│  │ Evaluation   │  │  Explain     │                     │
│  │ - Metrics    │  │ - Attribution│                     │
│  │ - Plots      │  │ - Text Gen   │                     │
│  └──────────────┘  └──────────────┘                     │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│                    Data Storage                          │
│  - Raw EHR Data (Parquet)                               │
│  - Processed Features (Parquet/NPY)                     │
│  - Model Artifacts (PyTorch/Joblib)                     │
└─────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Data Layer

**Purpose**: Store and manage EHR data

**Components**:
- **Synthetic Data Generator** (`ehrtriage/synthetic_data.py`): Creates realistic EHR-like data
- **Raw Data Store** (`data/raw/`): Stores original data files
- **Processed Data Store** (`data/processed/`): Stores engineered features

**Data Flow**:
1. Raw admissions, events, demographics → Parquet files
2. Cohort builder filters and labels cases
3. Feature engineer creates model inputs
4. Saved as Parquet (tabular) or NPY (arrays)

### 2. Core Processing Layer

**Purpose**: Transform raw data into model inputs

#### Cohort Builder (`ehrtriage/cohort.py`)

- **Readmission Cohort**: Identifies index admissions and 30-day readmissions
- **ICU Mortality Cohort**: Identifies ICU stays with 48+ hours of data
- Handles time windows, exclusion criteria, label generation

#### Feature Engineer

**Snapshot Features** (`ehrtriage/features.py`):
- Aggregates events in a time window
- Statistical features: mean, min, max, std, count
- One row per stay

**Sequence Features** (`ehrtriage/sequence_builder.py`):
- Time-binned representation (default: 4-hour bins)
- Fixed-length sequences with masking
- 3D tensor: [N, T, D]

### 3. Model Layer

**Purpose**: Train and serve prediction models

#### Baseline Model
- **Logistic Regression** (`ehrtriage/models/baselines.py`)
  - Snapshot features → Risk score
  - Coefficients used for interpretability
  - Fast training and inference

#### Sequence Models (`ehrtriage/models/sequence.py`)
- **GRU**: Bidirectional GRU with static feature fusion
- **Transformer**: Multi-head attention encoder with positional encoding
- Both output single risk score via prediction head

**Training Pipeline** (`ehrtriage/scripts/train_all.py`):
1. Load/generate data
2. Build cohorts
3. Engineer features
4. Train models with validation
5. Evaluate on test set
6. Save artifacts and metrics

### 4. Interpretability Layer

**Purpose**: Explain model predictions

#### Attribution (`ehrtriage/explain/attribution.py`)
- **Logistic**: Coefficient × feature value
- **Sequence Models**: Gradient-based temporal attribution
- Identifies top features and time periods

#### Text Generation (`ehrtriage/explain/text_generator.py`)
- Converts attributions to natural language
- Task-specific templates
- Includes disclaimer

### 5. API Layer

**Purpose**: Serve predictions via REST API

**Endpoints** (`api/routers/predict.py`):
- `POST /api/predict/{task}`: Make prediction
- `GET /api/models`: List available models
- `GET /api/example/{task}`: Get example timeline

**Request Flow**:
1. Client sends PatientTimeline (events + static features)
2. API converts to feature representation
3. Loads appropriate model from cache
4. Generates prediction and explanation
5. Returns PredictionResponse

### 6. Frontend Layer

**Purpose**: Interactive UI for exploring predictions

**Features** (not yet implemented):
- Timeline visualization
- Risk score display
- Contributing events highlighted
- Model comparison

## Data Schemas

### Event Schema
```python
{
  "time": "2024-01-01T08:00:00",
  "type": "vital|lab|medication",
  "code": "heart_rate|lactate|vasopressor",
  "value": 85.0
}
```

### Prediction Response Schema
```python
{
  "task": "readmission|icu_mortality",
  "risk_score": 0.75,
  "risk_label": "high",
  "explanation": "High risk of 30-day readmission...",
  "contributing_events": [...]
}
```

## Configuration Management

**Config Files** (`configs/`):
- `features.yml`: Feature definitions, time bins, normalization
- `data.yml`: Data paths, split ratios, synthetic generation params

**Config Loader** (`ehrtriage/config.py`):
- Centralized configuration access
- Default fallbacks
- Type-safe Config objects

## Model Artifacts

**Structure**:
```
models/artifacts/{task}/
  ├── logistic_model.joblib
  ├── logistic_scaler.joblib
  ├── logistic_metadata.json
  ├── logistic_metrics.json
  ├── gru.pt
  ├── gru_config.json
  ├── plots/
  │   ├── {task}_roc.png
  │   ├── {task}_pr.png
  │   └── {task}_calibration.png
```

## Deployment Architecture

### Development
- Local Python environment
- Uvicorn for API
- React dev server for frontend

### Production (Future)
- Docker containers
- API behind reverse proxy
- Model versioning system
- Monitoring and logging

## Security Considerations

1. **No PHI**: Only synthetic or deidentified data
2. **Disclaimer**: Prominent research-only warnings
3. **API Rate Limiting**: Prevent abuse (future)
4. **Model Versioning**: Track model provenance

## Scalability

**Current Limits**:
- Designed for moderate datasets (1K-100K admissions)
- Single-machine training and inference
- In-memory feature engineering

**Future Scaling**:
- Batch prediction API
- Model serving with caching
- Distributed training
- Feature store
