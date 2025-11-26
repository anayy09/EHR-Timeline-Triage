# EHR Timeline Triage

> **⚠️ RESEARCH PROTOTYPE - NOT FOR CLINICAL USE**

EHR Timeline Triage is an open-source research project that converts longitudinal Electronic Health Record (EHR) data from hospital stays into structured timelines and generates risk predictions with human-readable explanations. The system supports two critical prediction tasks:

1. **30-Day Readmission Risk**: Predicts whether a patient will be readmitted within 30 days of discharge
2. **48-Hour ICU Mortality Risk**: Predicts in-hospital mortality risk after the first 48 hours of ICU admission

The project demonstrates how temporal sequence modeling (GRU, Transformer) can capture clinical trajectories alongside traditional baseline models (Logistic Regression). Each prediction includes interpretable explanations that reference specific time periods and clinical signals.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           EHR Timeline Triage                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │   Frontend   │    │   REST API   │    │     ML Models            │  │
│  │   (Next.js)  │───►│  (FastAPI)   │───►│  • Logistic Regression   │  │
│  │   Port 3000  │    │   Port 8000  │    │  • GRU Neural Network    │  │
│  └──────────────┘    └──────────────┘    │  • Transformer           │  │
│                                          └──────────────────────────┘  │
│                             │                        │                  │
│                             ▼                        ▼                  │
│                    ┌──────────────────────────────────────┐            │
│                    │         Data & Model Artifacts       │            │
│                    │  • Synthetic EHR Data (Parquet)      │            │
│                    │  • Trained Models (.pt, .joblib)     │            │
│                    │  • Sequence Features (.npy)          │            │
│                    └──────────────────────────────────────┘            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Features

- **Timeline Construction**: Converts raw EHR events into structured timelines with configurable 4-hour time bins
- **Multiple Model Types**: 
  - **Logistic Regression** (scikit-learn): Fast baseline with coefficient-based explanations
  - **GRU Neural Network** (PyTorch): Sequence model for temporal pattern recognition
  - **Transformer** (PyTorch): Attention-based architecture for complex dependencies
- **Interpretability**: Feature attribution and natural language explanations for all predictions
- **REST API**: FastAPI backend with Swagger documentation at `/docs`
- **Web Dashboard**: Modern React/Next.js UI with real-time risk visualization
- **Synthetic Data Generation**: Built-in synthetic EHR data generator for demonstration
- **Comprehensive Evaluation**: AUROC, AUPRC, Brier score, calibration metrics, and plots

## Tech Stack

### Backend & ML
| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.11+ | Core runtime |
| PyTorch | 2.7.1 | Deep learning framework |
| FastAPI | 0.121.1 | REST API framework |
| Uvicorn | 0.38.0 | ASGI server |
| Pydantic | 2.12.4 | Data validation |
| scikit-learn | 1.7.0 | ML utilities & Logistic Regression |
| pandas | 2.3.0 | Data manipulation |
| DuckDB | 1.3.1 | Local SQL queries |

### Frontend
| Component | Version | Purpose |
|-----------|---------|---------|
| Next.js | 16.0.1 | React framework |
| React | 19.2.0 | UI library |
| TypeScript | 5 | Type safety |
| Tailwind CSS | 4 | Styling |

### Infrastructure
| Component | Purpose |
|-----------|---------|
| Docker | Containerization |
| Docker Compose | Multi-container orchestration |

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+ (for frontend)
- Docker and Docker Compose (recommended)

### Option 1: Using Docker (Recommended)

The easiest way to run the entire stack:

```bash
# Clone the repository
git clone https://github.com/anayy09/ehr-timeline-triage.git
cd ehr-timeline-triage

# Build and start all services
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

This starts:
- **API** at http://localhost:8000 (Swagger docs at http://localhost:8000/docs)
- **Frontend** at http://localhost:3000

### Option 2: Manual Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/anayy09/ehr-timeline-triage.git
   cd ehr-timeline-triage
   ```

2. **Set up Python environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   pip install -r requirements.txt
   ```

3. **Generate synthetic data**:
   ```bash
   python -m ehrtriage.scripts.generate_synthetic
   ```

4. **Train models**:
   ```bash
   python -m ehrtriage.scripts.train_all
   ```

5. **Start the API server**:
   ```bash
   uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
   ```

6. **Start the frontend** (in a new terminal):
   ```bash
   cd web
   npm install
   npm run dev
   ```

7. **Access the application**:
   - Frontend: http://localhost:3000
   - API docs: http://localhost:8000/docs

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API root with version info |
| `GET` | `/health` | Health check |
| `POST` | `/api/predict/readmission` | 30-day readmission prediction |
| `POST` | `/api/predict/icu_mortality` | 48-hour ICU mortality prediction |
| `GET` | `/api/models` | List available models and metrics |
| `GET` | `/api/example/{task}` | Get example patient timelines |

### Example Request

```bash
curl -X POST "http://localhost:8000/api/predict/readmission?model_type=gru" \
  -H "Content-Type: application/json" \
  -d '{
    "subject_id": "P001",
    "stay_id": "H001",
    "events": [
      {"time": "2024-01-01T08:00:00", "type": "vital", "code": "heart_rate", "value": 85},
      {"time": "2024-01-01T08:00:00", "type": "vital", "code": "sbp", "value": 130},
      {"time": "2024-01-01T09:00:00", "type": "lab", "code": "lactate", "value": 1.5}
    ],
    "static_features": {
      "age": 65,
      "sex": "M",
      "comorbidity_diabetes": true
    }
  }'
```

## Model Performance

> **Note**: Models are trained on synthetic data for demonstration purposes. Real clinical validation would be required for any production use.

### 30-Day Readmission

| Model | Test AUROC | Test AUPRC | Brier Score |
|-------|------------|------------|-------------|
| Logistic Regression | 0.497 | 0.051 | 0.048 |
| GRU | 0.468 | 0.050 | 0.279 |

### 48-Hour ICU Mortality

| Model | Test AUROC | Test AUPRC | Brier Score |
|-------|------------|------------|-------------|
| Logistic Regression | 0.259 | 0.009 | 0.009 |
| GRU | 0.194 | 0.011 | 0.615 |

*Performance metrics are limited due to synthetic data characteristics and class imbalance.*

## Datasets

### Synthetic Data (Default)

This project ships with a synthetic data generator that creates realistic EHR-like events including:
- **Admissions**: Hospital and ICU stays with timestamps
- **Vitals**: Heart Rate, Blood Pressure (SBP/DBP/MAP), Respiratory Rate, SpO2, Temperature
- **Labs**: Lactate, Creatinine, WBC, Hemoglobin, Platelets, Sodium, Potassium, Glucose
- **Medications**: Vasopressors, Antibiotics, Sedatives
- **Demographics**: Age, Sex, Comorbidities (CHF, Renal, Liver, COPD, Diabetes)

### MIMIC-IV (Optional)

If you have access to MIMIC-IV, you can use real de-identified data:

1. Request access at https://physionet.org/content/mimiciv/
2. Download and place files in `data/raw/`
3. Configure paths in `configs/data.yml`
4. Run cohort building scripts

**Note**: This repository does NOT include any MIMIC-IV data. You must obtain it separately following PhysioNet's requirements.

## Project Structure

```
ehr-timeline-triage/
├── ehrtriage/               # Core Python package (v0.1.0)
│   ├── cohort.py            # Cohort definitions
│   ├── features.py          # Feature engineering
│   ├── sequence_builder.py  # Sequence data builder
│   ├── synthetic_data.py    # Synthetic data generator
│   ├── models/              # ML models
│   │   ├── baselines.py     # Logistic Regression
│   │   └── sequence.py      # GRU, Transformer
│   ├── explain/             # Interpretability
│   │   ├── attribution.py   # Feature attribution
│   │   └── text_generator.py# Explanation generation
│   ├── evaluation/          # Metrics and plots
│   └── scripts/             # Training & data scripts
├── api/                     # FastAPI backend
│   ├── app.py               # Main application
│   ├── models.py            # Pydantic schemas
│   └── routers/             # API endpoints
├── web/                     # Next.js frontend
│   ├── app/                 # App router pages
│   ├── components/          # React components
│   └── types/               # TypeScript types
├── data/                    # Data directories
│   ├── raw/                 # Raw data (empty)
│   ├── processed/           # Processed features
│   └── synthetic/           # Generated synthetic data
├── models/artifacts/        # Trained model files
│   ├── readmission/         # 30-day readmission models
│   └── icu_mortality/       # ICU mortality models
├── notebooks/               # Jupyter notebooks
├── tests/                   # Unit tests
├── docs/                    # Documentation
└── configs/                 # Configuration files
```

## Documentation

- [System Design](docs/system_design.md) - Architecture and component overview
- [Data Pipeline](docs/data_pipeline.md) - Data processing and cohort building
- [Models](docs/models.md) - Model descriptions and training
- [Interpretability](docs/interpretability.md) - Explanation methods
- [Limitations](docs/limitations.md) - Known limitations and ethical considerations
- [Deployment Guide](docs/deployment.md) - Deployment options and production setup

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ehrtriage --cov=api

# Run specific test file
pytest tests/test_api.py -v
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Ensure all tests pass (`pytest`)
5. Run linting (`ruff check .`)
6. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details

---

## ⚠️ Disclaimer

**THIS SOFTWARE IS FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY.**

This is a research prototype and demonstration tool. It has **NOT** been validated for clinical use and should **NOT** be used for:
- Making clinical decisions
- Diagnosing patients
- Determining treatment plans
- Any patient care activities

The models are trained on synthetic or de-identified data and may not generalize to real clinical settings. No warranty or guarantee of accuracy is provided. Always consult qualified healthcare professionals for medical decisions.

**Generated by a research model on de-identified or synthetic data. Not for clinical use.**
