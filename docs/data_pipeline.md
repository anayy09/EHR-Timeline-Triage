# Data Pipeline

## Overview

The data pipeline transforms raw EHR data into labeled cohorts and engineered features suitable for machine learning models.

## Pipeline Stages

### 1. Data Input

#### Option A: Synthetic Data (Default)

**Generator**: `ehrtriage/synthetic_data.py`

**Process**:
1. Generate patient demographics (age, sex, comorbidities)
2. Generate hospital admissions with realistic timing
3. Generate ICU stays (subset of admissions)
4. Generate clinical events (vitals, labs, medications)

**Parameters** (from `configs/data.yml`):
- `n_patients`: Number of patients (default: 1000)
- `icu_admission_rate`: Fraction with ICU stays (default: 0.3)
- `readmission_rate`: Probability of readmission (default: 0.15)
- `mortality_rate`: ICU mortality rate (default: 0.10)
- `event_frequency_hours`: Range for event generation (default: [1, 4])

**Output Files**:
- `data/synthetic/patients.parquet`: Demographics and comorbidities
- `data/synthetic/admissions.parquet`: Hospital admissions with timestamps
- `data/synthetic/icustays.parquet`: ICU stays
- `data/synthetic/events.parquet`: All clinical events

**Synthetic Data Schema**:

**Patients**:
```
subject_id: string
sex: M/F
birth_year: int
comorbidity_chf: bool
comorbidity_renal: bool
comorbidity_liver: bool
comorbidity_copd: bool
comorbidity_diabetes: bool
```

**Admissions**:
```
subject_id: string
hadm_id: string
admittime: datetime
dischtime: datetime
deathtime: datetime (nullable)
age: int
```

**ICU Stays**:
```
stay_id: string
subject_id: string
hadm_id: string
intime: datetime
outtime: datetime
deathtime: datetime (nullable)
```

**Events**:
```
subject_id: string
hadm_id: string
stay_id: string (nullable)
time: datetime
type: vital|lab|medication
code: string (e.g., 'heart_rate', 'lactate')
value: float
```

#### Option B: MIMIC-IV Data

**Prerequisites**:
1. Access granted via PhysioNet
2. Downloaded files in `data/raw/mimic/`
3. Update `configs/data.yml` to set `use_synthetic: false`

**Required Tables**:
- admissions.csv
- patients.csv
- icustays.csv
- chartevents.csv (vitals)
- labevents.csv
- prescriptions.csv (medications)

**Mapping**:
The system expects similar schema. Custom loaders can be implemented in `ehrtriage/data_loaders.py` (not included in current version).

### 2. Cohort Definition

#### Task A: 30-Day Readmission

**Definition** (`ehrtriage/cohort.py::build_readmission_cohort`):

**Population**:
- Adult patients (≥18 years)
- Hospital admissions with ≥24 hour LOS
- Exclude: admissions ending in death

**Index Admission**: Each hospital discharge is a prediction point

**Label**: `readmit_30d`
- 1 if patient readmitted within 30 days
- 0 otherwise

**Output**:
```
subject_id: Patient identifier
hadm_id: Admission identifier
index_admit_time: Admission timestamp
discharge_time: Discharge timestamp
readmit_30d: Binary label (0/1)
days_to_readmit: Days until readmission (if applicable)
los_hours: Length of stay in hours
```

**Labeling Logic**:
```python
for each admission:
    next_admission = find_next_admission(patient, after=discharge_time)
    if next_admission:
        days_gap = (next_admission.time - discharge_time).days
        readmit_30d = 1 if days_gap <= 30 else 0
    else:
        readmit_30d = 0
```

#### Task B: 48-Hour ICU Mortality

**Definition** (`ehrtriage/cohort.py::build_icu_mortality_cohort`):

**Population**:
- ICU stays with ≥48 hours of observed data
- One row per ICU stay

**Prediction Time**: 48 hours after ICU admission

**Label**: `mortality_label`
- 1 if patient dies in hospital AFTER the 48-hour mark
- 0 otherwise

**Output**:
```
stay_id: ICU stay identifier
subject_id: Patient identifier
hadm_id: Hospital admission identifier
icu_intime: ICU admission timestamp
prediction_time_48h: intime + 48 hours
mortality_label: Binary label (0/1)
actual_death_time: Death timestamp (if applicable)
icu_los_hours: ICU length of stay
```

**Labeling Logic**:
```python
for each ICU stay:
    if icu_los_hours < 48:
        exclude  # Not enough observation time
    
    prediction_time = icu_intime + 48 hours
    
    if patient died and death_time > prediction_time:
        mortality_label = 1
    else:
        mortality_label = 0
```

### 3. Data Splitting

**Strategy**: Random split by stay (not by patient)

**Ratios** (configurable):
- Train: 70%
- Validation: 15%
- Test: 15%

**Implementation** (`ehrtriage/cohort.py::split_cohort`):
- Shuffle with fixed random seed (42)
- Deterministic splits for reproducibility

### 4. Feature Engineering

#### Snapshot Features (for Logistic Regression)

**Purpose**: Aggregate events in a time window into a single feature vector

**Window Definition**:
- **Readmission**: Last 48 hours before discharge
- **ICU Mortality**: First 48 hours of ICU stay

**Feature Types**:

1. **Vital Signs**: heart_rate, sbp, dbp, map, respiratory_rate, spo2, temperature
   - Aggregations: mean, min, max, std, count

2. **Lab Values**: lactate, creatinine, wbc, hemoglobin, platelet, sodium, potassium, bun
   - Aggregations: mean, min, max, last (most recent), count

3. **Medications**: vasopressor, antibiotic, sedative, diuretic, anticoagulant
   - Binary: given (0/1)
   - Count: number of administrations

4. **Static Features**: age, sex, comorbidities
   - One-hot encoding for categorical

**Output Shape**: `[N, D]` where N=stays, D=features (~100-150 features)

**Example Feature Names**:
```
heart_rate_mean
heart_rate_min
heart_rate_max
lactate_mean
lactate_last
med_vasopressor_given
age
sex_M
comorbidity_chf
```

#### Sequence Features (for GRU/Transformer)

**Purpose**: Represent temporal evolution of clinical state

**Time Binning**:
- Configurable bin size (default: 4 hours)
- Fixed number of bins (default: 12 bins = 48 hours)

**Process**:
1. Divide time window into bins
2. For each bin, aggregate events:
   - Vitals: mean value in bin
   - Labs: most recent value in bin
   - Medications: binary presence in bin
3. Concatenate static features

**Output**:
- **Sequences**: `[N, T, D]` where T=12 (timesteps), D=features per timestep
- **Masks**: `[N, T]` indicating valid timesteps (not padding)
- **Static**: `[N, S]` static features
- **Labels**: `[N]` binary outcomes

**Padding**:
- If actual sequence < T: pad with zeros, mask=0
- If actual sequence > T: truncate to most recent T bins

**Normalization**:
- Features normalized using training set statistics
- Standard scaling (zero mean, unit variance)
- Applied consistently to train/val/test

### 5. Data Storage

**Formats**:
- **Cohorts**: Parquet (efficient columnar storage)
- **Snapshot Features**: Parquet
- **Sequence Features**: NumPy arrays (.npy) + JSON metadata

**Directory Structure**:
```
data/
├── raw/
│   └── [source files]
├── synthetic/
│   ├── admissions.parquet
│   ├── patients.parquet
│   ├── icustays.parquet
│   └── events.parquet
└── processed/
    ├── cohort_readmission.parquet
    ├── cohort_icu_mortality.parquet
    ├── features_readmission_snapshot.parquet
    ├── features_icu_mortality_snapshot.parquet
    ├── readmission_sequence_train/
    │   ├── sequences.npy
    │   ├── masks.npy
    │   ├── labels.npy
    │   ├── static_features.npy
    │   └── metadata.json
    └── [similar for val/test splits]
```

## Data Quality Checks

1. **Missing Values**: Filled with 0 or median
2. **Outliers**: Clipped at ±5 standard deviations
3. **Temporal Ordering**: Verified in cohort building
4. **Label Balance**: Monitored, class weights used if needed

## Privacy and Ethics

- **No Real PHI**: All examples use synthetic data
- **Deidentification**: If using MIMIC, follow PhysioNet requirements
- **Disclaimer**: Clearly labeled as research-only

## Running the Pipeline

```bash
# Generate synthetic data
python -m ehrtriage.scripts.generate_synthetic

# Or use make command (if Makefile created)
# Build cohorts and features as part of training
python -m ehrtriage.scripts.train_all
```
