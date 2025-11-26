"""
Prediction endpoints for the API.

Handles risk prediction requests for different tasks.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Any, Tuple

import numpy as np
import pandas as pd
import torch
from fastapi import APIRouter, HTTPException, Query

from api.models import (
    PatientTimeline,
    PredictionResponse,
    AvailableModels,
    ModelInfo,
    ContributingEvent,
)
from ehrtriage.config import MODELS_DIR, get_feature_config
from ehrtriage.schemas import TaskType, RiskLevel
from ehrtriage.explain.text_generator import explain_prediction

router = APIRouter()

# Global model cache
_model_cache: Dict[str, Any] = {}


def load_model_from_cache(task: str, model_type: str = "logistic") -> Tuple[Any, Optional[Dict]]:
    """Load model from cache or disk. Returns (model, config) tuple."""
    cache_key = f"{task}_{model_type}"

    if cache_key in _model_cache:
        return _model_cache[cache_key]

    # Model path
    model_dir = MODELS_DIR / "artifacts" / task

    if model_type == "logistic":
        from ehrtriage.models.baselines import LogisticBaseline

        try:
            model = LogisticBaseline.load(model_dir, "logistic")
            _model_cache[cache_key] = (model, None)
            return model, None
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=f"Model not found for {task}/{model_type}: {str(e)}",
            )
    elif model_type == "gru":
        from ehrtriage.models.sequence import GRURiskModel, load_sequence_model

        try:
            model, config = load_sequence_model(GRURiskModel, model_dir, "gru", device="cpu")
            _model_cache[cache_key] = (model, config)
            return model, config
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=f"GRU model not found for {task}: {str(e)}",
            )
    elif model_type == "transformer":
        from ehrtriage.models.sequence import TransformerRiskModel, load_sequence_model

        try:
            model, config = load_sequence_model(TransformerRiskModel, model_dir, "transformer", device="cpu")
            _model_cache[cache_key] = (model, config)
            return model, config
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=f"Transformer model not found for {task}: {str(e)}",
            )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model type: {model_type}. Supported: logistic, gru, transformer",
        )


def convert_timeline_to_features(
    timeline: PatientTimeline, task: str
) -> pd.DataFrame:
    """
    Convert patient timeline to feature vector for prediction.

    This is a simplified version for demo purposes.
    In production, you'd use the same feature engineering pipeline as training.
    """
    config = get_feature_config()

    # Extract vital/lab/med features
    features = {}

    # Static features - handle both dict and StaticFeatures object
    static = timeline.static_features
    if static is not None:
        if isinstance(static, dict):
            features["age"] = static.get("age", 50) or 50
            sex = static.get("sex", "U")
            features["sex_M"] = 1.0 if sex == "M" else 0.0
            features["sex_F"] = 1.0 if sex == "F" else 0.0
            features["comorbidity_chf"] = float(static.get("comorbidity_chf", False) or False)
            features["comorbidity_renal"] = float(static.get("comorbidity_renal", False) or False)
            features["comorbidity_liver"] = float(static.get("comorbidity_liver", False) or False)
            features["comorbidity_copd"] = float(static.get("comorbidity_copd", False) or False)
            features["comorbidity_diabetes"] = float(static.get("comorbidity_diabetes", False) or False)
        else:
            features["age"] = static.age or 50
            features["sex_M"] = 1.0 if static.sex == "M" else 0.0
            features["sex_F"] = 1.0 if static.sex == "F" else 0.0
            features["comorbidity_chf"] = float(static.comorbidity_chf)
            features["comorbidity_renal"] = float(static.comorbidity_renal)
            features["comorbidity_liver"] = float(static.comorbidity_liver)
            features["comorbidity_copd"] = float(static.comorbidity_copd)
            features["comorbidity_diabetes"] = float(static.comorbidity_diabetes)
    else:
        # Defaults
        features["age"] = 50
        features["sex_M"] = 0.5
        features["sex_F"] = 0.5
        features["comorbidity_chf"] = 0.0
        features["comorbidity_renal"] = 0.0
        features["comorbidity_liver"] = 0.0
        features["comorbidity_copd"] = 0.0
        features["comorbidity_diabetes"] = 0.0

    # Aggregate events
    vitals_dict = {code: [] for code in config["features"]["vitals"].keys()}
    labs_dict = {code: [] for code in config["features"]["labs"].keys()}
    meds_dict = {code: 0 for code in config["features"]["medications"]}

    for event in timeline.events:
        if event.type == "vital" and event.code in vitals_dict:
            vitals_dict[event.code].append(event.value)
        elif event.type == "lab" and event.code in labs_dict:
            labs_dict[event.code].append(event.value)
        elif event.type == "medication" and event.code in meds_dict:
            meds_dict[event.code] = 1

    # Aggregate vital stats
    for code, values in vitals_dict.items():
        if values:
            features[f"{code}_mean"] = np.mean(values)
            features[f"{code}_min"] = np.min(values)
            features[f"{code}_max"] = np.max(values)
            features[f"{code}_std"] = np.std(values) if len(values) > 1 else 0.0
            features[f"{code}_count"] = len(values)
        else:
            features[f"{code}_mean"] = 0.0
            features[f"{code}_min"] = 0.0
            features[f"{code}_max"] = 0.0
            features[f"{code}_std"] = 0.0
            features[f"{code}_count"] = 0

    # Aggregate lab stats
    for code, values in labs_dict.items():
        if values:
            features[f"{code}_mean"] = np.mean(values)
            features[f"{code}_min"] = np.min(values)
            features[f"{code}_max"] = np.max(values)
            features[f"{code}_last"] = values[-1]
            features[f"{code}_count"] = len(values)
        else:
            features[f"{code}_mean"] = 0.0
            features[f"{code}_min"] = 0.0
            features[f"{code}_max"] = 0.0
            features[f"{code}_last"] = 0.0
            features[f"{code}_count"] = 0

    # Medication features
    for code, given in meds_dict.items():
        features[f"med_{code}_given"] = float(given)
        features[f"med_{code}_count"] = float(given)

    # Convert to DataFrame
    feature_df = pd.DataFrame([features])

    return feature_df


def convert_timeline_to_sequence(
    timeline: PatientTimeline, task: str, model_config: Optional[Dict]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Convert patient timeline to sequence format for GRU/Transformer models.
    
    Returns:
        Tuple of (sequence, static, mask, feature_names)
    """
    config = get_feature_config()
    
    # Default parameters
    seq_len = model_config.get("seq_len", 12) if model_config else 12
    input_dim = model_config.get("input_dim", 10) if model_config else 10
    static_dim = model_config.get("static_dim", 8) if model_config else 8
    
    # Get vital and lab codes
    vital_codes = list(config["features"]["vitals"].keys())
    lab_codes = list(config["features"]["labs"].keys())
    med_codes = config["features"]["medications"]
    
    # Feature names for sequence
    feature_names = vital_codes + lab_codes + med_codes
    
    # Sort events by time
    sorted_events = sorted(timeline.events, key=lambda e: e.time)
    
    # Create sequence tensor - shape [seq_len, input_dim]
    sequence = np.zeros((seq_len, len(feature_names)), dtype=np.float32)
    mask = np.zeros(seq_len, dtype=np.float32)
    
    # Group events by time bin (simplified - just use first seq_len events)
    time_idx = 0
    last_time = None
    
    for event in sorted_events:
        if time_idx >= seq_len:
            break
            
        # Check if we should advance time bin
        if last_time is not None and event.time != last_time:
            time_idx += 1
            if time_idx >= seq_len:
                break
        
        last_time = event.time
        mask[time_idx] = 1.0
        
        # Find feature index
        if event.code in feature_names:
            feat_idx = feature_names.index(event.code)
            if event.value is not None:
                sequence[time_idx, feat_idx] = float(event.value)
            elif event.type == "medication":
                sequence[time_idx, feat_idx] = 1.0
    
    # If no events, set at least one mask position
    if mask.sum() == 0:
        mask[0] = 1.0
    
    # Create static features
    static = np.zeros(static_dim, dtype=np.float32)
    sf = timeline.static_features
    
    if sf is not None:
        if isinstance(sf, dict):
            static[0] = float(sf.get("age", 50) or 50) / 100.0
            static[1] = 1.0 if sf.get("sex") == "M" else 0.0
            static[2] = float(sf.get("comorbidity_chf", False) or False)
            static[3] = float(sf.get("comorbidity_renal", False) or False)
            static[4] = float(sf.get("comorbidity_liver", False) or False)
            static[5] = float(sf.get("comorbidity_copd", False) or False)
            static[6] = float(sf.get("comorbidity_diabetes", False) or False)
        else:
            static[0] = float(sf.age or 50) / 100.0
            static[1] = 1.0 if sf.sex == "M" else 0.0
            static[2] = float(sf.comorbidity_chf)
            static[3] = float(sf.comorbidity_renal)
            static[4] = float(sf.comorbidity_liver)
            static[5] = float(sf.comorbidity_copd)
            static[6] = float(sf.comorbidity_diabetes)
    else:
        static[0] = 0.5  # default age
    
    return sequence, static, mask, feature_names


@router.post("/predict/{task}", response_model=PredictionResponse)
async def predict(
    task: TaskType,
    timeline: PatientTimeline,
    model_type: str = Query("logistic", description="Model type to use"),
):
    """
    Make a risk prediction for a patient timeline.

    Args:
        task: Prediction task (readmission or icu_mortality)
        timeline: Patient timeline with events
        model_type: Type of model to use (default: logistic)

    Returns:
        PredictionResponse with risk score and explanation
    """
    try:
        # Load model
        model, model_config = load_model_from_cache(task.value, model_type)

        # Convert timeline to features
        features_df = convert_timeline_to_features(timeline, task.value)

        # Make prediction
        if model_type == "logistic":
            from ehrtriage.explain.attribution import get_logistic_attributions

            # Get feature names from model
            if hasattr(model, "feature_names") and model.feature_names:
                # Align features with model's expected features
                model_features = model.feature_names
                missing_features = set(model_features) - set(features_df.columns)

                # Add missing features with zeros
                for feat in missing_features:
                    features_df[feat] = 0.0

                # Reorder to match model
                features_df = features_df[model_features]

            proba = model.predict_proba(features_df)
            risk_score = float(proba[0, 1])

            # Get attributions
            attributions = get_logistic_attributions(
                model,
                features_df.values[0],
                list(features_df.columns),
                top_k=10,
            )

            # Generate explanation
            explanation_text, contributing = explain_prediction(
                model_type="logistic",
                risk_score=risk_score,
                task=task.value,
                attributions=attributions,
            )

        else:
            # Sequence model prediction (GRU or Transformer)
            from ehrtriage.explain.attribution import explain_sequence_prediction
            
            # Convert features to sequence format
            sequence, static, mask, feature_names = convert_timeline_to_sequence(
                timeline, task.value, model_config
            )
            
            # Get prediction with explanations
            result = explain_sequence_prediction(
                model=model,
                sequence=sequence,
                static=static,
                mask=mask,
                feature_names=feature_names,
                top_k_timesteps=3,
                top_k_features=5,
                device="cpu",
            )
            
            risk_score = result["risk_score"]
            important_events = result["important_events"]
            
            # Generate explanation
            explanation_text, contributing = explain_prediction(
                model_type=model_type,
                risk_score=risk_score,
                task=task.value,
                important_events=important_events,
            )

        # Determine risk level
        if risk_score < 0.3:
            risk_label = RiskLevel.LOW
        elif risk_score < 0.7:
            risk_label = RiskLevel.MEDIUM
        else:
            risk_label = RiskLevel.HIGH

        # Convert contributing events to ContributingEvent objects
        contributing_events = [
            ContributingEvent(**event) for event in contributing[:10]
        ]

        return PredictionResponse(
            task=task,
            risk_score=risk_score,
            risk_label=risk_label,
            explanation=explanation_text,
            contributing_events=contributing_events,
            model_name=f"{model_type}_{task.value}",
            model_version="0.1.0",
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.get("/models", response_model=AvailableModels)
async def get_available_models():
    """
    Get list of available models and their metrics.

    Returns:
        AvailableModels with model information
    """
    models = []

    # Check for available models
    artifacts_dir = MODELS_DIR / "artifacts"

    if artifacts_dir.exists():
        for task_dir in artifacts_dir.iterdir():
            if task_dir.is_dir():
                task_name = task_dir.name

                # Check for model files
                for model_file in task_dir.glob("*_metadata.json"):
                    try:
                        with open(model_file, "r") as f:
                            metadata = json.load(f)

                        model_name = model_file.stem.replace("_metadata", "")

                        # Load metrics if available
                        metrics_file = task_dir / f"{model_name}_metrics.json"
                        if metrics_file.exists():
                            with open(metrics_file, "r") as f:
                                metrics = json.load(f)
                        else:
                            metrics = {}

                        models.append(
                            ModelInfo(
                                task=TaskType(task_name),
                                model_name=model_name,
                                model_type=metadata.get("model_type", "unknown"),
                                version="0.1.0",
                                metrics=metrics,
                            )
                        )
                    except Exception as e:
                        print(f"Error loading model metadata: {e}")
                        continue

    if not models:
        # Return placeholder if no models found
        models = [
            ModelInfo(
                task=TaskType.READMISSION,
                model_name="logistic_placeholder",
                model_type="logistic_regression",
                version="0.1.0",
                metrics={"note": "No trained models found. Run training scripts first."},
            )
        ]

    return AvailableModels(models=models)


@router.get("/example/{task}")
async def get_example_timeline(task: TaskType):
    """
    Get an example patient timeline for a task.

    Args:
        task: Prediction task

    Returns:
        Example PatientTimeline
    """
    # Create example timeline
    if task == TaskType.READMISSION:
        examples = [
            {
                "name": "Low Risk Patient",
                "description": "65-year-old male with controlled diabetes, stable vitals",
                "expected_risk": "Low",
                "timeline": {
                    "subject_id": "P000001",
                    "stay_id": "H00000001",
                    "events": [
                        {"time": "2024-01-01T08:00:00", "type": "vital", "code": "heart_rate", "value": 75.0},
                        {"time": "2024-01-01T08:00:00", "type": "vital", "code": "sbp", "value": 130.0},
                        {"time": "2024-01-01T08:00:00", "type": "vital", "code": "spo2", "value": 98.0},
                        {"time": "2024-01-01T08:30:00", "type": "lab", "code": "lactate", "value": 1.0},
                        {"time": "2024-01-01T08:30:00", "type": "lab", "code": "creatinine", "value": 0.9},
                    ],
                    "static_features": {
                        "age": 65,
                        "sex": "M",
                        "comorbidity_diabetes": True,
                    },
                },
            },
            {
                "name": "High Risk Patient",
                "description": "72-year-old female with multiple comorbidities, abnormal vitals",
                "expected_risk": "High",
                "timeline": {
                    "subject_id": "P000002",
                    "stay_id": "H00000002",
                    "events": [
                        {"time": "2024-01-01T08:00:00", "type": "vital", "code": "heart_rate", "value": 110.0},
                        {"time": "2024-01-01T08:00:00", "type": "vital", "code": "sbp", "value": 95.0},
                        {"time": "2024-01-01T08:00:00", "type": "vital", "code": "spo2", "value": 88.0},
                        {"time": "2024-01-01T08:30:00", "type": "lab", "code": "lactate", "value": 3.5},
                        {"time": "2024-01-01T08:30:00", "type": "lab", "code": "creatinine", "value": 2.2},
                        {"time": "2024-01-01T09:00:00", "type": "medication", "code": "vasopressor", "value": 1.0},
                    ],
                    "static_features": {
                        "age": 72,
                        "sex": "F",
                        "comorbidity_chf": True,
                        "comorbidity_renal": True,
                        "comorbidity_copd": True,
                    },
                },
            },
        ]
    else:  # ICU_MORTALITY
        examples = [
            {
                "name": "Stable ICU Patient",
                "description": "55-year-old male, stable vitals, no vasopressors",
                "expected_risk": "Low",
                "timeline": {
                    "subject_id": "P000003",
                    "stay_id": "ICU00000001",
                    "events": [
                        {"time": "2024-01-01T00:00:00", "type": "vital", "code": "heart_rate", "value": 85.0},
                        {"time": "2024-01-01T00:00:00", "type": "vital", "code": "map", "value": 75.0},
                        {"time": "2024-01-01T00:00:00", "type": "vital", "code": "spo2", "value": 96.0},
                        {"time": "2024-01-01T01:00:00", "type": "lab", "code": "lactate", "value": 1.5},
                    ],
                    "static_features": {
                        "age": 55,
                        "sex": "M",
                        "comorbidity_diabetes": True,
                    },
                },
            },
            {
                "name": "Critical ICU Patient",
                "description": "68-year-old female, on vasopressors, abnormal labs",
                "expected_risk": "High",
                "timeline": {
                    "subject_id": "P000004",
                    "stay_id": "ICU00000002",
                    "events": [
                        {"time": "2024-01-01T00:00:00", "type": "vital", "code": "heart_rate", "value": 125.0},
                        {"time": "2024-01-01T00:00:00", "type": "vital", "code": "map", "value": 55.0},
                        {"time": "2024-01-01T00:00:00", "type": "vital", "code": "spo2", "value": 85.0},
                        {"time": "2024-01-01T01:00:00", "type": "lab", "code": "lactate", "value": 5.2},
                        {"time": "2024-01-01T01:00:00", "type": "medication", "code": "vasopressor", "value": 1.0},
                        {"time": "2024-01-01T02:00:00", "type": "vital", "code": "heart_rate", "value": 135.0},
                    ],
                    "static_features": {
                        "age": 68,
                        "sex": "F",
                        "comorbidity_chf": True,
                        "comorbidity_renal": True,
                    },
                },
            },
        ]

    return {"examples": examples}
