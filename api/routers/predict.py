"""
Prediction endpoints for the API.

Handles risk prediction requests for different tasks.
"""

import json
from pathlib import Path
from typing import Dict, Optional

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
_model_cache: Dict[str, any] = {}


def load_model_from_cache(task: str, model_type: str = "logistic"):
    """Load model from cache or disk."""
    cache_key = f"{task}_{model_type}"

    if cache_key in _model_cache:
        return _model_cache[cache_key]

    # Model path
    model_dir = MODELS_DIR / "artifacts" / task

    if model_type == "logistic":
        from ehrtriage.models.baselines import LogisticBaseline

        try:
            model = LogisticBaseline.load(model_dir, "logistic")
            _model_cache[cache_key] = model
            return model
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=f"Model not found for {task}/{model_type}: {str(e)}",
            )
    else:
        # Sequence models would be loaded here
        raise HTTPException(
            status_code=501,
            detail=f"Sequence models not yet implemented in API",
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

    # Static features
    if timeline.static_features:
        features["age"] = timeline.static_features.age or 50
        features["sex_M"] = 1.0 if timeline.static_features.sex == "M" else 0.0
        features["sex_F"] = 1.0 if timeline.static_features.sex == "F" else 0.0
        features["comorbidity_chf"] = float(timeline.static_features.comorbidity_chf)
        features["comorbidity_renal"] = float(timeline.static_features.comorbidity_renal)
        features["comorbidity_liver"] = float(timeline.static_features.comorbidity_liver)
        features["comorbidity_copd"] = float(timeline.static_features.comorbidity_copd)
        features["comorbidity_diabetes"] = float(timeline.static_features.comorbidity_diabetes)
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
        model = load_model_from_cache(task.value, model_type)

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
            # Sequence model prediction (placeholder)
            raise HTTPException(
                status_code=501,
                detail="Sequence model predictions not yet implemented",
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

    except Exception as e:
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
        return {
            "subject_id": "P000001",
            "stay_id": "H00000001",
            "events": [
                {"time": "2024-01-01T08:00:00", "type": "vital", "code": "heart_rate", "value": 85.0},
                {"time": "2024-01-01T08:00:00", "type": "vital", "code": "sbp", "value": 120.0},
                {"time": "2024-01-01T08:00:00", "type": "vital", "code": "spo2", "value": 97.0},
                {"time": "2024-01-01T08:30:00", "type": "lab", "code": "lactate", "value": 1.2},
                {"time": "2024-01-01T08:30:00", "type": "lab", "code": "creatinine", "value": 1.0},
                {"time": "2024-01-01T09:00:00", "type": "medication", "code": "antibiotic", "value": 1.0},
            ],
            "static_features": {
                "age": 65,
                "sex": "M",
                "comorbidity_chf": True,
                "comorbidity_diabetes": True,
            },
        }
    else:  # ICU_MORTALITY
        return {
            "subject_id": "P000002",
            "stay_id": "ICU00000001",
            "events": [
                {"time": "2024-01-01T00:00:00", "type": "vital", "code": "heart_rate", "value": 110.0},
                {"time": "2024-01-01T00:00:00", "type": "vital", "code": "map", "value": 65.0},
                {"time": "2024-01-01T00:00:00", "type": "vital", "code": "spo2", "value": 90.0},
                {"time": "2024-01-01T01:00:00", "type": "lab", "code": "lactate", "value": 3.5},
                {"time": "2024-01-01T01:00:00", "type": "medication", "code": "vasopressor", "value": 1.0},
                {"time": "2024-01-01T02:00:00", "type": "vital", "code": "heart_rate", "value": 125.0},
            ],
            "static_features": {
                "age": 72,
                "sex": "F",
                "comorbidity_renal": True,
                "comorbidity_copd": True,
            },
        }
