"""
Configuration management for EHR Timeline Triage.

This module handles loading and validating configuration from YAML files.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


class Config:
    """Configuration container with validation."""

    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._config


def load_config(config_name: str = "features") -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_name: Name of the config file (without .yml extension)

    Returns:
        Config object with loaded settings

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    config_path = CONFIG_DIR / f"{config_name}.yml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)


def get_default_feature_config() -> Dict[str, Any]:
    """
    Return default feature engineering configuration.

    This is used when no config file exists yet or for fallback.
    """
    return {
        "time_bin_hours": 4,
        "tasks": {
            "readmission": {
                "lookback_hours": 48,
                "prediction_window_days": 30,
            },
            "icu_mortality": {
                "lookback_hours": 48,
                "prediction_window_hours": 48,
            },
        },
        "features": {
            "vitals": {
                "heart_rate": {"min": 0, "max": 250, "unit": "bpm"},
                "sbp": {"min": 0, "max": 300, "unit": "mmHg"},
                "dbp": {"min": 0, "max": 200, "unit": "mmHg"},
                "map": {"min": 0, "max": 250, "unit": "mmHg"},
                "respiratory_rate": {"min": 0, "max": 60, "unit": "breaths/min"},
                "spo2": {"min": 0, "max": 100, "unit": "%"},
                "temperature": {"min": 30, "max": 43, "unit": "C"},
            },
            "labs": {
                "lactate": {"min": 0, "max": 30, "unit": "mmol/L"},
                "creatinine": {"min": 0, "max": 20, "unit": "mg/dL"},
                "wbc": {"min": 0, "max": 100, "unit": "K/uL"},
                "hemoglobin": {"min": 0, "max": 25, "unit": "g/dL"},
                "platelet": {"min": 0, "max": 1000, "unit": "K/uL"},
                "sodium": {"min": 100, "max": 180, "unit": "mEq/L"},
                "potassium": {"min": 1, "max": 10, "unit": "mEq/L"},
                "bun": {"min": 0, "max": 200, "unit": "mg/dL"},
            },
            "medications": [
                "vasopressor",
                "antibiotic",
                "sedative",
                "diuretic",
                "anticoagulant",
            ],
            "static": [
                "age",
                "sex",
                "comorbidity_chf",
                "comorbidity_renal",
                "comorbidity_liver",
                "comorbidity_copd",
                "comorbidity_diabetes",
            ],
        },
        "normalization": {
            "method": "standard",  # standard, minmax, robust
            "clip_outliers": True,
            "outlier_std": 5,
        },
        "sequence": {
            "max_length": 12,  # 48 hours / 4 hour bins
            "padding_value": 0.0,
            "mask_padding": True,
        },
    }


def get_default_model_config(model_type: str) -> Dict[str, Any]:
    """
    Return default model configuration.

    Args:
        model_type: Type of model (logistic, gru, transformer)

    Returns:
        Dictionary with model hyperparameters
    """
    configs = {
        "logistic": {
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 1000,
            "class_weight": "balanced",
            "random_state": 42,
        },
        "gru": {
            "input_dim": None,  # Set dynamically based on features
            "hidden_dim": 64,
            "num_layers": 2,
            "dropout": 0.3,
            "bidirectional": True,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 50,
            "early_stopping_patience": 10,
            "weight_decay": 0.0001,
            "random_state": 42,
        },
        "transformer": {
            "input_dim": None,  # Set dynamically
            "d_model": 64,
            "nhead": 4,
            "num_layers": 3,
            "dim_feedforward": 256,
            "dropout": 0.3,
            "learning_rate": 0.0005,
            "batch_size": 32,
            "epochs": 50,
            "early_stopping_patience": 10,
            "weight_decay": 0.0001,
            "random_state": 42,
        },
    }

    return configs.get(model_type, {})


# Module-level config instances (lazy loaded)
_feature_config: Optional[Config] = None


def get_feature_config() -> Config:
    """Get or create feature configuration."""
    global _feature_config
    if _feature_config is None:
        try:
            _feature_config = load_config("features")
        except FileNotFoundError:
            # Use default config if file doesn't exist
            _feature_config = Config(get_default_feature_config())
    return _feature_config
