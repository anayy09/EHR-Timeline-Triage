"""
Data schemas using Pydantic for validation and serialization.

These schemas define the structure of data flowing through the system.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class Sex(str, Enum):
    """Patient sex."""

    MALE = "M"
    FEMALE = "F"
    OTHER = "O"
    UNKNOWN = "U"


class EventType(str, Enum):
    """Type of clinical event."""

    VITAL = "vital"
    LAB = "lab"
    MEDICATION = "medication"
    PROCEDURE = "procedure"
    DIAGNOSIS = "diagnosis"


class TaskType(str, Enum):
    """Prediction task type."""

    READMISSION = "readmission"
    ICU_MORTALITY = "icu_mortality"


class Event(BaseModel):
    """A single clinical event in the timeline."""

    time: str = Field(description="ISO format timestamp or offset in hours")
    type: str = Field(description="Event type (vital, lab, medication, etc.)")
    code: Optional[str] = Field(None, description="Event code or name")
    value: Optional[float] = Field(None, description="Numeric value for the event")
    unit: Optional[str] = Field(None, description="Unit of measurement")

    @field_validator("time")
    @classmethod
    def validate_time(cls, v: str) -> str:
        """Validate time format."""
        # Try parsing as ISO format
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
            return v
        except (ValueError, AttributeError):
            # Check if it's a numeric offset
            try:
                float(v)
                return v
            except ValueError:
                raise ValueError(f"Time must be ISO format or numeric offset: {v}")


class StaticFeatures(BaseModel):
    """Static patient features."""

    age: Optional[int] = Field(None, ge=0, le=120)
    sex: Optional[Sex] = None
    comorbidity_chf: bool = False
    comorbidity_renal: bool = False
    comorbidity_liver: bool = False
    comorbidity_copd: bool = False
    comorbidity_diabetes: bool = False


class PatientTimeline(BaseModel):
    """Complete patient timeline with events."""

    subject_id: str = Field(description="Unique patient identifier")
    stay_id: Optional[str] = Field(None, description="Hospital stay or ICU stay ID")
    events: List[Event] = Field(description="List of clinical events")
    static_features: Optional[Dict[str, Any]] = Field(
        None, description="Static patient features as key-value pairs"
    )

    @field_validator("events")
    @classmethod
    def validate_events(cls, v: List[Event]) -> List[Event]:
        """Ensure events list is not empty."""
        if not v:
            raise ValueError("Events list cannot be empty")
        return v


class RiskLevel(str, Enum):
    """Categorical risk level."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ContributingEvent(BaseModel):
    """An event that contributed to the prediction."""

    time: str = Field(description="Event timestamp")
    type: str = Field(description="Event type")
    code: str = Field(description="Event code/name")
    value: Optional[float] = None
    contribution_score: float = Field(description="Contribution to prediction")


class PredictionResponse(BaseModel):
    """Response from prediction API."""

    task: TaskType = Field(description="Prediction task")
    risk_score: float = Field(ge=0.0, le=1.0, description="Risk probability [0-1]")
    risk_label: RiskLevel = Field(description="Categorical risk level")
    explanation: str = Field(description="Human-readable explanation")
    contributing_events: List[ContributingEvent] = Field(
        description="Events that contributed most to prediction"
    )
    model_name: str = Field(description="Name of model used")
    model_version: Optional[str] = Field(None, description="Model version")

    @field_validator("risk_label", mode="before")
    @classmethod
    def compute_risk_label(cls, v: Any, info) -> RiskLevel:
        """Compute risk label from risk score if not provided."""
        if isinstance(v, RiskLevel):
            return v

        # Get risk_score from values
        risk_score = info.data.get("risk_score", 0.5)

        if risk_score < 0.3:
            return RiskLevel.LOW
        elif risk_score < 0.7:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.HIGH


class ModelInfo(BaseModel):
    """Information about an available model."""

    task: TaskType
    model_name: str
    model_type: str  # logistic, gru, transformer
    version: str
    metrics: Dict[str, Any]  # train/val/test metrics with nested dicts
    trained_date: Optional[str] = None


class AvailableModels(BaseModel):
    """Response listing available models."""

    models: List[ModelInfo]


# Cohort data schemas
class ReadmissionCohortEntry(BaseModel):
    """A single entry in the readmission cohort."""

    subject_id: str
    hadm_id: str
    index_admit_time: datetime
    discharge_time: datetime
    readmit_30d: int = Field(ge=0, le=1)
    los_hours: Optional[float] = None


class ICUMortalityCohortEntry(BaseModel):
    """A single entry in the ICU mortality cohort."""

    stay_id: str
    subject_id: str
    icu_intime: datetime
    prediction_time_48h: datetime
    mortality_label: int = Field(ge=0, le=1)
    actual_death_time: Optional[datetime] = None
