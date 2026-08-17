"""
Pydantic request/response schemas.

The observation schemas are BUILT AT IMPORT TIME from the saved
`flood_features.json` / `landslide_features.json` contracts rather than
being retyped by hand. That means the API's input contract cannot drift
from the models it serves: if a feature list changes in the ML layer,
the request schema and its Swagger documentation change with it.

Categorical fields are typed as Literals drawn from the saved category
schema, so an invalid land_cover or lithology is rejected by validation
before it ever reaches a model.
"""

from __future__ import annotations

from datetime import date as date_type
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model


def build_observation_model(name: str, features: list[str],
                            category_schema: dict[str, list[str]],
                            doc: str) -> type[BaseModel]:
    """Create an observation schema from a saved feature contract."""
    fields: dict[str, Any] = {
        "date": (date_type, Field(..., description="Observation date (YYYY-MM-DD)")),
        "latitude": (float, Field(..., ge=-90, le=90)),
        "longitude": (float, Field(..., ge=-180, le=180)),
    }
    for feat in features:
        if feat in category_schema:
            cats = tuple(category_schema[feat])
            fields[feat] = (Literal[cats],  # type: ignore[valid-type]
                            Field(..., description=f"Categorical. Allowed: {list(cats)}"))
        else:
            fields[feat] = (float, Field(..., description=f"Numeric feature '{feat}'"))
    model = create_model(name, __doc__=doc, **fields)
    model.model_config = ConfigDict(extra="forbid")
    return model


# ---------------------------------------------------------------------
# Generic responses
# ---------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = Field(..., description="ok | degraded")
    api_version: str
    data_mode: str = Field(..., description="synthetic | real")
    synthetic_data_warning: str | None
    ready_for_prediction: bool
    flood_model_loaded: bool
    landslide_model_loaded: bool
    tcdl_available: bool
    master_dataset_loaded: bool
    errors: list[str]


class ModelStatusResponse(BaseModel):
    status: str
    data_mode: str
    ready_for_prediction: bool
    versions: dict[str, str]
    flood_model: dict[str, Any]
    landslide_model: dict[str, Any]
    tcdl: dict[str, Any]
    artifacts: dict[str, Any]
    errors: list[str]


class HazardPrediction(BaseModel):
    date: date_type
    latitude: float
    longitude: float
    probability: float = Field(..., description="P(hazard = 1) from the frozen model")
    prediction: int = Field(..., description="1 if probability >= decision threshold")
    warning_status: str = Field(..., description="Warning | No Warning")


class HazardPredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    hazard: str
    model_version: str
    decision_threshold: float
    data_mode: str
    synthetic_data_warning: str | None
    n_observations: int
    predictions: list[HazardPrediction]
    calibration_note: str


class TrendSignals(BaseModel):
    """The temporal signals TCDL consumes. Null where the window is not yet full."""
    flood_probability: float | None = None
    landslide_probability: float | None = None
    coupled_probability: float | None = None
    flood_prob_ma: float | None = None
    landslide_prob_ma: float | None = None
    coupled_prob_ma: float | None = None
    rainfall_ma: float | None = None
    soil_moisture_ma: float | None = None
    flood_prob_rate: float | None = None
    landslide_prob_rate: float | None = None
    coupled_prob_rate: float | None = None
    rainfall_rate: float | None = None
    soil_moisture_rate: float | None = None
    flood_trend_up: bool | None = None
    landslide_trend_up: bool | None = None
    rainfall_trend_up: bool | None = None
    soil_trend_up: bool | None = None


class RuleDetail(BaseModel):
    rule_id: str
    description: str
    rule_class: str
    values: dict[str, Any]
    thresholds: dict[str, Any]


class WarningExplanation(BaseModel):
    triggered_rules: list[str]
    n_rules_triggered: int
    rule_details: list[RuleDetail]
    summary: str


class CoupledWarningResponse(BaseModel):
    data_mode: str
    synthetic_data_warning: str | None
    warning_status: str = Field(..., description="Warning | No Warning")
    warning_type: str = Field(...,
        description="No Warning | Flood Warning | Landslide Warning | Coupled Hazard Warning")
    warning_timestamp: str | None
    timestamp_resolution: str
    location: dict[str, Any]
    flood_probability: float | None
    landslide_probability: float | None
    coupled_probability: float | None
    trends: TrendSignals
    explanation: WarningExplanation
    baselines: dict[str, Any]
    tcdl_version: str
    notes: list[str]


class LeadTimeRecord(BaseModel):
    system: str
    hazard_type: str
    location_id: str
    latitude: float
    longitude: float
    event_time: str | None
    warning_time: str | None
    detected: int
    lead_time_hours: float | None = Field(...,
        description="Earliest-warning variant, in hours")
    lead_time_hours_contiguous: float | None = Field(...,
        description="Unbroken alert run ending at onset, in hours")


class LeadTimeResponse(BaseModel):
    data_mode: str
    lead_time_resolution: str
    lead_time_note: str
    lead_time_definition: dict[str, Any]
    event_source: dict[str, Any]
    n_records: int
    records: list[LeadTimeRecord]
