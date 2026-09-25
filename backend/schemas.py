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
                            doc: str, nullable: set[str] | None = None,
                            with_district: bool = False) -> type[BaseModel]:
    """Create an observation schema from a saved feature contract."""
    nullable = nullable or set()
    fields: dict[str, Any] = {
        "date": (date_type, Field(..., description="Observation date (YYYY-MM-DD)")),
        "latitude": (float, Field(..., ge=-90, le=90)),
        "longitude": (float, Field(..., ge=-180, le=180)),
    }
    if with_district:
        fields["district"] = (str | None, Field(
            None, description="Identifier only, never a model feature. If latitude/"
                              "longitude are omitted, the district's representative "
                              "point is used."))
        fields["latitude"] = (float | None, Field(None, ge=-90, le=90))
        fields["longitude"] = (float | None, Field(None, ge=-180, le=180))
    for feat in features:
        if feat in category_schema:
            cats = tuple(category_schema[feat])
            fields[feat] = (Literal[cats],  # type: ignore[valid-type]
                            Field(..., description=f"Categorical. Allowed: {list(cats)}"))
        elif feat in nullable:
            fields[feat] = (float | None, Field(
                ..., description=f"Numeric feature '{feat}'. May be null: missing in the "
                                 "model's training data, scored by XGBoost's learned "
                                 "default branch (never imputed)."))
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
    # Added with the real-model integration (additive; older clients ignore them).
    prediction_model_set: str = Field("real", description=(
        "Model set behind /predict/flood and /predict/landslide. data_mode above "
        "describes the dashboard / TCDL / SHAP endpoints."))
    synthetic_models_loaded: bool = Field(False, description=(
        "Synthetic reference models (served by ?model_set=synthetic)"))
    districts_loaded: bool = False
    dashboard_model_set: str = Field("real", description=(
        "Model set behind the dashboard, TCDL and SHAP endpoints"))
    shap_available: bool = False
    data_notice: str | None = Field(None, description=(
        "The caveat every dashboard response of this data_mode carries"))


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
    prediction_model_set: str = "real"
    synthetic_models: dict[str, Any] = Field(default_factory=dict)
    districts: dict[str, Any] = Field(default_factory=dict)
    dashboard_model_set: str = "real"
    shap: dict[str, Any] = Field(default_factory=dict)


class HazardPrediction(BaseModel):
    date: date_type
    latitude: float
    longitude: float
    probability: float = Field(..., description="P(hazard = 1) from the frozen model")
    prediction: int = Field(..., description="1 if probability >= decision threshold")
    warning_status: str = Field(..., description="Warning | No Warning")
    district: str | None = Field(None, description="Identifier only; not a model feature")
    missing_features: list[str] = Field(default_factory=list, description=(
        "Features supplied as null, scored through XGBoost's learned default branch"))


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
    model_set: str = Field("synthetic", description="real | synthetic")
    model_artifact: str | None = Field(None, description="Booster file that scored the input")
    model_note: str | None = None


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
    data_notice: str | None = None


class LeadTimeRecord(BaseModel):
    system: str
    hazard_type: str
    location_id: str
    latitude: float
    longitude: float
    district: str | None = Field(None, description="Identifier only (real data)")
    event_time: str | None
    warning_time: str | None
    detected: int
    lead_time_hours: float | None = Field(...,
        description=("Earliest-warning variant. DATE-QUANTISED: whole days x 24 (the data has "
                     "daily resolution; not hour-level timing). Name kept for compatibility; "
                     "prefer lead_time_days."))
    lead_time_hours_contiguous: float | None = Field(...,
        description=("Unbroken alert run ending at onset. DATE-QUANTISED: whole days x 24; "
                     "prefer lead_time_days_contiguous."))
    lead_time_days: float | None = Field(None,
        description="Earliest-warning variant in whole days (daily-resolution data)")
    lead_time_days_contiguous: float | None = Field(None,
        description="Unbroken alert run ending at onset, in whole days")


class LeadTimeResponse(BaseModel):
    data_mode: str
    data_notice: str | None = None
    lead_time_resolution: str
    lead_time_note: str
    lead_time_definition: dict[str, Any]
    event_source: dict[str, Any]
    n_records: int
    records: list[LeadTimeRecord]
