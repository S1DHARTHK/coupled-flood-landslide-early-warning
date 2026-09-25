/**
 * Type contracts for the Kerala Flood-Landslide Early Warning backend.
 *
 * These mirror the FastAPI response shapes. The frontend performs NO ML and
 * NO TCDL computation -- it only renders what the backend returns.
 *
 * `district` is optional on location-bearing types. The real pipeline sends
 * it (one representative point per Kerala district); the synthetic reference
 * pipeline does not. It is never fabricated or inferred from coordinates.
 */

/** The seven TCDL rules. Defined by the ML layer; the frontend never adds to this. */
export const TCDL_RULES = [
  "R1_FLOOD_LEVEL",
  "R2_LANDSLIDE_LEVEL",
  "R3_SUSTAINED_JOINT",
  "R4_RISING_FLOOD",
  "R5_RISING_LANDSLIDE",
  "R6_ENV_PRECURSOR",
  "R7_JOINT_MODERATE",
] as const;
export type TcdlRuleId = (typeof TCDL_RULES)[number];

/** Human-readable labels. Descriptions themselves come from the backend. */
export const RULE_LABELS: Record<TcdlRuleId, string> = {
  R1_FLOOD_LEVEL: "Flood level",
  R2_LANDSLIDE_LEVEL: "Landslide level",
  R3_SUSTAINED_JOINT: "Sustained joint risk",
  R4_RISING_FLOOD: "Rising flood",
  R5_RISING_LANDSLIDE: "Rising landslide",
  R6_ENV_PRECURSOR: "Environmental precursor",
  R7_JOINT_MODERATE: "Joint moderate risk",
};

/** Rules that replicate the single-model baselines vs. genuine coupling rules. */
export const BASELINE_RULES: TcdlRuleId[] = ["R1_FLOOD_LEVEL", "R2_LANDSLIDE_LEVEL"];

export type WarningType =
  | "No Warning"
  | "Flood Warning"
  | "Landslide Warning"
  | "Coupled Hazard Warning";

export type WarningStatus = "Warning" | "No Warning";

/** Map marker severity. Derived from backend values only -- never invented. */
export type HazardLevel = "normal" | "watch" | "warning" | "critical";

export interface LocationRef {
  location_id: string;
  latitude: number;
  longitude: number;
  /** Sent by the real pipeline; absent in synthetic. Never inferred. */
  district?: string | null;
  district_name?: string | null;
}

export interface EnvironmentBlock {
  date: string;
  latitude: number;
  longitude: number;
  common_environmental: Record<string, number | string | null>;
  flood_specific: Record<string, number | string | null>;
  landslide_specific: Record<string, number | string | null>;
  unavailable_fields: string[];
  note: string;
  district?: string | null;
}

export interface TrendSignals {
  flood_probability: number | null;
  landslide_probability: number | null;
  coupled_probability: number | null;
  flood_prob_ma: number | null;
  landslide_prob_ma: number | null;
  coupled_prob_ma: number | null;
  rainfall_ma: number | null;
  soil_moisture_ma: number | null;
  flood_prob_rate: number | null;
  landslide_prob_rate: number | null;
  coupled_prob_rate: number | null;
  rainfall_rate: number | null;
  soil_moisture_rate: number | null;
  flood_trend_up: boolean | null;
  landslide_trend_up: boolean | null;
  rainfall_trend_up: boolean | null;
  soil_trend_up: boolean | null;
}

export interface HazardReading {
  probability: number | null;
  prediction: number;
  warning_status: WarningStatus;
}

export interface CurrentLocation {
  location_id: string;
  split?: string;
  environment: EnvironmentBlock;
  flood: HazardReading;
  landslide: HazardReading;
  trends: TrendSignals;
  tcdl: {
    warning_status: WarningStatus;
    warning_type: WarningType;
    triggered_rules: TcdlRuleId[];
    warning_timestamp: string | null;
  };
  district?: string | null;
  /** The dataset's event labels for this day (0 = not reported, not proven absence). */
  recorded_labels?: { flood?: number; landslide?: number };
}

export interface CurrentResponse {
  data_mode: string;
  synthetic_data_warning: string | null;
  data_notice?: string | null;
  as_of: string;
  /** First and last day the pipeline outputs cover (for history replay). */
  available_date_range?: [string, string];
  n_locations: number;
  locations: CurrentLocation[];
}

export interface WarningRecord {
  date: string;
  latitude: number;
  longitude: number;
  location_id: string;
  split?: string;
  flood_probability: number | null;
  landslide_probability: number | null;
  coupled_probability: number | null;
  coupled_prob_ma: number | null;
  warning_status: WarningStatus;
  warning_type: WarningType;
  triggered_rules: TcdlRuleId[];
  warning_timestamp: string | null;
  baselines: { flood_only: number; landslide_only: number };
  actual: { flood: number; landslide: number };
  district?: string | null;
}

export interface WarningsResponse {
  data_mode: string;
  synthetic_data_warning: string | null;
  data_notice?: string | null;
  warning_types: WarningType[];
  timestamp_resolution: string;
  /** Location-days in the filtered range (warning or not). */
  n_location_days_in_range?: number;
  /** Records matching the filters before the limit was applied. */
  n_matching?: number;
  /** True when n_matching exceeds the records returned. */
  truncated?: boolean;
  n_records: number;
  records: WarningRecord[];
}

export interface TrendPoint extends TrendSignals {
  date: string;
  /** TCDL decision and recorded labels for the day (real pipeline). */
  tcdl_warning?: number;
  flood?: number;
  landslide?: number;
}

export interface TrendsResponse {
  data_mode: string;
  location_id: string;
  district?: string | null;
  n_records: number;
  parameters: Record<string, number | null>;
  source: string;
  series: TrendPoint[];
}

/**
 * One evaluated warning system in the lead-time comparison.
 *
 * The *_hours fields are DATE-QUANTISED (whole days x 24): the data has daily
 * resolution. Names are kept for API compatibility; the UI shows them in days.
 */
export interface SystemEvaluation {
  n_events: number;
  n_detected: number;
  detection_rate: number;
  mean_lead_time_hours_earliest: number | null;
  median_lead_time_hours_earliest: number | null;
  max_lead_time_hours_earliest: number | null;
  mean_lead_time_hours_contiguous: number | null;
  median_lead_time_hours_contiguous: number | null;
  n_zero_lead_nowcast: number;
  n_warning_days: number;
  time_in_warning_rate: number | null;
  n_false_alarm_days: number;
  false_alarm_day_rate: number | null;
  useful_horizon_days: number;
  false_alarm_day_rate_lookback_window: number | null;
  lookback_window_days: number;
}

export interface EvaluationResponse {
  data_mode: string;
  synthetic_data_warning: string | null;
  data_notice?: string | null;
  hazard_type: string;
  evaluation_period: {
    split: string;
    date_range: [string, string];
    n_rows: number;
    n_locations: number;
    note: string;
  };
  systems: Record<string, SystemEvaluation>;
  warning_counts: Record<string, number>;
  rule_firing_counts: Record<string, number>;
  parameters: Record<string, unknown>;
  rules: Record<string, string>;
  rule_classification: { baseline_replicating: string[]; coupling: string[] };
  interpretation_notes: Record<string, string>;
  source: string;
  /** Explains that *_hours lead-time fields are date-quantised. */
  lead_time_unit_note?: string;
}

export interface LeadTimeRecord {
  system: string;
  hazard_type: string;
  location_id: string;
  district?: string | null;
  latitude: number;
  longitude: number;
  event_time: string | null;
  warning_time: string | null;
  detected: number;
  /** Date-quantised (whole days x 24); prefer lead_time_days. */
  lead_time_hours: number | null;
  /** Date-quantised (whole days x 24); prefer lead_time_days_contiguous. */
  lead_time_hours_contiguous: number | null;
  /** Same values in the data's own daily resolution. */
  lead_time_days?: number | null;
  lead_time_days_contiguous?: number | null;
}

export interface LeadTimeResponse {
  data_mode: string;
  data_notice?: string | null;
  lead_time_resolution: string;
  lead_time_note: string;
  lead_time_definition: Record<string, unknown>;
  event_source: Record<string, unknown>;
  n_records: number;
  records: LeadTimeRecord[];
}

export interface SplitMetrics {
  split: string;
  n_rows: number;
  n_actual_positives: number;
  n_predicted_positives: number;
  threshold: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number | null;
  pr_auc: number | null;
  brier_score: number;
  mean_predicted_probability: number;
  observed_positive_rate: number;
  confusion_matrix: {
    true_negatives: number;
    false_positives: number;
    false_negatives: number;
    true_positives: number;
  };
}

export interface ModelPerformance {
  model_version: string;
  target: string;
  decision_threshold: number;
  best_iteration: number | null;
  class_imbalance_handling: Record<string, unknown> | null;
  splits: { train: SplitMetrics; validation: SplitMetrics; test: SplitMetrics };
  calibration_summary: Record<string, Record<string, number>> | null;
  split_periods: Record<string, [string, string] | null>;
}

export interface PerformanceResponse {
  data_mode: string;
  synthetic_data_warning: string | null;
  models: { flood: ModelPerformance | null; landslide: ModelPerformance | null };
  source: string[];
  note: string;
  /** "real" (default, the models behind /predict/*) | "synthetic". */
  model_set?: string;
  model_note?: string;
}

export interface FeatureImportanceRow {
  feature: string;
  importance_gain: number;
  importance_normalised: number;
  split_count: number;
}

export interface FeaturesResponse {
  data_mode: string;
  feature_contract: Record<
    string,
    {
      feature_order: string[];
      n_features: number;
      common_features: string[];
      model_specific_features: string[];
      categorical_features: string[];
      decision_threshold: number;
      category_schema?: Record<string, string[]>;
      /** Features the model was trained with missing values for (real flood: river_level_m). */
      nullable_features?: string[];
    }
  >;
  feature_importance: Record<
    string,
    {
      features: string[];
      importance: FeatureImportanceRow[];
      categorical_features: string[];
      category_schema?: Record<string, string[]>;
    }
  >;
  source: string[];
  note: string;
}

export interface HealthResponse {
  status: string;
  api_version: string;
  data_mode: string;
  synthetic_data_warning: string | null;
  ready_for_prediction: boolean;
  flood_model_loaded: boolean;
  landslide_model_loaded: boolean;
  tcdl_available: boolean;
  master_dataset_loaded: boolean;
  errors: string[];
  prediction_model_set?: string;
  dashboard_model_set?: string;
  synthetic_models_loaded?: boolean;
  districts_loaded?: boolean;
  shap_available?: boolean;
  data_notice?: string | null;
}

export interface ModelsStatusResponse {
  status: string;
  data_mode: string;
  ready_for_prediction: boolean;
  versions: Record<string, string>;
  flood_model: Record<string, unknown>;
  landslide_model: Record<string, unknown>;
  tcdl: Record<string, unknown>;
  artifacts: Record<string, { path: string; exists: boolean }>;
  errors: string[];
  prediction_model_set?: string;
  dashboard_model_set?: string;
  shap?: Record<string, unknown>;
}

export interface LocationsResponse {
  data_mode: string;
  available_date_range?: [string, string];
  locations: Array<
    LocationRef & { n_records: number; first_date: string; last_date: string }
  >;
}

// ---------------------------------------------------------------------
// Real-data prediction (GET /districts, GET /districts/{d}/observation,
// POST /predict/flood, POST /predict/landslide)
//
// The inputs are real rows of the real Master Dataset, read unchanged by the
// backend. `district` is an identifier only -- it is never a model feature.
// ---------------------------------------------------------------------

export interface DistrictInfo {
  district: string;
  latitude: number;
  longitude: number;
  first_date: string;
  last_date: string;
  n_days: number;
  /** Days with an observed CWC river level (0 = no gauge in this district). */
  river_level_days: number;
}

export interface DistrictsResponse {
  data_mode: string;
  n_districts: number;
  districts: DistrictInfo[];
  source: string[];
  note: string;
}

/** One observation in the format POST /predict/* accepts. */
export type Observation = {
  date: string;
  district: string;
  latitude: number;
  longitude: number;
} & Record<string, number | string | null>;

export interface DistrictObservationResponse {
  data_mode: string;
  district: string;
  date: string;
  observation: Observation;
  /** Features that are null in the dataset (never filled in). */
  missing_features: string[];
  /** The dataset's event labels for this day (0 = not reported). */
  recorded_labels: { flood?: number; landslide?: number };
  /** Chronological split the day belongs to (train = in-sample for the models). */
  split: "train" | "validation" | "test" | null;
  source: string;
  note: string;
}

export interface HazardPredictionItem {
  date: string;
  latitude: number;
  longitude: number;
  probability: number;
  prediction: number;
  warning_status: WarningStatus;
  district?: string | null;
  /** Features sent as null, scored through XGBoost's learned default branch. */
  missing_features?: string[];
}

export interface HazardPredictionResponse {
  hazard: "flood" | "landslide";
  model_version: string;
  decision_threshold: number;
  data_mode: string;
  synthetic_data_warning: string | null;
  n_observations: number;
  predictions: HazardPredictionItem[];
  calibration_note: string;
  model_set?: string;
  model_artifact?: string | null;
  model_note?: string | null;
}

/** A location prepared for the Kerala map. Purely presentational. */
export interface MapLocation extends LocationRef {
  floodProbability: number | null;
  landslideProbability: number | null;
  coupledProbability: number | null;
  warningStatus: WarningStatus;
  warningType: WarningType;
  triggeredRules: TcdlRuleId[];
  level: HazardLevel;
  date?: string;
}

// ---------------------------------------------------------------------
// SHAP explainability (served by /explain/global and /explain/current)
//
// The frontend NEVER computes these values. They are produced by
// scripts/analysis/shap_explainability.py and served verbatim by the FastAPI backend.
// SHAP values are in log-odds (margin) space and describe MODEL BEHAVIOUR,
// not physical causation.
// ---------------------------------------------------------------------

/** One row of global mean|SHAP| importance for a model. */
export interface ShapImportanceRow {
  feature: string;
  mean_abs_shap: number;
  share_of_total: number;
  mean_signed_shap: number;
  is_categorical: boolean;
}

export interface ShapGlobalResponse {
  data_mode: string;
  synthetic_data_warning: string | null;
  data_notice?: string | null;
  model: "flood" | "landslide";
  model_label: string;
  n_features: number;
  categorical_features: string[];
  shap_space: string;
  importance: ShapImportanceRow[];
  interpretation_note: string;
  source: string;
}

/** One feature's contribution to a single prediction. */
export interface ShapContribution {
  feature: string;
  value: number | string | null;
  shap_value: number;
  abs_shap_value: number;
  direction: string;
  effect: "increases risk" | "decreases risk" | "neutral";
  magnitude: number;
  is_categorical: boolean;
}

export interface ShapExplanationResponse {
  data_mode?: string;
  synthetic_data_warning?: string | null;
  model: "flood" | "landslide";
  model_label: string;
  target: string | null;
  probability: number;
  prediction: number;
  decision_threshold: number;
  raw_margin: number;
  base_value: number;
  base_value_probability: number;
  sum_shap_values: number;
  shap_space: string;
  additivity_check: {
    reconstructed_margin: number;
    model_margin: number;
    max_abs_difference: number;
    passed: boolean;
  };
  n_features: number;
  contributions: ShapContribution[];
  interpretation_note: string;
  sample?: {
    location_id?: string;
    mode?: "current" | "peak";
    date: string;
    latitude: number;
    longitude: number;
    district?: string | null;
    /** train | validation | test -- whether the model saw this day in training. */
    split?: string | null;
    actual_label: number | null;
  };
  source?: string;
}
