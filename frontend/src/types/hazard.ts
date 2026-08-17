/**
 * Type contracts for the Kerala Flood-Landslide Early Warning backend.
 *
 * These mirror the FastAPI response shapes. The frontend performs NO ML and
 * NO TCDL computation -- it only renders what the backend returns.
 *
 * FUTURE REAL DATA: `district` is declared optional on location-bearing types.
 * The current synthetic dataset does not contain it, so it is never fabricated
 * or inferred from coordinates. When the backend starts returning it, the UI
 * displays it automatically with no redesign.
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
  /** Not present in the synthetic dataset. Rendered only if the backend sends it. */
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
}

export interface CurrentResponse {
  data_mode: string;
  synthetic_data_warning: string | null;
  as_of: string;
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
  warning_types: WarningType[];
  timestamp_resolution: string;
  n_records: number;
  records: WarningRecord[];
}

export interface TrendPoint extends TrendSignals {
  date: string;
}

export interface TrendsResponse {
  data_mode: string;
  location_id: string;
  n_records: number;
  parameters: Record<string, number | null>;
  source: string;
  series: TrendPoint[];
}

/** One evaluated warning system in the lead-time comparison. */
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
}

export interface LeadTimeRecord {
  system: string;
  hazard_type: string;
  location_id: string;
  latitude: number;
  longitude: number;
  event_time: string | null;
  warning_time: string | null;
  detected: number;
  lead_time_hours: number | null;
  lead_time_hours_contiguous: number | null;
}

export interface LeadTimeResponse {
  data_mode: string;
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
}

export interface LocationsResponse {
  data_mode: string;
  locations: Array<
    LocationRef & { n_records: number; first_date: string; last_date: string }
  >;
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
