"""
TCDL V1.0 supplement: thresholds, rule-trigger and warning statistics, baseline
comparison (incl. minimum lead time), coupling analysis and reproducibility metadata.

Read-only over the TCDL V1.0 outputs written by scripts/models/tcdl_v1.py (artifacts/tcdl/tcdl_*).
It recomputes nothing that affects a warning: no threshold, window, rule or event is
changed, and the test period is only summarised.

Run from the project root, after `python scripts/models/tcdl_v1.py`:
    python scripts/analysis/16_tcdl_v1_supplement.py
"""

from __future__ import annotations

import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "artifacts" / "tcdl"                  # TCDL outputs (read + written here)
RM = ROOT / "artifacts" / "models"               # model files + model metadata (read only)
EXPECTED_SHA = "832145448d3b96330f70a229b1fc94db16f3fdc8767ecc6aae4903c900f46c94"
RULES = ["R1_FLOOD_LEVEL", "R2_LANDSLIDE_LEVEL", "R3_SUSTAINED_JOINT", "R4_RISING_FLOOD",
         "R5_RISING_LANDSLIDE", "R6_ENV_PRECURSOR", "R7_JOINT_MODERATE"]
COUPLING = RULES[2:]
TEMPORAL_PRECURSOR = ["R4_RISING_FLOOD", "R5_RISING_LANDSLIDE", "R6_ENV_PRECURSOR"]
SYSTEM_BASE = ["flood_only", "landslide_only", "tcdl_coupled", "tcdl_coupling_rules_only",
               "always_warn_control"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    if sha(ROOT / "dataset" / "real_master_dataset.csv") != EXPECTED_SHA:
        print("dataset hash mismatch")
        return 1
    res = json.loads((R / "tcdl_results.json").read_text(encoding="utf-8"))
    ts = pd.read_csv(R / "tcdl_timeseries.csv", low_memory=False, parse_dates=["date"])
    lt = pd.read_csv(R / "tcdl_lead_time.csv")
    for c in RULES:
        ts[c] = ts[c].astype(str).str.lower().eq("true")

    # ---- thresholds -------------------------------------------------------------
    q = res["parameters"]["threshold_quantiles"]
    th = res["parameters"]["resolved_thresholds"]
    thr = {name: {"signal": q[name]["signal"], "training_quantile": q[name]["quantile"],
                  "value": th[name]} for name in th}
    (R / "tcdl_thresholds.json").write_text(json.dumps({
        "computed_on": "TRAINING split only (2012-01-30 .. 2021-02-14), V1.0 model probabilities",
        "baseline_threshold": res["parameters"]["baseline_threshold"],
        "thresholds": thr}, indent=2), encoding="utf-8")

    # ---- rule triggers per split -------------------------------------------------
    rows = []
    for sp in ("train", "validation", "test"):
        d = ts[ts["split"] == sp]
        for r in RULES:
            fired = d[r]
            only = fired & (d[RULES].sum(axis=1) == 1)
            rows.append(dict(split=sp, rule=r, days_fired=int(fired.sum()),
                             pct_of_district_days=round(100 * fired.mean(), 2),
                             days_fired_as_only_rule=int(only.sum())))
    pd.DataFrame(rows).to_csv(R / "tcdl_rule_trigger_summary.csv", index=False)

    # ---- warning statistics per split -------------------------------------------
    ws = []
    for sp in ("train", "validation", "test"):
        d = ts[ts["split"] == sp]
        n = len(d)
        for name, col in (("tcdl_coupled", "tcdl_warning"),
                          ("tcdl_coupling_rules_only", "tcdl_coupling_only_warning"),
                          ("flood_only", "flood_only_warning"),
                          ("landslide_only", "landslide_only_warning")):
            k = int(d[col].sum())
            ws.append(dict(split=sp, system=name, district_days=n, warning_days=k,
                           pct_in_warning=round(100 * k / n, 2)))
    pd.DataFrame(ws).to_csv(R / "tcdl_warning_statistics.csv", index=False)

    # ---- baseline comparison (test), incl. min lead -------------------------------
    comp = []
    for scope, suffix in (("any_hazard", ""), ("flood", "__vs_flood"),
                          ("landslide", "__vs_landslide")):
        for s in SYSTEM_BASE:
            sub = lt[lt["system"] == s + suffix]
            det = sub[sub["detected"] == 1]
            summ = res["results"][scope][s]
            comp.append(dict(
                scope=scope, system=s, n_events=len(sub), n_detected=len(det),
                detection_rate=summ["detection_rate"],
                lead_earliest_days_mean=round(det["lead_time_days_earliest"].mean(), 2) if len(det) else None,
                lead_earliest_days_median=det["lead_time_days_earliest"].median() if len(det) else None,
                lead_earliest_days_min=det["lead_time_days_earliest"].min() if len(det) else None,
                lead_earliest_days_max=det["lead_time_days_earliest"].max() if len(det) else None,
                lead_contiguous_days_mean=round(det["lead_time_days_contiguous"].mean(), 2) if len(det) else None,
                lead_contiguous_days_median=det["lead_time_days_contiguous"].median() if len(det) else None,
                lead_contiguous_days_min=det["lead_time_days_contiguous"].min() if len(det) else None,
                lead_contiguous_days_max=det["lead_time_days_contiguous"].max() if len(det) else None,
                n_zero_lead_nowcast=summ["n_zero_lead_nowcast"],
                n_detected_with_lead_ge_1_day=int((det["lead_time_days_earliest"] >= 1).sum()),
                warning_days=summ["n_warning_days"],
                time_in_warning_rate=summ["time_in_warning_rate"],
                false_alarm_days_3d=summ["n_false_alarm_days"],
                false_alarm_rate_3d=summ["false_alarm_day_rate"],
                useful_warning_days_3d=summ["n_warning_days"] - summ["n_false_alarm_days"],
                false_alarm_rate_14d=summ["false_alarm_day_rate_lookback_window"]))
    comp = pd.DataFrame(comp)
    comp.to_csv(R / "tcdl_baseline_comparison.csv", index=False)

    # ---- coupling analysis (test) ---------------------------------------------------
    t = ts[ts["split"] == "test"].copy()
    tw = t[t["tcdl_warning"] == 1]
    no_base = tw[~tw["R1_FLOOD_LEVEL"] & ~tw["R2_LANDSLIDE_LEVEL"]]
    prec = tw[tw[TEMPORAL_PRECURSOR].any(axis=1)]
    prec_only = no_base[no_base[TEMPORAL_PRECURSOR].any(axis=1)]

    # per-event: does TCDL (or coupling rules) extend contiguous lead beyond the
    # independent baselines? any_hazard scope.
    def per_event(system):
        s = lt[lt["system"] == system].copy()
        s["key"] = s["location_id"] + "|" + s["event_timestamp"]
        return s.set_index("key")
    fo, lo, tc, cr = (per_event(x) for x in ("flood_only", "landslide_only",
                                            "tcdl_coupled", "tcdl_coupling_rules_only"))
    base_contig = pd.concat([fo["lead_time_days_contiguous"].fillna(-1),
                             lo["lead_time_days_contiguous"].fillna(-1)], axis=1).max(axis=1)
    base_det = (fo["detected"] | lo["detected"]).astype(int)
    tc_contig = tc["lead_time_days_contiguous"].fillna(-1)
    cr_contig = cr["lead_time_days_contiguous"].fillna(-1)
    ev = pd.DataFrame({"baseline_union_detected": base_det,
                       "baseline_union_contig": base_contig,
                       "tcdl_detected": tc["detected"], "tcdl_contig": tc_contig,
                       "coupling_only_detected": cr["detected"],
                       "coupling_only_contig": cr_contig})
    coupling = {
        "scope": "test period (2023-01-24 .. 2024-12-31), 9,912 district-days",
        "tcdl_warning_days": int(len(tw)),
        "tcdl_warning_days_without_R1_or_R2": int(len(no_base)),
        "share_of_tcdl_warnings_without_R1_or_R2": round(len(no_base) / len(tw), 4) if len(tw) else None,
        "rules_on_days_without_R1_or_R2": {r: int(no_base[r].sum()) for r in COUPLING},
        "tcdl_warning_days_with_a_temporal_precursor_rule_R4_R5_R6": int(len(prec)),
        "temporal_precursor_days_without_R1_or_R2": int(len(prec_only)),
        "rule_firing_days_test": {r: int(t[r].sum()) for r in RULES},
        "events_any_hazard": int(len(ev)),
        "events_detected_by_baseline_union_R1_or_R2": int(ev["baseline_union_detected"].sum()),
        "events_detected_by_tcdl": int(ev["tcdl_detected"].sum()),
        "events_detected_only_via_coupling_rules": int(((ev["tcdl_detected"] == 1)
                                                       & (ev["baseline_union_detected"] == 0)).sum()),
        "events_where_tcdl_contiguous_lead_exceeds_best_baseline": int(
            (ev["tcdl_contig"] > ev["baseline_union_contig"]).sum()),
        "events_where_coupling_only_contiguous_lead_exceeds_best_baseline": int(
            (ev["coupling_only_contig"] > ev["baseline_union_contig"]).sum()),
        "mean_extra_contiguous_days_tcdl_over_best_baseline_on_those_events": (
            round(float((ev["tcdl_contig"] - ev["baseline_union_contig"])
                        [ev["tcdl_contig"] > ev["baseline_union_contig"]].mean()), 2)
            if (ev["tcdl_contig"] > ev["baseline_union_contig"]).any() else 0.0),
        "note": ("tcdl_coupled contains R1 and R2, so it cannot detect fewer events or warn later "
                 "than either baseline by construction; the coupling-only rows isolate R3-R7."),
    }
    (R / "tcdl_coupling_analysis.json").write_text(json.dumps(coupling, indent=2), encoding="utf-8")

    # ---- metadata -----------------------------------------------------------------
    mmeta = json.loads((RM / "MODEL_METADATA_MasterDatasetV1.0.json").read_text(encoding="utf-8"))
    master = pd.read_csv(ROOT / "dataset" / "real_master_dataset.csv", usecols=["date", "flood", "landslide"])
    ev_counts = pd.Series(res["event_source"].get("n_events"))
    meta = {
        "generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "tcdl_version": res.get("version"),
        "layer_type": res.get("layer_type"),
        "dataset": {"name": "Master Dataset V1.0", "sha256": EXPECTED_SHA,
                    "district_days": int(len(master)),
                    "flood_positive_district_days": int(master["flood"].sum()),
                    "landslide_positive_district_days": int(master["landslide"].sum())},
        "models": {k: {"file": f"artifacts/models/{k}_xgboost_model.json",
                       "sha256": sha(RM / f"{k}_xgboost_model.json"),
                       "features": mmeta["models"][k]["features"],
                       "scale_pos_weight": mmeta["models"][k]["scale_pos_weight"],
                       "best_iteration": mmeta["models"][k]["best_iteration"]}
                   for k in ("flood", "landslide")},
        "rules": res["rules"],
        "thresholds": thr,
        "baseline_threshold": res["parameters"]["baseline_threshold"],
        "smooth_window_days": res["parameters"]["smooth_window_days"],
        "rate_window_days": res["parameters"]["rate_window_days"],
        "sustain_days": res["parameters"]["sustain_days"],
        "lead_time_lookback_days": res["parameters"]["max_lookback_days"],
        "useful_horizon_days": 3,
        "periods": {"train": ["2012-01-30", "2021-02-14"],
                    "validation": ["2021-02-15", "2023-01-23"],
                    "test": ["2023-01-24", "2024-12-31"]},
        "event_source": res["event_source"],
        "test_onsets": {h: int((lt["system"] == "flood_only"
                                if h == "any_hazard" else
                                lt["system"] == f"flood_only__vs_{h}").sum())
                        for h in ("any_hazard", "flood", "landslide")},
        "software": mmeta["software"],
    }
    (R / "TCDL_V1.0_METADATA.json").write_text(json.dumps(meta, indent=2, default=str),
                                               encoding="utf-8")
    print(comp[comp["scope"] == "any_hazard"].to_string(index=False))
    print(json.dumps(coupling, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
