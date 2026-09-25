"""
SHAP V1.0 supplement: importance CSVs, dependence plots, directional statistics and
reproducibility metadata for the V1.0 Flood / Landslide XGBoost models.

It adds NO new explanation method. Models, the test population and the SHAP values all
come from ml/shap_explainability.py (exact TreeSHAP via XGBoost pred_contribs, the
project's established method), bound to the real dataset with its own configure().
Nothing is trained, re-thresholded or written back to a model or the dataset.

Run from the project root, after `python scripts/analysis/shap_explainability.py`:
    python scripts/analysis/15_shap_v1_supplement.py
"""

from __future__ import annotations

import datetime
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset" / "real_master_dataset.csv"
EXPECTED_SHA = "832145448d3b96330f70a229b1fc94db16f3fdc8767ecc6aae4903c900f46c94"
OUT = ROOT / "artifacts" / "shap"
N_DEPENDENCE = 4                                    # dependence plots per model (top numeric)


def load_module():
    spec = importlib.util.spec_from_file_location("shap_explainability",
                                                  ROOT / "scripts" / "analysis" / "shap_explainability.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.configure("real")
    return mod


def directional(values: np.ndarray, X: pd.DataFrame, feats: list[str], cats: dict) -> list[dict]:
    """How feature value relates to SHAP sign, on the explained population."""
    rows = []
    for j, f in enumerate(feats):
        s = values[:, j]
        if f in cats:                                   # categorical: mean SHAP per category
            per = (pd.DataFrame({"cat": X[f].astype(str), "shap": s})
                   .groupby("cat")["shap"].agg(["mean", "size"]).sort_values("mean"))
            rows.append(dict(feature=f, kind="categorical", spearman_value_vs_shap=None,
                             mean_shap_low_quartile=None, mean_shap_high_quartile=None,
                             detail="; ".join(f"{c}: {m:+.3f} (n={n})"
                                              for c, (m, n) in per.iterrows())))
            continue
        v = pd.to_numeric(X[f], errors="coerce")
        ok = v.notna()
        rho = (pd.Series(v[ok]).rank().corr(pd.Series(s[ok.values]).rank())
               if ok.sum() > 2 and v[ok].nunique() > 1 else np.nan)
        if v[ok].nunique() >= 4:
            q1, q3 = v[ok].quantile([0.25, 0.75])
            lo = float(s[(v <= q1).values].mean())
            hi = float(s[(v >= q3).values].mean())
        else:
            lo = hi = np.nan
        miss = float(s[(~ok).values].mean()) if (~ok).any() else None
        rows.append(dict(feature=f, kind="numeric",
                         spearman_value_vs_shap=None if pd.isna(rho) else round(float(rho), 3),
                         mean_shap_low_quartile=None if pd.isna(lo) else round(lo, 4),
                         mean_shap_high_quartile=None if pd.isna(hi) else round(hi, 4),
                         detail=(f"distinct values in population: {int(v[ok].nunique())}"
                                 + (f"; missing rows: {int((~ok).sum())}, mean SHAP when "
                                    f"missing {miss:+.4f}" if miss is not None else ""))))
    return rows


def main() -> int:
    sha = hashlib.sha256(DATASET.read_bytes()).hexdigest()
    if sha != EXPECTED_SHA:
        print(f"dataset hash mismatch: {sha}")
        return 1
    mod = load_module()
    test = mod.load_test_split()
    meta = {"generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "dataset": {"name": "Master Dataset V1.0", "path": "real_master_dataset.csv",
                        "sha256": sha},
            "explainer": ("exact TreeSHAP via xgboost Booster.predict(pred_contribs=True) "
                          "(ml/shap_explainability.py)"),
            "shap_space": "log-odds (margin)",
            "population": {"split": "test (held-out)",
                           "date_range": [str(test["date"].min().date()),
                                          str(test["date"].max().date())],
                           "n_observations": int(len(test))},
            "software": {"python": sys.version.split()[0], "shap": shap.__version__,
                         "xgboost": xgb.__version__, "pandas": pd.__version__,
                         "numpy": np.__version__},
            "random_seed": "not applicable (exact TreeSHAP is deterministic)",
            "models": {}}

    for key in ("flood", "landslide"):
        m = mod.get_model(key)
        X = m.frame(test)
        values, base = m.shap_values(X)                 # additivity asserted inside
        feats = m.features
        cats = m.category_schema
        mean_abs = np.abs(values).mean(axis=0)
        total = mean_abs.sum() or 1.0
        imp = (pd.DataFrame({"feature": feats, "mean_abs_shap": mean_abs,
                             "share_of_total": mean_abs / total,
                             "mean_signed_shap": values.mean(axis=0),
                             "is_categorical": [f in cats for f in feats]})
               .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
        imp.insert(0, "rank", np.arange(1, len(imp) + 1))
        imp.round(6).to_csv(OUT / f"{key}_shap_importance.csv", index=False)

        d = pd.DataFrame(directional(values, X, feats, cats))
        d = d.set_index("feature").loc[imp["feature"]].reset_index()
        d.to_csv(OUT / f"{key}_shap_directional.csv", index=False)

        Xn = m.numeric_view(X)
        top_numeric = [f for f in imp["feature"] if f not in cats][:N_DEPENDENCE]
        plots = []
        for f in top_numeric:
            plt.figure()
            shap.dependence_plot(f, values, Xn, feature_names=feats, show=False,
                                 interaction_index=None)
            plt.title(f"{m.label} - SHAP dependence: {f}\nREAL DATA - Master Dataset V1.0, "
                      f"test period", fontsize=9)
            plt.tight_layout()
            p = OUT / f"{key}_shap_dependence_{f}.png"
            plt.savefig(p, dpi=150, bbox_inches="tight")
            plt.close()
            plots.append(p.name)

        booster_path = ROOT / "artifacts" / "models" / f"{key}_xgboost_model.json"
        meta["models"][key] = {
            "model_file": str(booster_path.relative_to(ROOT)).replace("\\", "/"),
            "model_sha256": hashlib.sha256(booster_path.read_bytes()).hexdigest(),
            "n_trees": int(m.booster.num_boosted_rounds()),
            "features": feats, "n_features": len(feats),
            "categorical_features": list(cats),
            "threshold": m.threshold,
            "base_value_logodds_mean": round(float(np.mean(base)), 6),
            "additivity_max_drift": float(getattr(m, "last_additivity_drift", 0.0)),
            "n_observations_explained": int(len(X)),
            "n_positive_in_population": int(test[key].sum()),
            "outputs": [f"{key}_shap_summary.png", f"{key}_shap_bar.png",
                        f"{key}_shap_importance.json", f"{key}_shap_importance.csv",
                        f"{key}_shap_directional.csv",
                        f"{key}_shap_example_explanation.json"] + plots,
        }
        print(f"{key}: {len(X)} obs explained, top = {imp['feature'].head(5).tolist()}, "
              f"dependence plots: {top_numeric}")

    (OUT / "SHAP_V1.0_METADATA.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("metadata ->", OUT / "SHAP_V1.0_METADATA.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
