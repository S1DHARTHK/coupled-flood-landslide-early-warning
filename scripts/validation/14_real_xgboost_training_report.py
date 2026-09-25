"""
14 - Final validation + training report for the REAL-data XGBoost models.

Run after training (nothing is trained here):
    python scripts/models/flood_xgboost.py        # real data -> artifacts/models + artifacts/results
    python scripts/models/landslide_xgboost.py    # real data -> artifacts/models + artifacts/results
    python scripts/validation/14_real_xgboost_training_report.py

Reads (read-only): artifacts/{models,results}/*, synthetic/results/*_model_results.json, both master datasets,
                   collected_datasets/master_dataset/VALIDATION_RESULTS.csv, scripts/models/{flood,landslide}_xgboost.py
Writes:            artifacts/results/REAL_MODEL_VALIDATION.csv          final validation checks
                   artifacts/results/real_evaluation_supplement.json    bootstrap intervals + reference rankings
                   documents/models/REAL_XGBOOST_TRAINING_REPORT.docx

The supplement quantifies how reliable the real-data metrics are; it does not change any model:
  * date-clustered bootstrap 95 % intervals for ROC-AUC / PR-AUC (all rows of a day resampled together,
    because districts on the same day share one storm), fixed seed;
  * Wilson 95 % intervals for recall / precision at the fixed 0.50 threshold;
  * reference rankings that use ONLY training labels and no model: the training positive rate of the
    row's district, of its calendar month, and of district x month. They show how much of a score
    "where and when" alone already gives.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

U = Path(__file__).resolve().parents[2] / "collected_datasets"
ROOT = U.parent
REAL_MODELS = ROOT / "artifacts" / "models"      # boosters + feature files
REAL_RESULTS = ROOT / "artifacts" / "results"    # metrics, predictions, plots, validation
SYN_MODELS = ROOT / "synthetic" / "models"
SYN_RESULTS = ROOT / "synthetic" / "results"
REPORT = ROOT / "documents" / "models" / "REAL_XGBOOST_TRAINING_REPORT.docx"


def _real(name: str) -> Path:
    """Folder of a real-data artifact: boosters/feature files vs. evaluation outputs."""
    return (REAL_MODELS if name.endswith(("_xgboost_model.json", "_features.json")) else REAL_RESULTS) / name
SPLITS = ("train", "validation", "test")
EXPECTED_BOUNDS = {"train": ("2012-01-30", "2021-02-14"), "validation": ("2021-02-15", "2023-01-23"),
                   "test": ("2023-01-24", "2024-12-31")}
N_BOOT, SEED = 2000, 20260912
SYNTHETIC_ARTIFACTS = [f"synthetic/{'models' if s in ('xgboost_model.json', 'features.json') else 'results'}/{h}_{s}"
                       for h in ("flood", "landslide") for s in
                       ("xgboost_model.json", "features.json", "model_results.json", "test_predictions.csv",
                        "feature_importance.png")] + ["synthetic/dataset/synthetic_master_dataset.csv"]


def load_ml(name: str):
    spec = importlib.util.spec_from_file_location(f"_ml_{name}", ROOT / "scripts" / "models" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    prev, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(mod)
            mod.configure("real")
    finally:
        sys.dont_write_bytecode = prev
    return mod


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def cluster_bootstrap(y, p, dates, rng):
    y, p = np.asarray(y), np.asarray(p)
    codes, uniq = pd.factorize(pd.Series(dates))
    groups = [np.flatnonzero(codes == i) for i in range(len(uniq))]
    roc, pr, skipped = [], [], 0
    for _ in range(N_BOOT):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        yy = y[idx]
        if yy.min() == yy.max():
            skipped += 1
            continue
        roc.append(roc_auc_score(yy, p[idx]))
        pr.append(average_precision_score(yy, p[idx]))
    q = lambda a: [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]
    return dict(roc_auc_95ci=q(roc), pr_auc_95ci=q(pr), n_boot_used=len(roc), n_boot_without_positives=skipped)


def reference_rankings(train, part, target):
    """Rank rows by training-label rates only (no model, no features): where / when / both."""
    out = {}
    tr = train.assign(month=train["date"].dt.month)
    pt = part.assign(month=part["date"].dt.month)
    base = tr[target].mean()
    for name, keys in (("district rate", ["district"]), ("calendar-month rate", ["month"]),
                       ("district x month rate", ["district", "month"])):
        rate = tr.groupby(keys)[target].mean().rename("r").reset_index()
        s = pt.merge(rate, on=keys, how="left")["r"].fillna(base).to_numpy()
        y = pt[target].to_numpy()
        out[name] = dict(roc_auc=round(float(roc_auc_score(y, s)), 3),
                         pr_auc=round(float(average_precision_score(y, s)), 3)) if y.min() != y.max() else None
    return out


class Checks:
    def __init__(self):
        self.rows = []

    def ok(self, check, cond, expected, observed, detail=""):
        self.rows.append(dict(check_id=f"R{len(self.rows) + 1:02d}", check=check, expected=str(expected),
                              observed=str(observed), status="PASS" if cond else "FAIL", detail=str(detail)))

    def frame(self):
        return pd.DataFrame(self.rows)


def main() -> int:
    rng = np.random.default_rng(SEED)
    fl, ls = load_ml("flood_xgboost"), load_ml("landslide_xgboost")
    real_path = fl.DATA_PATH
    df = pd.read_csv(real_path, parse_dates=["date"])
    syn = pd.read_csv(ROOT / "synthetic" / "dataset" / "synthetic_master_dataset.csv", parse_dates=["date"])
    with contextlib.redirect_stdout(io.StringIO()):
        tr, va, te = fl.chronological_split(df)
    parts = {"train": tr, "validation": va, "test": te}

    models = {
        "flood": dict(mod=fl, features=fl.FLOOD_FEATURES, target="flood", categorical=[]),
        "landslide": dict(mod=ls, features=ls.LANDSLIDE_FEATURES, target="landslide",
                          categorical=ls.CATEGORICAL_FEATURES),
    }
    C = Checks()
    supplement = {}
    for h, spec in models.items():
        res = json.loads((REAL_RESULTS / f"{h}_model_results.json").read_text(encoding="utf-8"))
        feat = json.loads((REAL_MODELS / f"{h}_features.json").read_text(encoding="utf-8"))
        syn_res = json.loads((SYN_RESULTS / f"{h}_model_results.json").read_text(encoding="utf-8"))
        booster = xgb.Booster()
        booster.load_model(str(REAL_MODELS / f"{h}_xgboost_model.json"))
        spec.update(res=res, feat=feat, syn_res=syn_res, booster=booster)
        F, T = spec["features"], spec["target"]

        # ---- trained, saved separately, reproducible ------------------------------------------
        C.ok(f"{h}: model trained and saved in models/real/", (REAL_MODELS / f"{h}_xgboost_model.json").exists(),
             "file exists", f"{len(booster.get_dump())} trees")
        C.ok(f"{h}: real artifacts are separate from the synthetic ones",
             (REAL_MODELS / f"{h}_xgboost_model.json").resolve() != (SYN_MODELS / f"{h}_xgboost_model.json").resolve(),
             "different paths", str((REAL_MODELS / f"{h}_xgboost_model.json").relative_to(ROOT)))
        C.ok(f"{h}: results file records the real dataset", res["dataset"]["path"] == real_path.name,
             real_path.name, res["dataset"]["path"])

        # ---- features ---------------------------------------------------------------------------
        C.ok(f"{h}: booster features = features.json feature_order = code list",
             list(booster.feature_names) == feat["feature_order"] == F, "|".join(F), "|".join(booster.feature_names))
        C.ok(f"{h}: every feature is a column of the real Master Dataset", set(F) <= set(df.columns), "all present",
             "all present" if set(F) <= set(df.columns) else sorted(set(F) - set(df.columns)))
        banned = {"district", "date", "latitude", "longitude", "flood", "landslide"}
        C.ok(f"{h}: no identifier, date, coordinate or target in the features", not (banned & set(F)),
             "none", sorted(banned & set(F)) or "none")

        # ---- split -------------------------------------------------------------------------------
        obs = {s: (res["split"][f"{s if s != 'validation' else 'validation'}_date_range"]) for s in SPLITS}
        C.ok(f"{h}: chronological split boundaries", all(tuple(obs[s]) == EXPECTED_BOUNDS[s] for s in SPLITS),
             " | ".join(f"{s} {a}..{b}" for s, (a, b) in EXPECTED_BOUNDS.items()),
             " | ".join(f"{s} {a}..{b}" for s, (a, b) in obs.items()))
        C.ok(f"{h}: no date in more than one split, train < validation < test",
             tr["date"].max() < va["date"].min() and va["date"].max() < te["date"].min(), "disjoint, ordered",
             "disjoint, ordered")

        # ---- leakage -----------------------------------------------------------------------------
        spw = (len(tr) - tr[T].sum()) / tr[T].sum()
        C.ok(f"{h}: scale_pos_weight computed from training labels only",
             abs(res["class_imbalance_handling"]["scale_pos_weight"] - spw) < 1e-9, f"{spw:.4f}",
             f"{res['class_imbalance_handling']['scale_pos_weight']:.4f}")
        same = [f for f in F if pd.api.types.is_numeric_dtype(df[f]) and df[f].fillna(-1e9).equals(df[T].astype(float))]
        C.ok(f"{h}: no feature column equals the target", not same, "none", same or "none")

        # ---- predictions: reload and re-score every split ------------------------------------------
        scored = {}
        for s, part in parts.items():
            X = part[F]
            if spec["categorical"]:
                X = ls.prepare_categoricals(X.assign(**{T: 0}))[F]
            p = booster.predict(xgb.DMatrix(X, enable_categorical=bool(spec["categorical"])))
            scored[s] = p
            ref_auc = res["metrics"][s]["roc_auc"]
            C.ok(f"{h}: reloaded model reproduces the reported {s} ROC-AUC",
                 abs(roc_auc_score(part[T], p) - ref_auc) < 1e-9, f"{ref_auc:.6f}", f"{roc_auc_score(part[T], p):.6f}")
        saved = pd.read_csv(REAL_RESULTS / f"{h}_test_predictions.csv")
        C.ok(f"{h}: saved test predictions carry district (mapping) and match the model",
             "district" in saved.columns and len(saved) == len(te) and
             np.allclose(np.sort(saved[f"{T}_probability"].to_numpy()), np.sort(np.round(scored["test"], 6)), atol=1e-6),
             f"{len(te)} rows", f"{len(saved)} rows, columns {'|'.join(saved.columns)}")

        # ---- missing values ------------------------------------------------------------------------
        if "river_level_m" in F:
            mb = res["preprocessing"].get("missing_values_by_split", {}).get("river_level_m", {})
            actual = {s: int(parts[s]["river_level_m"].isna().sum()) for s in SPLITS}
            C.ok(f"{h}: river_level_m NaN passed to XGBoost unchanged (recorded per split = dataset)",
                 all(mb.get(s, {}).get("n_missing") == actual[s] for s in SPLITS),
                 " | ".join(f"{s} {actual[s]}" for s in SPLITS),
                 " | ".join(f"{s} {mb.get(s, {}).get('n_missing')}" for s in SPLITS), "no imputation, no rows dropped")
            tdf = booster.trees_to_dataframe()
            rl = tdf[tdf["Feature"] == "river_level_m"]
            nan_rows = parts["test"][parts["test"]["river_level_m"].isna()]
            C.ok(f"{h}: trees route missing river_level_m through learned default branches",
                 len(rl) > 0 and rl["Missing"].notna().all(), "default branch on every river_level_m split",
                 f"{len(rl)} splits on river_level_m; {len(nan_rows)} test rows with NaN scored")
        else:
            C.ok(f"{h}: feature set has no missing value", not df[F].isna().any().any(), 0, int(df[F].isna().sum().sum()))

        # ---- reliability supplement -----------------------------------------------------------------
        sup = {}
        for s, part in parts.items():
            y = part[T].to_numpy()
            m = res["metrics"][s]
            cm = m["confusion_matrix"]
            sup[s] = dict(
                n_positive=int(y.sum()), n_positive_dates=int(part.loc[part[T] == 1, "date"].nunique()),
                base_rate=round(float(y.mean()), 5),
                recall_95ci=wilson(cm["true_positives"], cm["true_positives"] + cm["false_negatives"]),
                precision_95ci=wilson(cm["true_positives"], cm["true_positives"] + cm["false_positives"]),
                **(cluster_bootstrap(y, scored[s], part["date"], rng) if s != "train" else {}),
                reference_rankings=reference_rankings(tr, part, T) if s != "train" else None)
        supplement[h] = sup

    # ---- district not a feature anywhere in the ML code --------------------------------------------
    hits = [f"{n}:{l}" for n, mod in (("flood", fl), ("landslide", ls))
            for l in ("FLOOD_FEATURES", "LANDSLIDE_FEATURES", "COMMON_FEATURES") if "district" in getattr(mod, l, [])]
    C.ok("district is in no feature list of either script", not hits, "none", hits or "none")
    C.ok("district is declared an identifier (forbidden as a feature) in both scripts",
         "district" in fl.FORBIDDEN_FEATURES and "district" in ls.FORBIDDEN_FEATURES, "yes", "yes")

    # ---- synthetic artifacts untouched -------------------------------------------------------------
    st = subprocess.run(["git", "status", "--porcelain", "--"] + SYNTHETIC_ARTIFACTS, cwd=ROOT,
                        capture_output=True, text=True)
    changed = [l for l in st.stdout.splitlines() if l.strip()]
    C.ok("synthetic dataset and synthetic model artifacts identical to git HEAD", st.returncode == 0 and not changed,
         "no change", changed or "no change", "; ".join(SYNTHETIC_ARTIFACTS))
    C.ok("the backend still points at the synthetic artifacts (not integrated yet)",
         "synthetic_master_dataset.csv" in (ROOT / "backend" / "config.py").read_text(encoding="utf-8"), "yes", "yes")

    vf = C.frame()
    vf.to_csv(REAL_RESULTS / "REAL_MODEL_VALIDATION.csv", index=False)
    (REAL_RESULTS / "real_evaluation_supplement.json").write_text(json.dumps(
        dict(_note="Reliability supplement for the real-data models; no model was changed. Bootstrap: "
                   f"{N_BOOT} date-clustered resamples, seed {SEED}. Reference rankings use training labels only.",
             **supplement), indent=2), encoding="utf-8")
    build_report(df, syn, models, supplement, vf, parts)
    n_fail = int((vf["status"] == "FAIL").sum())
    print(f"{len(vf)} checks: {len(vf) - n_fail} PASS, {n_fail} FAIL")
    if n_fail:
        print(vf[vf["status"] == "FAIL"].to_string(index=False))
    print(f"Report: {REPORT}")
    return 1 if n_fail else 0


# ---------------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------------
COLUMN_NOTES = {
    "date": "Split key only. Real: 4,720 days; synthetic: 1,096.",
    "latitude": "Metadata. Real: 14 district points; synthetic: 9 cells.",
    "longitude": "Metadata (as latitude).",
    "district": "Real only. Identifier / mapping column; declared forbidden as a feature; ignored by the model.",
    "soil_moisture": "Same name, different quantity: real = NASA POWER GWETROOT saturation fraction (2 decimals, "
                     "77 distinct values); synthetic = volumetric content.",
    "river_level_m": "Real: stage relative to the gauge's training median (can be negative) and 38.9 % NaN "
                     "(no gauge / before 2015 / no reading) -> passed to XGBoost as missing.",
    "elevation_m": "Static per location: 14 distinct values = one per district, so it identifies the district.",
    "distance_to_river_km": "Static per district (14 values).",
    "drainage_density": "Static per district (14 values); real range much narrower.",
    "slope_degree": "Static per district; district means, so the range is narrow (1.2–14.6°).",
    "aspect_degree": "Static per district (13 distinct).",
    "curvature": "Static per district; district means are ~0 (|c| < 0.002).",
    "land_cover": "Real: 'forest' in all 14 districts -> constant; the model cannot use it (gain 0).",
    "lithology": "Real: 2 of 6 classes (granite_gneiss x13, alluvium x1).",
    "flood": "Real: 4.73 % positive, falling from 5.97 % (train) to 1.24 % (test). Synthetic: 7.98 %.",
    "landslide": "Real: 0.17 % positive (114 district-days). Synthetic: 7.34 %.",
}


def build_report(df, syn, models, supplement, vf, parts):
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor

    NAVY, GREY = RGBColor(0x1F, 0x38, 0x64), RGBColor(0x59, 0x59, 0x59)
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, side, Cm(1.9))
    st = doc.styles["Normal"]
    st.font.name, st.font.size = "Calibri", Pt(10)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    st.paragraph_format.space_after = Pt(4)
    for lvl in (1, 2):
        hs = doc.styles[f"Heading {lvl}"]
        hs.font.name, hs.font.color.rgb, hs.font.size = "Calibri", NAVY, Pt(14 if lvl == 1 else 11.5)
        hs.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")

    def shade(cell, fill):
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear"), shd.set(qn("w:color"), "auto"), shd.set(qn("w:fill"), fill)
        cell._tc.get_or_add_tcPr().append(shd)

    def table(header, rows, widths, size=8.5, status_col=None, bold_first=False):
        t = doc.add_table(rows=1, cols=len(header))
        t.style, t.alignment, t.autofit = "Table Grid", WD_TABLE_ALIGNMENT.CENTER, False
        for i, h in enumerate(header):
            c = t.rows[0].cells[i]
            r = c.paragraphs[0].add_run(str(h))
            r.bold, r.font.size, r.font.color.rgb = True, Pt(size), RGBColor(0xFF, 0xFF, 0xFF)
            shade(c, "1F3864")
        for row in rows:
            cells = t.add_row().cells
            for i, v in enumerate(row):
                r = cells[i].paragraphs[0].add_run(str(v))
                r.font.size = Pt(size)
                r.bold = bold_first and i == 0
                if status_col == i:
                    shade(cells[i], {"PASS": "D9EAD3", "FAIL": "F4CCCC"}.get(str(v), "FFFFFF"))
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
                row.cells[i].paragraphs[0].paragraph_format.space_after = Pt(0)
        doc.add_paragraph()

    def para(text, lead=None, size=None, color=None, italic=False):
        p = doc.add_paragraph()
        if lead:
            p.add_run(lead).bold = True
        r = p.add_run(text)
        r.italic = italic
        if size:
            r.font.size = Pt(size)
        if color:
            r.font.color.rgb = color
        return p

    def bullet(text, lead=None):
        p = doc.add_paragraph(style="List Bullet")
        if lead:
            p.add_run(lead).bold = True
        p.add_run(text)
        p.paragraph_format.space_after = Pt(2)

    f3 = lambda v: "–" if v is None else f"{v:.3f}"
    ci = lambda c: "–" if not c or c[0] is None else f"{c[0]:.2f}–{c[1]:.2f}"
    fr, lr = models["flood"]["res"], models["landslide"]["res"]
    fs, lsu = supplement["flood"], supplement["landslide"]
    n_fail = int((vf["status"] == "FAIL").sum())
    # numbers quoted in the narrative, computed so the text stays true after a rebuild
    keys = pd.read_csv(U / "master_dataset" / "real_master_dataset_keys.csv", low_memory=False,
                       usecols=["date", "split", "flood_long_period_only"])
    n_lp_train = int(((keys["split"] == "train") & (keys["flood_long_period_only"] == 1)).sum())
    ft = fr["metrics"]["test"]
    flagged_pct = 100 * ft["n_predicted_positives"] / ft["n_rows"]
    lpos = parts["test"][parts["test"]["landslide"] == 1]
    lpos_txt = "; ".join(f"{d.date()} {', '.join(sorted(g['district']))}" for d, g in lpos.groupby("date"))
    biggest = lpos.groupby("date").size()
    two_day = max((int(biggest.iloc[i:i + 2].sum()) if (len(biggest) > i + 1 and
                   (biggest.index[i + 1] - biggest.index[i]).days == 1) else int(biggest.iloc[i]))
                  for i in range(len(biggest))) if len(biggest) else 0

    t = doc.add_paragraph()
    r = t.add_run("REAL XGBOOST TRAINING REPORT")
    r.bold, r.font.size, r.font.color.rgb = True, Pt(20), NAVY
    para("Flood and Landslide XGBoost V1.0 retrained on the real Kerala district-day Master Dataset",
         size=11, color=GREY)
    para(f"{_date.today().strftime('%d %B %Y')} · models/real/ · generated by "
         "updated_data/scripts/14_real_xgboost_training_report.py", size=9, color=GREY)

    table(["Item", "Result"], [
        ["What changed", "ml/flood_xgboost.py and ml/landslide_xgboost.py gained a --dataset {real, synthetic} switch "
                         "(default real) and --output-dir. Real artifacts go to models/real/. Model, features, split, "
                         "parameters and metrics are unchanged. District is declared an identifier and forbidden as "
                         "a feature; a post-training assertion checks the booster's feature list."],
        ["Synthetic V1.0", "Still reproducible: --dataset synthetic regenerates all 10 committed ml/ artifacts byte for "
                           "byte (verified into a scratch folder). The committed synthetic models, the synthetic dataset, "
                           "backend, frontend, TCDL and SHAP are untouched."],
        ["Flood (test)", f"ROC-AUC {f3(fr['metrics']['test']['roc_auc'])} (95 % CI {ci(fs['test']['roc_auc_95ci'])}), "
                         f"PR-AUC {f3(fr['metrics']['test']['pr_auc'])} vs base rate {fs['test']['base_rate']:.4f}; "
                         f"recall {fr['metrics']['test']['recall']:.2f}, precision {fr['metrics']['test']['precision']:.3f} "
                         f"at 0.50 ({fs['test']['n_positive']} positives)"],
        ["Landslide (test)", f"ROC-AUC {f3(lr['metrics']['test']['roc_auc'])} (95 % CI {ci(lsu['test']['roc_auc_95ci'])}), "
                             f"PR-AUC {f3(lr['metrics']['test']['pr_auc'])} vs base rate {lsu['test']['base_rate']:.4f}; "
                             f"{lr['metrics']['test']['confusion_matrix']['true_positives']} of "
                             f"{lsu['test']['n_positive']} positives found — only {lsu['test']['n_positive']} test "
                             f"positives on {lsu['test']['n_positive_dates']} dates: NOT a reliable test estimate"],
        ["Final validation", f"{len(vf)} checks, {len(vf) - n_fail} PASS, {n_fail} FAIL (models/real/REAL_MODEL_VALIDATION.csv)"],
    ], [3.2, 14.0], size=9, bold_first=True)

    # 1 dataset
    doc.add_heading("1. Dataset used", level=1)
    bv = pd.read_csv(U / "master_dataset" / "VALIDATION_RESULTS.csv")
    para(f"updated_data/master_dataset/real_master_dataset.csv — {len(df):,} rows × {df.shape[1]} columns: 14 Kerala "
         f"districts × {df['date'].nunique():,} days ({df['date'].min().date()} → {df['date'].max().date()}); built by "
         f"updated_data/scripts/13_build_master_dataset.py ({len(bv)} checks, {int((bv['status'] == 'FAIL').sum())} FAIL). "
         "Documentation: REAL_DATA_DICTIONARY.md, MASTER_DATASET_REPORT.docx.")
    doc.add_heading("Synthetic vs real Master Dataset (column by column)", level=2)
    rows = []
    for c in dict.fromkeys(list(syn.columns) + list(df.columns)):
        a = syn[c] if c in syn else None
        b = df[c] if c in df else None
        dt = lambda x: "absent" if x is None else ("date" if c == "date" else str(x.dtype).replace("64", "").replace("object", "str"))
        rows.append([c, dt(a), dt(b), "–" if a is None else int(a.isna().sum()), "–" if b is None else f"{int(b.isna().sum()):,}",
                     "–" if a is None else a.nunique(), "–" if b is None else f"{b.nunique():,}", COLUMN_NOTES.get(c, "")])
    table(["Column", "Syn type", "Real type", "Syn NaN", "Real NaN", "Syn uniq", "Real uniq", "Difference that matters"],
          rows, [2.9, 1.3, 1.3, 1.1, 1.4, 1.2, 1.3, 6.7], size=7)
    para("Columns in both: all 22 synthetic columns (same names, same order). Only in real: district (after longitude). "
         "Only in synthetic: none. Dates are ISO strings in both and parsed by read_csv(parse_dates=['date']). "
         "Consequences for the pipeline: (1) the extra district column is ignored because features are selected by "
         "explicit lists; (2) river_level_m has NaN — handled natively by XGBoost, no code change needed; (3) "
         "land_cover/lithology values stay inside the fixed CATEGORY_SCHEMA; (4) targets are int 0/1 without NaN; "
         "(5) the split, computed on unique dates, lands exactly on the agreed boundaries. None required a change "
         "to the features or the model.", size=9)

    # 2 features/targets
    doc.add_heading("2. Features and targets", level=1)
    table(["Model", "Target", "Features (in model order)", "Not features"], [
        ["Flood", "flood (0/1)", ", ".join(models["flood"]["features"]),
         "landslide, district, date, latitude, longitude, landslide-specific terrain, categoricals"],
        ["Landslide", "landslide (0/1)", ", ".join(models["landslide"]["features"]) + " (land_cover, lithology categorical)",
         "flood, river_level_m, distance_to_river_km, drainage_density, district, date, latitude, longitude"],
    ], [2.0, 2.2, 8.4, 4.6], size=8, bold_first=True)
    para("Feature definitions are the V1.0 lists, unchanged. The real data dictionary shows land_cover is constant and "
         "lithology near-constant; that makes them useless (gain 0) but not harmful, so no change was necessary. "
         "Target definitions: flood = IMD-reported damaging flood/heavy-rain district-day (0 = not reported); "
         "landslide = exact-dated GSI / IMD landslide record (presence-only).", size=9)

    # 3 preprocessing, split, missing, config
    doc.add_heading("3. Preprocessing, split, missing values, configuration", level=1)
    bullet(" none fitted. No scaling (trees are scale-invariant), no imputation, no rows dropped. land_cover/lithology bound "
           "to the fixed CATEGORY_SCHEMA (pandas CategoricalDtype, XGBoost native categorical, no one-hot).", "Preprocessing —")
    bullet(" chronological_split(): 70/15/15 on unique dates, no shuffling. " + " · ".join(
        f"{s} {EXPECTED_BOUNDS[s][0]} → {EXPECTED_BOUNDS[s][1]} ({len(parts[s]):,} rows)" for s in SPLITS)
        + ". Both scripts cut identically (landslide script cross-checks the flood results of the same dataset).",
        "Split —")
    mb = fr["preprocessing"]["missing_values_by_split"]["river_level_m"]
    bullet(" the V1.0 code already passes NaN to XGBoost (XGBClassifier missing = NaN, tree_method = hist): each split "
           "learns a default branch for missing values from the TRAINING rows, so no statistic from validation/test is "
           "used and nothing is invented. Imputation was therefore not needed. river_level_m NaN: " + " · ".join(
               f"{s} {v['n_missing']:,} ({v['pct_missing']:.1f} %)" for s, v in mb.items())
           + ". Landslide features have no NaN.", "Missing river_level_m —")
    bullet(" no feature uses future data: rainfall sums are trailing windows ending on day t; river_level_m subtracts a "
           "median computed from training dates only (build validation V078–V080); scale_pos_weight from training labels "
           "only; early stopping on validation; the test split is used only for final scoring.", "Leakage —")
    p = fr["model_parameters"]
    bullet(f" objective binary:logistic, eval_metric aucpr, n_estimators 500 with early stopping 50 on validation, "
           f"learning_rate {p['learning_rate']}, max_depth {p['max_depth']}, min_child_weight {p['min_child_weight']}, "
           f"subsample {p['subsample']}, colsample_bytree {p['colsample_bytree']}, reg_lambda {p['reg_lambda']}, "
           f"random_state 42; landslide adds enable_categorical, max_cat_to_onehot 1. scale_pos_weight (train neg/pos): "
           f"flood {fr['class_imbalance_handling']['scale_pos_weight']:.2f}, landslide "
           f"{lr['class_imbalance_handling']['scale_pos_weight']:.2f}. Decision threshold fixed at 0.50.", "XGBoost (V1.0, unchanged) —")

    # 4/5 results
    def results_section(h, title):
        res, sup = models[h]["res"], supplement[h]
        doc.add_heading(title, level=1)
        rows = []
        for s in SPLITS:
            m = res["metrics"][s]
            cm = m["confusion_matrix"]
            rows.append([s, f"{m['n_rows']:,}", f"{m['n_actual_positives']:,} ({sup[s]['n_positive_dates']} dates)",
                         f"{m['precision']:.3f} ({ci(sup[s]['precision_95ci'])})", f"{m['recall']:.3f} ({ci(sup[s]['recall_95ci'])})",
                         f"{m['f1']:.3f}", f3(m["roc_auc"]) + ("" if s == "train" else f" ({ci(sup[s]['roc_auc_95ci'])})"),
                         f3(m["pr_auc"]) + ("" if s == "train" else f" ({ci(sup[s]['pr_auc_95ci'])})"),
                         f"{sup[s]['base_rate']:.4f}", f"TN {cm['true_negatives']:,} FP {cm['false_positives']:,} "
                                                       f"FN {cm['false_negatives']} TP {cm['true_positives']}"])
        table(["Split", "Rows", "Positives", "Precision (95 % CI)", "Recall (95 % CI)", "F1", "ROC-AUC (95 % CI)",
               "PR-AUC (95 % CI)", "Base rate", "Confusion matrix @0.50"], rows,
              [1.5, 1.2, 1.9, 2.0, 2.0, 1.0, 2.2, 2.2, 1.3, 2.9], size=7.5, bold_first=True)
        best = res.get("best_iteration", res.get("early_stopping", {}).get("best_iteration"))
        para(f"Early stopping kept {best + 1} trees. Accuracy is not shown as a headline: with base rates of "
             f"{sup['test']['base_rate'] * 100:.2f} % a model predicting 'no event' everywhere would score "
             f"{(1 - sup['test']['base_rate']) * 100:.1f} % on test. PR-AUC should be read against the base rate. "
             "CIs: Wilson for precision/recall; date-clustered bootstrap for ROC/PR-AUC.", size=8.5, italic=True)
        rr = sup["test"]["reference_rankings"]
        rv = sup["validation"]["reference_rankings"]
        table(["Ranking used on validation / test", "ROC-AUC val", "PR-AUC val", "ROC-AUC test", "PR-AUC test"],
              [[f"XGBoost ({h})", f3(res["metrics"]["validation"]["roc_auc"]), f3(res["metrics"]["validation"]["pr_auc"]),
                f3(res["metrics"]["test"]["roc_auc"]), f3(res["metrics"]["test"]["pr_auc"])]]
              + [[f"Reference: training {k} (no model)", f3((rv[k] or {}).get("roc_auc")), f3((rv[k] or {}).get("pr_auc")),
                  f3((rr[k] or {}).get("roc_auc")), f3((rr[k] or {}).get("pr_auc"))] for k in rr],
              [6.2, 2.6, 2.6, 2.6, 2.6], size=8, bold_first=True)
        imp = pd.DataFrame(res["feature_importance"])
        table(["Feature", "Gain", "Share", "Splits"],
              [[r.feature, f"{r.importance_gain:.1f}", f"{r.importance_normalised * 100:.1f} %", int(r.split_count)]
               for r in imp.itertuples()], [5.0, 2.5, 2.5, 2.0], size=8)
        png = REAL_RESULTS / f"{h}_feature_importance.png"
        if png.exists():
            doc.add_picture(str(png), width=Cm(12.5))

    results_section("flood", "4. Flood XGBoost — real data")
    fm = fr["metrics"]
    para(f"Read with care: (1) the positive rate falls from {fs['train']['base_rate'] * 100:.2f} % (train) to "
         f"{fs['validation']['base_rate'] * 100:.2f} % and {fs['test']['base_rate'] * 100:.2f} %: {n_lp_train:,} training "
         "positives come only from IMD's 2013/2015/2018 multi-week summary periods (e.g. all 14 districts, 29 May–30 Aug 2018), "
         "which have no counterpart later. Validation PR-AUC peaked after a handful of trees, so early stopping kept a "
         f"very small model ({(fr.get('best_iteration') or 0) + 1} trees). (2) scale_pos_weight inflates probabilities "
         f"(mean predicted {fm['test']['mean_predicted_probability']:.2f} vs observed {fm['test']['observed_positive_rate']:.3f} "
         f"on test), so at the fixed 0.50 threshold the model flags {flagged_pct:.0f} % of test district-days: high "
         f"recall, precision {ft['precision'] * 100:.1f} %. (3) Ranking skill above the 'district × month' reference is the part attributable to the "
         "weather features; river_level_m is used, but its absence also marks three districts and the pre-2015 years.",
         size=9)
    results_section("landslide", "5. Landslide XGBoost — real data")
    lm = lr["metrics"]
    para(f"TEST RELIABILITY WARNING — the test split contains only {lsu['test']['n_positive']} landslide district-days on "
         f"{lsu['test']['n_positive_dates']} dates ({lpos_txt}); {two_day} of them fall on one storm's two consecutive "
         "days. The test metrics therefore describe only a handful of independent events: recall "
         f"{lm['test']['recall']:.2f} is "
         f"{lm['test']['confusion_matrix']['true_positives']} of {lsu['test']['n_positive']} (95 % CI "
         f"{ci(lsu['test']['recall_95ci'])}); one event more or less would move ROC-AUC and PR-AUC substantially "
         f"(bootstrap ROC-AUC CI {ci(lsu['test']['roc_auc_95ci'])}, PR-AUC CI {ci(lsu['test']['pr_auc_95ci'])}). "
         f"Validation has {lsu['validation']['n_positive']} positives and also drives early stopping. Training fit is "
         f"near-perfect (ROC-AUC {f3(lm['train']['roc_auc'])}, recall {lm['train']['recall']:.2f}) on 95 positives, "
         "dominated by the August 2018 and 2019 events — a sign of memorisation. land_cover and lithology received zero "
         "gain; the terrain features are one value per district, so their importance mostly encodes which districts "
         "(Idukki, Wayanad, …) had recorded landslides. These numbers must not be reported as landslide-prediction skill.",
         size=9)

    # 6 comparison
    doc.add_heading("6. Comparison with the synthetic-data models", level=1)
    rows = []
    notes = {
        ("flood", "synthetic"): "9 simulated cells, labels drawn from the features (7.98 % positive); no gaps.",
        ("flood", "real"): "IMD-reported events; base rate falls 5.97 → 1.24 %; river_level_m 39 % NaN; tiny early-stopped model.",
        ("landslide", "synthetic"): "Labels drawn from the features, "
                                    f"{models['landslide']['syn_res']['metrics']['test']['n_actual_positives']} test positives.",
        ("landslide", "real"): f"{lsu['test']['n_positive']} test positives on {lsu['test']['n_positive_dates']} dates; "
                               "static terrain = district identity; train fit ≈ perfect.",
    }
    for h in ("flood", "landslide"):
        for name, res in (("Synthetic", models[h]["syn_res"]), ("Real", models[h]["res"])):
            cell = lambda s: (f"ROC {f3(res['metrics'][s]['roc_auc'])} · PR {f3(res['metrics'][s]['pr_auc'])} · "
                              f"F1 {res['metrics'][s]['f1']:.3f} · pos {res['metrics'][s]['n_actual_positives']}")
            rows.append([f"{h.capitalize()} XGBoost", name, cell("train"), cell("validation"), cell("test"),
                         notes[(h, name.lower())]])
    table(["Model", "Dataset", "Train", "Validation", "Test", "Important difference"], rows,
          [2.2, 1.5, 3.3, 3.3, 3.3, 3.8], size=7.5, bold_first=True)
    para("The numbers are not comparable as a contest. The synthetic labels are generated from the same features the "
         "model sees, so high scores there only show that the pipeline can learn a planted signal. Real labels are "
         "reports of damaging events at district level: absence of a report is not absence of an event, the static "
         "features cannot separate places within a district, and the flood label changes character over time. Real "
         "PR-AUC values are low in absolute terms but must be read against base rates 6–100× lower than in the synthetic "
         "file; the real landslide test ROC-AUC is higher than the synthetic one, yet it rests on "
         f"{lsu['test']['n_positive']} positives on {lsu['test']['n_positive_dates']} dates and a strong district/season "
         "prior (calendar month alone ranks the test set at ROC-AUC "
         f"{f3(lsu['test']['reference_rankings']['calendar-month rate']['roc_auc'])}) and is the least trustworthy "
         "number in this report.", size=9)

    # 7 limitations
    doc.add_heading("7. Limitations", level=1)
    for x in [
        "District-day unit: 14 locations. Every static feature (elevation, slope, aspect, curvature, distance to river, "
        "drainage density) is a district identifier in disguise; importances of static features are not process evidence.",
        "Landslide target: 114 positives in total, 8 in test on 4 dates, 11 in validation; presence-only. Test metrics are "
        "anecdotal; validation-based early stopping is noisy.",
        f"Flood target: LONG_PERIOD_SUMMARY periods ({n_lp_train:,} training positives) mark whole seasons, inflating the training "
        "positive rate and blurring what a 'flood day' is; 0 = not reported by IMD.",
        "Probabilities are not calibrated (scale_pos_weight, as in V1.0) and the 0.50 threshold is not tuned — both "
        "matter before the TCDL consumes these outputs.",
        "river_level_m missingness is structured (3 districts, pre-June-2015 for 7 more), so 'missing' is partly a "
        "district/period marker; one gauge represents a whole district.",
        "Same-day features: rainfall, humidity, temperature and soil wetness of day t are used to score day t "
        "(as in V1.0) — a nowcast, not a forecast; lead time is not evaluated here.",
    ]:
        bullet(" " + x)

    # 8 files
    doc.add_heading("8. Model files", level=1)
    files = [[str(_real(f).relative_to(ROOT)).replace("\\", "/"), d] for f, d in [
        ("flood_xgboost_model.json", "flood booster (trees up to best iteration)"),
        ("flood_features.json", "flood feature order + metadata"),
        ("flood_model_results.json", "flood metrics, split, parameters, importance"),
        ("flood_test_predictions.csv", "flood test probabilities (with district)"),
        ("flood_feature_importance.png", "flood importance plot"),
        ("flood_training_log.txt", "flood console log"),
        ("landslide_xgboost_model.json", "landslide booster"),
        ("landslide_features.json", "landslide feature order + category schema"),
        ("landslide_model_results.json", "landslide metrics, split, parameters, importance"),
        ("landslide_test_predictions.csv", "landslide test probabilities (with district)"),
        ("landslide_feature_importance.png", "landslide importance plot"),
        ("landslide_training_log.txt", "landslide console log"),
        ("REAL_MODEL_VALIDATION.csv", "final validation checks (this report, §9)"),
        ("real_evaluation_supplement.json", "bootstrap CIs, Wilson CIs, reference rankings")]]
    files.append(["ml/*.json, ml/*.csv, ml/*.png", "synthetic V1.0 artifacts — unchanged, still used by backend/TCDL/SHAP"])
    table(["Path", "Content"], files, [8.5, 8.7], size=8)
    para("Re-train: python ml/flood_xgboost.py && python ml/landslide_xgboost.py (real, default). Synthetic V1.0: add "
         "--dataset synthetic. Then python updated_data/scripts/14_real_xgboost_training_report.py.", size=9)

    # 9 validation
    doc.add_heading("9. Final validation", level=1)
    table(["ID", "Check", "Observed", "Status"],
          [[r.check_id, r.check, (r.observed[:95] + "…") if len(r.observed) > 95 else r.observed, r.status]
           for r in vf.itertuples()], [1.0, 8.2, 6.8, 1.3], size=7.5, status_col=3)

    # 10 before TCDL
    doc.add_heading("10. Open issues before TCDL", level=1)
    for x in [
        "Calibrate probabilities (e.g. on validation) and choose thresholds deliberately; V1.0 probabilities are "
        "inflated by scale_pos_weight.",
        "Decide on LONG_PERIOD_SUMMARY flood days (sensitivity run with flood_long_period_only from the keys file).",
        "Landslide: more exact-dated events are needed (GSI Bhukosh inventory) before any landslide or coupled metric "
        "can be reported; consider event-level rather than district-day evaluation.",
        "TCDL derives event onsets from runs of consecutive label days; 2018 summary periods would become single "
        "64/30-day 'events'. Review before coupling.",
        "Point TCDL, SHAP and the backend at models/real/ only when integration is intended (their paths still target ml/).",
    ]:
        bullet(" " + x)
    doc.save(REPORT)


if __name__ == "__main__":
    sys.exit(main())
