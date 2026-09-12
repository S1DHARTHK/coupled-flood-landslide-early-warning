# Current Project Status — 12 September 2026

## Completed

| Item | Status | Evidence |
|---|---|---|
| Real Master Dataset | 66,080 rows (14 districts × 4,720 days, 2012-01-30 → 2024-12-31), 23 columns; 87 build checks, 0 failed | `updated_data/master_dataset/`, `documents/master_dataset/` |
| Flood-data verification | All 288 Kerala IFI records (2012–2023) checked against IMD DWE: 41 corrected, 0 unresolved; the 18 flagged records re-verified by content, 18/18 agree; extended to 2024 with 25 IMD events | `documents/data_collection/DATA_VERIFICATION_REPORT.docx`, `updated_data/DATA_SOURCE_AND_CORRECTION_REPORT.docx` |
| Real Flood XGBoost | Trained (V1.0 settings unchanged). Test ROC-AUC 0.866, PR-AUC 0.075 against a 1.24 % base rate | `models/real/` |
| Real Landslide XGBoost | Trained. Test ROC-AUC 0.954, but only 8 test positives on 4 dates, so this is **not a reliable estimate** | `models/real/` |
| Real-model validation | 35/35 checks passed: chronological split, no leakage, `district` not used as a feature, NaN handled natively, synthetic artifacts unchanged | `models/real/REAL_MODEL_VALIDATION.csv`, `documents/models/REAL_XGBOOST_TRAINING_REPORT.docx` |
| Synthetic-data pipeline | Complete and reproducible: dataset, both models, TCDL, SHAP, backend, frontend. `--dataset synthetic` rebuilds the committed models byte for byte | `synthetic_master_dataset.csv`, `ml/` |

Split (both models): train 2012-01-30 → 2021-02-14 · validation → 2023-01-23 · test → 2024-12-31.

## Current Components

| Component | Status |
|---|---|
| Flood XGBoost | **Completed** on real data; not yet connected to TCDL or the backend |
| Landslide XGBoost | **Completed** on real data; evaluation limited by very few positives |
| TCDL | **Partially integrated**: implemented and evaluated only with the synthetic models and data (`ml/tcdl_v1.py`, `ml/tcdl_results.json`); a real-data run is pending |
| Backend | **Partially integrated**: works, but serves the synthetic models, TCDL and SHAP outputs from `ml/` (`backend/config.py`) |
| Frontend | **Partially integrated**: the dashboard reads everything from the backend, so it currently shows synthetic-pipeline outputs |

## Current Data Limitations

- **River level:** there's no usable CWC gauge for Alappuzha, Wayanad or Kottayam, and 7 more districts have data only from June 2015. `river_level_m` is missing for 38.9 % of district-days (45.6 % in train). One point gauge stands in for a whole district.
- **Landslides:** only exact-dated records are used, giving 114 positive district-days (95 train / 11 validation / 8 test). The 8 test positives fall on 4 dates, 6 of them in one storm. The data is presence-only (0 means no dated record). The GSI Bhukosh inventory couldn't be obtained and needs a manual download.
- **Lithology:** the district-dominant class takes only 2 values (granite_gneiss ×13, alluvium ×1). The source is the CGWB aquifer map; GSI 1:50k geology was unavailable.
- **Land cover:** "forest" in all 14 districts, so the feature is constant and gets zero importance.
- **Spatial unit:** all static features have one value per district, so they effectively encode district identity.
- **Flood labels:** 1,416 training positives come only from IMD multi-week summary periods (2013, 2015, 2018). The positive rate falls from 5.97 % (train) to 1.24 % (test). A 0 means "not reported by IMD", not proven absence.
- **Soil moisture:** NASA POWER saturation fraction (coarse grid), not volumetric water content.
- **Probabilities:** not calibrated (inflated by `scale_pos_weight`), and the 0.50 threshold is untuned.

## Next Steps (before final integration / research evaluation)

1. Calibrate both models' probabilities on the validation split and choose warning thresholds.
2. Decide how to treat the LONG_PERIOD_SUMMARY flood days (the keys file marks them for a comparison run).
3. Obtain the GSI Bhukosh landslide inventory, then rebuild with `updated_data/scripts/13_build_master_dataset.py` and retrain.
4. Run TCDL on the real models. Check how it derives event onsets, since the 2018 summary periods would become single multi-week events. Evaluate lead time against dated historical events.
5. Point TCDL, SHAP and the backend at `models/real/` and the real dataset. This needs path/config changes in `ml/tcdl_v1.py`, `ml/shap_explainability.py` and `backend/config.py`.
6. Commit the current work: the two changed `ml/*.py` files, `models/real/` and the new scripts aren't committed yet.

## Notes on the Folder Layout

- `synthetic_master_dataset.csv`, `DATA_DICTIONARY.md`, `ml/` (synthetic models) and `updated_data/` stay where they are because the code reads them from those paths. `updated_data/DATA_SOURCE_AND_CORRECTION_REPORT.docx` stays because `updated_data/scripts/08_quality_checks.py` reads it.
- Re-running scripts 13 and 14 writes `REAL_DATA_DICTIONARY.md`, `MASTER_DATASET_REPORT.docx` and `REAL_XGBOOST_TRAINING_REPORT.docx` back into `updated_data/master_dataset/`. Copy them into `documents/` after a rebuild.
