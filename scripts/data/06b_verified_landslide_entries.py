"""
Write the web-verified landslide district-days into the processed events folder
so the master-dataset build can consume them like any other label source.

Source of truth: the verification report produced by the 2018-2024 verification
passes (updated_data/landslide_events/verified_2018_2024/). Only rows whose
label was changed 0 -> 1 there are written here; nothing is invented, and this
script adds no event of its own.

Output:
    landslide_events/processed/VERIFIED_web_landslide_entries.csv

Run from the project root:
    python scripts/data/06b_verified_landslide_entries.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

U = Path(__file__).resolve().parents[2] / "collected_datasets"                  # collected_datasets/
VERIFIED_DIR = U / "landslide_events" / "verified_2018_2024"
LABELS = VERIFIED_DIR / "landslide_labels_verified_2018_2024.csv"
REPORT = VERIFIED_DIR / "landslide_verification_report.csv"
OUT = U / "landslide_events" / "processed" / "VERIFIED_web_landslide_entries.csv"

# Short, stable tags for the sources behind these labels (used in source_id).
TAGS = {
    "Kerala State Disaster Management Authority (KSDMA)": "KSDMA_EVENTREPORT_OCT2021",
    "Onmanorama (Malayala Manorama)": "ONMANORAMA",
}


def main() -> int:
    for f in (LABELS, REPORT):
        if not f.exists():
            print(f"missing input: {f}")
            return 1

    labels = pd.read_csv(LABELS, low_memory=False)
    report = pd.read_csv(REPORT)
    changed = labels[labels["label_changed"] == True].copy()      # noqa: E712

    rep = {r["dataset_record"]: r for _, r in report.iterrows()
           if isinstance(r.get("dataset_record"), str) and "|" in r["dataset_record"]}

    rows = []
    for _, r in changed.iterrows():
        rec = f"{r['date']}|{r['district']}"
        ev = rep.get(rec)
        if ev is None:
            print(f"no evidence row for {rec}; refusing to emit it")
            return 1
        tag = TAGS.get(ev["source"], "VERIFIED")
        rows.append({
            "district": r["district"],                 # already the CHIRPS spelling
            "date": r["date"],
            "date_precision": "EXACT_DATE",
            "source": "VERIFIED",
            "source_id": f"VERIFIED:{tag}:{r['date']}:{r['district']}",
            "source_name": ev["source"],
            "source_title": ev["source_title"],
            "source_url": ev["source_url"],
            "source_tier": ev["source_tier"],
            "reported_event_date": ev["reported_event_date"],
            "reported_location": ev["reported_location"],
            "evidence_summary": ev["evidence_summary"],
            "verification_status": ev["status"],
        })

    out = pd.DataFrame(rows).sort_values(["date", "district"])
    if out.duplicated(["date", "district"]).any():
        print("duplicate district-day in the verified entries; refusing to write")
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {len(out)} verified landslide district-days -> "
          f"{OUT.relative_to(U.parent)}")
    print(out.groupby(out["date"].str[:4]).size().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
