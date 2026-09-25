"""
District service -- district vocabulary and real observations.

`district` is an IDENTIFIER in this system, never a model feature. This
service only:
  * resolves a district name to the canonical CHIRPS spelling used by the
    real Master Dataset (e.g. "Pathanamthitta" -> "Pattanamtitta"),
  * supplies the district's representative point (the latitude/longitude
    the real dataset uses for every row of that district),
  * returns the REAL observed feature row for a district and date, read
    unchanged from the real Master Dataset.

Nothing is estimated: a feature that is missing in the dataset (e.g.
river_level_m for a district without a CWC gauge) is returned as null.
"""

from __future__ import annotations

from datetime import date as date_type
from typing import Any

import pandas as pd

from .. import config
from .model_service import ModelService


class DistrictService:
    def __init__(self, real_models: ModelService) -> None:
        self.errors: list[str] = []
        self.ready = False
        self.models = real_models
        self.points: dict[str, dict[str, float]] = {}
        self.aliases: dict[str, str] = {}
        self.ranges: dict[str, dict[str, Any]] = {}
        self.split_ranges: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
        try:
            self._load_reference()
            self._load_ranges()
            self.ready = True
        except Exception as exc:
            self.errors.append(f"districts: {exc}")

    # -----------------------------------------------------------------
    def _load_reference(self) -> None:
        path = config.DISTRICT_REFERENCE
        if not path.exists():
            raise FileNotFoundError(f"District reference not found at {path}")
        ref = pd.read_csv(path)
        for r in ref.itertuples():
            canonical = str(r.district_chirps).strip()
            self.points[canonical] = {"latitude": float(r.latitude),
                                      "longitude": float(r.longitude)}
            for alias in (r.district_chirps, r.district_geoboundaries):
                self.aliases[str(alias).strip().lower()] = canonical

    def _load_ranges(self) -> None:
        if not self.models.dataset_ready or self.models.master is None:
            raise RuntimeError("real master dataset not loaded: " + "; ".join(self.models.errors))
        m = self.models.master
        if "district" not in m.columns:
            raise RuntimeError("real master dataset has no 'district' column")
        unknown = sorted(set(m["district"]) - set(self.points))
        if unknown:
            raise RuntimeError(f"master dataset districts not in the reference: {unknown}")
        for d, g in m.groupby("district"):
            self.ranges[d] = {
                "first_date": g["date"].min().strftime("%Y-%m-%d"),
                "last_date": g["date"].max().strftime("%Y-%m-%d"),
                "n_days": int(len(g)),
                "river_level_days": int(g["river_level_m"].notna().sum())
                if "river_level_m" in g else 0,
            }
        split = self.models.flood_results.get("split", {})
        for name in ("train", "validation", "test"):
            rng = split.get(f"{name}_date_range")
            if rng:
                self.split_ranges[name] = (pd.Timestamp(rng[0]), pd.Timestamp(rng[1]))

    # -----------------------------------------------------------------
    def resolve(self, name: Any) -> str:
        """Canonical district name, or KeyError listing the valid names."""
        key = str(name).strip().lower() if name is not None else ""
        if key not in self.aliases:
            raise KeyError(f"Unknown district {name!r}. Valid districts: {self.names()}")
        return self.aliases[key]

    def names(self) -> list[str]:
        return sorted(self.points)

    def point(self, district: str) -> dict[str, float]:
        return self.points[district]

    def split_of(self, day: pd.Timestamp) -> str | None:
        for name, (lo, hi) in self.split_ranges.items():
            if lo <= day <= hi:
                return name
        return None

    def listing(self) -> list[dict]:
        return [{"district": d, **self.points[d], **self.ranges.get(d, {})}
                for d in self.names()]

    def observation(self, district: str, day: str | None) -> dict[str, Any]:
        """The real feature row for one district-day, exactly as stored."""
        canonical = self.resolve(district)
        m = self.models.master
        assert m is not None
        rows = m[m["district"] == canonical]
        if day is None:
            target = rows["date"].max()
        else:
            try:
                target = pd.Timestamp(date_type.fromisoformat(str(day)))
            except ValueError as exc:
                raise ValueError(f"date must be YYYY-MM-DD, got {day!r}") from exc
        row = rows[rows["date"] == target]
        if row.empty:
            rng = self.ranges[canonical]
            raise LookupError(f"No real observation for {canonical} on {day}; available "
                              f"{rng['first_date']} .. {rng['last_date']}")
        r = row.iloc[0]
        features = list(dict.fromkeys(self.models.flood_features + self.models.landslide_features))

        def value(col: str) -> Any:
            v = r[col]
            if isinstance(v, str):
                return v
            return None if pd.isna(v) else float(v)

        obs = {"date": target.strftime("%Y-%m-%d"), "district": canonical,
               "latitude": float(r["latitude"]), "longitude": float(r["longitude"])}
        obs.update({f: value(f) for f in features})
        labels = {h: int(r[h]) for h in ("flood", "landslide") if h in r.index and pd.notna(r[h])}
        return {"observation": obs,
                "missing_features": [f for f in features if obs[f] is None],
                "recorded_labels": labels,
                "split": self.split_of(target)}
