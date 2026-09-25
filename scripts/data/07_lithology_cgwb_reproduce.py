"""
Reproduce the district lithology table from the original CGWB Major Principal Aquifer map,
to verify lithology/Kerala_District_Lithology_CGWB.csv (whose derivation script was not kept).

Inputs (unchanged): lithology/cgwb_source/major_principal_aquifer_CGWB.zip (read in place)
                    reference/geoBoundaries-IND-ADM2_simplified.geojson, Kerala_district_reference.csv
Output: lithology/processed/Kerala_District_Lithology_CGWB_reproduced.csv
        lithology/processed/Lithology_reproduction_check.csv   (reproduced vs existing file)

Mapping of CGWB principal-aquifer rock names (pas_name) to the six model levels is the one
documented in lithology/STATUS_COLLECTED.md; any other rock name stays unmapped.
Areas are computed in an Albers equal-area projection centred on Kerala.
"""
from pathlib import Path
import io
import json
import zipfile
import pandas as pd
import shapefile
from pyproj import CRS, Transformer
from shapely.geometry import shape, box
from shapely.ops import transform, unary_union

U = Path(__file__).resolve().parents[2] / "collected_datasets"
OUT = U / "lithology" / "processed"
OUT.mkdir(parents=True, exist_ok=True)
MAP = {"Gneiss": "granite_gneiss", "Granite": "granite_gneiss", "Charnockite": "granite_gneiss",
       "Khondalites": "granite_gneiss", "Banded Gneissic Complex": "granite_gneiss",
       "Laterite": "laterite", "Schist": "schist", "Sandstone": "sandstone", "Shale": "shale",
       "Alluvium": "alluvium"}
LEVELS = ["granite_gneiss", "laterite", "schist", "sandstone", "shale", "alluvium"]
AEA = CRS.from_proj4("+proj=aea +lat_1=8 +lat_2=13 +lat_0=10.5 +lon_0=76.5 +datum=WGS84 +units=m")


def main():
    z = zipfile.ZipFile(U / "lithology" / "cgwb_source" / "major_principal_aquifer_CGWB.zip")
    src_crs = CRS.from_wkt(z.read("Major_Principal_Aquifer.prj").decode())
    r = shapefile.Reader(shp=io.BytesIO(z.read("Major_Principal_Aquifer.shp")),
                         shx=io.BytesIO(z.read("Major_Principal_Aquifer.shx")),
                         dbf=io.BytesIO(z.read("Major_Principal_Aquifer.dbf")))
    to_aea = Transformer.from_crs(src_crs, AEA, always_xy=True).transform
    wgs_to_aea = Transformer.from_crs("EPSG:4326", AEA, always_xy=True).transform
    wgs_to_src = Transformer.from_crs("EPSG:4326", src_crs, always_xy=True).transform

    ref = pd.read_csv(U / "reference" / "Kerala_district_reference.csv")
    gb = json.load(open(U / "reference" / "geoBoundaries-IND-ADM2_simplified.geojson", encoding="utf-8"))
    ids = dict(zip(ref["shape_id"], ref["district_chirps"]))
    dist = {ids[f["properties"]["shapeID"]]: transform(wgs_to_aea, shape(f["geometry"]))
            for f in gb["features"] if f["properties"]["shapeID"] in ids}
    kerala_src = transform(wgs_to_src, box(74.7, 8.1, 77.6, 12.95))

    pieces = {}
    for sr in r.iterShapeRecords():
        name = sr.record["pas_name"].strip()
        g = shape(sr.shape.__geo_interface__)
        if not g.intersects(kerala_src):
            continue
        g = transform(to_aea, g.intersection(kerala_src))
        pieces.setdefault(name, []).append(g)
    units = {k: unary_union(v).buffer(0) for k, v in pieces.items()}

    rows = []
    for d, poly in dist.items():
        area = poly.area
        fr = {k: poly.intersection(g).area / area for k, g in units.items()}
        covered = sum(fr.values())
        lv = {l: sum(v for k, v in fr.items() if MAP.get(k) == l) for l in LEVELS}
        rows.append(dict(district=d, district_coverage_pct=round(100 * covered, 2),
                         lithology=max(lv, key=lv.get),
                         **{f"frac_{l}": round(lv[l], 5) for l in LEVELS},
                         frac_unmapped=round(sum(v for k, v in fr.items() if k not in MAP), 5),
                         cgwb_classes_present="; ".join(f"{k} {100 * v:.1f}%" for k, v in
                                                        sorted(fr.items(), key=lambda x: -x[1]) if v > 0.0005)))
    rep = pd.DataFrame(rows)
    rep.to_csv(OUT / "Kerala_District_Lithology_CGWB_reproduced.csv", index=False)

    old = pd.read_csv(U / "lithology" / "Kerala_District_Lithology_CGWB.csv")
    cmp_ = rep.merge(old, on="district", suffixes=("_reproduced", "_existing"))
    chk = pd.DataFrame(dict(
        district=cmp_["district"],
        lithology_reproduced=cmp_["lithology_reproduced"], lithology_existing=cmp_["lithology_existing"],
        same_dominant_class=cmp_["lithology_reproduced"] == cmp_["lithology_existing"],
        max_abs_fraction_diff=[max(abs(r[f"frac_{l}_reproduced"] - r[f"frac_{l}_existing"]) for l in LEVELS)
                               for _, r in cmp_.iterrows()],
        coverage_reproduced=cmp_["district_coverage_pct_reproduced"],
        coverage_existing=cmp_["district_coverage_pct_existing"]))
    chk.to_csv(OUT / "Lithology_reproduction_check.csv", index=False)
    pd.set_option("display.width", 200)
    print(chk.to_string(index=False))
    print("distinct dominant classes:", sorted(rep["lithology"].unique()))


if __name__ == "__main__":
    main()
