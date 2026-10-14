#!/usr/bin/env python
"""Village layer for the village-level backtest (PLAN_erakde §11).

  python scripts/build_villages.py [--shp data/village_shp/VILLAGE_NLSC_1150817]

Writes data_processed/villages.parquet   — every village in the NLSC boundary file (TWD97 lon/lat): code, names, centroid
                                            (lon, lat → km in the kde.py frame), area km², county (臺 → 台), town
       data_processed/village_cases.parquet — local confirmed cases 2008+ with onset / report day, count, lon / lat and the
                                            village they fall in (point-in-polygon on the 最小統計區 centroid; cases without
                                            coordinates fall back to county + town + village name; the rest are dropped)
Village assignment uses the CURRENT boundaries (2026-08-17), so the 2018 台南 / 高雄 village mergers are handled by geometry.
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import shapefile
from matplotlib.path import Path as MPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dengue_ewarn.data import load_line_list  # noqa: E402
from dengue_ewarn.kde import to_km  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--shp", default=str(ROOT / "data" / "village_shp" / "VILLAGE_NLSC_1150817"))
args = ap.parse_args()
out = ROOT / "data_processed"; out.mkdir(exist_ok=True)

t0 = time.time()
r = shapefile.Reader(args.shp, encoding="utf-8")
fields = [f[0] for f in r.fields[1:]]
rows, polys = [], []
for sr in r.iterShapeRecords():
    rec = dict(zip(fields, sr.record)); shp = sr.shape
    pts = np.asarray(shp.points); parts = list(shp.parts) + [len(pts)]
    rings = [pts[parts[i]: parts[i + 1]] for i in range(len(parts) - 1) if parts[i + 1] - parts[i] >= 3]
    # area (shoelace in km²) and centroid of the largest ring (outer rings are counter-clockwise? NLSC uses clockwise; use |area|)
    best, area_tot, cx, cy, wsum = None, 0.0, 0.0, 0.0, 0.0
    for ring in rings:
        x, y = to_km(ring[:, 0], ring[:, 1])
        a = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        if abs(a) < 1e-9:
            continue
        # treat every ring as a part (holes are rare for villages); weight centroid by |area|
        ccx = np.sum((x + np.roll(x, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y)) / (6 * a)
        ccy = np.sum((y + np.roll(y, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y)) / (6 * a)
        area_tot += abs(a); cx += ccx * abs(a); cy += ccy * abs(a); wsum += abs(a)
    if wsum == 0:
        continue
    rows.append({"villcode": rec["VILLCODE"], "county": rec["COUNTYNAME"].replace("臺", "台"), "town": rec["TOWNNAME"], "village": rec["VILLNAME"],
                 "towncode": rec["TOWNCODE"], "x_km": cx / wsum, "y_km": cy / wsum, "area_km2": area_tot,
                 "bx0": pts[:, 0].min(), "bx1": pts[:, 0].max(), "by0": pts[:, 1].min(), "by1": pts[:, 1].max()})
    polys.append([MPath(ring) for ring in rings])
vil = pd.DataFrame(rows)
vil["lon"] = vil.x_km / (111.32 * np.cos(np.radians(23.0))) + 120.5; vil["lat"] = vil.y_km / 111.0 + 23.0
print(f"{len(vil)} villages read in {time.time()-t0:.0f}s | unnamed {int((vil.village == '').sum())} | area km² median {vil.area_km2.median():.2f}, p90 {vil.area_km2.quantile(.9):.1f}")

# ---- cases
raw = pd.read_csv(ROOT / "data" / "Dengue_Daily.csv", encoding="utf-8-sig", dtype=str, low_memory=False)
ll = load_line_list(); raw = raw.loc[ll.index]
ll["lon"] = pd.to_numeric(raw["最小統計區中心點X"], errors="coerce"); ll["lat"] = pd.to_numeric(raw["最小統計區中心點Y"], errors="coerce")
ll["vname"] = raw["居住縣市"].str.replace("臺", "台") + raw["居住鄉鎮"] + raw["居住村里"].fillna("")
loc = ll[(~ll.imported) & (ll.onset >= "2008-01-01")].copy()
print(f"local cases 2008+: {len(loc):,} | with coordinates {loc.lon.notna().mean():.3%}")

# point-in-polygon: candidate villages by bounding box, then ring test (any ring hit → inside; holes ignored)
t0 = time.time()
assign = np.full(len(loc), -1)
has = loc.lon.notna().to_numpy(); px = loc.lon.to_numpy(); py = loc.lat.to_numpy()
bx0, bx1, by0, by1 = vil.bx0.to_numpy(), vil.bx1.to_numpy(), vil.by0.to_numpy(), vil.by1.to_numpy()
# group identical coordinates (統計區 centroids repeat a lot) to cut the work
coords = pd.DataFrame({"x": px[has], "y": py[has]}).drop_duplicates()
hit = {}
for x, y in zip(coords.x.to_numpy(), coords.y.to_numpy()):
    cand = np.where((bx0 <= x) & (x <= bx1) & (by0 <= y) & (y <= by1))[0]
    for j in cand:
        if any(p.contains_point((x, y)) for p in polys[j]):
            hit[(x, y)] = j; break
keys = list(zip(px, py))
assign = np.array([hit.get(k, -1) if h else -1 for k, h in zip(keys, has)])
print(f"point-in-polygon: {len(coords):,} distinct coordinates in {time.time()-t0:.0f}s | assigned {np.mean(assign >= 0):.3%} of cases")
# priority: the file's own county + town + village name when it is a current village (the address is authoritative and the
# 最小統計區 grid, drawn in 2010 on the pre-merger villages, straddles the post-2018 boundaries); geometry otherwise
name_idx = {k: i for i, k in enumerate(vil.county + vil.town + vil.village)}
by_name = np.array([name_idx.get(k, -1) for k in loc.vname.to_numpy()])
geo = assign.copy(); assign = np.where(by_name >= 0, by_name, geo)
print(f"by name {np.mean(by_name >= 0):.3%}, by geometry for the rest → assigned {np.mean(assign >= 0):.3%}; dropped {int((assign < 0).sum())} cases; "
      f"name vs geometry agree where both exist: {np.mean(by_name[(by_name >= 0) & (geo >= 0)] == geo[(by_name >= 0) & (geo >= 0)]):.3%}")
loc = loc[assign >= 0].copy(); loc["villcode"] = vil.villcode.to_numpy()[assign[assign >= 0]]
loc["by_geometry"] = (by_name < 0)[assign >= 0] if False else np.where(by_name >= 0, False, True)[assign >= 0]
print("assigned by geometry (old / missing village names) by year:", loc.groupby(loc.onset.dt.year).by_geometry.mean().round(2).to_dict())
cases = loc[["onset", "report", "n", "county", "township", "lon", "lat", "villcode", "by_geometry"]].reset_index(drop=True)
cases.to_parquet(out / "village_cases.parquet"); vil.drop(columns=["bx0", "bx1", "by0", "by1"]).to_parquet(out / "villages.parquet")
print("villages with ≥1 case:", cases.villcode.nunique(), "| cases by county (top):", cases.county.value_counts().head(4).to_dict())
print("saved", out / "villages.parquet", out / "village_cases.parquet")
