"""Rebuild every Sen1Floods11 hand-labeled chip from Planetary Computer
Sentinel-1 RTC: same acquisition (date, orbit direction, relative orbit),
loaded onto the chip's exact pixel grid, so the hand labels still apply.

This puts the training and evaluation data in the domain TerraSignal runs on.
Writes data/sen1floods11/PCHand/<id>_PC.tif (VV, VH in dB, NaN = missing)
and pc_index.csv with per-chip alignment (VV correlation with the original).

  python train/build_pc_chips.py
"""
from __future__ import annotations

import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import odc.stac
import planetary_computer as pc
import pystac_client
import rasterio
from odc.geo.geobox import GeoBox

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s1f11 import ROOT as S1F11  # noqa: E402

from terrasignal.preprocess import to_db  # noqa: E402

OUT = S1F11 / "PCHand"
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def event_meta() -> dict:
    feats = json.loads((S1F11 / "metadata.geojson").read_text())["features"]
    m = {f["properties"]["location"]: f["properties"] for f in feats}
    m["Mekong"] = m["Cambodia"]
    return m


def build_one(path: Path, meta: dict, cat) -> dict:
    cid = path.name.replace("_S1Hand.tif", "")
    dest = OUT / f"{cid}_PC.tif"
    with rasterio.open(path) as r:
        orig = r.read().astype(np.float32)
        profile, transform, crs, shape = r.profile, r.transform, r.crs, r.shape
        b = r.bounds
    m = meta[cid.split("_")[0]]
    if not dest.exists():
        d = datetime.strptime(m["s1_date"], "%Y/%m/%d")
        items = [
            i for i in cat.search(
                collections=["sentinel-1-rtc"], bbox=[b.left, b.bottom, b.right, b.top],
                datetime=f"{(d - timedelta(days=1)).date()}/{(d + timedelta(days=1)).date()}",
            ).items()
            if i.properties["sat:orbit_state"] == m["orbit"].lower()
            and int(i.properties["sat:relative_orbit"]) == int(m["rel_orbit_num"])
        ]
        if not items:
            return {"id": cid, "status": "no_scene"}
        # nearest acquisition day to the event date, all frames of that pass
        best = min(items, key=lambda i: abs((i.datetime.replace(tzinfo=None) - d).total_seconds()))
        same = [i for i in items if i.datetime.date() == best.datetime.date()]
        gb = GeoBox(shape, transform, crs)
        ds = odc.stac.load(same, bands=["vv", "vh"], geobox=gb, groupby="solar_day",
                           dtype="float32", nodata=np.nan, resampling="bilinear").isel(time=0)
        db = np.stack([to_db(ds.vv.values), to_db(ds.vh.values)]).astype(np.float32)
        prof = {**profile, "count": 2, "dtype": "float32", "nodata": np.nan, "compress": "deflate"}
        with rasterio.open(dest, "w", **prof) as w:
            w.write(db)
    with rasterio.open(dest) as r:
        db = r.read()
    ok = np.isfinite(db[0]) & np.isfinite(orig[0])
    corr = float(np.corrcoef(db[0][ok], orig[0][ok])[0, 1]) if ok.sum() > 1000 else float("nan")
    return {"id": cid, "status": "ok", "valid_frac": round(float(ok.mean()), 3), "vv_corr": round(corr, 3),
            "vv_offset_db": round(float(np.median(db[0][ok] - orig[0][ok])), 2) if ok.any() else None}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = event_meta()
    cat = pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)
    chips = sorted((S1F11 / "S1Hand").glob("*_S1Hand.tif"))
    rows = []
    with ThreadPoolExecutor(4) as ex:
        for i, r in enumerate(ex.map(lambda p: build_one(p, meta, cat), chips), 1):
            rows.append(r)
            if i % 25 == 0 or i == len(chips):
                ok = [x for x in rows if x["status"] == "ok"]
                print(f"  {i}/{len(chips)} chips, {len(ok)} ok", flush=True)
    with open(S1F11 / "pc_index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "status", "valid_frac", "vv_corr", "vv_offset_db"])
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r["status"] == "ok"]
    corr = np.array([r["vv_corr"] for r in ok], float)
    off = np.array([r["vv_offset_db"] for r in ok if r["vv_offset_db"] is not None], float)
    print(f"built {len(ok)}/{len(rows)}; VV correlation median {np.nanmedian(corr):.3f} "
          f"(p10 {np.nanpercentile(corr, 10):.3f}); RTC - GEE VV offset median {np.median(off):+.2f} dB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
