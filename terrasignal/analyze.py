"""One call from (bbox, dates) to map-ready outputs:

  flood.tif        uint8 classes (COG): 0 none, 1 open-water flood, 2 permanent
                   water, 3 masked steep, 4 probable flood, 255 no data
  confidence.tif   uint8 % probability of new open water (COG), 255 no data
  vv_before.tif,   uint8 VV backdrops for the swipe view (COG):
  vv_after.tif     value = (dB + 25) / 25 * 254, 255 no data
  flood.geojson    WGS84 polygons with class, area_km2, mean confidence
  summary.json     scenes, dates, areas, model, timings, caveats
"""
from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import rasterio.features
import rioxarray  # noqa: F401  (registers .rio; the fetch path never imports it otherwise)
import xarray as xr
from pyproj import Transformer
from scipy import ndimage
from shapely.geometry import mapping, shape
from shapely.ops import transform as shp_transform
from shapely.ops import unary_union

from .aoi import AOI
from .baseline import FLOOD, PROBABLE, run_baseline
from .fetch import fetch_stack
from .preprocess import to_db

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = ROOT / "checkpoints" / "water_unet.pt"
MAX_FEATURES = 5000
CLASS_NAMES = {FLOOD: "open_water_flood", PROBABLE: "probable_flood"}

_detector_cache: dict = {}


def load_detector(ckpt: Path | None = DEFAULT_CKPT):
    """(model, meta) or None when no checkpoint exists, in which case the
    classical baseline is used."""
    if ckpt is None or not Path(ckpt).exists():
        return None
    key = str(Path(ckpt).resolve())
    if key not in _detector_cache:
        import torch

        from .model import load_checkpoint
        _detector_cache[key] = load_checkpoint(Path(ckpt), "cuda" if torch.cuda.is_available() else "cpu")
    return _detector_cache[key]


def _template(ds: xr.Dataset) -> xr.DataArray:
    return ds.vv_after.rio.write_crs(f"EPSG:{ds.attrs['epsg']}")


def write_cog(arr: np.ndarray, template: xr.DataArray, path: Path, nodata: int = 255) -> None:
    da = template.copy(data=arr.astype(np.uint8))
    da.attrs = {}
    da.rio.write_nodata(nodata, inplace=True)
    da.rio.to_raster(path, driver="COG", compress="DEFLATE")


def backdrop(linear: np.ndarray) -> np.ndarray:
    db = to_db(linear)
    out = np.clip((db + 25) / 25 * 254, 0, 254)
    return np.where(np.isfinite(db), np.round(out), 255).astype(np.uint8)


def polygons(
    classes: np.ndarray,
    confidence: np.ndarray | None,
    transform,
    epsg: int,
    resolution: float,
    simplify_m: float | None = None,
    max_features: int = MAX_FEATURES,
) -> tuple[dict, int]:
    """Vectorize flood classes into a WGS84 FeatureCollection. Areas come
    from pixel counts in the projected grid, so they are exact."""
    to_wgs = Transformer.from_crs(epsg, 4326, always_xy=True).transform
    simplify_m = resolution if simplify_m is None else simplify_m
    px_km2 = resolution * resolution / 1e6
    feats, total = [], 0
    for cls, name in CLASS_NAMES.items():
        mask = classes == cls
        if not mask.any():
            continue
        lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
        total += n
        ids = np.arange(1, n + 1)
        sizes = ndimage.sum(mask, lab, ids)
        mean_conf = (
            ndimage.mean(np.nan_to_num(confidence), lab, ids) if confidence is not None and cls == FLOOD else None
        )
        keep = set(ids[np.argsort(sizes)[::-1][:max_features]].tolist())
        # Trace with the same 8-connectivity used for labelling and merge the
        # pieces, so each region becomes exactly one feature carrying its area once.
        parts: dict[int, list] = {}
        for geom, val in rasterio.features.shapes(
            lab.astype(np.int32), mask=mask, transform=transform, connectivity=8
        ):
            if int(val) in keep:
                parts.setdefault(int(val), []).append(shape(geom))
        for i, geoms in parts.items():
            g = (geoms[0] if len(geoms) == 1 else unary_union(geoms)).simplify(simplify_m, preserve_topology=True)
            props = {"class": name, "area_km2": round(float(sizes[i - 1]) * px_km2, 4)}
            if mean_conf is not None:
                props["confidence"] = round(float(mean_conf[i - 1]), 3)
            feats.append({"type": "Feature", "geometry": mapping(shp_transform(to_wgs, g)), "properties": props})
    feats.sort(key=lambda f: -f["properties"]["area_km2"])
    feats = feats[:max_features]
    return {"type": "FeatureCollection", "features": feats}, total


def caveats(stats: dict, method: str) -> list[str]:
    out = []
    if method == "baseline":
        out.append("No trained model found; the classical threshold baseline was used.")
    if stats.get("masked_steep_fraction", 0) > 0.5:
        out.append(f"{stats['masked_steep_fraction']:.0%} of the area is steep terrain and was masked; "
                   "no flood found there is not evidence of no flood.")
    if stats.get("probable_flood_km2"):
        out.append("'Probable flood' comes from a backscatter-drop heuristic (water among buildings or "
                   "vegetation); it can also flag paddy fields being flooded for planting.")
    out.append("Radar sees open water best; flash floods that drain before the satellite pass are missed.")
    return out


def analyze_stack(ds: xr.Dataset, out: Path, ckpt: Path | None = DEFAULT_CKPT, method: str = "auto") -> dict:
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    res = float(ds.attrs["resolution"])
    det = load_detector(ckpt) if method in ("auto", "hybrid") else None
    if method == "hybrid" and det is None:
        raise FileNotFoundError(f"hybrid method needs a checkpoint at {ckpt}")

    confidence = None
    if det is not None:
        from .detect import run_model
        classes, stats, layers = run_model(
            det[0], det[1], ds.vv_before.values, ds.vh_before.values, ds.vv_after.values,
            ds.vh_after.values, ds.dem.values, ds.jrc_occurrence.values, res,
        )
        confidence = layers["confidence"]
        used = "hybrid"
    else:
        classes, stats = run_baseline(ds.vv_before.values, ds.vv_after.values, ds.dem.values,
                                      ds.jrc_occurrence.values, res)
        used = "baseline"

    tpl = _template(ds)
    write_cog(classes, tpl, out / "flood.tif")
    if confidence is not None:
        pct = np.where(np.isfinite(confidence), np.round(np.nan_to_num(confidence) * 100), 255)
        write_cog(pct, tpl, out / "confidence.tif")
    write_cog(backdrop(ds.vv_before.values), tpl, out / "vv_before.tif")
    write_cog(backdrop(ds.vv_after.values), tpl, out / "vv_after.tif")

    fc, n_regions = polygons(classes, confidence, tpl.rio.transform(), int(ds.attrs["epsg"]), res)
    (out / "flood.geojson").write_text(json.dumps(fc))

    a = ds.attrs
    summary = {
        "bbox": list(a["bbox"]),
        "before": {"date": a["before_date"], "scenes": str(a["before_scenes"]).split(",")},
        "after": {"date": a["after_date"], "scenes": str(a["after_scenes"]).split(",")},
        "orbit": {"state": a["orbit_state"], "relative": int(a["relative_orbit"])},
        "grid": {"epsg": int(a["epsg"]), "resolution_m": res, "shape": [int(ds.sizes["y"]), int(ds.sizes["x"])]},
        "method": used,
        "areas_km2": {
            "open_water_flood": stats["flood_km2"],
            "probable_flood": stats.get("probable_flood_km2"),
            "permanent_water": stats["permanent_water_km2"],
            "valid": stats["valid_km2"],
        },
        "open_water_mean_confidence": stats.get("open_water_mean_confidence"),
        "masked_steep_fraction": stats["masked_steep_fraction"],
        "flood_regions": n_regions,
        "polygons_written": len(fc["features"]),
        "model": stats.get("model"),
        "caveats": caveats(stats, used),
        "files": sorted(p.name for p in out.iterdir() if p.suffix in (".tif", ".geojson")) + ["summary.json"],
        "seconds": {"detect_and_export": round(time.time() - t0, 1)},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def analyze(
    bbox, before: date, after: date, out: Path,
    ckpt: Path | None = DEFAULT_CKPT, method: str = "auto", window_days: int = 14,
) -> dict:
    """Fetch imagery for any AOI and produce all outputs in `out`."""
    t0 = time.time()
    aoi = AOI.from_list(bbox)
    ds = fetch_stack(aoi, before, after, window_days=window_days)
    t_fetch = time.time() - t0
    summary = analyze_stack(ds, out, ckpt, method)
    summary["seconds"]["fetch"] = round(t_fetch, 1)
    summary["seconds"]["total"] = round(time.time() - t0, 1)
    summary["aoi_km2"] = round(aoi.area_km2, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
