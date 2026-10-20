"""Classical flood baseline: split-based Otsu water thresholding on filtered
VV backscatter, differenced in time and masked by terrain and permanent water.

This is the bar the learned model has to beat.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import label
from skimage.filters import threshold_otsu

from .preprocess import lee_filter, permanent_water, slope_degrees, to_db

# Open water in VV is typically -15 to -22 dB; clamping stops a water-free
# scene from producing a threshold that splits two kinds of land.
THRESHOLD_RANGE_DB = (-23.0, -14.0)
FALLBACK_THRESHOLD_DB = -18.0


@dataclass
class Threshold:
    value_db: float
    method: str          # "tiles", "fallback"
    n_tiles: int = 0
    raw_db: float | None = None


def ashman_d(a: np.ndarray, b: np.ndarray) -> float:
    s = np.sqrt(a.var() + b.var())
    return float(np.sqrt(2) * abs(a.mean() - b.mean()) / s) if s > 0 else 0.0


def split_otsu(
    db: np.ndarray,
    exclude: np.ndarray | None = None,
    tile: int = 256,
    min_ad: float = 2.0,
    min_frac: float = 0.1,
    min_valid: float = 0.8,
    max_samples: int = 2_000_000,
    seed: int = 0,
) -> Threshold:
    """Otsu on pooled pixels from tiles whose histograms are clearly bimodal
    (Ashman's D above `min_ad`, each class at least `min_frac` of the tile,
    the darker class water-like). Falls back to a fixed threshold when no
    tile qualifies, e.g. a scene with no open water."""
    ok = np.isfinite(db) if exclude is None else np.isfinite(db) & ~exclude
    picked = []
    H, W = db.shape
    for y in range(0, H - tile + 1, tile):
        for x in range(0, W - tile + 1, tile):
            m = ok[y:y + tile, x:x + tile]
            if m.mean() < min_valid:
                continue
            v = db[y:y + tile, x:x + tile][m]
            t = threshold_otsu(v)
            lo, hi = v[v < t], v[v >= t]
            frac = lo.size / v.size
            if not (min_frac <= frac <= 1 - min_frac):
                continue
            if lo.mean() > THRESHOLD_RANGE_DB[1] or ashman_d(lo, hi) < min_ad:
                continue
            picked.append(v)
    if not picked:
        return Threshold(FALLBACK_THRESHOLD_DB, "fallback")
    pool = np.concatenate(picked)
    if pool.size > max_samples:
        pool = np.random.default_rng(seed).choice(pool, max_samples, replace=False)
    raw = float(threshold_otsu(pool))
    return Threshold(float(np.clip(raw, *THRESHOLD_RANGE_DB)), "tiles", len(picked), raw)


def remove_small(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    if min_pixels <= 1:
        return mask
    lab, n = label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_pixels
    keep[0] = False
    return keep[lab]


@dataclass
class BaselineParams:
    filter_size: int = 7
    max_slope_deg: float = 5.0
    min_drop_db: float = 3.0
    perm_occurrence: int = 50
    min_area_m2: float = 5000.0


# Output classes in flood.tif
NO_FLOOD, FLOOD, PERMANENT, MASKED, NODATA = 0, 1, 2, 3, 255


def run_baseline(
    vv_before: np.ndarray,
    vv_after: np.ndarray,
    dem: np.ndarray,
    jrc_occurrence: np.ndarray,
    resolution: float,
    params: BaselineParams | None = None,
) -> tuple[np.ndarray, dict]:
    """Return (class map, stats). Inputs are linear VV gamma0 arrays on one grid."""
    p = params or BaselineParams()
    db_b = to_db(lee_filter(vv_before, p.filter_size))
    db_a = to_db(lee_filter(vv_after, p.filter_size))

    valid = np.isfinite(db_b) & np.isfinite(db_a)
    perm = permanent_water(jrc_occurrence, p.perm_occurrence)
    slope = slope_degrees(dem, resolution)
    steep = ~(slope <= p.max_slope_deg)  # NaN slope counts as steep

    t_b = split_otsu(db_b, exclude=steep)
    t_a = split_otsu(db_a, exclude=steep)
    water_b = valid & (db_b < t_b.value_db)
    water_a = valid & (db_a < t_a.value_db)
    dropped = (db_a - db_b) <= -p.min_drop_db

    classes, stats = compose_flood(water_b, water_a & dropped, valid, perm, steep, resolution, p.min_area_m2)
    stats.update(threshold_before=asdict(t_b), threshold_after=asdict(t_a), params=asdict(p))
    return classes, stats


def compose_flood(
    water_b: np.ndarray,
    water_a: np.ndarray,
    valid: np.ndarray,
    perm: np.ndarray,
    steep: np.ndarray,
    resolution: float,
    min_area_m2: float,
) -> tuple[np.ndarray, dict]:
    """Shared by every detector: flood = new water outside permanent water
    and steep terrain, minus tiny clusters; returns class map and areas."""
    flood = water_a & ~water_b & ~(steep | perm) & valid
    px_area = resolution * resolution
    flood = remove_small(flood, int(np.ceil(min_area_m2 / px_area)))

    classes = np.full(valid.shape, NO_FLOOD, np.uint8)
    classes[steep & valid] = MASKED
    classes[perm & valid] = PERMANENT
    classes[flood] = FLOOD
    classes[~valid] = NODATA

    n_valid = int(valid.sum())
    stats = {
        "flood_km2": round(float(flood.sum()) * px_area / 1e6, 3),
        "permanent_water_km2": round(float((perm & valid).sum()) * px_area / 1e6, 3),
        "water_after_km2": round(float((water_a & ~steep).sum()) * px_area / 1e6, 3),
        "valid_km2": round(n_valid * px_area / 1e6, 3),
        "masked_steep_fraction": round(float((steep & valid).sum()) / max(n_valid, 1), 4),
    }
    return classes, stats
