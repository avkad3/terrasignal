"""SAR preprocessing and auxiliary masks."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

# Sentinel-1 IW GRD high-resolution products have an ENL of about 4.4
DEFAULT_ENL = 4.4


def to_db(linear: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10 * np.log10(np.where(linear > 0, linear, np.nan))


def _nan_mean_filter(img: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Local mean that ignores NaNs (normalized convolution)."""
    valid = np.isfinite(img)
    filled = np.where(valid, img, 0.0)
    w = uniform_filter(valid.astype(np.float32), size)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = uniform_filter(filled, size) / w
    return m, w


def lee_filter(linear: np.ndarray, size: int = 7, enl: float = DEFAULT_ENL) -> np.ndarray:
    """Lee speckle filter on linear backscatter. NaNs stay NaN and do not
    leak into their neighbours."""
    img = linear.astype(np.float32)
    mean, _ = _nan_mean_filter(img, size)
    sq, _ = _nan_mean_filter(img * img, size)
    var = np.maximum(sq - mean * mean, 0)
    cu2 = 1.0 / enl  # squared coefficient of variation of pure speckle
    with np.errstate(invalid="ignore", divide="ignore"):
        k = (var - mean * mean * cu2) / (var * (1 + cu2))
    k = np.clip(np.nan_to_num(k, nan=0.0), 0, 1)
    out = mean + k * (img - mean)
    out[~np.isfinite(img)] = np.nan
    return out


def slope_degrees(dem: np.ndarray, resolution: float, smooth_m: float = 30.0) -> np.ndarray:
    """Terrain slope. Copernicus DEM is a surface model, so building edges
    look like cliffs; smoothing to about its native 30 m stops cities from
    being masked as steep terrain."""
    d = np.where(np.isfinite(dem), dem, np.nanmean(dem) if np.isfinite(dem).any() else 0)
    if smooth_m > 0:
        d = gaussian_filter(d.astype(np.float32), sigma=smooth_m / resolution)
    gy, gx = np.gradient(d.astype(np.float32), resolution)
    s = np.degrees(np.arctan(np.hypot(gx, gy)))
    s[~np.isfinite(dem)] = np.nan
    return s


def permanent_water(jrc_occurrence: np.ndarray, min_occurrence: int = 50) -> np.ndarray:
    """Pixels that held water in at least `min_occurrence`% of observations
    (1984-2020). 255 is JRC's no-data value."""
    occ = jrc_occurrence.astype(np.int16)
    return (occ != 255) & (occ >= min_occurrence)
