"""Learned flood detector: U-Net water maps for both dates, combined with the
same terrain and permanent-water masks as the baseline."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from torch import nn

from .baseline import NO_FLOOD, PROBABLE, compose_flood, remove_small
from .infer import water_probability
from .preprocess import lee_filter, permanent_water, slope_degrees, to_db


@dataclass
class DetectParams:
    water_threshold: float = 0.5
    max_slope_deg: float = 5.0
    perm_occurrence: int = 50
    min_area_m2: float = 5000.0
    mc_samples: int = 0
    # Change-only "probable flood": water among buildings or vegetation is
    # brighter than open water, so the U-Net (trained on open water) misses
    # it, but both polarizations still drop. Physics-motivated heuristics,
    # not fitted to labels.
    change: bool = True
    min_vv_drop_db: float = 2.0
    min_vh_drop_db: float = 3.0
    max_vh_after_db: float = -18.0
    filter_size: int = 7


def change_signature(vv_b, vh_b, vv_a, vh_a, p: DetectParams) -> np.ndarray:
    vvb, vva = to_db(lee_filter(vv_b, p.filter_size)), to_db(lee_filter(vv_a, p.filter_size))
    vhb, vha = to_db(lee_filter(vh_b, p.filter_size)), to_db(lee_filter(vh_a, p.filter_size))
    with np.errstate(invalid="ignore"):
        return (
            (vva - vvb <= -p.min_vv_drop_db)
            & (vha - vhb <= -p.min_vh_drop_db)
            & (vha <= p.max_vh_after_db)
        )


def run_model(
    model: nn.Module,
    meta: dict,
    vv_before, vh_before, vv_after, vh_after, dem, jrc_occurrence,
    resolution: float,
    params: DetectParams | None = None,
) -> tuple[np.ndarray, dict, dict]:
    """Return (class map, stats, probability layers)."""
    p = params or DetectParams()
    prob_b, std_b = water_probability(model, meta, vv_before, vh_before, mc_samples=p.mc_samples)
    prob_a, std_a = water_probability(model, meta, vv_after, vh_after, mc_samples=p.mc_samples)
    valid = np.isfinite(prob_b) & np.isfinite(prob_a)
    water_b = valid & (prob_b >= p.water_threshold)
    water_a = valid & (prob_a >= p.water_threshold)
    perm = permanent_water(jrc_occurrence, p.perm_occurrence)
    steep = ~(slope_degrees(dem, resolution) <= p.max_slope_deg)

    classes, stats = compose_flood(water_b, water_a, valid, perm, steep, resolution, p.min_area_m2)

    stats["probable_flood_km2"] = 0.0
    if p.change:
        sig = change_signature(vv_before, vh_before, vv_after, vh_after, p)
        probable = sig & (classes == NO_FLOOD) & ~water_b
        px = resolution * resolution
        probable = remove_small(probable, int(np.ceil(p.min_area_m2 / px)))
        classes[probable] = PROBABLE
        stats["probable_flood_km2"] = round(float(probable.sum()) * px / 1e6, 3)
    stats.update(params=asdict(p), model={k: meta.get(k) for k in ("encoder", "epoch", "valid")})
    layers = {"prob_before": prob_b, "prob_after": prob_a, "std_before": std_b, "std_after": std_a}
    return classes, stats, layers
