"""Learned flood detector: U-Net water maps for both dates, combined with the
same terrain and permanent-water masks as the baseline."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from torch import nn

from .baseline import compose_flood
from .infer import water_probability
from .preprocess import permanent_water, slope_degrees


@dataclass
class DetectParams:
    water_threshold: float = 0.5
    max_slope_deg: float = 5.0
    perm_occurrence: int = 50
    min_area_m2: float = 5000.0
    mc_samples: int = 0


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
    stats.update(params=asdict(p), model={k: meta[k] for k in ("encoder", "epoch", "valid")})
    layers = {"prob_before": prob_b, "prob_after": prob_a, "std_before": std_b, "std_after": std_a}
    return classes, stats, layers
