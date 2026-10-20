"""Fetch an aligned before/after Sentinel-1 stack, plus terrain and permanent
water layers, for any AOI from Microsoft Planetary Computer.

Before/after scenes must share orbit direction and relative orbit; otherwise
viewing-geometry differences show up as false change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import numpy as np
import odc.stac
import planetary_computer as pc
import pystac
import pystac_client
import xarray as xr
from shapely.geometry import box, shape
from shapely.ops import unary_union

from .aoi import AOI, DEFAULT_MAX_AREA_KM2

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S1 = "sentinel-1-rtc"
DEM = "cop-dem-glo-30"
JRC = "jrc-gsw"
MIN_COVERAGE = 0.9


class NoPairError(RuntimeError):
    pass


@dataclass
class Pass:
    """All scenes from one satellite pass (adjacent frames share a pass)."""

    orbit_state: str
    relative_orbit: int
    day: date
    items: list[pystac.Item] = field(default_factory=list)
    coverage: float = 0.0

    @property
    def key(self) -> tuple[str, int]:
        return (self.orbit_state, self.relative_orbit)

    @property
    def ids(self) -> list[str]:
        return [it.id for it in self.items]


def _client() -> pystac_client.Client:
    return pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)


def group_passes(items: list[pystac.Item], aoi: AOI) -> list[Pass]:
    groups: dict[tuple, Pass] = {}
    for it in items:
        p = it.properties
        k = (p["sat:orbit_state"], int(p["sat:relative_orbit"]), it.datetime.date())
        groups.setdefault(k, Pass(*k)).items.append(it)
    target = box(*aoi.bbox)
    for g in groups.values():
        foot = unary_union([shape(it.geometry) for it in g.items])
        g.coverage = foot.intersection(target).area / target.area
    return list(groups.values())


def pick_pair(
    passes: list[Pass],
    before: date,
    after: date,
    window_days: int = 14,
    min_coverage: float = MIN_COVERAGE,
) -> tuple[Pass, Pass]:
    """Choose the after pass nearest `after`, then the same-geometry pass
    nearest `before` that is strictly earlier. Each role only draws from
    passes within `window_days` of its own date. Pure function; no network."""
    ok = [p for p in passes if p.coverage >= min_coverage]

    def near(p: Pass, d: date) -> bool:
        return abs((p.day - d).days) <= window_days

    afters = sorted((p for p in ok if near(p, after)), key=lambda p: abs((p.day - after).days))
    for a in afters:
        befores = [p for p in ok if p.key == a.key and p.day < a.day and near(p, before)]
        if befores:
            b = min(befores, key=lambda p: abs((p.day - before).days))
            return b, a
    seen = ", ".join(
        f"{p.day} {p.orbit_state[:3]}{p.relative_orbit} {p.coverage:.0%}"
        for p in sorted(passes, key=lambda p: p.day)
    )
    raise NoPairError(
        f"no same-orbit Sentinel-1 pair covering ≥{min_coverage:.0%} of the AOI within "
        f"±{window_days} days of {before} and {after}. Passes found: {seen or 'none'}. "
        "Try a wider --window or a smaller box"
    )


def find_pair(
    aoi: AOI, before: date, after: date, window_days: int = 14
) -> tuple[Pass, Pass]:
    if before >= after:
        raise ValueError("before date must be earlier than after date")
    w = timedelta(days=window_days)
    rng = f"{(before - w).isoformat()}/{(after + w).isoformat()}"
    items = list(_client().search(collections=[S1], bbox=aoi.bbox, datetime=rng).items())
    if not items:
        raise NoPairError(f"no Sentinel-1 scenes over the AOI in {rng}")
    return pick_pair(group_passes(items, aoi), before, after, window_days)


def _load_s1(p: Pass, aoi: AOI, resolution: float, geobox=None) -> xr.Dataset:
    kw = dict(geobox=geobox) if geobox is not None else dict(
        bbox=aoi.bbox, crs=f"EPSG:{aoi.utm_epsg}", resolution=resolution
    )
    ds = odc.stac.load(
        p.items, bands=["vv", "vh"], groupby="solar_day", dtype="float32",
        nodata=np.nan, resampling="bilinear", **kw,
    )
    return ds.isel(time=0, drop=True)


def _load_static(collection: str, band: str, aoi: AOI, geobox, dtype: str, nodata) -> xr.DataArray:
    items = list(_client().search(collections=[collection], bbox=aoi.bbox).items())
    if not items:
        raise NoPairError(f"no {collection} data over the AOI")
    # static tiles share one datetime, so solar_day grouping mosaics them
    ds = odc.stac.load(
        items, bands=[band], geobox=geobox, groupby="solar_day", dtype=dtype,
        nodata=nodata, resampling="bilinear",
    )
    return ds[band].isel(time=0, drop=True)


def fetch_stack(
    aoi: AOI,
    before: date,
    after: date,
    resolution: float = 10.0,
    window_days: int = 14,
    max_area_km2: float = DEFAULT_MAX_AREA_KM2,
) -> xr.Dataset:
    """Return one Dataset on a common UTM grid with variables
    vv_before, vh_before, vv_after, vh_after (linear gamma0),
    dem (m), and jrc_occurrence (0-100 % of years with water)."""
    aoi.check_size(max_area_km2)
    b, a = find_pair(aoi, before, after, window_days)

    s1a = _load_s1(a, aoi, resolution)
    gb = s1a.odc.geobox
    s1b = _load_s1(b, aoi, resolution, geobox=gb)
    dem = _load_static(DEM, "data", aoi, gb, "float32", np.nan)
    jrc = _load_static(JRC, "occurrence", aoi, gb, "uint8", 255)

    ds = xr.Dataset(
        {
            "vv_before": s1b.vv, "vh_before": s1b.vh,
            "vv_after": s1a.vv, "vh_after": s1a.vh,
            "dem": dem, "jrc_occurrence": jrc,
        }
    )
    ds.attrs.update(
        bbox=aoi.bbox, epsg=aoi.utm_epsg, resolution=resolution,
        before_date=b.day.isoformat(), after_date=a.day.isoformat(),
        orbit_state=a.orbit_state, relative_orbit=a.relative_orbit,
        before_scenes=",".join(b.ids), after_scenes=",".join(a.ids),
        fetched_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )
    return ds.compute()
