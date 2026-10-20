"""Save a fetched stack to disk: one GeoTIFF per variable, metadata JSON,
and a quicklook PNG for eyeballing."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr


def to_db(linear: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10 * np.log10(np.where(linear > 0, linear, np.nan))


def save_stack(ds: xr.Dataset, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    for name, da in ds.data_vars.items():
        da = da.rio.write_crs(f"EPSG:{ds.attrs['epsg']}")
        da.rio.to_raster(out / f"{name}.tif", compress="deflate", tiled=True)
    (out / "meta.json").write_text(json.dumps(ds.attrs, indent=2, default=str))
    quicklook(ds, out / "quicklook.png")
    return out


def quicklook(ds: xr.Dataset, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vvb, vva = to_db(ds.vv_before.values), to_db(ds.vv_after.values)
    panels = [
        (vvb, f"VV before {ds.attrs['before_date']} (dB)", "gray", (-25, 0)),
        (vva, f"VV after {ds.attrs['after_date']} (dB)", "gray", (-25, 0)),
        (vva - vvb, "VV change (dB)", "RdBu", (-8, 8)),
        (ds.dem.values, "Copernicus DEM (m)", "terrain", None),
        (np.where(ds.jrc_occurrence.values == 255, np.nan, ds.jrc_occurrence.values),
         "JRC water occurrence (%)", "Blues", (0, 100)),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    for ax, (img, title, cmap, lim) in zip(axes, panels):
        im = ax.imshow(img, cmap=cmap, **({} if lim is None else dict(vmin=lim[0], vmax=lim[1])))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"bbox {ds.attrs['bbox']}  ·  {ds.attrs['orbit_state']} orbit {ds.attrs['relative_orbit']}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
