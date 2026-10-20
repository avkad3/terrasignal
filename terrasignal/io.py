"""Save and load stacks and results: GeoTIFFs, metadata JSON, quicklook PNGs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr

from .preprocess import to_db

STACK_VARS = ["vv_before", "vh_before", "vv_after", "vh_after", "dem", "jrc_occurrence"]


def save_stack(ds: xr.Dataset, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    for name, da in ds.data_vars.items():
        da = da.rio.write_crs(f"EPSG:{ds.attrs['epsg']}")
        da.rio.to_raster(out / f"{name}.tif", compress="deflate", tiled=True)
    (out / "meta.json").write_text(json.dumps(ds.attrs, indent=2, default=str))
    quicklook(ds, out / "quicklook.png")
    return out


def load_stack(folder: Path) -> xr.Dataset:
    missing = [v for v in STACK_VARS if not (folder / f"{v}.tif").exists()]
    if missing:
        raise FileNotFoundError(f"{folder} is missing {missing}; run `terrasignal fetch` first")
    ds = xr.Dataset(
        {v: rioxarray.open_rasterio(folder / f"{v}.tif", masked=False).squeeze("band", drop=True)
         for v in STACK_VARS}
    )
    ds.attrs.update(json.loads((folder / "meta.json").read_text()))
    return ds.load()


def save_classes(classes: np.ndarray, template: xr.DataArray, path: Path) -> None:
    da = template.copy(data=classes).astype("uint8")
    da.attrs = {}
    da = da.rio.write_nodata(255)
    da.rio.to_raster(path, compress="deflate", tiled=True)


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


def flood_overlay(
    ds: xr.Dataset, classes: np.ndarray, stats: dict, title: str, path: Path, label: str = "Baseline flood"
) -> None:
    """VV-after backdrop with flood in red and permanent water in blue."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from .baseline import FLOOD, MASKED, PERMANENT, PROBABLE

    vva = to_db(ds.vv_after.values)
    rgba = np.zeros(classes.shape + (4,), np.float32)
    rgba[classes == FLOOD] = (0.9, 0.1, 0.1, 0.85)
    rgba[classes == PROBABLE] = (1.0, 0.55, 0.0, 0.75)
    rgba[classes == PERMANENT] = (0.1, 0.35, 0.9, 0.85)
    rgba[classes == MASKED] = (1.0, 0.8, 0.2, 0.25)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(vva, cmap="gray", vmin=-25, vmax=0)
    axes[0].set_title(f"VV after {ds.attrs['after_date']} (dB)")
    axes[1].imshow(vva, cmap="gray", vmin=-25, vmax=0)
    axes[1].imshow(rgba)
    probable = stats.get("probable_flood_km2")
    title_txt = f"{label}: {stats['flood_km2']:.1f} km²"
    handles = [Patch(color=(0.9, 0.1, 0.1), label="new open-water flood")]
    if probable is not None:
        title_txt += f" open water + {probable:.1f} km² probable"
        handles.append(Patch(color=(1.0, 0.55, 0.0), label="probable flood (change only)"))
    handles += [Patch(color=(0.1, 0.35, 0.9), label="permanent water (JRC)"),
                Patch(color=(1.0, 0.8, 0.2), alpha=0.5, label="masked: slope > 5°")]
    axes[1].set_title(title_txt)
    axes[1].legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.85)
    for ax in axes:
        ax.axis("off")
    sub = f"{title}  ·  {ds.attrs['before_date']} → {ds.attrs['after_date']}"
    if "threshold_before" in stats:
        tb, ta = stats["threshold_before"], stats["threshold_after"]
        sub += (f"  ·  water threshold {tb['value_db']:.1f} / {ta['value_db']:.1f} dB "
                f"({tb['method']} / {ta['method']})")
    fig.suptitle(sub)
    fig.tight_layout(rect=(0, 0, 1, 0.95))  # keep the suptitle clear of panel titles
    fig.savefig(path, dpi=110)
    plt.close(fig)
