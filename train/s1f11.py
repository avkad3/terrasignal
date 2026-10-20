"""Sen1Floods11 hand-labeled loader. Each chip is 512x512 at 10 m:
S1Hand = VV, VH sigma0 in dB; LabelHand = -1 no data, 0 land, 1 water."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parent.parent / "data" / "sen1floods11"
SPLITS = {
    "train": "flood_train_data.csv",
    "valid": "flood_valid_data.csv",
    "test": "flood_test_data.csv",
    "bolivia": "flood_bolivia_data.csv",  # held-out country, never trained on
}


@dataclass
class Split:
    x: np.ndarray        # (N, 2, 512, 512) float32 dB, NaN = missing
    y: np.ndarray        # (N, 512, 512) int8
    ids: list[str]

    @property
    def countries(self) -> list[str]:
        return [i.split("_")[0] for i in self.ids]


def chip_ids(split: str) -> list[str]:
    path = ROOT / "splits" / SPLITS[split]
    with open(path, newline="") as f:
        rows = [r for r in csv.reader(f) if r and r[0].endswith(".tif")]
    return [r[0].replace("_S1Hand.tif", "") for r in rows]


# Rebuilt chips are compared with the original on 9x9-smoothed VV (speckle
# decorrelates raw pixels even when grids agree). A chip is kept if, after
# its best integer shift within +-3 px, the smoothed correlation is >= 0.6.
MIN_PC_CORR = 0.6


def pc_index(min_corr: float = MIN_PC_CORR) -> dict[str, tuple[int, int]]:
    """chip id -> (dy, dx) co-registration shift, for usable rebuilt chips."""
    idx = ROOT / "pc_index.csv"
    if not idx.exists():
        return {}
    out = {}
    with open(idx, newline="") as f:
        for r in csv.DictReader(f):
            if r["status"] != "ok" or r.get("best_corr") in (None, "", "nan"):
                continue
            if float(r["best_corr"]) >= min_corr:
                dy, dx = map(int, r["best_shift"].split(","))
                out[r["id"]] = (dy, dx)
    return out


def shift_nan(x: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift (C, H, W) by whole pixels, filling uncovered edges with NaN."""
    out = np.full_like(x, np.nan)
    H, W = x.shape[1:]
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    out[:, yd, xd] = x[:, ys, xs]
    return out


def load_split(split: str, source: str = "gee") -> Split:
    """source: 'gee' = original GEE sigma0 chips, 'pc' = the same chips rebuilt
    from Planetary Computer RTC (train/build_pc_chips.py), 'both' = each chip
    once from each source."""
    if source == "both":
        a, b = load_split(split, "gee"), load_split(split, "pc")
        return Split(np.concatenate([a.x, b.x]), np.concatenate([a.y, b.y]), a.ids + b.ids)
    ids = chip_ids(split)
    shifts = pc_index() if source == "pc" else {}
    if source == "pc":
        ids = [i for i in ids if i in shifts]
    xs, ys = [], []
    for cid in ids:
        src = ROOT / "S1Hand" / f"{cid}_S1Hand.tif" if source == "gee" else ROOT / "PCHand" / f"{cid}_PC.tif"
        with rasterio.open(src) as r:
            x = r.read().astype(np.float32)
        if source == "pc" and shifts[cid] != (0, 0):
            x = shift_nan(x, *shifts[cid])  # co-register to the labels
        with rasterio.open(ROOT / "LabelHand" / f"{cid}_LabelHand.tif") as r:
            y = r.read(1).astype(np.int8)
        x[~np.isfinite(x)] = np.nan
        y[~np.isfinite(x).all(axis=0)] = -1
        xs.append(x)
        ys.append(y)
    return Split(np.stack(xs), np.stack(ys), ids)
