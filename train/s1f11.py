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


def load_split(split: str) -> Split:
    ids = chip_ids(split)
    xs, ys = [], []
    for cid in ids:
        with rasterio.open(ROOT / "S1Hand" / f"{cid}_S1Hand.tif") as r:
            x = r.read().astype(np.float32)
        with rasterio.open(ROOT / "LabelHand" / f"{cid}_LabelHand.tif") as r:
            y = r.read(1).astype(np.int8)
        x[~np.isfinite(x)] = np.nan
        y[~np.isfinite(x).all(axis=0)] = -1
        xs.append(x)
        ys.append(y)
    return Split(np.stack(xs), np.stack(ys), ids)
