"""Compare the U-Net with the classical Otsu baseline on Sen1Floods11
test and Bolivia (held-out country) chips, per country.

  python train/evaluate_water.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s1f11 import load_split  # noqa: E402

from terrasignal.baseline import split_otsu  # noqa: E402
from terrasignal.model import amp_ok, load_checkpoint, model_input_db, normalize  # noqa: E402
from terrasignal.preprocess import lee_filter, to_db  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


class Counts:
    def __init__(self):
        self.tp = self.fp = self.fn = 0

    def add(self, pred, y):
        m, t = y >= 0, y == 1
        self.tp += int((pred & t & m).sum())
        self.fp += int((pred & ~t & m).sum())
        self.fn += int((~pred & t & m).sum())

    def scores(self):
        return {
            "iou": round(self.tp / max(self.tp + self.fp + self.fn, 1), 4),
            "precision": round(self.tp / max(self.tp + self.fp, 1), 4),
            "recall": round(self.tp / max(self.tp + self.fn, 1), 4),
        }


def baseline_water(x_db: np.ndarray) -> np.ndarray:
    vv = to_db(lee_filter(10 ** (x_db[0] / 10)))
    t = split_otsu(vv)
    return np.isfinite(vv) & (vv < t.value_db)


@torch.no_grad()
def model_water(model, meta, x_db, device, shift_db=0.0) -> np.ndarray:
    xn, valid = normalize(model_input_db(x_db + shift_db, meta), meta["mean"], meta["std"])
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp_ok(device)):
        logits = model(torch.from_numpy(xn)[None].to(device))[0, 0].float().cpu().numpy()
    return (logits > 0) & valid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "water_unet.pt"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "results_water.json"))
    ap.add_argument("--source", choices=["gee", "pc"], default="gee")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = load_checkpoint(Path(args.ckpt), device)
    methods = ["otsu_baseline", "unet", "unet_shift_+1dB"]
    results = {}
    for split in ["test", "bolivia"]:
        s = load_split(split, args.source)
        per = defaultdict(lambda: {m: Counts() for m in methods})
        for x, y, country in zip(s.x, s.y, s.countries):
            preds = {
                "otsu_baseline": baseline_water(x),
                "unet": model_water(model, meta, x, device),
                "unet_shift_+1dB": model_water(model, meta, x, device, shift_db=1.0),
            }
            for m, p in preds.items():
                per[country][m].add(p, y)
                per["ALL"][m].add(p, y)
        results[split] = {c: {m: v.scores() for m, v in d.items()} for c, d in sorted(per.items())}

    results["checkpoint"] = {k: meta[k] for k in ("epoch", "valid", "encoder", "offset_db")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))

    for split in ["test", "bolivia"]:
        print(f"\n{split.upper():10s} {'IoU otsu':>9s} {'IoU unet':>9s} {'unet +1dB':>10s}")
        for c, d in results[split].items():
            print(f"{c:10s} {d['otsu_baseline']['iou']:9.3f} {d['unet']['iou']:9.3f} {d['unet_shift_+1dB']['iou']:10.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
