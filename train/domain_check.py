"""Does the model see water in the data it actually runs on?

Planetary Computer RTC has no flood labels, but JRC Global Surface Water
gives a proxy: pixels with water in >= 90% of 1984-2020 observations should be
water, and pixels never seen as water (occurrence 0) on flat ground should not.
Checked on the *before* image of each fetched case stack.

  python train/domain_check.py --ckpt checkpoints/water_unet.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from terrasignal.infer import water_probability
from terrasignal.io import load_stack
from terrasignal.model import load_checkpoint
from terrasignal.preprocess import slope_degrees

ROOT = Path(__file__).resolve().parent.parent
CASES = ["assam_2024", "vijayawada_2024", "pakistan_2022", "derna_2023", "emilia_2023"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "water_unet.pt"))
    ap.add_argument("--out")
    args = ap.parse_args()
    model, meta = load_checkpoint(Path(args.ckpt), "cuda")
    res = {}
    for case in CASES:
        folder = ROOT / "data" / case
        if not (folder / "meta.json").exists():
            continue
        ds = load_stack(folder)
        p, _ = water_probability(model, meta, ds.vv_before.values, ds.vh_before.values)
        jrc = ds.jrc_occurrence.values
        flat = slope_degrees(ds.dem.values, float(ds.attrs["resolution"])) <= 5
        perm = (jrc != 255) & (jrc >= 90) & np.isfinite(p)
        dry = (jrc == 0) & flat & np.isfinite(p)
        r = {
            "perm_px": int(perm.sum()),
            "perm_recall": round(float((p[perm] > 0.5).mean()), 3) if perm.any() else None,
            "dry_fp_rate": round(float((p[dry] > 0.5).mean()), 4),
        }
        res[case] = r
        print(f"{case:16s} permanent-water recall {r['perm_recall']} (n={r['perm_px']:,}) · "
              f"dry-land false positives {r['dry_fp_rate']:.4f}", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
