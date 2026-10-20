"""How much does water IoU depend on absolute calibration? Shift every input
by a fixed number of dB and re-score on labeled test and Bolivia chips.

  python train/robustness.py --ckpt checkpoints/water_unet.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_water import Counts, model_water  # noqa: E402
from s1f11 import load_split  # noqa: E402

from terrasignal.model import load_checkpoint  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SHIFTS = [-4, -3, -2, -1, 0, 1, 2, 3, 4]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "water_unet.pt"))
    ap.add_argument("--out")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = load_checkpoint(Path(args.ckpt), device)
    res = {}
    for split in ("test", "bolivia"):
        s = load_split(split)
        res[split] = {}
        for sh in SHIFTS:
            c = Counts()
            for x, y in zip(s.x, s.y):
                c.add(model_water(model, meta, x, device, shift_db=sh), y)
            res[split][sh] = c.scores()["iou"]
        print(f"{split:8s} " + "  ".join(f"{sh:+d}:{res[split][sh]:.3f}" for sh in SHIFTS), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
