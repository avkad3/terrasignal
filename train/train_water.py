"""Train the water segmentation U-Net on Sen1Floods11 hand-labeled chips.

  python train/train_water.py --epochs 80
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s1f11 import load_split  # noqa: E402

from terrasignal.model import CLIP_DB, amp_ok, build_model, model_input_db, normalize, save_checkpoint  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def channel_stats(x: np.ndarray) -> tuple[list[float], list[float]]:
    mean, std = [], []
    for c in range(x.shape[1]):
        v = x[:, c][np.isfinite(x[:, c])]
        v = np.clip(v, *CLIP_DB)
        mean.append(float(v.mean()))
        std.append(float(v.std()))
    return mean, std


class Chips(Dataset):
    def __init__(self, x, y, mean, std, crop=None, augment=False, repeats=1, offset_db=2.0,
                 channel_jitter_db=0.5, seed=0):
        self.x, self.y, self.mean, self.std = x, y, mean, std
        self.crop, self.augment, self.repeats, self.offset_db = crop, augment, repeats, offset_db
        self.channel_jitter_db = channel_jitter_db
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.x) * self.repeats

    def __getitem__(self, i):
        i %= len(self.x)
        x, y = self.x[i].copy(), self.y[i].copy()
        if self.crop:
            H, W = y.shape
            for _ in range(10):  # prefer crops that contain labels
                r, c = self.rng.integers(0, H - self.crop + 1), self.rng.integers(0, W - self.crop + 1)
                yc = y[r:r + self.crop, c:c + self.crop]
                if (yc >= 0).mean() > 0.5:
                    break
            x, y = x[:, r:r + self.crop, c:c + self.crop], yc
        if self.augment:
            # Calibration shift: GEE sigma0 (training) vs RTC gamma0 (inference).
            # A shared random offset plus a per-channel one (VH over water reads
            # ~3 dB brighter in RTC) keeps the model from keying on absolute levels.
            x = x + self.rng.uniform(-self.offset_db, self.offset_db)
            x = x + self.rng.uniform(-self.channel_jitter_db, self.channel_jitter_db, size=(2, 1, 1))
            k = self.rng.integers(4)
            x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
            if self.rng.random() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
        xn, valid = normalize(np.ascontiguousarray(x), self.mean, self.std)
        y = np.where(valid, y, -1)
        return torch.from_numpy(xn), torch.from_numpy(np.ascontiguousarray(y).astype(np.int64))


def masked_loss(logits, y):
    m = (y >= 0).float()
    t = y.clamp(min=0).float()
    bce = (F.binary_cross_entropy_with_logits(logits, t, reduction="none") * m).sum() / m.sum().clamp(min=1)
    p = torch.sigmoid(logits) * m
    dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + (t * m).sum() + 1)
    return bce + dice


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    tp = fp = fn = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp_ok(device)):
            pred = model(x)[:, 0] > 0
        m = y >= 0
        t = y == 1
        tp += (pred & t & m).sum().item()
        fp += (pred & ~t & m).sum().item()
        fn += (~pred & t & m).sum().item()
    return {"iou": tp / max(tp + fp + fn, 1), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--offset-db", type=float, default=2.0)
    ap.add_argument("--channel-jitter-db", type=float, default=0.5)
    ap.add_argument("--lee", type=int, default=0, help="Lee filter size applied to inputs (0 = off)")
    ap.add_argument("--source", choices=["gee", "pc", "both"], default="gee", help="training chips")
    ap.add_argument("--valid-source", choices=["gee", "pc", "both"], help="validation chips (default: --source)")
    ap.add_argument("--out", default=str(ROOT / "checkpoints" / "water_unet.pt"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    tr, va = load_split("train", args.source), load_split("valid", args.valid_source or args.source)
    if args.lee:  # harmonize texture with the inference data (see model_input_db)
        tr.x = np.stack([model_input_db(x, {"lee": args.lee}) for x in tr.x]).astype(np.float32)
        va.x = np.stack([model_input_db(x, {"lee": args.lee}) for x in va.x]).astype(np.float32)
    mean, std = channel_stats(tr.x)
    print(f"[{args.source}/{args.valid_source or args.source}] train {len(tr.ids)} chips, valid {len(va.ids)}; mean {np.round(mean, 2)} std {np.round(std, 2)}; "
          f"water fraction {(tr.y == 1).sum() / (tr.y >= 0).sum():.3f}; loaded in {time.time() - t0:.0f}s", flush=True)

    train_dl = DataLoader(
        Chips(tr.x, tr.y, mean, std, crop=args.crop, augment=True, repeats=4, offset_db=args.offset_db,
              channel_jitter_db=args.channel_jitter_db, seed=args.seed),
        batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True,
    )
    valid_dl = DataLoader(Chips(va.x, va.y, mean, std), batch_size=2, num_workers=0)

    model = build_model(args.encoder, "imagenet", args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train_dl), pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=amp_ok(device))

    log_path = Path(args.out).with_suffix(".log.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    with open(log_path, "w", newline="") as lf:
        log = csv.writer(lf)
        log.writerow(["epoch", "loss", "valid_iou", "valid_precision", "valid_recall", "seconds"])
        for ep in range(1, args.epochs + 1):
            model.train()
            te, losses = time.time(), []
            for x, y in train_dl:
                x, y = x.to(device), y.to(device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp_ok(device)):
                    logits = model(x)[:, 0]
                loss = masked_loss(logits.float(), y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                losses.append(loss.item())
            v = evaluate(model, valid_dl, device)
            log.writerow([ep, f"{np.mean(losses):.4f}", f"{v['iou']:.4f}", f"{v['precision']:.4f}",
                          f"{v['recall']:.4f}", f"{time.time() - te:.1f}"])
            lf.flush()
            mark = ""
            if v["iou"] > best:
                best = v["iou"]
                save_checkpoint(Path(args.out), model, {
                    "encoder": args.encoder, "dropout": args.dropout, "mean": mean, "std": std,
                    "epoch": ep, "valid": v, "offset_db": args.offset_db, "trained_on": "sen1floods11-hand",
                    "lee": args.lee, "channel_jitter_db": args.channel_jitter_db,
                    "source": args.source, "valid_source": args.valid_source or args.source,
                })
                mark = " *"
            print(f"ep {ep:3d}  loss {np.mean(losses):.4f}  valid IoU {v['iou']:.4f} "
                  f"P {v['precision']:.3f} R {v['recall']:.3f}  {time.time() - te:.0f}s{mark}", flush=True)
    print(f"best valid IoU {best:.4f} -> {args.out}  (total {(time.time() - t0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
