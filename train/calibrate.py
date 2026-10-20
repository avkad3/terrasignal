"""Calibrate the water model and measure how trustworthy its probabilities are.

1. Fit one temperature T on the *valid* split (minimizing pixel NLL).
2. On test and Bolivia (never used for fitting), compare
   raw / temperature-scaled / MC-dropout + temperature probabilities:
   IoU, NLL, Brier, ECE and reliability curves.
3. Check whether uncertainty flags errors (AUROC of uncertainty vs error).
4. Store T in the checkpoint so inference uses it.

  python train/calibrate.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s1f11 import load_split  # noqa: E402

from terrasignal.model import load_checkpoint, logit_samples, normalize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
N_BINS = 15
# validated categorical slots 1-3 (dataviz reference palette, light mode)
COLORS = {"raw": "#2a78d6", "temperature": "#eb6834", "mc+temperature": "#1baf7a"}
MARKERS = {"raw": "o", "temperature": "s", "mc+temperature": "^"}


@torch.no_grad()
def collect(model, meta, split, device, mc_samples: int, temperature: float = 1.0):
    """Per labeled pixel: deterministic logit, MC mean prob, MC std, label."""
    s = load_split(split)
    logits, mc_mean, mc_std, labels = [], [], [], []
    for x_db, y in zip(s.x, s.y):
        xn, valid = normalize(x_db, meta["mean"], meta["std"])
        xb = torch.from_numpy(xn)[None].to(device)
        det = logit_samples(model, xb)[0, 0].cpu().numpy()
        m = (y >= 0) & valid
        logits.append(det[m])
        labels.append(y[m].astype(np.uint8))
        if mc_samples:
            p = torch.sigmoid(logit_samples(model, xb, mc_samples, temperature))[:, 0]
            mc_mean.append(p.mean(0).cpu().numpy()[m])
            mc_std.append(p.std(0, unbiased=False).cpu().numpy()[m])
    out = {"logit": np.concatenate(logits), "y": np.concatenate(labels)}
    if mc_samples:
        out["mc_mean"], out["mc_std"] = np.concatenate(mc_mean), np.concatenate(mc_std)
    return out


def fit_temperature(logit: np.ndarray, y: np.ndarray, device: str, max_px: int = 4_000_000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), min(len(y), max_px), replace=False)
    l = torch.from_numpy(logit[idx]).float().to(device)
    t = torch.from_numpy(y[idx]).float().to(device)
    log_t = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(l / log_t.exp(), t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def metrics(p: np.ndarray, y: np.ndarray) -> dict:
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    yb = y.astype(bool)
    pred = p >= 0.5
    tp, fp, fn = (pred & yb).sum(), (pred & ~yb).sum(), (~pred & yb).sum()
    bins = np.minimum((p * N_BINS).astype(int), N_BINS - 1)
    cnt = np.bincount(bins, minlength=N_BINS)
    conf = np.bincount(bins, weights=p, minlength=N_BINS)
    pos = np.bincount(bins, weights=yb, minlength=N_BINS)
    nz = cnt > 0
    ece = float((np.abs(conf[nz] - pos[nz])).sum() / len(p))
    return {
        "iou": round(float(tp / max(tp + fp + fn, 1)), 4),
        "nll": round(float(-(yb * np.log(p) + (~yb) * np.log(1 - p)).mean()), 4),
        "brier": round(float(((p - yb) ** 2).mean()), 4),
        "ece": round(ece, 4),
        "reliability": {
            "confidence": [round(float(v), 4) for v in (conf[nz] / cnt[nz])],
            "observed": [round(float(v), 4) for v in (pos[nz] / cnt[nz])],
            "count": [int(v) for v in cnt[nz]],
        },
    }


def auroc(score: np.ndarray, positive: np.ndarray, max_px: int = 2_000_000, seed: int = 0) -> float:
    """Probability a random error pixel gets a higher uncertainty score than a
    random correct pixel (Mann-Whitney U)."""
    rng = np.random.default_rng(seed)
    if len(score) > max_px:
        idx = rng.choice(len(score), max_px, replace=False)
        score, positive = score[idx], positive[idx]
    n1, n0 = positive.sum(), (~positive).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks for ties
    s_sorted = score[order]
    _, first, counts = np.unique(s_sorted, return_index=True, return_counts=True)
    for f, c in zip(first, counts):
        if c > 1:
            ranks[order[f:f + c]] = f + (c + 1) / 2
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def plot(results: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = [s for s in ("test", "bolivia") if s in results]
    fig, axes = plt.subplots(1, len(splits), figsize=(6.2 * len(splits), 5.6), sharey=True)
    for ax, split in zip(np.atleast_1d(axes), splits):
        ax.plot([0, 1], [0, 1], color="#9a9890", lw=1, ls="--", zorder=1, label="perfect calibration")
        for name, r in results[split]["methods"].items():
            rel = r["reliability"]
            ax.plot(rel["confidence"], rel["observed"], color=COLORS[name], lw=2, marker=MARKERS[name],
                    ms=6, markeredgecolor="white", markeredgewidth=1.2, zorder=3,
                    label=f"{name}   ECE {r['ece']:.3f} · NLL {r['nll']:.3f}")
        ax.set_title(f"{split.capitalize()} ({'10 countries' if split == 'test' else 'never seen in training'})",
                     fontsize=11, color="#21201c", loc="left")
        ax.set_xlabel("predicted P(water)", color="#4a4943")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(color="#e6e5e0", lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#c9c8c0")
        ax.tick_params(colors="#6b6a63")
        ax.legend(loc="upper left", fontsize=8.5, frameon=False, labelcolor="#21201c")
    np.atleast_1d(axes)[0].set_ylabel("observed water fraction", color="#4a4943")
    fig.suptitle(f"Reliability of water probabilities (T = {results['temperature']:.2f}, "
                 f"{results['mc_samples']} MC-dropout samples)", fontsize=12, color="#21201c", x=0.01, ha="left")
    fig.patch.set_facecolor("#fcfcfb")
    for ax in np.atleast_1d(axes):
        ax.set_facecolor("#fcfcfb")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "water_unet.pt"))
    ap.add_argument("--mc", type=int, default=20)
    ap.add_argument("--no-write", action="store_true", help="do not store T in the checkpoint")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = Path(args.ckpt)
    model, meta = load_checkpoint(ckpt, device)

    va = collect(model, meta, "valid", device, mc_samples=0)
    T = fit_temperature(va["logit"], va["y"], device)
    print(f"temperature fitted on valid ({len(va['y']):,} px): T = {T:.3f}")

    results = {"temperature": round(T, 4), "mc_samples": args.mc}
    for split in ("test", "bolivia"):
        d = collect(model, meta, split, device, mc_samples=args.mc, temperature=T)
        y = d["y"]
        p_raw = 1 / (1 + np.exp(-d["logit"]))
        p_t = 1 / (1 + np.exp(-d["logit"] / T))
        methods = {"raw": metrics(p_raw, y), "temperature": metrics(p_t, y), "mc+temperature": metrics(d["mc_mean"], y)}
        err_t = (p_t >= 0.5) != y.astype(bool)
        err_mc = (d["mc_mean"] >= 0.5) != y.astype(bool)
        unc = {
            "auroc_margin_temperature": round(auroc(1 - np.abs(2 * p_t - 1), err_t), 4),
            "auroc_mc_std": round(auroc(d["mc_std"], err_mc), 4),
            "error_rate": round(float(err_t.mean()), 4),
            "mean_mc_std": round(float(d["mc_std"].mean()), 4),
        }
        results[split] = {"pixels": int(len(y)), "methods": methods, "uncertainty": unc}
        print(f"\n{split.upper()} ({len(y):,} labeled px)")
        print(f"  {'method':16s} {'IoU':>6s} {'NLL':>7s} {'Brier':>7s} {'ECE':>7s}")
        for name, r in methods.items():
            print(f"  {name:16s} {r['iou']:6.3f} {r['nll']:7.4f} {r['brier']:7.4f} {r['ece']:7.4f}")
        print(f"  error-detection AUROC: margin {unc['auroc_margin_temperature']:.3f}, "
              f"MC std {unc['auroc_mc_std']:.3f}  (error rate {unc['error_rate']:.3f})")

    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "calibration.json").write_text(json.dumps(results, indent=2))
    plot(results, ROOT / "docs" / "img" / "reliability.png")

    if not args.no_write:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        ck["meta"]["temperature"] = T
        ck["meta"]["calibrated_on"] = "sen1floods11-hand valid"
        torch.save(ck, ckpt)
        print(f"\nstored T = {T:.3f} in {ckpt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
