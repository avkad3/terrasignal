"""Run the water model over a scene of any size with overlapping tiles."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .model import amp_ok, normalize
from .preprocess import to_db


def _window(n: int) -> np.ndarray:
    """2-D blending weights that fade toward tile edges, so seams vanish."""
    w = np.hanning(n + 2)[1:-1]
    return np.outer(w, w).astype(np.float32) + 1e-3


def _tiles(H: int, W: int, size: int, stride: int):
    ys = list(range(0, max(H - size, 0) + 1, stride))
    xs = list(range(0, max(W - size, 0) + 1, stride))
    if ys[-1] + size < H:
        ys.append(H - size)
    if xs[-1] + size < W:
        xs.append(W - size)
    return [(y, x) for y in ys for x in xs]


def _enable_mc_dropout(model: nn.Module) -> None:
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


@torch.no_grad()
def water_probability(
    model: nn.Module,
    meta: dict,
    vv_linear: np.ndarray,
    vh_linear: np.ndarray,
    tile: int = 512,
    overlap: int = 128,
    batch: int = 4,
    mc_samples: int = 0,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (mean water probability, std across MC-dropout samples or None).
    Missing pixels come back as NaN."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    db = np.stack([to_db(vv_linear), to_db(vh_linear)])
    H, W = db.shape[1:]
    pad_h, pad_w = max(tile - H, 0), max(tile - W, 0)
    if pad_h or pad_w:
        db = np.pad(db, ((0, 0), (0, pad_h), (0, pad_w)), constant_values=np.nan)
    x, valid = normalize(db, meta["mean"], meta["std"])
    Hp, Wp = x.shape[1:]

    samples = max(mc_samples, 1)
    if mc_samples:
        _enable_mc_dropout(model)
    else:
        model.eval()

    win = _window(tile)
    acc = np.zeros((Hp, Wp), np.float32)
    acc2 = np.zeros((Hp, Wp), np.float32) if mc_samples else None
    wsum = np.zeros((Hp, Wp), np.float32)
    coords = _tiles(Hp, Wp, tile, tile - overlap)
    for i in range(0, len(coords), batch):
        chunk = coords[i:i + batch]
        xb = torch.from_numpy(np.stack([x[:, y:y + tile, c:c + tile] for y, c in chunk])).to(device)
        s1 = np.zeros((len(chunk), tile, tile), np.float32)
        s2 = np.zeros_like(s1)
        for _ in range(samples):
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp_ok(device)):
                p = torch.sigmoid(model(xb)[:, 0].float()).cpu().numpy()
            s1 += p
            s2 += p * p
        mean_t = s1 / samples
        for (y, c), m, sq in zip(chunk, mean_t, s2 / samples):
            acc[y:y + tile, c:c + tile] += m * win
            if acc2 is not None:
                acc2[y:y + tile, c:c + tile] += np.sqrt(np.maximum(sq - m * m, 0)) * win
            wsum[y:y + tile, c:c + tile] += win
    model.eval()

    prob = (acc / wsum)[:H, :W]
    prob[~valid[:H, :W]] = np.nan
    std = None
    if acc2 is not None:
        std = (acc2 / wsum)[:H, :W]
        std[~valid[:H, :W]] = np.nan
    return prob, std
