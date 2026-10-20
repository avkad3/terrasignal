"""Water segmentation network and the input normalization shared by training
and inference."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

# Inputs are VV and VH backscatter in dB. Statistics come from the
# Sen1Floods11 hand-labeled train split (see train/train_water.py) and are
# stored in each checkpoint; these are only defaults.
CLIP_DB = (-40.0, 5.0)
DEFAULT_MEAN = (-10.5, -17.5)
DEFAULT_STD = (4.0, 4.5)


def amp_ok(device: str) -> bool:
    """Mixed precision only where it is safe and useful. GTX 16xx (Turing
    without tensor cores) returns NaNs from fp16 convolutions and gains no
    speed from them."""
    if device != "cuda" or not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 7 and "GTX 16" not in torch.cuda.get_device_name()


def build_model(encoder: str = "resnet34", weights: str | None = "imagenet", dropout: float = 0.2) -> nn.Module:
    import segmentation_models_pytorch as smp

    m = smp.Unet(encoder_name=encoder, encoder_weights=weights, in_channels=2, classes=1)
    # Dropout right before the 1x1 head: regularizes training and lets
    # MC dropout sample predictions at inference time.
    m.segmentation_head = nn.Sequential(nn.Dropout2d(dropout), m.segmentation_head)
    return m


def normalize(db: np.ndarray, mean, std) -> tuple[np.ndarray, np.ndarray]:
    """db: (2, H, W) VV/VH in dB, NaN = missing. Returns (normalized float32
    with missing set to 0, valid mask)."""
    valid = np.isfinite(db).all(axis=0)
    x = np.clip(np.nan_to_num(db, nan=0.0), *CLIP_DB)
    x = (x - np.asarray(mean, np.float32)[:, None, None]) / np.asarray(std, np.float32)[:, None, None]
    x[:, ~valid] = 0.0
    return x.astype(np.float32), valid


def save_checkpoint(path: Path, model: nn.Module, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[nn.Module, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    meta = ck["meta"]
    m = build_model(meta.get("encoder", "resnet34"), weights=None, dropout=meta.get("dropout", 0.2))
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval(), meta
