"""Water segmentation network and the input normalization shared by training
and inference."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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


def logit_samples(model: nn.Module, x: torch.Tensor, mc_samples: int = 0, temperature: float = 1.0) -> torch.Tensor:
    """Water logits, shape (S, B, H, W), divided by `temperature`.

    mc_samples=0 gives one deterministic pass. Otherwise dropout stays on and
    S = mc_samples. For our U-Net the dropout sits right before the 1x1 head,
    so encoder and decoder run once and only the cheap head is resampled."""
    head = getattr(model, "segmentation_head", None)
    split = (
        hasattr(model, "encoder") and hasattr(model, "decoder")
        and isinstance(head, nn.Sequential) and isinstance(head[0], nn.Dropout2d)
    )
    if split:
        dec = model.decoder(model.encoder(x))
        if not mc_samples:
            out = head[1](dec)[None]
        else:
            p = head[0].p
            out = torch.stack([head[1](F.dropout2d(dec, p, training=True)) for _ in range(mc_samples)])
        return out[:, :, 0] / temperature
    # generic model: full forward passes, dropout layers switched on for MC
    was = [(m, m.training) for m in model.modules() if isinstance(m, (nn.Dropout, nn.Dropout2d))]
    for m, _ in was:
        m.train(bool(mc_samples))
    try:
        out = torch.stack([model(x)[:, 0] for _ in range(max(mc_samples, 1))])
    finally:
        for m, t in was:
            m.train(t)
    return out / temperature


def model_input_db(db: np.ndarray, meta: dict) -> np.ndarray:
    """Apply the checkpoint's input harmonization to a (2, H, W) dB stack.

    GEE sigma0 (training) is noisy over water while Planetary Computer RTC is
    smooth there; a shared Lee filter makes texture mean the same in both."""
    size = int(meta.get("lee", 0) or 0)
    if not size:
        return db
    from .preprocess import lee_filter, to_db
    return np.stack([to_db(lee_filter(10 ** (c / 10), size)) for c in db])


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


WEIGHTS_FILE, CONFIG_FILE = "model.safetensors", "config.json"


def export_checkpoint(path: Path, out_dir: Path) -> None:
    """Write weights as safetensors (no pickle, safe to load from anywhere)
    plus the metadata as config.json."""
    import json

    from safetensors.torch import save_file

    ck = torch.load(path, map_location="cpu", weights_only=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in ck["state_dict"].items()}, out_dir / WEIGHTS_FILE)
    (out_dir / CONFIG_FILE).write_text(json.dumps(ck["meta"], indent=2))


def _resolve(path) -> Path:
    """'hf://user/repo' or 'hf://user/repo/subfolder' -> local folder (downloaded and cached)."""
    s = str(path)
    if not s.startswith("hf://"):
        return Path(path)
    from huggingface_hub import hf_hub_download

    parts = s[len("hf://"):].strip("/").split("/")
    repo, sub = "/".join(parts[:2]), "/".join(parts[2:]) or None
    cfg = hf_hub_download(repo, CONFIG_FILE, subfolder=sub)
    hf_hub_download(repo, WEIGHTS_FILE, subfolder=sub)
    return Path(cfg).parent


def load_checkpoint(path, device: str = "cpu") -> tuple[nn.Module, dict]:
    """Load from a training .pt file, an exported folder (model.safetensors +
    config.json), or a Hugging Face repo given as hf://user/repo[/subfolder]."""
    p = _resolve(path)
    if p.is_dir():
        import json

        from safetensors.torch import load_file

        meta = json.loads((p / CONFIG_FILE).read_text())
        state = load_file(p / WEIGHTS_FILE)
    else:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        meta, state = ck["meta"], ck["state_dict"]
    m = build_model(meta.get("encoder", "resnet34"), weights=None, dropout=meta.get("dropout", 0.2))
    m.load_state_dict(state)
    return m.to(device).eval(), meta
