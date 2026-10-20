---
license: cc-by-4.0
library_name: segmentation-models-pytorch
pipeline_tag: image-segmentation
tags:
  - remote-sensing
  - sentinel-1
  - sar
  - flood-mapping
  - water-segmentation
  - earth-observation
  - geospatial
datasets:
  - sen1floods11
metrics:
  - iou
---

# TerraSignal water U-Net (Sentinel-1 RTC)

Segments **open water** in Sentinel-1 SAR imagery (VV and VH backscatter). It's the water model behind [TerraSignal](https://github.com/avkad3/terrasignal), which maps floods by comparing water before and after an event anywhere on Earth.

It was trained and evaluated on **Planetary Computer `sentinel-1-rtc` imagery**, the data TerraSignal runs on, using the hand labels from [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11).

## Results

Water IoU on hand-labeled chips rebuilt from the same Planetary Computer acquisitions. The test set is held-out chips from 9 countries; Bolivia is a country never used in training.

| | Otsu threshold | Same U-Net trained on GEE σ⁰ (`gee_baseline/`) | **This model** |
|---|---|---|---|
| Test, 9 countries | 0.481 | 0.263 | **0.683** |
| India | 0.660 | 0.042 | **0.685** |
| Bolivia (unseen country) | 0.450 | 0.048 | **0.655** |

- It beats the Otsu threshold in all 9 test countries and in Bolivia.
- Probabilities are temperature-calibrated (T = 1.66, fitted on validation chips). Test ECE improves from 0.035 to 0.016, and Bolivia from 0.047 to 0.035.
- On live Planetary Computer scenes it detects 99–100% of permanent water (JRC occurrence ≥ 90%) in five case studies across India, Pakistan, Libya and Italy.

## Why there are two models: the domain gap

`gee_baseline/` is the same architecture trained on Sen1Floods11's original Google Earth Engine σ⁰ chips. On GEE data it scores 0.669 test IoU. On Planetary Computer RTC data it **collapses to 0.263** and misses permanent rivers entirely.

The cause is texture: over water, GEE imagery sits near the radar noise floor and is *noisier* than land, while RTC water is *smoother* than land. The GEE model learned "water = dark and noisy". This model was trained on chips rebuilt from the same acquisitions on Planetary Computer, co-registered to the original labels. **Use the model that matches your imagery source**, and keep `gee_baseline/` only for GEE σ⁰ data or for reproducing the comparison.

## Usage

Inputs are VV and VH **in dB**, shape (2, H, W), from Sentinel-1 RTC (γ⁰) at 10 m. `config.json` holds the normalization statistics and temperature.

```python
from terrasignal.model import load_checkpoint
from terrasignal.infer import water_probability

model, meta = load_checkpoint("hf://REPO_ID", "cpu")       # or "cuda"
prob, _ = water_probability(model, meta, vv_linear, vh_linear)  # any size, tiled; calibrated P(water)
```

Or run the full flood pipeline for any area:

```bash
pip install "terrasignal[ml] @ git+https://github.com/avkad3/terrasignal"
terrasignal analyze --bbox 80.52 16.45 80.75 16.65 --before 2024-08-20 --after 2024-09-01 \
    --name vijayawada --ckpt hf://REPO_ID
```

**Architecture:** U-Net with a ResNet34 encoder ([segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)), 2 input channels, 1 output logit, and Dropout2d(0.2) before the head. The weights are in `model.safetensors`.

```python
import json, segmentation_models_pytorch as smp, torch
from safetensors.torch import load_file
cfg = json.load(open("config.json"))
m = smp.Unet(cfg["encoder"], encoder_weights=None, in_channels=2, classes=1)
m.segmentation_head = torch.nn.Sequential(torch.nn.Dropout2d(cfg["dropout"]), m.segmentation_head)
m.load_state_dict(load_file("model.safetensors")); m.eval()
# input: (dB - cfg["mean"]) / cfg["std"] per channel, dB clipped to [-40, 5], missing pixels set to 0
# P(water) = sigmoid(logit / cfg["temperature"])
```

## Training

- **Data:** 204 Sen1Floods11 hand-labeled chips rebuilt from Planetary Computer RTC (same date, orbit direction and relative orbit; co-registered by best integer shift; chips under 0.6 smoothed correlation dropped). Validation used 73 chips, test 75, Bolivia 15. Paraguay isn't in Planetary Computer's archive.
- **Augmentation:** random 256 px crops, flips and rotations, a ±2 dB shared brightness offset, and ±0.5 dB per channel.
- **Loss:** masked BCE + Dice.
- **Optimizer:** AdamW with one-cycle LR (max 3e-4) for 80 epochs; the best epoch by validation IoU was 48.
- **Hardware:** a GTX 1650 (4 GB), in fp32 (fp16 returns NaN on GTX 16-series cards). Training took 38 minutes.

## Limitations

- **"Water" means any open water.** Rice paddies flooded for planting and waterlogged fields count. Flood mapping needs a before/after comparison and a permanent-water mask, as TerraSignal does.
- **Braided river corridors get filled in.** In Assam's dry season the model labels sandbars between channels as water (31% false positives on never-water flat land). It probably learned this from monsoon-season labels.
- **It doesn't see water under vegetation or among buildings**, because Sen1Floods11 labels open water only. TerraSignal adds a separate, heuristic "probable flood" class for that.
- **Mid-range probabilities (0.6–0.9) are still overconfident** after temperature scaling.
- **Steep terrain** (radar shadow and layover) is unreliable and should be masked.
- **Use the model that matches your data.** This model expects terrain-corrected γ⁰ (RTC); unprocessed σ⁰ from other sources will behave differently.

## Attribution

- Labels: Sen1Floods11 (Bonafilia et al., 2020, *Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1*, CVPR Workshops), CC BY 4.0.
- Imagery: contains modified Copernicus Sentinel data, processed by Microsoft Planetary Computer (`sentinel-1-rtc`).
- Weights released under CC BY 4.0.
