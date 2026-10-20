# TerraSignal

**Pick any place on Earth and two dates. TerraSignal shows what changed and how confident it is.**

TerraSignal compares free Sentinel-1 radar imagery over time to map floods and other surface change. Radar sees through clouds, which matters because floods arrive under monsoon skies. The pipeline works for any bounding box on the planet. Curated case studies cover India and the rest of the world.

> Status: Phase 1 in progress (flood mapping). The fetch pipeline is done; the baseline and the model are next.

## How it works

```
any bbox + before/after dates
        │
        ▼
STAC search (Planetary Computer) ──► pick a same-orbit Sentinel-1 pair
        │                              (same direction + relative orbit, ≥90% coverage)
        ▼
aligned stack on a local UTM grid, 10 m
  VV/VH before · VV/VH after · Copernicus DEM · JRC permanent water
        │
        ▼
water(after) − water(before) − permanent water − steep/shadowed terrain
        │
        ▼
flood map + confidence + area (km²)
```

Why same-orbit pairs: radar backscatter depends on viewing geometry. Comparing an ascending pass with a descending one makes every hillside look as if it changed.

## Data (all free)

| Layer | Source | Purpose |
|---|---|---|
| Sentinel-1 RTC (VV, VH) | Microsoft Planetary Computer | Radar backscatter that sees through clouds |
| Copernicus DEM GLO-30 | Planetary Computer | Terrain masks (radar shadow, steep slopes) |
| JRC Global Surface Water | Planetary Computer | Permanent rivers and lakes, which are not floods |
| Sen1Floods11 | Google Cloud public bucket | Training and hand-labeled test data |

## Case studies

| Case | Region | Event |
|---|---|---|
| `assam_2024` | India | Brahmaputra monsoon floods |
| `chennai_2023` | India | Cyclone Michaung urban flooding |
| `wayanad_2024` | India | Mundakkai landslide (a non-flood control case) |
| `pakistan_2022` | Pakistan | Indus floods near Manchar Lake |
| `derna_2023` | Libya | Storm Daniel dam-break flash flood (a hard case) |

## Quick start

```bash
python -m uv venv .venv --python 3.12
python -m uv pip install --python .venv/Scripts/python.exe -e ".[dev]"

# a curated case
.venv/Scripts/python.exe -m terrasignal.cli fetch --case wayanad_2024

# anywhere on Earth (area cap: 2,500 km²)
.venv/Scripts/python.exe -m terrasignal.cli fetch --bbox 92.1 26.25 92.5 26.55 --before 2024-03-01 --after 2024-07-04
```

Each run writes `data/<name>/` with one GeoTIFF per layer, `meta.json` (the scenes used, dates, orbit) and `quicklook.png`.

## Roadmap

- **Phase 1: flood mapping**
  - [x] 1.1 Fetch any AOI with same-orbit pairing, aligned DEM and permanent water
  - [ ] 1.2 Otsu baseline with terrain and permanent-water masks
  - [ ] 1.3 U-Net trained on Sen1Floods11, compared to the baseline per country
  - [ ] 1.4 Calibrated confidence (MC dropout + temperature scaling)
  - [ ] 1.5 CLI and API that output COG, GeoJSON and stats
- **Phase 2:** interactive map with a before/after swipe
- **Phase 3:** urban change detection (Sentinel-2, OSCD, Siamese network)
- **Phase 4:** evaluation page and write-up
- **Phase 5:** foundation-model fine-tune (Prithvi) and an Earth Engine global overview

## Hardware

Developed on a GTX 1650 (4 GB VRAM) with 8 GB of RAM. The area cap keeps one request to a few hundred MB in memory.
