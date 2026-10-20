"""Command line entry point.

  terrasignal fetch --case assam_2024
  terrasignal fetch --bbox 92.1 26.25 92.5 26.55 --before 2024-03-01 --after 2024-07-04
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import yaml

from .aoi import AOI, AOIError
from .fetch import NoPairError, fetch_stack
from .baseline import run_baseline
from .io import flood_overlay, load_stack, save_classes, save_stack

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "cases" / "cases.yaml"


def load_cases() -> dict:
    return yaml.safe_load(CASES.read_text())


def _fetch(args) -> int:
    if args.case:
        cases = load_cases()
        if args.case not in cases:
            print(f"unknown case {args.case!r}; known: {', '.join(cases)}", file=sys.stderr)
            return 2
        c = cases[args.case]
        bbox, before, after, name = c["bbox"], c["before"], c["after"], args.case
    else:
        if not (args.bbox and args.before and args.after):
            print("give --case, or all of --bbox, --before, --after", file=sys.stderr)
            return 2
        bbox, name = args.bbox, args.name or "custom"
        before, after = date.fromisoformat(args.before), date.fromisoformat(args.after)
    try:
        aoi = AOI.from_list(bbox)
        print(f"[{name}] AOI {aoi.area_km2:.0f} km², EPSG:{aoi.utm_epsg}; searching Sentinel-1…")
        t = time.time()
        ds = fetch_stack(aoi, before, after, resolution=args.resolution, window_days=args.window)
    except (AOIError, NoPairError, ValueError) as e:
        print(f"[{name}] error: {e}", file=sys.stderr)
        return 1
    out = save_stack(ds, Path(args.out) / name)
    a = ds.attrs
    print(
        f"[{name}] pair {a['before_date']} → {a['after_date']} "
        f"({a['orbit_state']} orbit {a['relative_orbit']}), grid {ds.sizes['y']}×{ds.sizes['x']}, "
        f"{time.time() - t:.0f}s → {out}"
    )
    return 0


def _baseline(args) -> int:
    names = list(load_cases()) if args.case == "all" else [args.case]
    data = Path(args.data)
    for name in names:
        folder = data / name
        try:
            ds = load_stack(folder)
        except FileNotFoundError as e:
            print(f"[{name}] {e}", file=sys.stderr)
            return 1
        t = time.time()
        classes, stats = run_baseline(
            ds.vv_before.values, ds.vv_after.values, ds.dem.values,
            ds.jrc_occurrence.values, float(ds.attrs["resolution"]),
        )
        save_classes(classes, ds.vv_after, folder / "baseline_flood.tif")
        (folder / "baseline_stats.json").write_text(json.dumps(stats, indent=2))
        title = load_cases().get(name, {}).get("title", name)
        flood_overlay(ds, classes, stats, title, folder / "baseline.png")
        tb, ta = stats["threshold_before"], stats["threshold_after"]
        print(
            f"[{name}] flood {stats['flood_km2']:.2f} km² of {stats['valid_km2']:.0f} km² · "
            f"thresholds {tb['value_db']:.1f} ({tb['method']},{tb['n_tiles']}) / "
            f"{ta['value_db']:.1f} ({ta['method']},{ta['n_tiles']}) dB · "
            f"steep {stats['masked_steep_fraction']:.0%} · {time.time() - t:.0f}s"
        )
    return 0


def _detect(args) -> int:
    from .detect import DetectParams, run_model
    from .model import load_checkpoint

    model, meta = load_checkpoint(args.ckpt, "cuda" if _cuda() else "cpu")
    names = list(load_cases()) if args.case == "all" else [args.case]
    for name in names:
        folder = Path(args.data) / name
        try:
            ds = load_stack(folder)
        except FileNotFoundError as e:
            print(f"[{name}] {e}", file=sys.stderr)
            return 1
        t = time.time()
        classes, stats, layers = run_model(
            model, meta, ds.vv_before.values, ds.vh_before.values, ds.vv_after.values,
            ds.vh_after.values, ds.dem.values, ds.jrc_occurrence.values,
            float(ds.attrs["resolution"]), DetectParams(),
        )
        save_classes(classes, ds.vv_after, folder / "model_flood.tif")
        conf = layers["confidence"]
        pct = np.where(np.isfinite(conf), np.round(np.nan_to_num(conf) * 100), 255).astype(np.uint8)
        save_classes(pct, ds.vv_after, folder / "model_confidence.tif")  # % new-water probability
        (folder / "model_stats.json").write_text(json.dumps(stats, indent=2))
        title = load_cases().get(name, {}).get("title", name)
        flood_overlay(ds, classes, stats, title, folder / "model.png", label="Hybrid")
        mc = stats["open_water_mean_confidence"]
        print(f"[{name}] open-water flood {stats['flood_km2']:.2f} km²"
              f"{f' (mean confidence {mc:.0%})' if mc is not None else ''} + probable "
              f"{stats['probable_flood_km2']:.2f} km² of {stats['valid_km2']:.0f} km² · "
              f"steep {stats['masked_steep_fraction']:.0%} · {time.time() - t:.0f}s")
    return 0


def _analyze(args) -> int:
    from .analyze import analyze, analyze_stack

    if args.case:
        cases = load_cases()
        if args.case not in cases:
            print(f"unknown case {args.case!r}; known: {', '.join(cases)}", file=sys.stderr)
            return 2
        c = cases[args.case]
        bbox, before, after, name = c["bbox"], c["before"], c["after"], args.case
    elif args.bbox and args.before and args.after:
        bbox, name = args.bbox, args.name or "custom"
        before, after = date.fromisoformat(args.before), date.fromisoformat(args.after)
    else:
        print("give --case, or all of --bbox, --before, --after", file=sys.stderr)
        return 2
    out = Path(args.out) if args.out else Path(args.data) / name / "analysis"
    ckpt = None if args.method == "baseline" else args.ckpt  # str: may be hf://user/repo
    stack_dir = Path(args.data) / name
    try:
        if args.case and not args.refetch and (stack_dir / "meta.json").exists():
            print(f"[{name}] reusing fetched stack in {stack_dir}")
            s = analyze_stack(load_stack(stack_dir), out, ckpt, args.method)
        else:
            print(f"[{name}] fetching and analyzing…")
            s = analyze(bbox, before, after, out, ckpt, args.method, args.window)
    except (AOIError, NoPairError, ValueError, FileNotFoundError) as e:
        print(f"[{name}] error: {e}", file=sys.stderr)
        return 1
    ar = s["areas_km2"]
    conf = s["open_water_mean_confidence"]
    conf_txt = f" ({conf:.0%} conf.)" if conf is not None else ""
    prob_txt = f" + probable {ar['probable_flood']:.1f} km²" if ar.get("probable_flood") else ""
    print(f"[{name}] {s['before']['date']} → {s['after']['date']} · {s['method']} · "
          f"open-water flood {ar['open_water_flood']:.1f} km²{conf_txt}{prob_txt} · "
          f"{s['polygons_written']} polygons → {out}")
    for c in s["caveats"]:
        print(f"  note: {c}")
    return 0


def _cuda() -> bool:
    import torch
    return torch.cuda.is_available()


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap =argparse.ArgumentParser(prog="terrasignal")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download an aligned before/after stack")
    f.add_argument("--case")
    f.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    f.add_argument("--before")
    f.add_argument("--after")
    f.add_argument("--name")
    f.add_argument("--resolution", type=float, default=10.0)
    f.add_argument("--window", type=int, default=14, help="days around each date to search")
    f.add_argument("--out", default=str(ROOT / "data"))
    b = sub.add_parser("baseline", help="classical Otsu flood map for a fetched stack")
    b.add_argument("--case", required=True, help="case name, a custom --name, or 'all'")
    b.add_argument("--data", default=str(ROOT / "data"))
    d = sub.add_parser("detect", help="U-Net flood map for a fetched stack")
    d.add_argument("--case", required=True, help="case name, a custom --name, or 'all'")
    d.add_argument("--data", default=str(ROOT / "data"))
    d.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "water_unet.pt"))
    a = sub.add_parser("analyze", help="fetch + detect + map-ready outputs (COG, GeoJSON, summary)")
    a.add_argument("--case")
    a.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    a.add_argument("--before")
    a.add_argument("--after")
    a.add_argument("--name")
    a.add_argument("--method", choices=["auto", "hybrid", "baseline"], default="auto")
    a.add_argument("--window", type=int, default=14)
    a.add_argument("--refetch", action="store_true", help="ignore a cached stack for --case")
    a.add_argument("--data", default=str(ROOT / "data"))
    a.add_argument("--out")
    a.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "water_unet.pt"))
    args = ap.parse_args(argv)
    return {"fetch": _fetch, "baseline": _baseline, "detect": _detect, "analyze": _analyze}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
