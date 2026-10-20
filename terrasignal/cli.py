"""Command line entry point.

  terrasignal fetch --case assam_2024
  terrasignal fetch --bbox 92.1 26.25 92.5 26.55 --before 2024-03-01 --after 2024-07-04
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import yaml

from .aoi import AOI, AOIError
from .fetch import NoPairError, fetch_stack
from .io import save_stack

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
    args = ap.parse_args(argv)
    return _fetch(args)


if __name__ == "__main__":
    sys.exit(main())
