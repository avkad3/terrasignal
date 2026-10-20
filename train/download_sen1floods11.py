"""Download the hand-labeled part of Sen1Floods11 (Bonafilia et al., 2020)
from its public Google Cloud bucket. Resumable: existing files are skipped.

  python train/download_sen1floods11.py
"""
from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BUCKET = "sen1floods11"
PREFIXES = [
    "v1.1/splits/flood_handlabeled/",
    "v1.1/data/flood_events/HandLabeled/S1Hand/",
    "v1.1/data/flood_events/HandLabeled/LabelHand/",
]
OUT = Path(__file__).resolve().parent.parent / "data" / "sen1floods11"


def list_objects(prefix: str) -> list[tuple[str, int]]:
    out, token = [], None
    while True:
        q = {"prefix": prefix, "maxResults": 1000, "fields": "items(name,size),nextPageToken"}
        if token:
            q["pageToken"] = token
        url = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?" + urllib.parse.urlencode(q)
        d = json.load(urllib.request.urlopen(url, timeout=60))
        out += [(i["name"], int(i["size"])) for i in d.get("items", [])]
        token = d.get("nextPageToken")
        if not token:
            return out


def fetch(obj: tuple[str, int]) -> int:
    name, size = obj
    dest = OUT / name.split("HandLabeled/")[-1].replace("v1.1/splits/flood_handlabeled/", "splits/")
    if dest.exists() and dest.stat().st_size == size:
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://storage.googleapis.com/{BUCKET}/{urllib.parse.quote(name)}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        f.write(r.read())
    tmp.replace(dest)
    return size


def main() -> int:
    objs = [o for p in PREFIXES for o in list_objects(p)]
    total = sum(s for _, s in objs)
    print(f"{len(objs)} files, {total / 1e6:.0f} MB -> {OUT}")
    done = 0
    with ThreadPoolExecutor(8) as ex:
        for i, n in enumerate(ex.map(fetch, objs), 1):
            done += n
            if i % 100 == 0 or i == len(objs):
                print(f"  {i}/{len(objs)} files, {done / 1e6:.0f} MB downloaded this run", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
