"""Download monster sprites from maplestory.io into monster/<name>/."""
from __future__ import annotations

import io
import json
import os
import urllib.request
import zipfile

BASE = "https://maplestory.io/api"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "maplebot/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def save_zip_frames(zip_bytes: bytes, out_root: str, name: str) -> int:
    """Write every PNG in the zip except death animations as out_root/name/name_<action>_<i>.png."""
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for info in sorted(z.infolist(), key=lambda i: i.filename):
            fn = info.filename
            if not fn.lower().endswith(".png"):
                continue
            parts = fn.replace("\\", "/").split("/")
            action = parts[-2] if len(parts) >= 2 else "frame"
            if action.lower().startswith("die"):
                continue
            idx = os.path.splitext(parts[-1])[0]
            with z.open(info) as f, open(os.path.join(out_dir, f"{name}_{action}_{idx}.png"), "wb") as o:
                o.write(f.read())
            n += 1
    return n


def fetch_mob(name: str, out_root: str = "monster", region: str = "GMS", version: str = "83") -> int:
    mobs = json.loads(_get(f"{BASE}/{region}/{version}/mob"))
    match = next((m for m in mobs if str(m.get("name", "")).lower() == name.lower()), None)
    if match is None:
        raise LookupError(f"no mob named {name!r} in {region}/{version}; browse https://maplestory.wiki/{region}/{version}/mob")
    data = _get(f"{BASE}/{region}/{version}/mob/{match['id']}/download")
    folder = name.lower().replace(" ", "_")
    return save_zip_frames(data, out_root, folder)
