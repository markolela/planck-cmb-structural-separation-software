# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Marko Lela (ORCID: 0009-0008-0768-5184)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PATCHD = REPO / "data" / "processed" / "astro" / "patches"


def read_meta(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_min_abs_b_from_tag(tag: str) -> Optional[float]:
    m = re.search(r"_b(\d+)_", tag)
    return float(m.group(1)) if m else None


def extract_b_list(meta: Dict[str, Any], tag: str) -> List[float]:
    """
    Robust extraction of Galactic latitude b in degrees.

    Priority order.
    1. meta["patches"] entries that contain b or latitude fields.
    2. meta["centers"] entries if they are dicts in a Galactic frame.
    3. fallback to common "centers_*" list formats that store lon, lat pairs.

    Then reduce to the actually used subset.
    1. If meta["subset"]["indices"] exists, index by it.
    2. Else fallback to the |b| threshold parsed from the dataset tag, for example "_b65_".
    3. Else return all b values.
    """
    pl = meta.get("patches")

    if isinstance(pl, list) and pl and isinstance(pl[0], dict):
        b_all: List[float] = []
        for p in pl:
            for key in ("b_deg", "lat_deg", "gal_b_deg", "galactic_b_deg", "glat_deg"):
                if key in p:
                    b_all.append(float(p[key]))
                    break
            else:
                if (
                    "center_gal_deg" in p
                    and isinstance(p["center_gal_deg"], (list, tuple))
                    and len(p["center_gal_deg"]) >= 2
                ):
                    b_all.append(float(p["center_gal_deg"][1]))
                elif (
                    "center" in p
                    and isinstance(p["center"], (list, tuple))
                    and len(p["center"]) >= 2
                    and str(p.get("coord", "galactic")).lower().startswith("gal")
                ):
                    b_all.append(float(p["center"][1]))
                else:
                    raise KeyError("Found a patch entry without a usable b or latitude field.")
    else:
        centers = meta.get("centers")

        if isinstance(centers, list) and centers and isinstance(centers[0], dict):
            frame = str(meta.get("frame", "")).lower()
            if "gal" in frame and "lat_deg" in centers[0]:
                b_all = [float(c["lat_deg"]) for c in centers]
            else:
                raise KeyError("centers exist, but the frame is not Galactic or lat_deg is missing.")
        else:
            b_all = []
            for k in (
                "centers_gal_deg",
                "centers_galactic_deg",
                "centers_gal_lonlat_deg",
                "centers_deg",
                "centers_lonlat_deg",
            ):
                v = meta.get(k)
                if isinstance(v, list) and v and isinstance(v[0], (list, tuple)) and len(v[0]) >= 2:
                    b_all = [float(p[1]) for p in v]
                    break
            if not b_all:
                raise KeyError("Could not extract Galactic latitudes b from meta.json.")

    sub = meta.get("subset") or {}
    idx = sub.get("indices")
    if isinstance(idx, list) and idx and isinstance(idx[0], int):
        return [b_all[i] for i in idx]

    bmin = _parse_min_abs_b_from_tag(tag)
    if bmin is not None:
        return [b for b in b_all if abs(b) >= bmin]

    return b_all


def subset_and_write(src_tag: str, hemi: str) -> str:
    src = PATCHD / src_tag
    if not src.exists():
        raise FileNotFoundError(f"Missing patch folder: {src}")

    meta = read_meta(src / "meta.json")
    b_list = extract_b_list(meta, src_tag)

    patch_files = sorted(
        src.glob("patch_*.npy"),
        key=lambda p: int(re.search(r"patch_(\d+)", p.stem).group(1))
        if re.search(r"patch_(\d+)", p.stem)
        else p.name,
    )

    if len(patch_files) != len(b_list):
        print(
            f"[DBG] src_tag={src_tag} subset={meta.get('subset', {})} parsed_bmin={_parse_min_abs_b_from_tag(src_tag)}"
        )

    if len(patch_files) != len(b_list):
        raise RuntimeError(f"Patch count ({len(patch_files)}) does not match b count ({len(b_list)}).")

    keep_idx = [i for i, b in enumerate(b_list) if (b >= 0.0) == (hemi == "N")]
    if not keep_idx:
        raise RuntimeError(f"Hemisphere {hemi}: no patches selected.")

    dst_tag = f"{src_tag}_hemi{hemi}"
    dst = PATCHD / dst_tag
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    stack: List[np.ndarray] = []
    for i in keep_idx:
        arr = np.load(patch_files[i])
        stack.append(arr)
        np.save(dst / patch_files[i].name, arr)

    np.savez_compressed(dst / f"{dst_tag}_stack.npz", patches=np.stack(stack, axis=0))

    meta_out = dict(meta)
    meta_out["dataset"] = dst_tag
    meta_out["parent_tag"] = src_tag
    meta_out["subset"] = {
        "type": "hemisphere",
        "rule": "gal_b>=0" if hemi == "N" else "gal_b<0",
        "parent_indices": keep_idx,
        "count": len(keep_idx),
        "source": "jackknife_hemi_t3",
    }
    meta_out["hemi"] = hemi

    bmin = int(_parse_min_abs_b_from_tag(src_tag) or 0)
    meta_out["mask_label"] = meta.get("mask_label", f"|b| >= {bmin}; FOV 12 deg")
    meta_out["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    (dst / "meta.json").write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
    return dst_tag


def run_t3(tag: str, trend: str, scales: str, null_n: int, null_seed: int) -> None:
    cmd = [
        sys.executable,
        "-m",
        "scripts.run_t3_on_patches",
        "--dataset",
        tag,
        "--scales",
        scales,
        "--trend",
        trend,
        "--agg",
        "median",
        "--jobs-data",
        "20",
        "--null",
        str(null_n),
        "--null-seed",
        str(null_seed),
        "--null-family",
        "phase_randomized",
        "--jobs-null",
        "20",
    ]
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=REPO)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True, help="Comma separated list: <tag1>,<tag2>,...")
    ap.add_argument("--trend", default="log")
    ap.add_argument("--scales", default="1,2,4")
    ap.add_argument("--null-n", type=int, default=200)
    ap.add_argument("--null-seed", type=int, default=12345)
    args = ap.parse_args()

    tags = [t.strip() for t in args.datasets.split(",") if t.strip()]
    for base_tag in tags:
        print(f"[INFO] Jackknife for: {base_tag}")
        n_tag = subset_and_write(base_tag, "N")
        s_tag = subset_and_write(base_tag, "S")
        for t in (n_tag, s_tag):
            run_t3(t, args.trend, args.scales, args.null_n, args.null_seed)


if __name__ == "__main__":
    main()