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

"""
Build patch datasets from half mission difference maps.

Goal.
Create a deterministic patch dataset for the half mission half difference map.
This is a standard negative control in CMB analyses.

Definition.
We build the difference map as:
  diff = 0.5 * (HM1 - HM2)

We then extract patches at the same patch centers as an existing base patch dataset.
This ensures that only the underlying sky map changes, not the patch geometry.

Outputs.
data/processed/astro/patches/<out_dataset>/
  - patch_XX.npy files
  - meta.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

import shutil

from hashlib import sha256

REPO = Path(__file__).resolve().parents[1]
PATCH_ROOT = REPO / "data" / "processed" / "astro" / "patches"


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _write_json(p: Path, obj: dict) -> None:
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def _first_key(meta: dict, keys: tuple[str, ...]):
    """Return the first existing meta value for a list of candidate keys."""
    for k in keys:
        if k in meta and meta[k] is not None:
            return meta[k]
    return None


def _require_number(meta: dict, keys: tuple[str, ...], name: str) -> float:
    """Read a numeric value from meta under multiple possible key names."""
    v = _first_key(meta, keys)
    if v is None:
        raise SystemExit(
            f"Base meta.json missing required numeric field {name!r}. "
            f"Tried keys: {keys}. Available keys: {sorted(meta.keys())}"
        )
    return float(v)


def _read_centers_any(meta: dict) -> list[dict]:
    """
    Read centers from meta.json with backward compatible key variants.
    Returns a list of dicts that must contain lon_deg and lat_deg.
    """
    centers_raw = _first_key(meta, ("centers", "patch_centers", "centers_deg", "patch_centers_deg"))
    if not isinstance(centers_raw, list) or not centers_raw:
        raise SystemExit(
            "Base meta.json does not contain a valid center list. "
            "Expected keys: centers, patch_centers, centers_deg, patch_centers_deg. "
            f"Available keys: {sorted(meta.keys())}"
        )

    # Normalize: accept list[dict] or list[(lon,lat)].
    if isinstance(centers_raw[0], dict):
        if "lon_deg" not in centers_raw[0] or "lat_deg" not in centers_raw[0]:
            raise SystemExit("Center dicts must contain lon_deg and lat_deg.")
        return [dict(c) for c in centers_raw]

    if isinstance(centers_raw[0], (list, tuple)) and len(centers_raw[0]) >= 2:
        return [{"lon_deg": float(x[0]), "lat_deg": float(x[1])} for x in centers_raw]

    raise SystemExit("Unsupported center format in base meta.json.")


def _sha256_file(path: Path) -> str:
    """Compute SHA256 over a file in a streaming fashion for reproducibility metadata."""
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dataset", required=True, help="Existing patch dataset id used only for centers and geometry.")
    ap.add_argument("--hm1-fits", required=True, help="Path to HM1 FITS file.")
    ap.add_argument("--hm2-fits", required=True, help="Path to HM2 FITS file.")
    ap.add_argument("--out-dataset", required=True, help="Output patch dataset id.")
    ap.add_argument("--factor", type=float, default=0.5, help="Scale factor for (HM1 - HM2). Default 0.5.")
    args = ap.parse_args()

    base_tag = str(args.base_dataset).strip()
    out_tag = str(args.out_dataset).strip()

    base_dir = PATCH_ROOT / base_tag
    if not base_dir.exists():
        raise SystemExit(f"Base patch dataset not found: {base_dir}")

    base_meta_path = base_dir / "meta.json"
    if not base_meta_path.exists():
        raise SystemExit(f"Base meta.json not found: {base_meta_path}")

    base_meta = _read_json(base_meta_path)
    base_frame = str(base_meta.get("frame", "")).strip().lower()

    if not base_frame:
        raise SystemExit("Base meta.json has no frame field. Abort to avoid silent frame mismatch.")

    centers = _read_centers_any(base_meta)
    n_patches = int(base_meta.get("n_patches", len(centers)))

    N = int(_require_number(base_meta, ("N", "patch_N", "n_pix"), name="patch size N"))
    fov_deg = _require_number(base_meta, ("fov_deg", "fov", "patch_fov_deg"), name="field of view (deg)")
    lat_cut_deg = float(base_meta.get("lat_cut_deg", 0.0))

    interp = str(base_meta.get("interp", "bilinear"))

    hm1 = Path(args.hm1_fits).resolve()
    hm2 = Path(args.hm2_fits).resolve()
    if not hm1.exists():
        raise SystemExit(f"HM1 FITS not found: {hm1}")
    if not hm2.exists():
        raise SystemExit(f"HM2 FITS not found: {hm2}")

    # Import existing map and projection utilities from your patch builder.
    # This keeps the sampling and projection consistent with the pipeline.
    from scripts.make_real_patches_standalone import (  # type: ignore
        read_healpix_map_single,
        hp_from_meta,
        _prepare_sampler,
        center_coord,
        gnomonic_patch,
    )

    v1, nside1, order1, frame1 = read_healpix_map_single(hm1)
    v2, nside2, order2, frame2 = read_healpix_map_single(hm2)

    if (nside1 != nside2) or (order1 != order2) or (frame1 != frame2) or (v1.size != v2.size):
        raise SystemExit(
            "HM1 and HM2 maps are not compatible. "
            f"HM1: nside={nside1}, order={order1}, frame={frame1}, npix={v1.size}. "
            f"HM2: nside={nside2}, order={order2}, frame={frame2}, npix={v2.size}."
        )

    hm_frame = str(frame1).strip().lower()
    if base_frame and hm_frame and base_frame != hm_frame:
        raise SystemExit(
            f"Frame mismatch. base_meta frame={base_frame}, HM frame={hm_frame}. "
            "Centers are interpreted in the base frame. Abort."
        )

    diff = float(args.factor) * (np.asarray(v1, dtype=np.float64) - np.asarray(v2, dtype=np.float64))

    hp_obj = hp_from_meta(nside1, order1, frame1)
    sampler = _prepare_sampler(hp_obj, diff, interp=interp)

    out_dir = PATCH_ROOT / out_tag
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    patch_paths: list[str] = []
    out_centers: list[dict] = []

    print(f"== build_hm_diff_patches :: out={out_tag} :: base={base_tag} ==")
    print(f"HM1: {hm1}")
    print(f"HM2: {hm2}")
    print(f"diff = {args.factor} * (HM1 - HM2)")
    print(f"NSIDE={nside1} ORDER={order1} FRAME={frame1} N={N} fov_deg={fov_deg} n_patches={n_patches}")

    if len(centers) != n_patches:
        print(f"[WARN] len(centers)={len(centers)} differs from n_patches={n_patches}. Using centers length.")
        n_patches = len(centers)

    for i in range(n_patches):
        c = centers[i]
        L = float(c["lon_deg"])
        B = float(c["lat_deg"])
        ctr = center_coord(L, B, frame1)

        patch = gnomonic_patch(
            diff,
            hp_obj,
            ctr,
            N=N,
            fov_deg=fov_deg,
            sampler=sampler,
            so3_R_inv=None,
        ).astype(np.float32)

        if not np.all(np.isfinite(patch)):
            raise SystemExit(f"Non finite values in patch. i={i} center=({L:.6f},{B:.6f})")

        fn = out_dir / f"patch_{i:02d}.npy"
        np.save(fn, patch)
        patch_paths.append(str(fn))
        out_centers.append(dict(c))

        if i < 3 or i == n_patches - 1:
            print(f"  patch_{i:02d}: center=({L:.2f},{B:.2f}) deg -> {fn.name}")

    meta_out = dict(base_meta)
    meta_out["dataset"] = out_tag
    meta_out["tag"] = out_tag
    meta_out["source"] = ""
    meta_out["source_note"] = "Half mission difference patches built from hm1 and hm2."

    meta_out["derived_from"] = {
        "base_patch_dataset": base_tag,
        "base_meta_sha256": sha256(base_meta_path.read_bytes()).hexdigest(),
        "hm1_fits": str(hm1),
        "hm2_fits": str(hm2),
        "hm1_sha256": _sha256_file(hm1),
        "hm2_sha256": _sha256_file(hm2),
        "operation": "factor * (HM1 - HM2)",
        "factor": float(args.factor),
        "interp": interp,
    }
    meta_out["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta_out["nside"] = int(nside1)
    meta_out["order"] = str(order1)
    meta_out["frame"] = str(frame1)
    meta_out["lat_cut_deg"] = float(lat_cut_deg)
    meta_out["N"] = int(N)
    meta_out["fov_deg"] = float(fov_deg)
    meta_out["interp"] = str(interp)

    meta_out["centers"] = out_centers
    meta_out["patches"] = patch_paths
    meta_out.pop("stack", None)

    meta_out["n_patches"] = int(n_patches)

    _write_json(out_dir / "meta.json", meta_out)
    print("[OK] HM difference patch dataset written.")
    print(out_dir)


if __name__ == "__main__":
    main()