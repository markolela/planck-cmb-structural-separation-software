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
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PATCHES = REPO / "data" / "processed" / "astro" / "patches"


def fft_gauss_blur(img: np.ndarray, sigma_px: float) -> np.ndarray:
    if sigma_px <= 0.0:
        return img
    h, w = img.shape
    fy = np.fft.fftfreq(h)
    fx = np.fft.fftfreq(w)
    fy2, fx2 = np.meshgrid(fy, fx, indexing="ij")
    g = np.exp(-2.0 * (np.pi**2) * (sigma_px**2) * (fx2**2 + fy2**2))
    f = np.fft.fft2(img)
    return np.fft.ifft2(f * g).real


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Dataset tag. Folder name under patches.")
    ap.add_argument("--target-fwhm-arcmin", type=float, required=True, help="Target beam FWHM in arcmin.")
    ap.add_argument(
        "--assume-native-fwhm-arcmin",
        type=float,
        default=5.0,
        help="Assumed native beam FWHM in arcmin. Planck CMB is about 5 arcmin.",
    )
    ap.add_argument("--out-suffix", default=None, help="Optional output suffix, for example _sm10am.")
    args = ap.parse_args()

    src = PATCHES / args.dataset
    if not src.exists():
        raise SystemExit(f"Missing dataset folder: {src}")

    meta_path = src / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    n = int(meta.get("N", 1024))
    fov_deg = float(meta.get("fov_deg", 12.0))

    # Pixel scale
    pix_arcmin = (fov_deg * 60.0) / n

    # Sigma in pixels
    fwhm_to_sigma = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    sigma_tgt_px = (args.target_fwhm_arcmin / pix_arcmin) * fwhm_to_sigma
    sigma_nat_px = (args.assume_native_fwhm_arcmin / pix_arcmin) * fwhm_to_sigma
    delta_sigma_px = max(0.0, math.sqrt(max(0.0, sigma_tgt_px**2 - sigma_nat_px**2)))

    out_suffix = args.out_suffix or f"_sm{int(round(args.target_fwhm_arcmin))}am"
    dst_tag = args.dataset + out_suffix
    dst = PATCHES / dst_tag
    dst.mkdir(parents=True, exist_ok=True)

    patch_files = sorted(src.glob("patch_*.npy"))
    smoothed: list[np.ndarray] = []

    for pf in patch_files:
        arr = np.load(pf)
        arr_sm = fft_gauss_blur(arr, delta_sigma_px)
        np.save(dst / pf.name, arr_sm.astype(arr.dtype))
        smoothed.append(arr_sm)

    stack_path = dst / f"{dst_tag}_stack.npz"
    if smoothed:
        np.savez_compressed(stack_path, patches=np.stack(smoothed, axis=0))

    meta_out = dict(meta)
    meta_out["dataset"] = dst_tag
    meta_out["derived_from"] = args.dataset
    meta_out["patches"] = [str(dst / pf.name) for pf in patch_files]
    meta_out["stack"] = str(stack_path)
    meta_out["smoothing"] = {
        "type": "gaussian_planar",
        "target_fwhm_arcmin": args.target_fwhm_arcmin,
        "assumed_native_fwhm_arcmin": args.assume_native_fwhm_arcmin,
        "delta_sigma_px": delta_sigma_px,
        "pix_arcmin": pix_arcmin,
    }
    (dst / "meta.json").write_text(json.dumps(meta_out, indent=2), encoding="utf-8")

    print(f"[OK] Smoothing finished: {dst_tag}")
    print(f"     Target FWHM = {args.target_fwhm_arcmin} arcmin. Assumed native = {args.assume_native_fwhm_arcmin} arcmin.")
    print(f"     Delta sigma px = {delta_sigma_px:.3f}. Pixel scale = {pix_arcmin:.3f} arcmin per px.")


if __name__ == "__main__":
    main()