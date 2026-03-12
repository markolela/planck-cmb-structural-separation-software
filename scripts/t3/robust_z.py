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
Robust Z standardization per patch and scale against the chosen null family.

Inputs:
- Real per patch CSV:    <prefix>_metrics_per_patch.csv
- Null per patch CSV:    <prefix>_null[_family]_metrics_per_patch.csv
  required columns at least: rep, patch, s, bytes_per_cell

Definition:
- center(p,s) = median_r( bytes_per_cell_null[p,s,r] )
- mad(p,s)    = median_r( |x - center| )
- scale(p,s)  = 1.4826 * mad(p,s)
- Z(p,s)      = (bytes_per_cell_real[p,s] - center(p,s)) / scale(p,s)

If mad == 0:
- scale is set to EPS and mad0_flag = 1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MAD_TO_SIGMA = 1.4826
EPS = 1e-12
METRIC_COL = "bytes_per_cell"


def _mad_np(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    m = float(np.median(x))
    return float(np.median(np.abs(x - m)))


def _derive_prefix_and_real_path(null_per_patch_csv: Path) -> tuple[str, Path]:
    """
    Derive (prefix, real_per_patch_csv_path) from a null per-patch CSV filename.

    We only require:
    - filename contains "_null"
    - filename contains "_metrics_per_patch"
    - filename ends with ".csv"

    This allows additional suffixes like "_R2" before ".csv".
    Example:
      <prefix>_xz_null_metrics_per_patch.csv
      <prefix>_xz_null_metrics_per_patch_R2.csv

    In both cases, the corresponding real per-patch file is:
      <prefix>_xz_metrics_per_patch.csv
    """
    name = null_per_patch_csv.name

    if not name.endswith(".csv"):
        raise ValueError(f"Expected a .csv file, got {name!r}")

    i_null = name.find("_null")
    if i_null < 0:
        raise ValueError(f"Expected '_null' in filename, got {name!r}")

    i_mpp = name.find("_metrics_per_patch")
    if i_mpp < 0:
        raise ValueError(f"Expected '_metrics_per_patch' in filename, got {name!r}")

    if i_mpp < i_null:
        raise ValueError(f"Filename order invalid (metrics_per_patch before null): {name!r}")

    prefix = name[:i_null]
    real_name = prefix + "_metrics_per_patch.csv"
    real_path = null_per_patch_csv.with_name(real_name)
    return prefix, real_path


def compute_z_from_csvs(
    *,
    real_per_patch_csv: Path,
    null_per_patch_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    real_df = pd.read_csv(real_per_patch_csv)
    null_df = pd.read_csv(null_per_patch_csv)

    required_real = {"patch", "s", METRIC_COL}
    required_null = {"rep", "patch", "s", METRIC_COL}

    miss_real = sorted(required_real - set(real_df.columns))
    miss_null = sorted(required_null - set(null_df.columns))
    if miss_real:
        raise KeyError(f"Missing columns in real CSV: {miss_real}")
    if miss_null:
        raise KeyError(f"Missing columns in null CSV: {miss_null}")

    real_df = real_df[["patch", "s", "n_cg", METRIC_COL]].copy() if "n_cg" in real_df.columns else real_df[["patch", "s", METRIC_COL]].copy()
    null_df = null_df[["rep", "patch", "s", METRIC_COL]].copy()

    real_df["patch"] = real_df["patch"].astype(int)
    real_df["s"] = real_df["s"].astype(int)
    null_df["patch"] = null_df["patch"].astype(int)
    null_df["s"] = null_df["s"].astype(int)
    null_df["rep"] = null_df["rep"].astype(int)

    g = null_df.groupby(["patch", "s"], as_index=False)

    centers = g[METRIC_COL].median().rename(columns={METRIC_COL: "null_center"})
    mads = g[METRIC_COL].apply(lambda x: _mad_np(x.to_numpy())).reset_index().rename(columns={METRIC_COL: "null_mad"})
    stats = centers.merge(mads, on=["patch", "s"], how="inner")

    stats["null_scale"] = (MAD_TO_SIGMA * stats["null_mad"]).astype(float)
    stats["mad0_flag"] = (stats["null_mad"].to_numpy(dtype=np.float64) == 0.0).astype(int)

    stats.loc[stats["null_scale"].to_numpy(dtype=np.float64) <= 0.0, "null_scale"] = float(EPS)

    z_df = real_df.merge(stats, on=["patch", "s"], how="inner")

    z_df["real_bpc"] = z_df[METRIC_COL].astype(float)
    z_df["Z_bpc"] = (z_df["real_bpc"] - z_df["null_center"].astype(float)) / z_df["null_scale"].astype(float)

    keep = ["patch", "s"]
    if "n_cg" in z_df.columns:
        keep.append("n_cg")
    keep += ["real_bpc", "null_center", "null_mad", "null_scale", "mad0_flag", "Z_bpc"]
    z_df = z_df[keep].sort_values(["patch", "s"]).reset_index(drop=True)

    z_by_patch = z_df.groupby("patch", as_index=False).agg(
        Z_s_med=("Z_bpc", "median"),
        Z_s_mad=("Z_bpc", lambda x: _mad_np(x.to_numpy()) * MAD_TO_SIGMA),
        mad0_any=("mad0_flag", "max"),
        n_scales=("s", "nunique"),
    )

    runinfo = {
        "metric_col": METRIC_COL,
        "mad_to_sigma": MAD_TO_SIGMA,
        "eps": EPS,
        "n_patches": int(z_by_patch.shape[0]),
        "n_scales": int(z_df["s"].nunique()),
        "null_reps_per_patch_in_file": int(null_df["rep"].nunique()),
    }

    return z_df, z_by_patch, runinfo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--null-per-patch", type=str, required=True, help="Path to *_null*_metrics_per_patch.csv")
    args = ap.parse_args()

    null_path = Path(args.null_per_patch).expanduser().resolve()
    if not null_path.exists():
        raise FileNotFoundError(null_path)

    prefix, real_path = _derive_prefix_and_real_path(null_path)
    if not real_path.exists():
        raise FileNotFoundError(f"Derived real per-patch CSV not found: {real_path}")

    z_df, z_by_patch, runinfo = compute_z_from_csvs(real_per_patch_csv=real_path, null_per_patch_csv=null_path)

    outdir = null_path.parent
    out_scale = outdir / f"{prefix}_zbpc_per_patch_scale.csv"
    out_patch = outdir / f"{prefix}_zbpc_by_patch.csv"
    out_json = outdir / f"{prefix}_zbpc_runinfo.json"

    z_df.to_csv(out_scale, index=False)
    z_by_patch.to_csv(out_patch, index=False)
    out_json.write_text(json.dumps(runinfo, indent=2), encoding="utf-8")

    mad0_cnt = int((z_df["mad0_flag"].to_numpy(dtype=int) > 0).sum())
    print(f"OK: real={real_path.name}")
    print(f"OK: null={null_path.name}")
    print(f"Wrote: {out_scale}")
    print(f"Wrote: {out_patch}")
    print(f"Wrote: {out_json}")
    print(f"mad0_flag_count_rows={mad0_cnt} of {int(z_df.shape[0])}")


if __name__ == "__main__":
    main()
