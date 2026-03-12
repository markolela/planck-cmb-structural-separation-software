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
Compressor aggregation based on robust Z scores.

Inputs in the same dataset directory:
- <prefix>_zbpc_by_patch.csv
- <prefix>_xz_zbpc_by_patch.csv
- <prefix>_zstd_zbpc_by_patch.csv

Each file is produced by scripts/t3/robust_z.py.

Per patch p:
- Z_s(p,c) is stored as Z_s_med in the input files.
- Z_pc(p) = median_c Z_s(p,c) for c in {gzip, xz, zstd}
- S_pc(p) = MAD_c Z_s(p,c)  (unscaled MAD, exactly as pre-registered)

Consistency rule (fixed):
- Sign consistency across all three compressors.
- Spread threshold: S_pc <= 1.0
- robust_flag = sign_consistent and S_pc <= 1.0
"""


from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SPREAD_THRESH = 1.0


def _mad_1d(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    m = float(np.median(x))
    mad = float(np.median(np.abs(x - m)))
    return float(mad)


def _sign3(a: float, b: float, c: float) -> int:
    sa = 1 if a > 0 else (-1 if a < 0 else 0)
    sb = 1 if b > 0 else (-1 if b < 0 else 0)
    sc = 1 if c > 0 else (-1 if c < 0 else 0)
    if sa == sb and sb == sc:
        return sa
    return 9


def _load_one(path: Path, comp: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"patch", "Z_s_med", "Z_s_mad", "mad0_any", "n_scales"}
    miss = sorted(need - set(df.columns))
    if miss:
        raise KeyError(f"Missing columns in {path.name}: {miss}")

    out = df[["patch", "Z_s_med", "Z_s_mad", "mad0_any", "n_scales"]].copy()
    out["patch"] = out["patch"].astype(int)

    out = out.rename(
        columns={
            "Z_s_med": f"Z_{comp}",
            "Z_s_mad": f"Zs_mad_{comp}",
            "mad0_any": f"mad0_any_{comp}",
            "n_scales": f"n_scales_{comp}",
        }
    )
    return out.sort_values("patch").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=str, required=True)
    ap.add_argument("--prefix", type=str, required=True)
    args = ap.parse_args()

    ddir = Path(args.dataset_dir).expanduser().resolve()
    if not ddir.exists():
        raise FileNotFoundError(ddir)

    prefix = str(args.prefix).strip()

    f_gzip = ddir / f"{prefix}_zbpc_by_patch.csv"
    f_xz = ddir / f"{prefix}_xz_zbpc_by_patch.csv"
    f_zstd = ddir / f"{prefix}_zstd_zbpc_by_patch.csv"

    for f in [f_gzip, f_xz, f_zstd]:
        if not f.exists():
            raise FileNotFoundError(f)

    g = _load_one(f_gzip, "gzip")
    x = _load_one(f_xz, "xz")
    z = _load_one(f_zstd, "zstd")

    m = g.merge(x, on="patch", how="inner").merge(z, on="patch", how="inner")

    if m.shape[0] != g.shape[0] or m.shape[0] != x.shape[0] or m.shape[0] != z.shape[0]:
        raise RuntimeError("Patch join is incomplete. Patch IDs do not match across inputs.")

    Zg = m["Z_gzip"].to_numpy(dtype=np.float64)
    Zx = m["Z_xz"].to_numpy(dtype=np.float64)
    Zz = m["Z_zstd"].to_numpy(dtype=np.float64)

    Z_stack = np.vstack([Zg, Zx, Zz]).T
    Z_pc = np.median(Z_stack, axis=1)
    S_pc = np.apply_along_axis(_mad_1d, 1, Z_stack)

    sign_code = np.array([_sign3(a, b, c) for a, b, c in zip(Zg, Zx, Zz)], dtype=int)
    sign_consistent = (sign_code != 9)
    robust_flag = sign_consistent & (S_pc <= float(SPREAD_THRESH))

    out = pd.DataFrame(
        {
            "patch": m["patch"].astype(int),
            "Z_gzip": Zg,
            "Z_xz": Zx,
            "Z_zstd": Zz,
            "Z_pc_med": Z_pc,
            "S_pc": S_pc,
            "sign_consistent": sign_consistent.astype(int),
            "robust_flag": robust_flag.astype(int),
            "spread_thresh": float(SPREAD_THRESH),
            "mad0_any_gzip": m["mad0_any_gzip"].astype(int),
            "mad0_any_xz": m["mad0_any_xz"].astype(int),
            "mad0_any_zstd": m["mad0_any_zstd"].astype(int),
        }
    ).sort_values("patch").reset_index(drop=True)

    out_csv = ddir / f"{prefix}_zpc_by_patch.csv"
    out_json = ddir / f"{prefix}_zpc_runinfo.json"

    runinfo = {
        "prefix": prefix,
        "files": {
            "gzip": f_gzip.name,
            "xz": f_xz.name,
            "zstd": f_zstd.name,
        },
        "spread_thresh": SPREAD_THRESH,
        "n_patches": int(out.shape[0]),
        "robust_count": int(out["robust_flag"].sum()),
        "sign_inconsistent_count": int((out["sign_consistent"] == 0).sum()),
        "spread_fail_count": int(((out["sign_consistent"] == 1) & (out["robust_flag"] == 0)).sum()),
        "mad0_any_counts": {
            "gzip": int(out["mad0_any_gzip"].sum()),
            "xz": int(out["mad0_any_xz"].sum()),
            "zstd": int(out["mad0_any_zstd"].sum()),
        },
    }

    out.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(runinfo, indent=2), encoding="utf-8")

    print(f"OK: {out_csv}")
    print(f"OK: {out_json}")
    print(f"robust_count={runinfo['robust_count']} of {runinfo['n_patches']}")
    print(f"sign_inconsistent_count={runinfo['sign_inconsistent_count']}")
    print(f"spread_fail_count={runinfo['spread_fail_count']}")


if __name__ == "__main__":
    main()
