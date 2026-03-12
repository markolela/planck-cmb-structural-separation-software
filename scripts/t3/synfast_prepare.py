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
Helpers for preparing gaussian_forward_synfast null ensembles.
New code is written in English by convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np

from scripts.t3.synfast_null import (
    read_healpix_map_single_for_cl,
    estimate_pseudo_cl_from_source,
    synfast_cl_cache_path,
    try_load_synfast_cl_cache,
    write_synfast_cl_cache,
)


class SynfastPrepResult(NamedTuple):
    centers_for_synfast: list[dict]
    patch_N: int
    patch_fov_deg: float
    cl: np.ndarray
    nside: int
    f_sky: float
    lmax: int
    lat_cut_deg: float | None
    cache_file: Path
    cache_hit: bool


def prepare_gaussian_forward_synfast(
    *,
    meta: dict,
    subset: list[int] | None,
    n_patches: int,
    outdir: Path,
    dataset: str,
    repo_root: Path,
    log: Callable[[str], None],
) -> SynfastPrepResult:
    centers_raw = (
        meta.get("centers")
        or meta.get("patch_centers")
        or meta.get("centers_deg")
        or meta.get("patch_centers_deg")
    )
    if not isinstance(centers_raw, list) or not centers_raw:
        raise ValueError(
            "gaussian_forward_synfast requires centers in meta.json. "
            "Expected keys: centers or patch_centers."
        )

    if subset:
        centers_for_synfast = [centers_raw[i] for i in subset]
    else:
        centers_for_synfast = centers_raw

    if len(centers_for_synfast) != int(n_patches):
        raise ValueError(
            "gaussian_forward_synfast: centers count does not match patches count. "
            f"len(centers)={len(centers_for_synfast)}, n_patches={int(n_patches)}"
        )

    if "N" not in meta or "fov_deg" not in meta:
        raise ValueError("gaussian_forward_synfast requires meta['N'] and meta['fov_deg'].")

    patch_N = int(meta["N"])
    patch_fov_deg = float(meta["fov_deg"])

    if "source" not in meta or not meta["source"]:
        raise ValueError("gaussian_forward_synfast requires meta['source'].")

    src_path = Path(str(meta["source"]))
    if not src_path.is_absolute():
        cand = repo_root / src_path
        if cand.exists():
            src_path = cand

    lat_cut_for_cl = meta.get("lat_cut_deg", None)
    lat_cut_deg = None if lat_cut_for_cl is None else float(lat_cut_for_cl)

    cache_file = synfast_cl_cache_path(outdir, dataset, lat_cut_deg)

    cl: np.ndarray | None = None
    f_sky: float | None = None
    lmax: int | None = None
    nside_src: int | None = None
    cache_hit = False

    if cache_file.exists():
        cl0, nside0, f_sky0, lmax0, hit = try_load_synfast_cl_cache(cache_file, src_path, lat_cut_deg)
        if hit:
            cache_hit = True
            cl = np.asarray(cl0, dtype=np.float64)
            nside_src = int(nside0)
            f_sky = float(f_sky0)
            lmax = int(lmax0)

    if cl is None or f_sky is None or lmax is None or nside_src is None:
        vals_src, nside_src2 = read_healpix_map_single_for_cl(src_path)
        cl2, f_sky2, lmax2 = estimate_pseudo_cl_from_source(vals_src, nside_src2, lat_cut_deg)

        cl = np.asarray(cl2, dtype=np.float64)
        nside_src = int(nside_src2)
        f_sky = float(f_sky2)
        lmax = int(lmax2)

        write_synfast_cl_cache(cache_file, cl, nside_src, f_sky, lmax, src_path, lat_cut_deg)
        cache_hit = False

    ch = "HIT" if cache_hit else "MISS"
    log(
        f"[NULL] synfast prepared. cache={ch}, file={cache_file.name}, "
        f"nside={int(nside_src)}, lmax={int(lmax)}, f_sky={float(f_sky):.3f}"
    )

    return SynfastPrepResult(
        centers_for_synfast=centers_for_synfast,
        patch_N=patch_N,
        patch_fov_deg=patch_fov_deg,
        cl=cl,
        nside=int(nside_src),
        f_sky=float(f_sky),
        lmax=int(lmax),
        lat_cut_deg=lat_cut_deg,
        cache_file=cache_file,
        cache_hit=bool(cache_hit),
    )
