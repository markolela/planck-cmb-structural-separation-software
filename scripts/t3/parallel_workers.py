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

from dataclasses import dataclass, field
from pathlib import Path

import os
import time
from datetime import datetime

from threading import Event, Thread, Lock

import csv

import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed

from scripts.t3.metrics import kappa_table_for_patch
from scripts.t3.surrogates import (
    rng_for_null,
    phase_surrogate2d,
    rotation90_surrogate2d,
    aaft_surrogate2d,
    iaaft_surrogate2d,
    ring_phase_shuffle_surrogate2d,
)
from scripts.t3.synfast_null import (
    sample_gnomonic_patch_healpy,
    synfast_map_from_cl,
)


def make_surrogate_patch_for_null(
    *,
    null_family: str,
    p: Path,
    idx: int,
    r: int,
    null_seed: int,
    paths: list[Path],
    synfast_map: np.ndarray | None,
    synfast_nside: int | None,
    centers_for_synfast: list[dict] | None,
    patch_N_for_synfast: int | None,
    patch_fov_for_synfast: float | None,
) -> tuple[np.ndarray, int | None]:
    """
    Create a surrogate patch for the null ensemble.
    Returns (surrogate_patch, src_idx). src_idx is only set for so3_rotation.
    """
    if null_family == "gaussian_forward_synfast":
        if centers_for_synfast is None or synfast_map is None or synfast_nside is None:
            raise ValueError("gaussian_forward_synfast: centers_for_synfast or synfast_map is missing.")
        if patch_N_for_synfast is None or patch_fov_for_synfast is None:
            raise ValueError("gaussian_forward_synfast: patch geometry is missing.")

        c = centers_for_synfast[int(idx)]
        if "lon_deg" in c and "lat_deg" in c:
            lon0 = float(c["lon_deg"])
            lat0 = float(c["lat_deg"])
        elif "lon" in c and "lat" in c:
            lon0 = float(c["lon"])
            lat0 = float(c["lat"])
        else:
            raise KeyError(f"Center dict has no lon/lat keys. Keys: {sorted(c.keys())}")

        sur = sample_gnomonic_patch_healpy(
            synfast_map,
            synfast_nside,
            lon0_deg=lon0,
            lat0_deg=lat0,
            N=patch_N_for_synfast,
            fov_deg=patch_fov_for_synfast,
        )
        return sur, None

    arr = np.load(p)

    rng_pr = rng_for_null(null_seed, null_family, idx, r)
    if null_family == "phase_randomized":
        sur = phase_surrogate2d(arr, rng_pr)
    elif null_family == "rotation90":
        sur = rotation90_surrogate2d(arr, rng_pr)
    elif null_family == "aaft":
        sur = aaft_surrogate2d(arr, rng_pr)
    elif null_family == "iaaft":
        sur = iaaft_surrogate2d(arr, rng_pr, n_iter=20)
    elif null_family == "ring_phase_shuffle":
        sur = ring_phase_shuffle_surrogate2d(arr, rng_pr)
    else:
        raise ValueError(f"Unknown null_family: {null_family}")

    return sur, None


def _filtered_kappa_kwargs(d: dict[str, object]) -> dict[str, object]:
    """
    Extra keyword arguments forwarded to kappa_table_for_patch.

    Important: Do not allow keys that are fixed by the worker itself.
    Otherwise we would pass duplicates and risk TypeError.
    """
    if not d:
        return {}
    banned = {"s_levels", "bpc0_cache"}
    return {k: v for k, v in d.items() if k not in banned}


@dataclass(frozen=True)
class NullRepTask:
    ds: str
    r: int
    null_seed: int
    null_family: str
    scales: tuple[int, ...]
    agg_stat: str
    paths: tuple[str, ...]
    out_csv: str

    synfast_centers: list[dict] | None
    patch_N_for_synfast: int | None
    patch_fov_for_synfast: float | None
    synfast_cl: np.ndarray | None
    synfast_nside: int | None

    kappa_kwargs: dict[str, object] = field(default_factory=dict)
    jobs_inner: int = 1


def run_null_rep_task(task: NullRepTask) -> tuple[int, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    paths = [Path(p) for p in task.paths]
    n_patches = int(len(paths))
    n_scales = int(len(task.scales))

    hb_s = float(os.environ.get("T3_HEARTBEAT_S", "60"))
    hb_stop: Event | None = None
    hb_thread: Thread | None = None
    hb_lock: Lock | None = None
    hb_done = 0
    hb_t0 = time.perf_counter()

    if hb_s > 0:
        hb_stop = Event()
        hb_lock = Lock()

        def _hb() -> None:
            while hb_stop is not None and not hb_stop.wait(hb_s):
                if hb_lock is not None:
                    with hb_lock:
                        done_local = int(hb_done)
                else:
                    done_local = int(hb_done)

                elapsed_s = time.perf_counter() - hb_t0
                rate = done_local / elapsed_s if elapsed_s > 0 else 0.0
                eta_s = (n_patches - done_local) / rate if rate > 0 else float("inf")

                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"[{now}] (+{elapsed_s:8.1f}s) [NULL] "
                    f"rep={int(task.r)} pid={os.getpid()} patches={done_local}/{n_patches} "
                    f"elapsed={elapsed_s:7.0f}s eta={eta_s:7.0f}s",
                    flush=True,
                )

        hb_thread = Thread(target=_hb, daemon=True)
        hb_thread.start()

    try:
        synfast_map: np.ndarray | None = None
        synfast_seed32: int | None = None
        if task.null_family == "gaussian_forward_synfast":
            if task.synfast_cl is None or task.synfast_nside is None:
                raise ValueError("gaussian_forward_synfast: synfast_cl or synfast_nside is None.")
            rng_map = rng_for_null(task.null_seed, task.null_family, -1, task.r)
            synfast_seed32 = int(rng_map.integers(0, 2**32 - 1, dtype=np.uint32))
            synfast_map = synfast_map_from_cl(task.synfast_cl, task.synfast_nside, synfast_seed32)

        kappa_patch = np.empty((n_patches, n_scales), dtype=np.float64)
        alpha_patch = np.empty((n_patches, n_scales), dtype=np.float64)
        omegaP_patch = np.empty((n_patches, n_scales), dtype=np.float64)
        omegaS_patch = np.empty((n_patches, n_scales), dtype=np.float64)

        out_csv_path = Path(task.out_csv)
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)

        use_inner = (task.null_family == "gaussian_forward_synfast") and (int(task.jobs_inner) > 1)
        jobs_inner_eff = int(task.jobs_inner)
        jobs_inner_eff = max(1, min(int(jobs_inner_eff), int(n_patches)))

        results_by_idx: list[tuple[pd.DataFrame, int | None] | None] = [None] * int(n_patches)

        # Stagger patch order across reps to avoid all processes hammering patch_00.npy on /mnt/c at once.
        patch_order = list(range(int(n_patches)))
        shift = int(task.r) % int(n_patches) if int(n_patches) > 0 else 0
        if shift:
            patch_order = patch_order[shift:] + patch_order[:shift]

        def _compute_one_patch(i: int) -> tuple[int, pd.DataFrame, int | None]:
            p = paths[int(i)]
            sur, src_idx = make_surrogate_patch_for_null(
                null_family=task.null_family,
                p=p,
                idx=int(i),
                r=int(task.r),
                null_seed=int(task.null_seed),
                paths=paths,
                synfast_map=synfast_map,
                synfast_nside=task.synfast_nside,
                centers_for_synfast=task.synfast_centers,
                patch_N_for_synfast=task.patch_N_for_synfast,
                patch_fov_for_synfast=task.patch_fov_for_synfast,
            )

            local_cache: dict[tuple[int, int, int, int], float] = {}
            df = kappa_table_for_patch(
                sur,
                s_levels=task.scales,
                bpc0_cache=local_cache,
                **_filtered_kappa_kwargs(dict(task.kappa_kwargs)),
            )

            return int(i), df, src_idx

        if use_inner:
            with ThreadPoolExecutor(max_workers=int(jobs_inner_eff)) as ex:
                futs = {ex.submit(_compute_one_patch, int(i)): int(i) for i in range(int(n_patches))}
                for fut in as_completed(futs):
                    i, df, src_idx = fut.result()
                    results_by_idx[int(i)] = (df, src_idx)

                    if hb_lock is not None:
                        with hb_lock:
                            hb_done += 1
        else:
            done_count = 0
            for i in patch_order:
                i2, df, src_idx = _compute_one_patch(int(i))
                results_by_idx[int(i2)] = (df, src_idx)

                done_count += 1
                if hb_lock is not None:
                    with hb_lock:
                        hb_done = int(done_count)

        if any(x is None for x in results_by_idx):
            raise RuntimeError("Null patch computation failed for at least one patch.")

        writer: csv.DictWriter | None = None
        with out_csv_path.open("w", newline="", encoding="utf-8") as f:
            for idx in range(int(n_patches)):
                df, src_idx = results_by_idx[int(idx)]  # type: ignore[misc]

                kappa_patch[int(idx), :] = df["kappa_nat_corr"].to_numpy(dtype=np.float64)
                alpha_patch[int(idx), :] = df["alpha_lmw"].to_numpy(dtype=np.float64)
                omegaP_patch[int(idx), :] = df["omegaP"].to_numpy(dtype=np.float64)
                omegaS_patch[int(idx), :] = df["omegaS"].to_numpy(dtype=np.float64)

                df = df.copy()
                df.insert(0, "dataset", str(task.ds))
                df.insert(1, "rep", int(task.r))
                df.insert(2, "patch", int(idx))

                if task.null_family == "gaussian_forward_synfast":
                    df.insert(3, "synfast_seed32", int(synfast_seed32 if synfast_seed32 is not None else -1))

                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(df.columns))
                    writer.writeheader()

                writer.writerows(df.to_dict(orient="records"))

        if task.agg_stat == "median":
            kappa_agg = np.nanmedian(kappa_patch, axis=0)
            alpha_agg = np.nanmedian(alpha_patch, axis=0)
            omegaP_agg = np.nanmedian(omegaP_patch, axis=0)
            omegaS_agg = np.nanmedian(omegaS_patch, axis=0)
        elif task.agg_stat == "mean":
            kappa_agg = np.nanmean(kappa_patch, axis=0)
            alpha_agg = np.nanmean(alpha_patch, axis=0)
            omegaP_agg = np.nanmean(omegaP_patch, axis=0)
            omegaS_agg = np.nanmean(omegaP_patch, axis=0)
        else:
            raise ValueError(f"agg_stat must be 'mean' or 'median', got {task.agg_stat!r}")

        return int(task.r), str(out_csv_path), kappa_agg, alpha_agg, omegaP_agg, omegaS_agg

    finally:
        if hb_stop is not None:
            hb_stop.set()
        if hb_thread is not None:
            hb_thread.join(timeout=2.0)


def concat_rep_csvs(rep_files: list[Path], out_file: Path) -> None:
    if not rep_files:
        raise ValueError("No rep CSV files to concatenate.")

    with out_file.open("w", encoding="utf-8", newline="") as out:
        for i, p in enumerate(rep_files):
            with p.open("r", encoding="utf-8") as inp:
                if i == 0:
                    out.write(inp.read())
                else:
                    inp.readline()
                    out.write(inp.read())


_NULL_CTX: dict[str, object] = {}

def init_null_rep_worker(init_ctx: dict[str, object]) -> None:
    global _NULL_CTX
    _NULL_CTX = init_ctx


def run_null_rep_worker(r: int, out_csv: str) -> tuple[int, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ctx = _NULL_CTX
    task = NullRepTask(
        ds=str(ctx["ds"]),
        r=int(r),
        null_seed=int(ctx["null_seed"]),
        null_family=str(ctx["null_family"]),
        scales=tuple(ctx["scales"]),  # type: ignore[arg-type]
        agg_stat=str(ctx["agg_stat"]),
        paths=tuple(ctx["paths"]),  # type: ignore[arg-type]
        out_csv=str(out_csv),
        kappa_kwargs=dict(ctx.get("kappa_kwargs", {})),
        synfast_centers=ctx["synfast_centers"],  # type: ignore[assignment]
        patch_N_for_synfast=ctx["patch_N_for_synfast"],  # type: ignore[assignment]
        patch_fov_for_synfast=ctx["patch_fov_for_synfast"],  # type: ignore[assignment]
        synfast_cl=ctx["synfast_cl"],  # type: ignore[assignment]
        synfast_nside=ctx["synfast_nside"],  # type: ignore[assignment]
        jobs_inner=int(_NULL_CTX.get("jobs_inner", 1)),
    )
    return run_null_rep_task(task)


@dataclass(frozen=True)
class PatchTask:
    idx: int
    path: str
    scales: tuple[int, ...]
    kappa_kwargs: dict[str, object] = field(default_factory=dict)


def run_patch_task(task: PatchTask) -> tuple[int, pd.DataFrame]:
    arr = np.load(task.path)
    df = kappa_table_for_patch(
        arr,
        s_levels=task.scales,
        bpc0_cache={},
        **_filtered_kappa_kwargs(dict(task.kappa_kwargs)),
    )
    df["patch"] = int(task.idx)
    return int(task.idx), df
