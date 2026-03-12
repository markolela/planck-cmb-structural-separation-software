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
Run T3/ARL-RG on real astro patches (Planck/WMAP/HI4PI).
Reads patch datasets produced by make_real_patches_standalone.py (meta.json),
aggregates across patches, fits the trend (log|inv|auto), computes the plateau,
and writes CSV/PNG/JSON to data/processed/astro/suite/<dataset>/.

Notes:
- Surrogate null test (--null N, --null-seed): phase surrogates per patch.
  Outputs *_null_metrics{_per_patch}.csv, *_null_report.png, *_null_summary.json
- Bugfix: unify ds/dataset_name usage in run_dataset()
- Plateau metric made robust (MAD-like via range / |mean|)
- AIC-based trend selection for --trend auto
"""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

import argparse
import json
import csv
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

# Robust default for headless/WSL runs.
# If the user explicitly sets MPLBACKEND, we respect it.
_is_wsl = bool(os.environ.get("WSL_DISTRO_NAME")) or os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop")
if not os.environ.get("MPLBACKEND"):
    if _is_wsl or (not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY")):
        matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

from hashlib import sha256

import time
from datetime import datetime

from scripts.t3.paths import REPO, PATCH, OUT, load_manifest, load_dataset_meta, resolve_patch_paths
from scripts.t3.metrics import S_LEVELS, kappa_table_for_patch, fit_trend, plateau_score
from scripts.t3.encoding import EncodingSpec
from scripts.t3.compressors import CompressorSpec
from scripts.t3.compressors import get_versions_snapshot

from scripts.t3.synfast_prepare import prepare_gaussian_forward_synfast
from scripts.t3.parallel_workers import (
    PatchTask,
    run_patch_task,
    NullRepTask,
    run_null_rep_task,
    concat_rep_csvs,
    init_null_rep_worker,
    run_null_rep_worker,
)


# ----------------------- Core run per dataset -----------------------

def _aggregate(df_all: pd.DataFrame, *, statistic: str = "mean") -> pd.DataFrame:
    if statistic not in ("mean", "median"):
        raise ValueError(f"statistic must be 'mean' or 'median', got {statistic!r}")

    cent = statistic  # pandas agg supports "mean" and "median" as strings

    return df_all.groupby("s", as_index=False).agg(
        n_cg=("n_cg", "first"),
        bytes_per_cell=("bytes_per_cell", cent),
        bpc_baseline=("bpc_baseline", cent),
        kappa_nat=("kappa_nat", cent),
        kappa_nat_corr=("kappa_nat_corr", cent),
        entropy_nat=("entropy_nat", cent),
        alpha_lmw=("alpha_lmw", cent),
        omegaP=("omegaP", cent),
        omegaS=("omegaS", cent),
        kappa_nat_std=("kappa_nat", "std"),
        kappa_nat_corr_std=("kappa_nat_corr", "std"),
        alpha_lmw_std=("alpha_lmw", "std"),
    )


def _plot_and_write(ds: str, outdir: Path, agg: pd.DataFrame, theta: float, chosen: str,
                    pscore: float, suffix: str = "") -> Path:
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    s_arr = agg["s"].to_numpy()

    # Panel 1: scale function with error bars and trend subtraction
    yerr = np.nan_to_num(agg["kappa_nat_corr_std"].to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    ax[0].errorbar(s_arr,
                   agg["kappa_nat_corr"].to_numpy(),
                   yerr=yerr, fmt="o-", capsize=3, label="kappa_nat_corr")
    ax[0].plot(s_arr, agg["kappa_theta"].to_numpy(), "s--", label="kappa_theta")
    ax[0].set_xlabel("s"); ax[0].set_ylabel("Nat/cell"); ax[0].legend()
    ax[0].set_title("Scale function")

    # Panel 2: LMW contraction
    ax[1].plot(s_arr, agg["alpha_lmw"].to_numpy(), "o-")
    ax[1].set_xlabel("s"); ax[1].set_ylabel("alpha_LMW")
    ax[1].set_title("LMW contraction")

    # Panel 3: long-range coupling (Ω proxies)
    ax[2].plot(s_arr, agg["omegaP"].to_numpy(), "o-", label="Ω Pearson")
    ax[2].plot(s_arr, agg["omegaS"].to_numpy(), "s--", label="Ω Spearman")
    ax[2].set_xlabel("s"); ax[2].set_ylabel("Long-range coupling")
    ax[2].set_title("LRC proxies"); ax[2].legend()

    title_tag = " (NULL)" if suffix else ""
    fig.suptitle(
        f"T3 on patches. {ds}{title_tag}\nθ({chosen})={theta:+.4f}, PP={pscore*100.0:.2f}%",
        y=0.98,
    )
    fig.tight_layout()
    png_path = outdir / f"{ds}{suffix}_report.png"
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    return png_path


def run_dataset(
    dataset_name: str,
    outdir: Path,
    scales,
    trend: str,
    agg_stat: str = "median",
    null_n: int = 0,
    null_seed: int = 20251,
    null_family: str = "phase_randomized",
    jobs_data: int = 0,
    jobs_null: int = 0,
    jobs_null_inner: int = 0,
    compressor: str = "gzip",
) -> None:
    ds = dataset_name  # use consistently

    compressor = str(compressor).strip().lower()

    if compressor not in ("gzip", "xz", "zstd"):
        raise ValueError(f"compressor must be one of gzip,xz,zstd, got {compressor!r}")

    level_by_name = {"gzip": 6, "xz": 6, "zstd": 10}
    enc_spec = EncodingSpec()
    comp_spec = CompressorSpec(compressor, int(level_by_name[compressor]), 1)

    kappa_kwargs: dict[str, object] = dict(
        enc_spec=enc_spec,
        comp_spec=comp_spec,
    )

    comp_suffix = "" if compressor == "gzip" else f"_{compressor}"

    if len(scales) < 2:
        raise ValueError("Need at least two scales for theta and plateau, e.g. scales=(1,2,4)")
    meta = load_dataset_meta(ds)

    meta_path = PATCH / ds / "meta.json"
    meta_sha12 = sha256(meta_path.read_bytes()).hexdigest()[:12]

    # Try to extract a meaningful mask/subset label
    mask_label = (
        meta.get("mask_label")
        or meta.get("mask")
        or (meta.get("subset") or {}).get("label")
        or (meta.get("params") or {}).get("label")
        or ""
    )

    # Patch paths (make robust)
    paths = resolve_patch_paths(ds, meta)
    subset = meta.get("subset", {}).get("indices")
    if subset:
        paths = [paths[i] for i in subset]
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------------- Original ----------------
    bpc0_cache: dict[tuple[int, int, int, int], float] = {}

    if not paths:
        raise ValueError(f"No patch files found for dataset: {ds}")

    patch_jobs = int(jobs_data) if int(jobs_data) > 0 else (os.cpu_count() or 1)
    patch_jobs = max(1, min(int(patch_jobs), int(len(paths))))

    if patch_jobs == 1:
        tables: list[pd.DataFrame] = []
        for idx, p in enumerate(paths):
            arr = np.load(p)
            df = kappa_table_for_patch(arr, s_levels=scales, bpc0_cache=bpc0_cache, **kappa_kwargs)
            df["patch"] = idx
            tables.append(df)
    else:
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"[DATA] Starting patch processing. dataset={ds}, patches={len(paths)}, jobs={patch_jobs}",
            flush=True,
        )

        tasks = [
            PatchTask(
                idx=int(i),
                path=str(p),
                scales=tuple(int(x) for x in scales),
                kappa_kwargs=dict(kappa_kwargs),
            )
            for i, p in enumerate(paths)
        ]

        tables_by_idx: list[pd.DataFrame | None] = [None] * int(len(paths))
        with ProcessPoolExecutor(max_workers=int(patch_jobs)) as ex:
            futs = {ex.submit(run_patch_task, t): int(t.idx) for t in tasks}
            for fut in as_completed(futs):
                idx, df = fut.result()
                tables_by_idx[int(idx)] = df

        if any(x is None for x in tables_by_idx):
            raise RuntimeError("Patch processing failed for at least one patch.")

        tables = [x for x in tables_by_idx if x is not None]

    df_all = pd.concat(tables, ignore_index=True)
    agg = _aggregate(df_all, statistic=agg_stat)

    # Trendabzug
    ktheta, theta, chosen = fit_trend(agg, ycol="kappa_nat_corr", trend=trend)
    agg["kappa_theta"] = ktheta
    pscore = plateau_score(agg["kappa_theta"], n_cg=agg["n_cg"], min_ncg=64)
    plateau_pct = float(pscore * 100.0)

    # Speichern
    per_patch_csv = outdir / f"{ds}{comp_suffix}_metrics_per_patch.csv"
    agg_csv = outdir / f"{ds}{comp_suffix}_metrics.csv"

    df_all.to_csv(per_patch_csv, index=False)
    agg.to_csv(agg_csv, index=False)

    # Plot & JSON
    png_path = _plot_and_write(ds, outdir, agg, theta, chosen, pscore, suffix=comp_suffix)
    summary = {
        "dataset": ds,
        "trend": chosen,
        "theta": float(theta),
        "pp": float(pscore),
        "plateau_pct": float(plateau_pct),
        "s_levels": list(map(int, agg["s"].to_numpy())),
        "alpha_lmw_max": float(agg["alpha_lmw"].max()),
        "omegaP_max": float(agg["omegaP"].max()),
        "omegaS_max": float(agg["omegaS"].max()),
        "csv_per_patch": str(per_patch_csv),
        "csv_agg": str(agg_csv),
        "png": str(png_path),
        "n_patches": int(len(paths)),
        "subset_size": int(len(meta.get("subset", {}).get("indices", [])) or len(paths)),
        "mask_label": mask_label,
        "meta_sha12": meta_sha12,

        "compressor_set": ["gzip", "xz", "zstd"],
        "compressor_params": {
            "gzip": {"level": 6, "threads": 1},
            "xz": {"level": 6, "threads": 1},
            "zstd": {"level": 10, "threads": 1},
        },
        "selected_compressor": str(comp_spec.name),
        "selected_compressor_spec": {
            "name": str(comp_spec.name),
            "level": int(comp_spec.level),
            "threads": int(comp_spec.threads),
        },
        "compressor_versions": get_versions_snapshot(
            [
                CompressorSpec("gzip", 6, 1),
                CompressorSpec("xz", 6, 1),
                CompressorSpec("zstd", 10, 1),
            ]
        ),
        "encoding_spec": json.loads(enc_spec.to_json()),
    }

    (outdir / f"{ds}{comp_suffix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Console output
    print(f"\n=== {ds} ===")
    print(agg[["s","n_cg","kappa_nat_corr","kappa_theta","alpha_lmw","omegaP","omegaS"]].to_string(index=False))
    print(f"θ ({chosen}): {theta:.6f}   PP: {pscore*100.0:.2f}%")
    print(f"CSV: {agg_csv}")
    print(f"PNG: {png_path}")

    # ---------------- Surrogate-Nulltest (optional) ----------------
    if null_n and null_n > 0:
        t0 = time.perf_counter()

        def _log(msg: str) -> None:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            dt = time.perf_counter() - t0
            print(f"[{now}] (+{dt:8.1f}s) {msg}", flush=True)

        # Freeze the trend for the null. With auto, the choice is determined by the data.
        trend_for_null = chosen

        # Dateinamen wie vorher.
        suffix = f"{comp_suffix}_null" if null_family == "phase_randomized" else f"{comp_suffix}_null_{null_family}"
        per_patch_csv_null = outdir / f"{ds}{suffix}_metrics_per_patch.csv"
        agg_csv_null = outdir / f"{ds}{suffix}_metrics.csv"
        effects_csv_null = outdir / f"{ds}{suffix}_effects.csv"

        n_patches = int(len(paths))
        n_scales = int(len(scales))

        centers_for_synfast: list[dict] | None = None
        patch_N_for_synfast: int | None = None
        patch_fov_for_synfast: float | None = None
        synfast_cl: np.ndarray | None = None
        synfast_nside: int | None = None
        synfast_f_sky: float | None = None
        synfast_lmax: int | None = None
        synfast_lat_cut_deg: float | None = None
        synfast_cl_cache_file: Path | None = None
        synfast_cl_cache_hit: bool | None = None

        if str(null_family) == "gaussian_forward_synfast":
            prep = prepare_gaussian_forward_synfast(
                meta=meta,
                subset=subset,
                n_patches=n_patches,
                outdir=outdir,
                dataset=ds,
                repo_root=REPO,
                log=_log,
            )

            centers_for_synfast = prep.centers_for_synfast
            patch_N_for_synfast = prep.patch_N
            patch_fov_for_synfast = prep.patch_fov_deg
            synfast_cl = prep.cl
            synfast_nside = prep.nside
            synfast_f_sky = prep.f_sky
            synfast_lmax = prep.lmax
            synfast_lat_cut_deg = prep.lat_cut_deg
            synfast_cl_cache_file = prep.cache_file
            synfast_cl_cache_hit = prep.cache_hit

        # Reference s and n_cg once.
        df_ref = kappa_table_for_patch(np.load(paths[0]), s_levels=scales, bpc0_cache={}, **kappa_kwargs)
        s_by_scale = df_ref["s"].to_numpy(dtype=np.int64)
        n_cg_by_scale = df_ref["n_cg"].to_numpy(dtype=np.int64)

        # Target arrays. Each repetition is already aggregated over patches.
        kappa_corr_rep = np.empty((int(null_n), int(n_scales)), dtype=np.float64)
        alpha_rep = np.empty((int(null_n), int(n_scales)), dtype=np.float64)
        omegaP_rep = np.empty((int(null_n), int(n_scales)), dtype=np.float64)
        omegaS_rep = np.empty((int(null_n), int(n_scales)), dtype=np.float64)

        jobs_eff = int(jobs_null) if int(jobs_null) > 0 else (os.cpu_count() or 1)
        jobs_eff = max(1, min(int(jobs_eff), int(null_n)))

        jobs_inner_eff = 1
        if str(null_family) == "gaussian_forward_synfast":
            if int(jobs_null_inner) > 0:
                jobs_inner_eff = int(jobs_null_inner)
            else:
                if int(jobs_eff) > 1:
                    jobs_inner_eff = 1
                else:
                    jobs_inner_eff = int(os.cpu_count() or 1)

            jobs_inner_eff = max(1, min(int(jobs_inner_eff), int(n_patches)))
            _log(f"[NULL] gaussian_forward_synfast. jobs_outer={jobs_eff}, jobs_inner={jobs_inner_eff}")

        tmp_dir = outdir / f".tmp_{ds}{suffix}_per_patch"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        rep_files: list[Path | None] = [None] * int(null_n)
        paths_str = tuple(str(p) for p in paths)

        tasks: list[NullRepTask] = []
        for r in range(int(null_n)):
            rep_csv = tmp_dir / f"rep_{r:06d}.csv"
            tasks.append(
                NullRepTask(
                    ds=str(ds),
                    r=int(r),
                    null_seed=int(null_seed),
                    null_family=str(null_family),
                    scales=tuple(int(x) for x in scales),
                    agg_stat=str(agg_stat),
                    paths=paths_str,
                    out_csv=str(rep_csv),
                    kappa_kwargs=dict(kappa_kwargs),
                    synfast_centers=centers_for_synfast,
                    patch_N_for_synfast=patch_N_for_synfast,
                    patch_fov_for_synfast=patch_fov_for_synfast,
                    synfast_cl=synfast_cl,
                    synfast_nside=synfast_nside,
                    jobs_inner=int(jobs_inner_eff),
                )
            )

        if jobs_eff == 1:
            for t in tasks:
                r, rep_csv, kappa_a, alpha_a, omegaP_a, omegaS_a = run_null_rep_task(t)
                kappa_corr_rep[r, :] = kappa_a
                alpha_rep[r, :] = alpha_a
                omegaP_rep[r, :] = omegaP_a
                omegaS_rep[r, :] = omegaS_a
                rep_files[r] = Path(rep_csv)

                if r % 10 == 0:
                    _log(f"[NULL] rep {r}/{null_n}")
        else:
            _log(f"[NULL] parallel. jobs={jobs_eff}, reps={null_n}, patches={n_patches}")
            done = 0

            init_ctx: dict[str, object] = dict(
                ds=str(ds),
                null_seed=int(null_seed),
                null_family=str(null_family),
                scales=tuple(int(x) for x in scales),
                agg_stat=str(agg_stat),
                paths=paths_str,
                kappa_kwargs=dict(kappa_kwargs),
                synfast_centers=centers_for_synfast,
                patch_N_for_synfast=patch_N_for_synfast,
                patch_fov_for_synfast=patch_fov_for_synfast,
                synfast_cl=synfast_cl,
                synfast_nside=synfast_nside,
                jobs_inner=int(jobs_inner_eff),
            )

            with ProcessPoolExecutor(
                max_workers=int(jobs_eff),
                initializer=init_null_rep_worker,
                initargs=(init_ctx,),
            ) as ex:
                futs = {}
                for r in range(int(null_n)):
                    rep_csv = tmp_dir / f"rep_{r:06d}.csv"
                    futs[ex.submit(run_null_rep_worker, int(r), str(rep_csv))] = int(r)

                for fut in as_completed(futs):
                    r, rep_csv, kappa_a, alpha_a, omegaP_a, omegaS_a = fut.result()
                    kappa_corr_rep[r, :] = kappa_a
                    alpha_rep[r, :] = alpha_a
                    omegaP_rep[r, :] = omegaP_a
                    omegaS_rep[r, :] = omegaS_a
                    rep_files[r] = Path(rep_csv)

                    done += 1
                    if done % max(1, int(jobs_eff)) == 0:
                        _log(f"[NULL] completed {done}/{null_n}")


        rep_files_ok = [p for p in rep_files if p is not None]
        if len(rep_files_ok) != int(null_n):
            raise ValueError("Not all null reps were produced.")

        rep_files_ok_sorted = [Path(tmp_dir / f"rep_{r:06d}.csv") for r in range(int(null_n))]
        concat_rep_csvs(rep_files_ok_sorted, per_patch_csv_null)

        shutil.rmtree(tmp_dir, ignore_errors=True)

        # theta and PP per rep.
        thetas = np.empty(null_n, dtype=np.float64)
        pps = np.empty(null_n, dtype=np.float64)

        for r in range(null_n):
            df_r = pd.DataFrame({
                "s": s_by_scale.astype(int),
                "n_cg": n_cg_by_scale.astype(int),
                "kappa_nat_corr": kappa_corr_rep[r, :].astype(float),
            })
            ktheta_r, theta_r, _ = fit_trend(df_r, ycol="kappa_nat_corr", trend=trend_for_null)
            pp_r = plateau_score(ktheta_r, n_cg=n_cg_by_scale, min_ncg=64)

            thetas[r] = float(theta_r)
            pps[r] = float(pp_r)

        # Null references and effect distributions.
        theta_data = float(theta)
        pp_data = float(pscore)
        pp_data_pct = float(pp_data * 100.0)

        theta_null_ref = float(np.median(thetas))
        pp_null_ref = float(np.median(pps))
        pp_null_ref_pct = float(pp_null_ref * 100.0)

        dtheta_med = float(theta_null_ref - theta_data)

        dtheta_dist = thetas - theta_data

        dpp_dist_pct = (pp_data - pps) * 100.0
        dpp_med = float(pp_data_pct - pp_null_ref_pct)

        dtheta_ci95 = [float(np.quantile(dtheta_dist, 0.025)), float(np.quantile(dtheta_dist, 0.975))]
        dpp_ci95 = [float(np.quantile(dpp_dist_pct, 0.025)), float(np.quantile(dpp_dist_pct, 0.975))]

        # Representative null curve. Median over reps.
        kappa_corr_med_over_reps = np.nanmedian(kappa_corr_rep, axis=0)
        alpha_med_over_reps = np.nanmedian(alpha_rep, axis=0)
        omegaP_med_over_reps = np.nanmedian(omegaP_rep, axis=0)
        omegaS_med_over_reps = np.nanmedian(omegaS_rep, axis=0)

        # Std over reps like pandas std, ddof=1.
        # For null_n == 1, ddof=1 would yield NaN because degrees of freedom are missing.
        # For null_n == 1, we use ddof=0, so std = 0 and the error band is defined.
        ddof_std = 1 if int(null_n) > 1 else 0
        kappa_corr_std_over_reps = np.nanstd(kappa_corr_rep, axis=0, ddof=ddof_std)

        agg_null = pd.DataFrame({
            "s": s_by_scale.astype(int),
            "n_cg": n_cg_by_scale.astype(int),
            "kappa_nat_corr": kappa_corr_med_over_reps.astype(float),
            "kappa_nat_corr_std": kappa_corr_std_over_reps.astype(float),
            "alpha_lmw": alpha_med_over_reps.astype(float),
            "omegaP": omegaP_med_over_reps.astype(float),
            "omegaS": omegaS_med_over_reps.astype(float),
        })

        ktheta_n, theta_n_curve, chosen_n = fit_trend(agg_null, ycol="kappa_nat_corr", trend=trend_for_null)
        agg_null["kappa_theta"] = ktheta_n
        pscore_n = plateau_score(agg_null["kappa_theta"], n_cg=agg_null["n_cg"], min_ncg=64)

        # Effekte als CSV.
        effects_df = pd.DataFrame({
            "dataset": [str(ds)] * int(null_n),
            "rep": np.arange(null_n, dtype=int),
            "theta_r": thetas.astype(float),
            "pp_r": pps.astype(float),
            "dtheta_r_minus_data": dtheta_dist.astype(float),
            "dpp_data_minus_r": dpp_dist_pct.astype(float),
        })
        effects_df.to_csv(effects_csv_null, index=False)

        # Aggregierte Null CSV.
        agg_null.insert(0, "dataset", str(ds))
        agg_null.to_csv(agg_csv_null, index=False)

        # Plot and JSON for null.
        png_null = _plot_and_write(ds, outdir, agg_null, theta_n_curve, chosen_n, pscore_n, suffix=suffix)
        summary_null = {
            "dataset": ds,
            "is_null": True,
            "null_family": str(null_family),
            "null_reps_per_patch": int(null_n),
            "null_seed": int(null_seed),
            "trend_used": str(chosen_n),
            "theta": float(theta_null_ref),
            "plateau_pct": float(pp_null_ref_pct),
            "theta_data": float(theta_data),
            "pp_data": float(pp_data),
            "theta_null_ref_median_over_reps": float(theta_null_ref),
            "pp_null_ref_median_over_reps": float(pp_null_ref),
            "dtheta_med": float(dtheta_med),
            "dtheta_ci95": dtheta_ci95,
            "dpp_med": float(dpp_med),
            "dpp_ci95": dpp_ci95,
            "csv_per_patch": str(per_patch_csv_null),
            "csv_agg": str(agg_csv_null),
            "csv_effects": str(effects_csv_null),
            "png": str(png_null),

            "compressor_set": ["gzip", "xz", "zstd"],
            "compressor_params": {
                "gzip": {"level": 6, "threads": 1},
                "xz": {"level": 6, "threads": 1},
                "zstd": {"level": 10, "threads": 1},
            },
            "selected_compressor": str(comp_spec.name),
            "selected_compressor_spec": {
                "name": str(comp_spec.name),
                "level": int(comp_spec.level),
                "threads": int(comp_spec.threads),
            },
            "compressor_versions": get_versions_snapshot(
                [
                    CompressorSpec("gzip", 6, 1),
                    CompressorSpec("xz", 6, 1),
                    CompressorSpec("zstd", 10, 1),
                ]
            ),
            "encoding_spec": json.loads(enc_spec.to_json()),
        }

        if null_family == "gaussian_forward_synfast":
            summary_null["synfast_nside"] = int(synfast_nside if synfast_nside is not None else -1)
            summary_null["synfast_lmax"] = int(synfast_lmax if synfast_lmax is not None else -1)
            summary_null["synfast_f_sky"] = float(synfast_f_sky if synfast_f_sky is not None else float("nan"))
            summary_null["synfast_lat_cut_deg"] = None if synfast_lat_cut_deg is None else float(synfast_lat_cut_deg)
            summary_null["synfast_cl_cache_file"] = None if synfast_cl_cache_file is None else str(synfast_cl_cache_file)
            summary_null["synfast_cl_cache_hit"] = None if synfast_cl_cache_hit is None else bool(synfast_cl_cache_hit)

        (outdir / f"{ds}{suffix}_summary.json").write_text(json.dumps(summary_null, indent=2), encoding="utf-8")

        # Console output for null.
        null_desc = f"NULL, surrogates per patch={null_n}, family={null_family}"
        print(f"\n=== {ds} ({null_desc}) ===")
        print(agg_null[["s","n_cg","kappa_nat_corr","kappa_theta","alpha_lmw","omegaP","omegaS"]].to_string(index=False))
        print(f"θ ({chosen_n}) [NULL]: {theta_n_curve:.6f}   PP: {pscore_n*100.0:.2f}%")
        print(f"CSV (NULL): {agg_csv_null}")
        print(f"PNG (NULL): {png_null}")


# ----------------------- CLI -----------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Run T3/ARL-RG on astro patches")
    ap.add_argument("--dataset", type=str, default=None,
                    help="e.g. planck_smica_cmb | planck_gnilc_dust | wmap_ilc9 | hi4pi_nhi (default: all in manifest)")
    ap.add_argument("--trend", type=str, default="log", choices=("auto","log","inv"),
                    help="Trend model for kappa(s): auto selects AIC-min(log, inv).")
    ap.add_argument("--scales", type=str, default=None,
                    help="Comma-separated scale list, e.g. 1,2,4,8,16,32,64 (default: script defaults).")
    ap.add_argument("--null", type=int, default=0,
                    help="Number of phase surrogates per patch (0 = off).")
    ap.add_argument("--null-seed", type=int, default=20251,
                    help="RNG seed for surrogates.")
    ap.add_argument(
        "--compressor",
        type=str,
        default="gzip",
        choices=("gzip", "xz", "zstd"),
        help="Compressor for the compression proxy. Fixed levels: gzip=6, xz=6, zstd=10. Threads forced to 1.",
    )
    ap.add_argument("--agg", type=str, default="median", choices=("mean", "median"),
                help="Aggregation across patches: mean or median.")
    ap.add_argument(
            "--null-family",
            type=str,
            default="phase_randomized",
            choices=("phase_randomized", "rotation90", "aaft", "iaaft", "ring_phase_shuffle", "gaussian_forward_synfast"),
            help="Surrogate family for the null ensemble."
        )
    ap.add_argument(
        "--jobs-data",
        type=int,
        default=0,
        help="Number of processes for data patch processing. 0 selects os.cpu_count().",
    )
    ap.add_argument(
        "--jobs-null",
        type=int,
        default=0,
        help="Number of processes for null ensemble processing. 0 selects os.cpu_count().",
    )
    ap.add_argument(
        "--jobs-null-inner",
        type=int,
        default=0,
        help="Number of parallel workers inside one null repetition for gaussian_forward_synfast. 0 selects an automatic default.",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="Deprecated. Use --jobs-data and --jobs-null. If set and both are 0, it applies to both.",
    )

    args = ap.parse_args()

    jobs_null_inner = int(args.jobs_null_inner)
    jobs_data = int(args.jobs_data)
    jobs_null = int(args.jobs_null)

    if int(args.jobs) > 0 and jobs_data == 0 and jobs_null == 0:
        jobs_data = int(args.jobs)
        jobs_null = int(args.jobs)

    # Parse scales
    if args.scales:
        scales = tuple(int(x) for x in args.scales.split(",") if x.strip())
    else:
        scales = S_LEVELS

    if len(scales) < 2:
        raise SystemExit("ERROR: --scales must contain at least two values, for example 1,2,4")

    mani = load_manifest()
    if args.dataset:
        todo = [args.dataset]
    else:
        # all datasets that have patches in the manifest; if the manifest is empty, use existing folders
        if mani:
            todo = [k for k, v in mani.items() if isinstance(v, dict) and v.get("patches")]
        else:
            todo = [p.name for p in PATCH.iterdir() if (p / "meta.json").exists()]

    OUT.mkdir(parents=True, exist_ok=True)
    for ds in todo:
        outdir = OUT / ds
        run_dataset(
            ds,
            outdir,
            scales=scales,
            trend=args.trend,
            agg_stat=args.agg,
            null_n=args.null,
            null_seed=args.null_seed,
            null_family=args.null_family,
            jobs_data=jobs_data,
            jobs_null=jobs_null,
            jobs_null_inner=jobs_null_inner,
            compressor=args.compressor,
        )

if __name__ == "__main__":
    main()