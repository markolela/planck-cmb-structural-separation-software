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

import gzip
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scripts.t3.encoding import EncodingSpec, encode_patch_to_bytes
from scripts.t3.compressors import CompressorSpec, compress_length


# Default scales. Can be overridden via CLI.
S_LEVELS = (1, 2, 4, 8, 16, 32, 64)

# Quantization and gzip parameters.
QBITS = 8
GZIP_LEVEL = 6

# LMW smoothing on the coarse grained field: sigma = SIGMA_BASE.
# Important:
# cg_field is already coarse grained at scale s.
# sigma is therefore expressed in units of the coarse grained grid.
# Do not multiply sigma by s.
# Otherwise you introduce an artificial second scale amplification and bias the scale comparison.
SIGMA_BASE = 1.0

# LRC / long-range coupling.
LRC_FAR_FRAC = 0.25  # "far" if d_cell > far_frac * n_cg


def _lrc_samples_for_scale(n_cg: int) -> int:
    # About 5% of all pairs, bounded for robustness across small and large scales.
    return int(min(100_000, max(5_000, 0.05 * n_cg * n_cg)))


# LRC pair cache.
# We generate random pairs and the "far" mask only once per n_cg.
# This is deterministic and saves time for many patches and null reps.
_LRC_PAIR_CACHE: dict[tuple[int, float, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _lrc_prepare_pairs(n: int, far_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    pairs = _lrc_samples_for_scale(n)

    i = rng.integers(0, n, size=(pairs, 2), dtype=np.int64)
    j = rng.integers(0, n, size=(pairs, 2), dtype=np.int64)

    dx = np.abs(i[:, 0] - j[:, 0])
    dy = np.abs(i[:, 1] - j[:, 1])
    dx = np.minimum(dx, n - dx)
    dy = np.minimum(dy, n - dy)

    d = np.hypot(dx, dy)
    mask = d > (far_frac * n)
    return i, j, mask


def _lrc_get_pairs(n: int, far_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = (int(n), float(far_frac), int(seed))
    hit = _LRC_PAIR_CACHE.get(key)
    if hit is not None:
        return hit
    val = _lrc_prepare_pairs(n, far_frac, seed)
    _LRC_PAIR_CACHE[key] = val
    return val


def _pearson_abs(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    mx = float(np.mean(x))
    my = float(np.mean(y))
    x0 = x - mx
    y0 = y - my
    vx = float(np.dot(x0, x0))
    vy = float(np.dot(y0, y0))
    if vx <= 0.0 or vy <= 0.0:
        return 0.0
    cov = float(np.dot(x0, y0))
    return abs(cov / np.sqrt(vx * vy))


def _quantize_u8(field: np.ndarray, qbits: int = QBITS) -> np.ndarray:
    f = np.asarray(field, float)
    fmin, fmax = float(np.min(f)), float(np.max(f))
    if not np.isfinite(fmin) or not np.isfinite(fmax) or fmax <= fmin:
        return np.zeros_like(f, dtype=np.uint8)
    qmax = (1 << qbits) - 1
    return np.rint((f - fmin) / (fmax - fmin) * qmax).astype(np.uint8)


def gzip_bpc(field: np.ndarray, qbits: int = QBITS, compresslevel: int = GZIP_LEVEL) -> float:
    spec = EncodingSpec()
    raw = encode_patch_to_bytes(field, spec)
    spec_c = CompressorSpec("gzip", int(compresslevel), 1)
    comp_len = compress_length(raw, spec_c)
    return comp_len / len(raw)

def bpc_for_patch_bytes(
    field: np.ndarray,
    *,
    enc_spec: EncodingSpec | None = None,
    comp_spec: CompressorSpec | None = None,
) -> float:
    """
    Bytes per cell for a patch under a fixed encoding and compressor spec.

    This is the new single entry point for compressor-agnostic compression proxy
    measurements. It is deterministic if:
    - enc_spec is fixed
    - comp_spec is fixed and threads=1
    - gzip uses mtime=0 inside compressors.py

    Default behavior matches the existing pipeline:
    - EncodingSpec() -> uint8 minmax quantization, C order
    - CompressorSpec("gzip", GZIP_LEVEL, 1)
    """
    if enc_spec is None:
        enc_spec = EncodingSpec()

    if comp_spec is None:
        comp_spec = CompressorSpec("gzip", int(GZIP_LEVEL), 1)

    raw = encode_patch_to_bytes(field, enc_spec)
    comp_len = compress_length(raw, comp_spec)

    # For uint8 encoding, len(raw) equals number of cells.
    return comp_len / len(raw)


def gzip_baseline_bpc(
    n_cg: int,
    qbits: int = QBITS,
    trials: int = 3,
    compresslevel: int = GZIP_LEVEL,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Machine baseline: gzip bytes per cell for i.i.d. random data.

    Purpose:
    The gzip format has a fixed header and block structure. For small arrays this can bias
    bytes-per-cell estimates. This baseline averages over a few i.i.d. trials to smooth
    such header and block effects.

    Determinism:
    - Uses a fixed default RNG seed when rng is not provided.
    - Uses gzip mtime=0 to suppress time-dependent headers.
    - Uses C-order contiguous bytes.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    vals: list[float] = []
    for _ in range(int(trials)):
        # i.i.d. uniform bytes in the quantized alphabet
        q = rng.integers(0, 1 << int(qbits), size=(int(n_cg), int(n_cg)), dtype=np.uint8)

        raw = q.tobytes(order="C")
        spec_c = CompressorSpec("gzip", int(compresslevel), 1)
        vals.append(compress_length(raw, spec_c) / len(raw))

    return float(np.mean(vals))


def empirical_entropy_nat(field: np.ndarray, qbits: int = QBITS) -> float:
    """Shannon entropy (nat/cell) of the quantized field."""
    q = _quantize_u8(field, qbits)
    cnt = np.bincount(q.ravel(), minlength=1 << qbits).astype(float)
    p = cnt / cnt.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def coarse_grain(field: np.ndarray, s: int) -> Tuple[np.ndarray, int]:
    """s×s averaging with wrap padding. Returns (field, |R_s|)."""
    n = field.shape[0]
    pad = (-n) % s
    if pad:
        field = np.pad(field, ((0, pad), (0, pad)), mode="wrap")
        n = field.shape[0]
    new_n = n // s
    cg = field.reshape(new_n, s, new_n, s).mean(axis=(1, 3))
    return cg, new_n * new_n


def alpha_lmw(cg_field: np.ndarray, s: int, sigma_base: float = SIGMA_BASE) -> float:
    """
    LMW proxy: variance reduction under Gaussian smoothing.
    sigma = sigma_base in the coarse grained field.
    cg_field is already coarse grained at scale s.
    Therefore do not use sigma_base * s.
    """
    var0 = float(np.var(cg_field))
    if var0 <= 0:
        return 0.0
    sm = gaussian_filter(cg_field, sigma=sigma_base, mode="wrap")
    a = float(np.var(sm) / var0)
    return max(0.0, min(a, 0.999999))


def lrc_penalties(cg_field: np.ndarray, far_frac: float = LRC_FAR_FRAC, seed: int = 1234) -> Tuple[float, float]:
    """Long-range coupling proxies: Pearson and Spearman on far pixel pairs. Pairs are cached."""
    n = int(cg_field.shape[0])

    i, j, mask = _lrc_get_pairs(n, far_frac, seed)
    if not np.any(mask):
        return 0.0, 0.0

    xi = cg_field[i[mask, 0], i[mask, 1]]
    xj = cg_field[j[mask, 0], j[mask, 1]]

    rho_p = _pearson_abs(xi, xj)

    # Spearman: rank transform, then Pearson on ranks.
    # mergesort is stable, making ranks deterministic with ties.
    ri = np.argsort(np.argsort(xi, kind="mergesort"), kind="mergesort")
    rj = np.argsort(np.argsort(xj, kind="mergesort"), kind="mergesort")
    rho_s = _pearson_abs(ri.astype(float), rj.astype(float))

    return float(rho_p), float(rho_s)


def baseline_bpc_iid_u8_for_compressor(
    n_cg: int,
    *,
    qbits: int = QBITS,
    trials: int = 3,
    comp_spec: CompressorSpec,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Machine baseline: compressed bytes per cell for i.i.d. random uint8 data.

    This generalizes the existing gzip baseline to any compressor spec.
    Deterministic if rng is fixed and comp_spec threads=1.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    vals: list[float] = []
    for _ in range(int(trials)):
        q = rng.integers(0, 1 << int(qbits), size=(int(n_cg), int(n_cg)), dtype=np.uint8)
        raw = q.tobytes(order="C")
        vals.append(compress_length(raw, comp_spec) / len(raw))

    return float(np.mean(vals))


def kappa_table_for_patch(
    field: np.ndarray,
    s_levels=S_LEVELS,
    bpc0_cache: dict[tuple, float] | None = None,
    *,
    enc_spec: EncodingSpec | None = None,
    comp_spec: CompressorSpec | None = None,
) -> pd.DataFrame:
    """
    Kappa, LMW, and LRC over scales for a single patch.

    Speed:
    bpc0 depends only on n_cg, encoding spec, compressor spec, and baseline trials.
    Therefore bpc0 is cached per key and generated deterministically.
    This saves a lot of time in null runs and avoids RNG ordering effects.

    Default behavior matches the existing pipeline:
    - EncodingSpec() with uint8 minmax encoding
    - CompressorSpec("gzip", GZIP_LEVEL, 1)
    """
    field = np.asarray(field)
    if field.ndim != 2:
        raise ValueError(f"Patch must be 2D, got shape={field.shape}")
    if field.shape[0] != field.shape[1]:
        raise ValueError(f"Patch must be square, got shape={field.shape}")

    if bpc0_cache is None:
        bpc0_cache = {}

    if enc_spec is None:
        enc_spec = EncodingSpec()

    if comp_spec is None:
        comp_spec = CompressorSpec("gzip", int(GZIP_LEVEL), 1)

    comp_spec.validate()

    rows = []
    base = field

    # Alphabet size factor for kappa in nat per cell.
    ln_alphabet = float(np.log(float(1 << int(enc_spec.qbits))))

    # Deterministic compressor code for baseline seeding.
    comp_code = {"gzip": 1, "xz": 2, "zstd": 3}[comp_spec.name]

    for s in s_levels:
        cg, size_rs = coarse_grain(base, int(s))
        n_cg = int(cg.shape[0])

        bpc = bpc_for_patch_bytes(cg, enc_spec=enc_spec, comp_spec=comp_spec)

        trials = 3
        key = (n_cg, int(enc_spec.qbits), comp_spec.name, int(comp_spec.level), int(comp_spec.threads), int(trials))
        bpc0 = bpc0_cache.get(key)
        if bpc0 is None:
            seed = (
                1_000_003
                + 1009 * n_cg
                + 9176 * int(enc_spec.qbits)
                + 37 * int(comp_spec.level)
                + 991 * int(comp_spec.threads)
                + 13 * int(trials)
                + 101 * int(comp_code)
            ) & 0xFFFFFFFF
            rng0 = np.random.default_rng(int(seed))

            if comp_spec.name == "gzip":
                bpc0 = gzip_baseline_bpc(n_cg, qbits=int(enc_spec.qbits), trials=trials, compresslevel=int(comp_spec.level), rng=rng0)
            else:
                bpc0 = baseline_bpc_iid_u8_for_compressor(
                    n_cg,
                    qbits=int(enc_spec.qbits),
                    trials=trials,
                    comp_spec=comp_spec,
                    rng=rng0,
                )

            bpc0_cache[key] = float(bpc0)

        ent = empirical_entropy_nat(cg)
        a = alpha_lmw(cg, int(s))
        om_p, om_s = lrc_penalties(cg)

        rows.append(
            dict(
                s=int(s),
                n_cg=int(n_cg),
                cells_rs=int(size_rs),
                bytes_per_cell=float(bpc),
                bpc_baseline=float(bpc0),
                kappa_nat=float(bpc * ln_alphabet),
                kappa_nat_corr=float((bpc - bpc0) * ln_alphabet),
                entropy_nat=float(ent),
                alpha_lmw=float(a),
                omegaP=float(om_p),
                omegaS=float(om_s),
            )
        )

    return pd.DataFrame(rows)


def kappa_theta_log(df: pd.DataFrame, ycol: str = "kappa_nat_corr") -> tuple[np.ndarray, float, np.ndarray]:
    """
    Log trend: y ~ a + theta * log(s)
    Returns:
    kappa_theta = y - theta * log(s) for PP and plots
    theta
    residual_full = y - (a + theta * log(s)) for AIC comparison only
    """
    s = df["s"].to_numpy(float)
    y = df[ycol].to_numpy(float).ravel()
    X = np.vstack([np.ones_like(s), np.log(s)]).T
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a = float(np.asarray(coef).ravel()[0])
    theta = float(np.asarray(coef).ravel()[1])

    ktheta = y - theta * np.log(s)
    res_full = y - (a + theta * np.log(s))
    return ktheta, theta, res_full


def kappa_theta_inv(df: pd.DataFrame, ycol: str = "kappa_nat_corr") -> tuple[np.ndarray, float, np.ndarray]:
    """
    Inverse trend: y ~ a + theta * (1/s)
    Returns:
    kappa_theta = y - theta / s for PP and plots
    theta
    residual_full = y - (a + theta / s) for AIC comparison only
    """
    s = df["s"].to_numpy(float)
    y = df[ycol].to_numpy(float).ravel()
    X = np.vstack([np.ones_like(s), 1.0 / s]).T
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a = float(np.asarray(coef).ravel()[0])
    theta = float(np.asarray(coef).ravel()[1])

    ktheta = y - theta / s
    res_full = y - (a + theta / s)
    return ktheta, theta, res_full


def _aic_of_residuals(residuals: np.ndarray, k_params: int = 2) -> float:
    """AIC = n * ln(RSS/n) + 2k (for comparing log vs inv)."""
    r = np.asarray(residuals, float).ravel()
    n = max(1, r.size)
    rss = float(np.sum(r * r))
    return n * np.log(max(rss / n, 1e-30)) + 2 * k_params


def fit_trend(df: pd.DataFrame, ycol: str, trend: str) -> tuple[np.ndarray, float, str]:
    """
    trend: 'log' | 'inv' | 'auto'
    Returns: (kappa_theta, theta, chosen_trend)
    """
    if trend == "log":
        ktheta, theta, _ = kappa_theta_log(df, ycol=ycol)
        return ktheta, theta, "log"
    if trend == "inv":
        ktheta, theta, _ = kappa_theta_inv(df, ycol=ycol)
        return ktheta, theta, "inv"

    # auto: compare AIC(log) vs AIC(inv)
    k_log, th_log, res_log = kappa_theta_log(df, ycol)
    k_inv, th_inv, res_inv = kappa_theta_inv(df, ycol)
    aic_log = _aic_of_residuals(res_log, 2)
    aic_inv = _aic_of_residuals(res_inv, 2)
    if aic_log <= aic_inv:
        return k_log, th_log, "log"
    return k_inv, th_inv, "inv"


def plateau_score(values, n_cg=None, min_ncg=64):
    v = np.asarray(values, dtype=float)
    if n_cg is not None:
        mask = np.asarray(n_cg, dtype=int) >= int(min_ncg)
        if mask.any():
            v = v[mask]
    if v.size == 0:
        return float("nan")
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    vmean = float(np.mean(v))
    denom = max(abs(vmean), 1e-12)
    return (vmax - vmin) / denom
