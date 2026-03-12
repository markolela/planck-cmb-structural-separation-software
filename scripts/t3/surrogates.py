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

from hashlib import sha256
import numpy as np


def rng_for_null(null_seed: int, null_family: str, patch_idx: int, rep_idx: int) -> np.random.Generator:
    """
    Deterministic RNG for null generation.
    The seed is derived from (null_seed, family, patch_idx, rep_idx) via sha256.
    This keeps results stable and independent of execution order.
    """
    key = f"{null_seed}|{null_family}|{patch_idx}|{rep_idx}".encode("utf-8")
    h = sha256(key).digest()
    seed64 = int.from_bytes(h[:8], "little", signed=False)
    return np.random.default_rng(seed64)


def phase_surrogate2d(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    2D phase surrogate.
    Keeps the amplitude spectrum exactly and randomizes phases.
    Enforces Hermitian symmetry so the inverse is real without a real-part hack.
    Vectorized to stay fast for large patches and many surrogates.
    """
    x0 = np.asarray(x)
    X = np.fft.fft2(x0)
    mag = np.abs(X)

    # Hermitian-consistent phase randomization.
    # Condition: phi[(-i) mod n0, (-j) mod n1] = -phi[i, j]
    phi = rng.uniform(-np.pi, np.pi, size=X.shape)
    phi_neg = np.roll(np.roll(phi[::-1, ::-1], 1, axis=0), 1, axis=1)
    phi = 0.5 * (phi - phi_neg)

    Y = mag * np.exp(1j * phi)

    # Self-conjugate frequencies must remain real.
    # Copy them from X to keep mean and sign stable.
    n0, n1 = X.shape
    Y[0, 0] = X[0, 0].real
    if n0 % 2 == 0:
        Y[n0 // 2, 0] = X[n0 // 2, 0].real
    if n1 % 2 == 0:
        Y[0, n1 // 2] = X[0, n1 // 2].real
    if (n0 % 2 == 0) and (n1 % 2 == 0):
        Y[n0 // 2, n1 // 2] = X[n0 // 2, n1 // 2].real

    y = np.fft.ifft2(Y).real
    return y.astype(x0.dtype, copy=False)


def rotation90_surrogate2d(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Patch rotation in 90 degree steps.
    Tests sensitivity to pixel grid orientation.
    """
    k = int(rng.integers(0, 4))
    return np.rot90(x, k=k).copy()


def _rank_match_to_sorted(x: np.ndarray, target_sorted: np.ndarray) -> np.ndarray:
    """
    Replace values in x by target_sorted according to the rank order of x.
    target_sorted must be sorted already and have the same length.
    """
    flat = np.asarray(x, float).ravel()
    idx = np.argsort(flat, kind="mergesort")
    out = np.empty_like(flat)
    out[idx] = target_sorted
    return out.reshape(x.shape)


def aaft_surrogate2d(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    2D AAFT surrogate.
    1) Gaussianize x via rank matching to N(0,1).
    2) Phase randomize in Fourier domain.
    3) Map back to the original amplitude distribution of x via rank matching.
    """
    x = np.asarray(x, float)
    x_sorted = np.sort(x.ravel())

    g_sorted = np.sort(rng.standard_normal(size=x.size))
    x_gauss = _rank_match_to_sorted(x, g_sorted)

    y = phase_surrogate2d(x_gauss, rng)

    y = _rank_match_to_sorted(y, x_sorted)
    return y.astype(x.dtype, copy=False)


def iaaft_surrogate2d(x: np.ndarray, rng: np.random.Generator, n_iter: int = 20) -> np.ndarray:
    """
    2D IAAFT surrogate.
    Keeps the amplitude distribution exactly and approximates the power spectrum
    much better than AAFT.
    n_iter is fixed to keep runtime controlled.
    """
    dtype0 = np.asarray(x).dtype
    x = np.asarray(x, float)

    x_sorted = np.sort(x.ravel())
    mag = np.abs(np.fft.rfft2(x))

    y = rng.permutation(x.ravel()).reshape(x.shape)

    for _ in range(int(n_iter)):
        Y = np.fft.rfft2(y)
        Y = mag * np.exp(1j * np.angle(Y))
        y = np.fft.irfft2(Y, s=x.shape)
        y = _rank_match_to_sorted(y, x_sorted)

    return y.astype(dtype0, copy=False)


# Ring phase shuffle cache. Key is (n0, n1).
_RING_SHUFFLE_CACHE: dict[tuple[int, int], dict[str, np.ndarray]] = {}


def _ring_shuffle_prepare(n0: int, n1: int) -> dict[str, np.ndarray]:
    key = (int(n0), int(n1))
    hit = _RING_SHUFFLE_CACHE.get(key)
    if hit is not None:
        return hit

    fy = np.fft.fftfreq(n0)[:, None]
    fx = np.fft.fftfreq(n1)[None, :]
    rad = np.sqrt(fy * fy + fx * fx)
    scale = max(n0, n1)
    rb = np.rint(rad * scale).astype(np.int32)

    reps_i = []
    reps_j = []
    reps_i2 = []
    reps_j2 = []
    fixed_i = []
    fixed_j = []

    for i in range(n0):
        i2 = (-i) % n0
        for j in range(n1):
            j2 = (-j) % n1
            if (i == i2) and (j == j2):
                fixed_i.append(i)
                fixed_j.append(j)
                continue
            if (i < i2) or (i == i2 and j < j2):
                reps_i.append(i)
                reps_j.append(j)
                reps_i2.append(i2)
                reps_j2.append(j2)

    reps_i = np.asarray(reps_i, dtype=np.int32)
    reps_j = np.asarray(reps_j, dtype=np.int32)
    reps_i2 = np.asarray(reps_i2, dtype=np.int32)
    reps_j2 = np.asarray(reps_j2, dtype=np.int32)

    bins = rb[reps_i, reps_j].astype(np.int32)

    out = {
        "reps_i": reps_i,
        "reps_j": reps_j,
        "reps_i2": reps_i2,
        "reps_j2": reps_j2,
        "fixed_i": np.asarray(fixed_i, dtype=np.int32),
        "fixed_j": np.asarray(fixed_j, dtype=np.int32),
        "bins": bins,
    }
    _RING_SHUFFLE_CACHE[key] = out
    return out


def ring_phase_shuffle_surrogate2d(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    FFT ring phase shuffle.
    Keeps the amplitude spectrum exactly.
    Permutes phases only within radial frequency rings.
    Enforces Hermitian symmetry so the inverse is real.
    """
    x0 = np.asarray(x)
    n0, n1 = x0.shape

    cache = _ring_shuffle_prepare(n0, n1)
    reps_i = cache["reps_i"]
    reps_j = cache["reps_j"]
    reps_i2 = cache["reps_i2"]
    reps_j2 = cache["reps_j2"]
    fixed_i = cache["fixed_i"]
    fixed_j = cache["fixed_j"]
    bins = cache["bins"]

    X = np.fft.fft2(x0)
    mag = np.abs(X)

    ph0 = np.angle(X[reps_i, reps_j])
    ph_new = ph0.copy()

    for b in np.unique(bins):
        idxs = np.where(bins == b)[0]
        if idxs.size <= 1:
            continue
        vals = ph0[idxs].copy()
        rng.shuffle(vals)
        ph_new[idxs] = vals

    phi = np.zeros((n0, n1), dtype=np.float64)
    phi[reps_i, reps_j] = ph_new
    phi[reps_i2, reps_j2] = -ph_new

    Y = mag * np.exp(1j * phi)

    if fixed_i.size:
        Y[fixed_i, fixed_j] = X[fixed_i, fixed_j].real

    y = np.fft.ifft2(Y).real
    return y.astype(x0.dtype, copy=False)
