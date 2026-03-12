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

from pathlib import Path
import numpy as np
import healpy as hp


# Cache for the tangent plane grid. Key is (N, fov_deg).
_TAN_UV_CACHE: dict[tuple[int, float], tuple[np.ndarray, np.ndarray]] = {}


def _get_tan_uv(N: int, fov_deg: float) -> tuple[np.ndarray, np.ndarray]:
    key = (int(N), float(fov_deg))
    hit = _TAN_UV_CACHE.get(key)
    if hit is not None:
        return hit

    half = 0.5 * float(fov_deg)
    umax = float(np.tan(np.deg2rad(half)))

    uu = np.linspace(-umax, +umax, int(N), dtype=np.float64)
    vv = np.linspace(-umax, +umax, int(N), dtype=np.float64)
    U, V = np.meshgrid(uu, vv)

    _TAN_UV_CACHE[key] = (U, V)
    return U, V


def gnomonic_lonlat_grid_numpy(lon0_deg: float, lat0_deg: float, N: int, fov_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Inverse gnomonic projection.
    Returns lon, lat in radians as (N,N) arrays.
    Orientation matches make_real_patches_standalone.
    """
    lon0 = float(np.deg2rad(lon0_deg))
    lat0 = float(np.deg2rad(lat0_deg))

    cl = float(np.cos(lat0))
    sl = float(np.sin(lat0))
    co = float(np.cos(lon0))
    so = float(np.sin(lon0))

    # Unit vector pointing to the patch center.
    cx = cl * co
    cy = cl * so
    cz = sl

    # Local basis at center.
    # East.
    ex = -so
    ey = co
    ez = 0.0

    # North.
    nx = -co * sl
    ny = -so * sl
    nz = cl

    U, V = _get_tan_uv(int(N), float(fov_deg))

    px = cx + U * ex + V * nx
    py = cy + U * ey + V * ny
    pz = cz + U * ez + V * nz

    norm = np.sqrt(px * px + py * py + pz * pz)
    px = px / norm
    py = py / norm
    pz = pz / norm

    lon = np.mod(np.arctan2(py, px), 2.0 * np.pi)
    lat = np.arcsin(np.clip(pz, -1.0, 1.0))
    return lon, lat


def sample_gnomonic_patch_healpy(m: np.ndarray, nside: int, lon0_deg: float, lat0_deg: float, N: int, fov_deg: float) -> np.ndarray:
    """
    Samples a gnomonic patch (N,N) from a HEALPix map m.
    Bilinear interpolation via hp.get_interp_val.
    """
    lon, lat = gnomonic_lonlat_grid_numpy(lon0_deg, lat0_deg, N=int(N), fov_deg=float(fov_deg))
    theta = 0.5 * np.pi - lat
    phi = lon
    vals = hp.get_interp_val(m, theta, phi)
    return np.asarray(vals, dtype=np.float32).reshape(int(N), int(N))


def read_healpix_map_single_for_cl(path: Path) -> tuple[np.ndarray, int]:
    """
    Robust reader.
    Reads field=0, detects NESTED from FITS header.
    Sets UNSEEN and nonfinite values to NaN.
    Returns (map_values, nside).
    """
    if not path.exists():
        raise FileNotFoundError(path)

    # Local import to avoid overhead when synfast null is not used.
    from astropy.io import fits

    try:
        hdr = fits.getheader(path.as_posix(), ext=1)
    except Exception:
        hdr = fits.getheader(path.as_posix(), ext=0)

    ordering = (hdr.get("ORDERING", "RING") or "RING").strip().upper()
    nest = ordering.startswith("NEST")

    vals = hp.read_map(path.as_posix(), field=0, nest=nest, verbose=False)
    vals = np.asarray(vals, dtype=float).ravel()

    bad = (~np.isfinite(vals)) | (vals == hp.UNSEEN)
    if np.any(bad):
        vals = vals.copy()
        vals[bad] = np.nan

    nside = int(hp.npix2nside(vals.size))
    return vals, nside


def estimate_pseudo_cl_from_source(vals: np.ndarray, nside: int, lat_cut_deg: float | None) -> tuple[np.ndarray, float, int]:
    """
    Pseudo-Cl via hp.anafast on a masked field.
    Mask is finite and optionally |lat| >= lat_cut_deg.
    Removes the masked mean so l=0 does not dominate.
    """
    npix = int(vals.size)
    theta, _phi = hp.pix2ang(int(nside), np.arange(npix, dtype=np.int64))
    lat_deg = 90.0 - np.degrees(theta)

    mask = np.isfinite(vals)
    if lat_cut_deg is not None:
        mask &= (np.abs(lat_deg) >= float(lat_cut_deg))

    if not np.any(mask):
        raise ValueError("Cl estimation failed. Mask is empty. Check lat_cut_deg and data.")

    f_sky = float(np.mean(mask.astype(np.float64)))

    mean = float(np.nanmean(vals[mask]))
    work = np.zeros_like(vals, dtype=np.float64)
    work[mask] = (vals[mask] - mean)

    lmax = int(3 * int(nside) - 1)
    cl = hp.anafast(work, lmax=lmax)

    if cl.size >= 2:
        cl[0] = 0.0
        cl[1] = 0.0

    return np.asarray(cl, dtype=np.float64), f_sky, lmax


def synfast_map_from_cl(cl: np.ndarray, nside: int, seed32: int) -> np.ndarray:
    """
    Deterministic synfast realization without permanently changing the global RNG.
    """
    state = np.random.get_state()
    np.random.seed(int(seed32) & 0xFFFFFFFF)
    try:
        m = hp.synfast(np.asarray(cl, dtype=np.float64), nside=int(nside), lmax=int(len(cl) - 1), new=True, verbose=False)
    finally:
        np.random.set_state(state)
    return np.asarray(m, dtype=np.float32)


def _synfast_lat_tag(lat_cut_deg: float | None) -> str:
    if lat_cut_deg is None:
        return "none"
    x = float(lat_cut_deg)
    s = f"{x:.6f}"
    s = s.replace("-", "m").replace(".", "p")
    return s


def synfast_cl_cache_path(outdir: Path, ds: str, lat_cut_deg: float | None) -> Path:
    tag = _synfast_lat_tag(lat_cut_deg)
    return outdir / f"{ds}_synfast_cl_cache_lat{tag}.npz"


def try_load_synfast_cl_cache(cache_path: Path, src_path: Path, lat_cut_deg: float | None) -> tuple[np.ndarray, int, float, int, bool]:
    """
    Returns (cl, nside, f_sky, lmax, cache_hit).
    cache_hit is True only if source and lat_cut match.
    """
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)

    z = np.load(cache_path, allow_pickle=False)

    src_size_cached = int(z["source_size"].item())
    src_mtime_ns_cached = int(z["source_mtime_ns"].item())
    lat_cut_cached = float(z["lat_cut_deg"].item())

    st = src_path.stat()
    if int(st.st_size) != src_size_cached:
        return (z["cl"], int(z["nside"].item()), float(z["f_sky"].item()), int(z["lmax"].item()), False)
    if int(st.st_mtime_ns) != src_mtime_ns_cached:
        return (z["cl"], int(z["nside"].item()), float(z["f_sky"].item()), int(z["lmax"].item()), False)

    lat_now = -1.0 if lat_cut_deg is None else float(lat_cut_deg)
    if abs(lat_now - lat_cut_cached) > 1e-12:
        return (z["cl"], int(z["nside"].item()), float(z["f_sky"].item()), int(z["lmax"].item()), False)

    return (z["cl"], int(z["nside"].item()), float(z["f_sky"].item()), int(z["lmax"].item()), True)


def write_synfast_cl_cache(cache_path: Path, cl: np.ndarray, nside: int, f_sky: float, lmax: int, src_path: Path, lat_cut_deg: float | None) -> None:
    st = src_path.stat()
    lat_val = -1.0 if lat_cut_deg is None else float(lat_cut_deg)

    np.savez_compressed(
        cache_path,
        cl=np.asarray(cl, dtype=np.float64),
        nside=np.asarray(int(nside), dtype=np.int32),
        f_sky=np.asarray(float(f_sky), dtype=np.float64),
        lmax=np.asarray(int(lmax), dtype=np.int32),
        lat_cut_deg=np.asarray(float(lat_val), dtype=np.float64),
        source_size=np.asarray(int(st.st_size), dtype=np.int64),
        source_mtime_ns=np.asarray(int(st.st_mtime_ns), dtype=np.int64),
        source_path=np.asarray(str(src_path), dtype="U"),
    )
