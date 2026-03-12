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

from pathlib import Path
import argparse, numpy as np
import healpy as hp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", nargs="+", required=True)
    ap.add_argument("--fwhm-arcmin", type=float, default=10.0)
    ap.add_argument("--nside", type=int, default=2048)
    ap.add_argument("--outdir", default=str(Path("data") / "raw" / "astro" / "planck" / "harmonized"))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fwhm_rad = np.deg2rad(args.fwhm_arcmin / 60.0)

    for src in args.inputs:
        src = Path(src)
        print(f"[i] Loading {src.name}")
        m = hp.read_map(src, field=None, verbose=False)  # I (or IQU, field=None loads all)

        # Smooth
        if isinstance(m, np.ndarray) and m.ndim == 2:  # IQU
            m_s = []
            for i in range(m.shape[0]):
                m_s.append(hp.sphtfunc.smoothing(m[i], fwhm=fwhm_rad, verbose=False))
            m = np.asarray(m_s)
        else:
            m = hp.sphtfunc.smoothing(m, fwhm=fwhm_rad, verbose=False)

        # Adjust NSIDE
        if isinstance(m, np.ndarray) and m.ndim == 2:
            m2 = []
            for i in range(m.shape[0]):
                m2.append(
                    hp.ud_grade(
                        m[i],
                        nside_out=args.nside,
                        pess=False,
                        order_in="RING",
                        order_out="RING",
                    )
                )
            m = np.asarray(m2)
        else:
            m = hp.ud_grade(m, nside_out=args.nside, pess=False, order_in="RING", order_out="RING")

        out = outdir / f"{src.stem}_beam{int(args.fwhm_arcmin)}arcmin_nside{args.nside}.fits"
        hp.write_map(out, m, overwrite=True, dtype=np.float32)
        print(f"[ok] Saved: {out}")


if __name__ == "__main__":
    main()