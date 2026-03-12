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
from pathlib import Path
import numpy as np

import healpy as hp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in",
        dest="src",
        required=True,
        help="Input FITS (Planck COM_CMB_IQU-..._halfmission-*.fits)",
    )
    ap.add_argument(
        "--out",
        dest="dest",
        required=True,
        help="Output FITS (..._sm10am.fits in .../planck/harmonized/)",
    )
    args = ap.parse_args()

    src = Path(args.src)
    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Field 0 is I or T
    print(f"[READ] {src}")
    m = hp.read_map(src.as_posix(), field=0, verbose=False)
    m = m.astype(np.float32, copy=False)

    # 10 arcmin FWHM in radians
    fwhm = np.deg2rad(10.0 / 60.0)
    print("[SMOOTH] FWHM = 10′")
    ms = hp.smoothing(m, fwhm=fwhm, verbose=False).astype(np.float32, copy=False)

    print(f"[WRITE] {dest}")
    hp.write_map(dest.as_posix(), ms, dtype=np.float32, overwrite=True)

    print("[OK] Done.")


if __name__ == "__main__":
    main()