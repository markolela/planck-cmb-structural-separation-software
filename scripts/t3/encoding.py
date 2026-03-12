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
Deterministic patch encoding to raw bytes.

Important
We never compress container formats here.
We compress raw bytes produced from a strictly defined encoding.

Current default matches the existing quantization behavior in scripts/t3/metrics.py:
- per patch min max scaling to uint8
- rounding via np.rint
- no explicit clipping
- non finite or constant fields map to all zeros
- row major C order, contiguous bytes
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict
import json
import numpy as np


@dataclass(frozen=True)
class EncodingSpec:
    qbits: int = 8
    dtype: str = "uint8"
    byteorder: str = "little"
    order: str = "C"
    scaling: str = "per_patch_minmax"
    rounding: str = "rint"
    clipping: str = "none"
    nonfinite_policy: str = "zeros"
    constant_policy: str = "zeros"

    def to_dict(self) -> Dict[str, Any]:
        return dict(asdict(self))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


def quantize_u8_minmax(field: np.ndarray, qbits: int = 8) -> np.ndarray:
    """
    Quantize to uint8 using per field min max scaling.

    This mirrors the existing behavior in scripts/t3/metrics.py.
    """
    f = np.asarray(field, float)
    fmin, fmax = float(np.min(f)), float(np.max(f))
    if not np.isfinite(fmin) or not np.isfinite(fmax) or fmax <= fmin:
        return np.zeros_like(f, dtype=np.uint8)
    qmax = (1 << int(qbits)) - 1
    return np.rint((f - fmin) / (fmax - fmin) * qmax).astype(np.uint8)


def encode_patch_to_bytes(field: np.ndarray, spec: EncodingSpec | None = None) -> bytes:
    """
    Encode a 2D patch to raw bytes according to EncodingSpec.

    This function performs quantization and returns raw bytes only.
    It does not compress.
    """
    if spec is None:
        spec = EncodingSpec()

    if spec.qbits != 8:
        raise ValueError(f"Unsupported qbits in this project baseline: {spec.qbits}")
    if spec.dtype != "uint8":
        raise ValueError(f"Unsupported dtype in this project baseline: {spec.dtype}")
    if spec.order != "C":
        raise ValueError(f"Unsupported order in this project baseline: {spec.order}")

    q = quantize_u8_minmax(field, spec.qbits)
    q = np.ascontiguousarray(q)
    return q.tobytes(order="C")
