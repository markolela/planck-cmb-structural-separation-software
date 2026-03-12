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
Deterministic compressor length measurement.

Goal
We measure compressed length as a proxy. This must be fast and reproducible.

Key points
- Compression runs in-process for performance.
- gzip uses gzip.compress with mtime=0 to freeze headers.
- xz uses lzma.compress with FORMAT_XZ.
- zstd uses the python package zstandard with threads=1.

Version freeze
We log both:
- CLI tool versions: gzip, xz, zstd
- In-process library versions: python zstandard and libzstd
This is important because CLI zstd and libzstd from wheels can differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Literal, Sequence
import gzip
import lzma
import os
import platform
import subprocess
import sys

try:
    import zstandard as zstd
except Exception:
    zstd = None


CompressorName = Literal["gzip", "xz", "zstd"]


_DETERMINISTIC_ENV_OVERRIDES: Dict[str, str] = {
    "LANG": "C",
    "LC_ALL": "C",
    "LANGUAGE": "",
    "TZ": "UTC",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


@dataclass(frozen=True)
class CompressorSpec:
    """
    Fixed compressor configuration.

    name: gzip, xz, zstd
    level: compressor level
    threads: must be 1 in this project
    """
    name: CompressorName
    level: int
    threads: int = 1

    def validate(self) -> None:
        if int(self.threads) != 1:
            raise ValueError(f"threads must be 1 for determinism, got {self.threads}")


def _build_env_for_subprocess() -> Dict[str, str]:
    env = dict(os.environ)
    env.update(_DETERMINISTIC_ENV_OVERRIDES)
    return env


def _gzip_compress(raw: bytes, level: int) -> bytes:
    return gzip.compress(raw, compresslevel=int(level), mtime=0)


def _xz_compress(raw: bytes, level: int) -> bytes:
    return lzma.compress(raw, format=lzma.FORMAT_XZ, preset=int(level))


@lru_cache(maxsize=64)
def _zstd_compressor(level: int, threads: int):
    if zstd is None:
        raise RuntimeError("Python package 'zstandard' is not available in this environment.")
    return zstd.ZstdCompressor(level=int(level), threads=int(threads))


def _zstd_compress(raw: bytes, level: int, threads: int) -> bytes:
    c = _zstd_compressor(int(level), int(threads))
    return c.compress(raw)


def compress_bytes(raw: bytes, spec: CompressorSpec) -> bytes:
    """
    Return compressed bytes for raw input bytes and the given spec.
    """
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise TypeError(f"raw must be bytes-like, got {type(raw)!r}")

    spec.validate()

    if spec.name == "gzip":
        return _gzip_compress(bytes(raw), spec.level)

    if spec.name == "xz":
        return _xz_compress(bytes(raw), spec.level)

    if spec.name == "zstd":
        return _zstd_compress(bytes(raw), spec.level, spec.threads)

    raise ValueError(f"Unsupported compressor name: {spec.name!r}")


def compress_length(raw: bytes, spec: CompressorSpec) -> int:
    """
    Return compressed length in bytes.
    """
    return int(len(compress_bytes(raw, spec)))


def get_compressor_versions_cli(specs: Sequence[CompressorSpec]) -> Dict[str, str]:
    """
    Return name -> first version line from the CLI tool.
    """
    names = sorted({s.name for s in specs})
    out: Dict[str, str] = {}

    for name in names:
        cmd = [name, "--version"]
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_build_env_for_subprocess(),
            check=False,
            timeout=10.0,
        )
        txt = (p.stdout + b"\n" + p.stderr).decode("utf-8", errors="replace").strip()
        first_line = txt.splitlines()[0].strip() if txt else ""
        out[name] = first_line

    return out


def get_compressor_versions_inproc() -> Dict[str, str]:
    """
    Return versions relevant for in-process compression.
    """
    out: Dict[str, str] = {}

    if zstd is None:
        out["py_zstandard"] = "MISSING"
        out["libzstd"] = "MISSING"
    else:
        out["py_zstandard"] = str(getattr(zstd, "__version__", "UNKNOWN"))
        out["libzstd"] = str(getattr(zstd, "ZSTD_VERSION", "UNKNOWN"))

    out["python"] = sys.version.split()[0]
    out["platform"] = platform.platform()
    out["machine"] = platform.machine()
    out["processor"] = platform.processor()

    return out


def get_versions_snapshot(specs: Sequence[CompressorSpec]) -> Dict[str, Dict[str, str]]:
    """
    Combined snapshot for run logs.
    """
    return {
        "cli": get_compressor_versions_cli(specs),
        "inproc": get_compressor_versions_inproc(),
    }
