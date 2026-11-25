# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Public pseudopotentials for the tests of the DFT on the processor: a few
small files of the library of Quantum ESPRESSO and of GBRV, one of each kind
that read_upf reads, downloaded once into the folder MDAPY_TEST_CACHE (default
~/.cache/mdapy/tests) and checked by their SHA-256. A test skips where a file
cannot be downloaded."""

import functools
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

import pytest

_QE = "https://pseudopotentials.quantum-espresso.org/upf_files/"
_GBRV = "https://www.physics.rutgers.edu/gbrv/"

FILES = {
    # norm-conserving, UPF 2: von Barth and Car, LDA
    "Al.pz-vbc.UPF": (
        _QE,
        "4eab06b63f87f07ede2d5a193e6d993a09107167fd6b8647afa807342501d6e5",
    ),
    # ultrasoft, UPF 2: pslibrary 1.0.0, s, p and d projectors
    "Al.pbe-n-rrkjus_psl.1.0.0.UPF": (
        _QE,
        "cc4f5dc6afe09c8f482dc7645e6e7cca546a55f8d907c71c825c62bf85a38d3e",
    ),
    # PAW, UPF 2: pslibrary 1.0.0, one-center densities up to l = 4
    "Fe.pbe-spn-kjpaw_psl.1.0.0.UPF": (
        _QE,
        "a8c4ca198bc91c217b6b05670605b6d0c7865e8d36eb0d2154e9c33ba58e279a",
    ),
    # ultrasoft, UPF 1: GBRV 1.0, the charges pseudized inside rinner
    "al_pbe_v1.uspp.F.UPF": (
        _GBRV,
        "0fdd0d7cf85ffb2b872041de4a9626c1f70646c4b81181b7dba92beb61f5811d",
    ),
}


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@functools.cache
def path(name: str) -> Path:
    """The path of the public file name in the cache, downloaded if it is not
    there; a skip if it cannot be downloaded, a failure if it is not the file
    of the SHA-256 above."""
    site, digest = FILES[name]
    default = Path.home() / ".cache" / "mdapy" / "tests"
    folder = Path(os.environ.get("MDAPY_TEST_CACHE") or default) / "pseudopotentials"
    target = folder / name

    def cached() -> bool:
        return target.is_file() and _digest(target.read_bytes()) == digest

    if cached():
        return target
    try:
        with urllib.request.urlopen(site + name, timeout=120) as response:
            data = response.read()
    except OSError as error:
        pytest.skip(f"{name} could not be downloaded from {site}: {error}")
    if _digest(data) != digest:
        pytest.fail(f"{site}{name} is not the file of SHA-256 {digest}")
    folder.mkdir(parents=True, exist_ok=True)
    # whole or not at all, for the workers of pytest-xdist that download it too
    with tempfile.NamedTemporaryFile(dir=folder, delete=False) as part:
        part.write(data)
    try:
        os.replace(part.name, target)
    except OSError:
        # on Windows, while another worker reads the one it put there
        os.unlink(part.name)
        if not cached():
            raise
    return target
