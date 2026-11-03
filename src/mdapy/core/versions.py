# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The versions and the machine, as an issue should tell them."""

import importlib.metadata
import os
import platform
import shutil
import sys
from typing import TextIO

import numpy as np

from mdapy import __version__, _core
from mdapy.core.threads import get_num_threads

# what MDAPY works with when installed, by the names pip knows them under
_OPTIONAL = (
    "ase",
    "freud-analysis",
    "lammps",
    "matplotlib",
    "ovito",
    "phonopy",
    "pymatgen-core",
    "seekpath",
    "spglib",
    "torch",
    "torchnep",
)


def _installed(package: str) -> str:
    # read from the files of the package, as importing torch alone takes seconds
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def show_versions(file: TextIO | None = None) -> None:
    """Print the versions of MDAPY, Python and the packages around them, and
    the processor, memory and GPU of this machine, to paste into an issue.

    Parameters
    ----------
    file : text stream, optional
        Where to print them; the standard output by default.
    """
    memory = _core.core.memory_bytes()
    blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"]
    blocks = (
        (
            ("MDAPY", __version__),
            ("Atom indices", f"{_core.index_bits} bits"),
            ("Python", sys.version.replace("\n", " ")),
            ("Platform", platform.platform()),
        ),
        (
            ("CPU", _core.core.cpu_name() or platform.machine()),
            (
                "Logical CPUs",
                f"{os.cpu_count()}, {_core.core.hardware_threads()} usable",
            ),
            ("MDAPY threads", str(get_num_threads())),
            ("Memory", f"{memory / 2**30:.1f} GB" if memory else "unknown"),
            ("GPU", _core.visualization.gpu_device() or "none found"),
            ("FFmpeg", "found" if shutil.which("ffmpeg") else "not found"),
        ),
        (
            ("numpy", f"{np.__version__} ({blas})"),
            ("polars", _installed("polars")),
            *((package, _installed(package)) for package in _OPTIONAL),
        ),
    )
    text = "\n\n".join(
        "\n".join(f"{name:<15}{value}" for name, value in block) for block in blocks
    )
    print(text, file=file)
