# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import re
import tomllib
from pathlib import Path

import mdapy


def test_version_is_pep440():
    assert re.fullmatch(r"\d+\.\d+\.\d+((a|b|rc)\d+)?(\.dev\d+)?", mdapy.__version__)


def test_public_names_exist():
    for name in mdapy.__all__:
        assert hasattr(mdapy, name), name


def test_errors_and_units_are_importable_as_modules():
    from mdapy.errors import Cancelled
    from mdapy.units import BOLTZMANN

    assert mdapy.errors.Cancelled is Cancelled
    assert mdapy.units.BOLTZMANN == BOLTZMANN


def test_the_gpu_part_is_of_the_version_of_mdapy():
    """mdapy-cuda12 takes its version from MDAPY, and the extra cuda12 of
    MDAPY asks for that version; the package carries the license of MDAPY."""
    root = Path(__file__).parent.parent
    project = tomllib.loads((root / "pyproject.toml").read_text())
    (requirement,) = project["project"]["optional-dependencies"]["cuda12"]
    assert requirement.startswith(f"mdapy-cuda12=={mdapy.__version__};")
    assert (
        "(platform_machine == 'x86_64' or platform_machine == 'aarch64')" in requirement
    )
    plugin = tomllib.loads((root / "cuda12" / "pyproject.toml").read_text())
    version = plugin["tool"]["scikit-build"]["metadata"]["version"]
    assert version["input"] == "../src/mdapy/__init__.py"
    assert (
        version["regex"]
        == project["tool"]["scikit-build"]["metadata"]["version"]["regex"]
    )
    assert (root / "cuda12" / "LICENSE").read_text() == (root / "LICENSE").read_text()
    # mdapy-rocm6, the same for AMD GPUs
    (requirement,) = project["project"]["optional-dependencies"]["rocm6"]
    assert requirement.startswith(f"mdapy-rocm6=={mdapy.__version__};")
    amd = tomllib.loads((root / "rocm6" / "pyproject.toml").read_text())
    assert (
        amd["tool"]["scikit-build"]["metadata"]
        == plugin["tool"]["scikit-build"]["metadata"]
    )
    assert (root / "rocm6" / "LICENSE").read_text() == (root / "LICENSE").read_text()
