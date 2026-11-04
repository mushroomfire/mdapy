# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import importlib.metadata
import inspect
import io
import os
import platform
import re
import shutil
import subprocess
import sys

import numpy as np
import polars as pl
import pytest

import mdapy as mp
from mdapy import _core
from mdapy.visualization import gpu_available


def report() -> dict[str, str]:
    stream = io.StringIO()
    mp.show_versions(file=stream)
    return {
        line[:15].strip(): line[15:] for line in stream.getvalue().splitlines() if line
    }


def test_the_report_tells_the_versions():
    rows = report()
    assert rows["MDAPY"] == mp.__version__
    assert rows["Python"].startswith(platform.python_version())
    assert rows["Platform"] == platform.platform()
    assert rows["polars"] == pl.__version__
    assert rows["numpy"].startswith(np.__version__ + " (")
    assert rows["Atom indices"] == f"{np.iinfo(mp.index_dtype).bits} bits"


def test_the_report_names_every_dependency_of_the_package():
    rows = report()
    for requirement in importlib.metadata.requires("mdapy"):
        if not re.search(r"extra\s*==\s*['\"](dev|docs)['\"]", requirement):
            assert re.match(r"[\w.-]+", requirement).group() in rows


def test_the_report_tells_the_machine():
    rows = report()
    assert rows["Logical CPUs"] == (
        f"{os.cpu_count()}, {_core.core.hardware_threads()} usable"
    )
    assert rows["MDAPY threads"] == str(mp.get_num_threads())
    with mp.num_threads(1):
        assert report()["MDAPY threads"] == "1"
    assert (rows["GPU"] != "none found") == gpu_available()


@pytest.mark.skipif(sys.platform == "win32", reason="the system's own numbers")
def test_the_processor_and_the_memory_are_those_of_the_system():
    bytes_ = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    assert _core.core.memory_bytes() == bytes_
    assert report()["Memory"] == f"{bytes_ / 2**30:.1f} GB"
    if sys.platform == "darwin":
        name = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    else:
        with open("/proc/cpuinfo") as info:
            found = re.search(r"^model name\s*:\s*(.*)$", info.read(), re.MULTILINE)
        name = found.group(1).strip() if found else ""
    assert _core.core.cpu_name() == name


def test_it_prints_to_the_standard_output(capsys):
    assert mp.show_versions() is None
    assert capsys.readouterr().out.startswith(f"MDAPY          {mp.__version__}\n")


def test_it_tells_what_is_missing(monkeypatch):
    real = importlib.metadata.version

    def without_ase(package):
        if package == "ase":
            raise importlib.metadata.PackageNotFoundError(package)
        return real(package) if package == "polars" else "1.2.3"

    monkeypatch.setattr(importlib.metadata, "version", without_ase)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(_core.core, "cpu_name", lambda: "")
    monkeypatch.setattr(_core.core, "memory_bytes", lambda: 0)
    monkeypatch.setattr(_core.visualization, "gpu_device", lambda: "")
    rows = report()
    assert rows["ase"] == "not installed"
    assert rows["spglib"] == "1.2.3"
    assert rows["FFmpeg"] == "not found"
    assert rows["CPU"] == platform.machine()
    assert rows["Memory"] == "unknown"
    assert rows["GPU"] == "none found"


def test_ffmpeg_on_the_path_is_found(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: "/bin/" + name)
    assert report()["FFmpeg"] == "found"


def test_mdapy_needs_nothing_optional_until_it_is_used():
    """All subpackages load with every optional package blocked, and a plot says
    what it needs only when it is called."""
    code = """
import sys
BLOCKED = {
    "ase", "dscribe", "freud", "lammps", "matplotlib", "networkx", "ovito",
    "phonopy", "pymatgen", "scipy", "seekpath", "spglib", "symfc", "torch",
}
for name in BLOCKED:
    sys.modules[name] = None
import mdapy
from mdapy.visualization import plot
loaded = {name for name, module in sys.modules.items() if module is not None}
assert not BLOCKED & {name.split(".")[0] for name in loaded}
try:
    plot.figure("single")
except ImportError as error:
    assert "matplotlib" in str(error)
else:
    raise SystemExit("a plot did not ask for matplotlib")
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


def test_the_top_level_holds_the_foundations():
    objects = {"Box", "CancelToken", "NeighborList", "System", "Topology"}
    functions = {
        "get_num_threads",
        "index_dtype",
        "num_threads",
        "set_num_threads",
        "show_versions",
    }
    modules = {
        "analysis",
        "build",
        "calculators",
        "elements",
        "errors",
        "identify",
        "interop",
        "io",
        "lattice_dynamics",
        "mlip",
        "neighbor",
        "simulation",
        "transform",
        "units",
        "visualization",
    }
    assert set(mp.__all__) == objects | functions | modules | {"__version__"}
    assert all(hasattr(mp, name) for name in mp.__all__)
    for module in modules:
        assert inspect.ismodule(getattr(mp, module))


def test_it_imports_nothing_heavy():
    # the versions come from the files of the packages: importing them would
    # take seconds
    code = (
        "import sys; import mdapy; mdapy.show_versions();"
        "assert not {'ovito', 'lammps', 'freud', 'matplotlib'} & set(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)
