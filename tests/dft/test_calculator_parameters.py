# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The parameters of the calculator DFT, checked when it is made and before
the GPU part is loaded, so that a machine without one says what is wrong with
them: each refused with its message, and those that are right reaching the
loader, here an MDAPY without a GPU part on any machine."""

import numpy as np
import pytest

from mdapy.calculators import DFT
from mdapy.errors import InvalidParameter

from dft.test_loader import not_built_in

NO_GPU_PART = "the DFT runs in the GPU part of MDAPY, which is not installed"
RANGES = "tolerance > 0, max_iterations >= 1, 0 < mixing <= 1 and kerker >= 0"
HUBBARD = "hubbard maps an element to the label of a pseudo-orbital and U >= 0 in eV"


class Communicator:
    """What DFT asks of a communicator of mpi4py, of one process."""

    def Get_size(self):  # noqa: N802
        return 1

    def Get_rank(self):  # noqa: N802
        return 0

    def Allreduce(self, *args):  # noqa: N802
        raise AssertionError("not called when the calculator is made")


@pytest.fixture(autouse=True)
def without_gpu_part(monkeypatch):
    not_built_in(monkeypatch)


@pytest.mark.parametrize(
    ("options", "message"),
    [({"encut": 0.0}, "encut must be positive"), ({"encut": float("nan")}, "encut must be positive"),
     ({"smearing": "cold"}, r"smearing must be one of \['fixed', 'gaussian'"),
     ({"sigma": 0.0}, "sigma must be positive"), ({"sigma": -0.1, "smearing": "fermi-dirac"}, "sigma must be positive"),
     ({"tolerance": 0.0}, RANGES), ({"max_iterations": 0}, RANGES), ({"mixing": 0.0}, RANGES),
     ({"mixing": 1.01}, RANGES), ({"kerker": -0.1}, RANGES),
     ({"ecutrho": 2399.0}, "ecutrho must be at least 4 encut"),
     ({"kpoints": (2, 2)}, "kpoints must be three integers of at least 1"),
     ({"kpoints": (2.0, 2.0, 2.0)}, "kpoints must be three integers"),
     ({"kpoints": (2, 0, 2)}, "kpoints must be three integers"),
     ({"kspacing": 0.0}, "kspacing must be positive"),
     ({"hubbard": {"Fe": "3d"}}, HUBBARD), ({"hubbard": {"Fe": ("3d",)}}, HUBBARD),
     ({"hubbard": {"Fe": (2, 4.0)}}, HUBBARD), ({"hubbard": {"Fe": ("3d", -1.0)}}, HUBBARD),
     ({"moment_radius": 0.0}, "moment_radius must be positive"),
     ({"precision": "single"}, "precision must be 'double', 'mixed' or 'auto'"),
     ({"kernels": "cufft"}, "kernels must be 'mdapy' or 'vendor'"),
     ({"kernels": "auto"}, "kernels must be 'mdapy' or 'vendor'"),
     ({"preconditioner": "tf"}, "preconditioner must be 'kerker' or 'ldos'"),
     ({"magnetic_mixing": 0.0}, "magnetic_mixing must be positive"),
     ({"restart": "first_order"}, "restart must be False, True, 'atomic' or 'second_order'"),
     ({"gpu": -1}, "gpu must be at least 0"),
     ({"comm": object()}, "comm must be a communicator of mpi4py, not <object")],
)  # fmt: skip
def test_what_is_refused(tmp_path, options, message):
    options = {"encut": 600.0, "pseudopotentials": tmp_path, **options}
    with pytest.raises(InvalidParameter, match=message):
        DFT(options.pop("encut"), **options)


def test_the_pseudopotentials(tmp_path, monkeypatch):
    """By element, a folder, or the folder of MDAPY_PSEUDO_DIR; a file is not a
    folder, and without them there is nothing to read."""
    monkeypatch.delenv("MDAPY_PSEUDO_DIR", raising=False)
    with pytest.raises(InvalidParameter, match="give pseudopotentials, or set MDAPY_PSEUDO_DIR"):  # fmt: skip
        DFT(600.0)
    monkeypatch.setenv("MDAPY_PSEUDO_DIR", str(tmp_path / "none"))
    with pytest.raises(InvalidParameter, match="none is not a folder"):
        DFT(600.0)
    monkeypatch.setenv("MDAPY_PSEUDO_DIR", str(tmp_path))
    with pytest.raises(InvalidParameter, match=NO_GPU_PART):
        DFT(600.0)
    (tmp_path / "Fe.upf").write_text("")
    with pytest.raises(InvalidParameter, match="Fe.upf is not a folder"):
        DFT(600.0, pseudopotentials=tmp_path / "Fe.upf")
    # by element, files not read before they are needed
    with pytest.raises(InvalidParameter, match=NO_GPU_PART):
        DFT(600.0, pseudopotentials={"Fe": tmp_path / "Fe.upf", "Ni": "missing.upf"})


@pytest.mark.parametrize(
    "options",
    [{"smearing": "fixed", "sigma": 0.0}, {"ecutrho": 2400.0, "kpoints": np.array([4, 4, 1])},
     {"kpoints": [1, 1, 1], "kspacing": 0.5}, {"mixing": 1.0, "kerker": 0.0, "max_iterations": 1},
     {"magmoms": {"Fe": 2.5}, "magnetic_mixing": 1.6, "moment_radius": 1.2},
     {"hubbard": {"Fe": ("3d", 4.0), "Ni": ["3d", 0]}}, {"precision": "auto", "kernels": "vendor"},
     {"preconditioner": "ldos", "restart": "second_order"}, {"restart": True, "symmetry": False},
     {"gpu": 0, "comm": Communicator()}, {"comm": Communicator(), "bands": 12, "seed": 3}],
)  # fmt: skip
def test_what_reaches_the_gpu_part(tmp_path, options):
    with pytest.raises(InvalidParameter, match=NO_GPU_PART):
        DFT(600.0, pseudopotentials=tmp_path, **options)
