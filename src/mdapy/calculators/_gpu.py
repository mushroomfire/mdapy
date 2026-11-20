# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Calculators on an NVIDIA GPU: the potential on the device, with the atoms
and the neighbor list of its last computation kept there."""

import threading
from collections.abc import Set
from typing import Any

import numpy as np

from mdapy.calculators._base import CalculatorResult
from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.errors import InvalidParameter

DEVICES = ("cpu", "cuda")
PRECISIONS = ("mixed", "double")
# the skin of the neighbor list kept on the device, in Å
SKIN = 1.0


_NO_GPU_PART = (
    "this MDAPY has no GPU part: on Linux on x86-64, install it with "
    'pip install "mdapy[cuda12]"; elsewhere, build MDAPY with the CUDA toolkit, '
    "pip install mdapy --no-binary mdapy -Ccmake.define.MDAPY_CUDA=ON"
)


def module() -> Any:
    """The GPU part: mdapy._cuda of an MDAPY built with the CUDA toolkit, or
    that of the package mdapy-cuda12 at the version of MDAPY; None without
    either."""
    try:
        from mdapy import _cuda

        return _cuda
    except ImportError:
        pass
    try:
        import mdapy_cuda12
    except ImportError:
        return None
    import mdapy

    if mdapy_cuda12.__version__ != mdapy.__version__:
        raise InvalidParameter(
            f"mdapy-cuda12 {mdapy_cuda12.__version__} is not the GPU part of MDAPY "
            f'{mdapy.__version__}: pip install --upgrade "mdapy[cuda12]"'
        )
    from mdapy_cuda12 import _cuda

    return _cuda


def cuda() -> Any:
    """The GPU part, with a device to run on."""
    found = module()
    if found is None:
        raise InvalidParameter(_NO_GPU_PART)
    try:
        found.device_name()
    except RuntimeError as error:
        raise InvalidParameter(str(error)) from None
    return found


def resolve(device: str, precision: str | None) -> tuple[str, str]:
    """The device and the precision of a calculator: by default double on
    the processor, mixed on the GPU."""
    if device not in DEVICES:
        raise InvalidParameter(f"device must be one of {DEVICES}, not {device!r}")
    if precision is None:
        precision = "mixed" if device == "cuda" else "double"
    if precision not in PRECISIONS:
        raise InvalidParameter(
            f"precision must be one of {PRECISIONS}, not {precision!r}"
        )
    if device == "cpu" and precision != "double":
        raise InvalidParameter("the processor computes in double precision only")
    return device, precision


def box_arguments(box: Box) -> tuple[np.ndarray, np.ndarray, tuple[bool, bool, bool]]:
    pbc = tuple(bool(p) for p in box.pbc)
    return (
        np.ascontiguousarray(box.matrix, dtype=np.float64),
        np.ascontiguousarray(box.origin, dtype=np.float64),
        pbc,  # type: ignore[return-value]
    )


class OnDevice:
    """A potential of mdapy._cuda, and the atoms and the neighbor list of its
    last computation, used again while the number of atoms stays."""

    def __init__(self, potential: Any, cutoff: float):
        self.potential = potential
        self.cutoff = float(cutoff)
        self.__atoms: Any = None
        self.__neighbors: Any = None
        self.__lock = threading.Lock()

    def compute(
        self, system: System, species: np.ndarray, wanted: Set[str], heat: bool = False
    ) -> CalculatorResult:
        """What wanted names; heat: the atomic energies and virials of the
        heat flux, of the potential's compute_heat."""
        module = cuda()
        positions = np.ascontiguousarray(system["positions"], dtype=np.float64)
        h, lo, pbc = box_arguments(system.box)
        n = len(system)
        with self.__lock:
            if self.__atoms is None or self.__atoms.size != n:
                self.__atoms = module.Atoms(n)
                self.__neighbors = module.Neighbors(self.cutoff, SKIN)
            atoms, neighbors = self.__atoms, self.__neighbors
            atoms.set_species(np.ascontiguousarray(species, dtype=np.int32))
            atoms.set_positions(positions)
            try:
                neighbors.update(atoms, h, lo, pbc)
            except ValueError as error:
                raise InvalidParameter(str(error)) from None
            energy = bool({"energy", "atomic_energies"} & wanted)
            atomic = bool({"atomic_energies", "atomic_virials"} & wanted)
            virial = bool({"virial", "atomic_virials"} & wanted)
            if heat:
                self.potential.compute_heat(atoms, neighbors, h, lo, pbc)
            else:
                self.potential.compute(
                    atoms, neighbors, h, lo, pbc, energy, virial, atomic
                )
            return CalculatorResult(
                energy=atoms.total_energy() if "energy" in wanted else None,
                atomic_energies=atoms.energies()
                if "atomic_energies" in wanted
                else None,
                forces=atoms.forces() if "forces" in wanted else None,
                virial=atoms.total_virial() if "virial" in wanted else None,
                atomic_virials=atoms.virials() if "atomic_virials" in wanted else None,
                volume=system.box.volume,
            )
