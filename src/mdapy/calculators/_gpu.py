# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Calculators on a GPU, NVIDIA's with CUDA or AMD's with HIP: the potential
on the device, with the atoms and the neighbor list of its last computation
kept there."""

import importlib
import os
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
    "this MDAPY has no GPU part: on Linux on x86-64 or ARM, install it with "
    'pip install "mdapy[cuda12]"; elsewhere, build MDAPY with the CUDA toolkit, '
    "pip install mdapy --no-binary mdapy -Ccmake.define.MDAPY_CUDA=ON; for an "
    "AMD GPU, build it with ROCm, -Ccmake.define.MDAPY_HIP=ON"
)

# the packages of the GPU part: the name pip knows, and the module in each
PACKAGES = {"mdapy-cuda12": "_cuda", "mdapy-rocm6": "_hip"}


def _built_in() -> Any:
    """The GPU part built with MDAPY: mdapy._cuda with the CUDA toolkit, or
    mdapy._hip with ROCm; None without either."""
    try:
        from mdapy import _cuda

        return _cuda
    except ImportError:
        pass
    try:
        from mdapy import _hip

        return _hip
    except ImportError:
        return None


def module() -> Any:
    """The GPU part: mdapy._cuda or mdapy._hip of an MDAPY built with the
    CUDA toolkit or with ROCm, or that of the package mdapy-cuda12 or
    mdapy-rocm6 at the version of MDAPY, of the two the first that sees a
    device; None without any."""
    found = _built_in()
    if found is not None:
        return found
    import mdapy

    parts = []
    for name, part in PACKAGES.items():
        try:
            package = importlib.import_module(name.replace("-", "_"))
        except ImportError:
            continue
        if package.__version__ != mdapy.__version__:
            raise InvalidParameter(
                f"{name} {package.__version__} is not the GPU part of MDAPY "
                f'{mdapy.__version__}: pip install --upgrade "mdapy[{name[6:]}]"'
            )
        parts.append(importlib.import_module(f"{package.__name__}.{part}"))
    if len(parts) > 1:
        for part in parts:
            if part.device_count() > 0:
                return part
    return parts[0] if parts else None


def _local_rank() -> int | None:
    """The rank of this process among those of MPI on its node, as mpirun or
    srun give it; None outside MPI."""
    for name in (
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MPI_LOCALRANKID",
        "MV2_COMM_WORLD_LOCAL_RANK",
        "SLURM_LOCALID",
    ):
        value = os.environ.get(name)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                continue
    return None


_chosen: list = []  # whether the device of this process was chosen


def cuda() -> Any:
    """The GPU part, with a device to run on: under MPI, the device of this
    process among those of its node, by its rank there, chosen before the
    first allocation."""
    found = module()
    if found is None:
        raise InvalidParameter(_NO_GPU_PART)
    try:
        found.device_name()
        if not _chosen:
            _chosen.append(True)
            rank, count = _local_rank(), int(found.device_count())
            if rank is not None and count > 1:
                found.use_device(rank % count)
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
