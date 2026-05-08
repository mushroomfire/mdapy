# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

# Resolve MDAPY_NUM_THREADS BEFORE importing any submodule that pulls in polars.
# Polars locks its thread pool at import time and only honours POLARS_MAX_THREADS,
# so we mirror MDAPY_NUM_THREADS to it here. We deliberately do NOT touch
# OMP_NUM_THREADS — OpenMP thread counts are passed explicitly into C++ functions
# via num_threads() clauses, and mutating OMP_NUM_THREADS would affect other
# OpenMP libraries (torch / sklearn / scipy) sharing the same interpreter.
import os as _os
import warnings as _warnings

_env = _os.environ.get("MDAPY_NUM_THREADS")
if _env is not None:
    try:
        _n = int(_env)
    except ValueError as _e:
        raise ValueError(
            f"MDAPY_NUM_THREADS must be a positive integer, got {_env!r}."
        ) from _e
    if _n <= 0:
        raise ValueError(f"MDAPY_NUM_THREADS must be > 0, got {_n}.")
    _cpu = _os.cpu_count() or 1
    if _n > _cpu:
        _warnings.warn(
            f"MDAPY_NUM_THREADS={_n} exceeds os.cpu_count()={_cpu}; "
            "oversubscription typically degrades performance.",
            stacklevel=2,
        )
    _os.environ["POLARS_MAX_THREADS"] = str(_n)
    del _n, _cpu
del _env, _os, _warnings

from mdapy.parallel import get_num_threads
from mdapy.system import System
from mdapy.box import Box
from mdapy.build_lattice import build_crystal, build_hea
from mdapy.create_polycrystal import CreatePolycrystal
from mdapy.eam import EAM, EAMAverage, EAMGenerator
from mdapy.nep import NEP
from mdapy.elastic import get_elastic_constant
from mdapy.mean_squared_displacement import MeanSquaredDisplacement
from mdapy.minimizer import FIRE
from mdapy.plotset import set_figure, save_figure
from mdapy.spline import Spline
from mdapy.pigz import compress_file
from mdapy.wigner_seitz_defect import WignerSeitzAnalysis
from mdapy.atomic_strain import AtomicStrain
from mdapy.trajectory import XYZTrajectory, Trajectory
from mdapy.lindemann_parameter import LindemannParameter
from mdapy.void_analysis import VoidAnalysis
from mdapy.orthogonal_cell import orthogonal_cell
from mdapy.sqs import SQS
from mdapy.bond_stiffness import BondStiffness
from mdapy.unwrap_trajectory import unwrap_trajectory

__all__ = [
    "get_num_threads",
    "System",
    "Box",
    "build_crystal",
    "build_hea",
    "CreatePolycrystal",
    "EAM",
    "EAMAverage",
    "EAMGenerator",
    "NEP",
    "get_elastic_constant",
    "MeanSquaredDisplacement",
    "FIRE",
    "set_figure",
    "save_figure",
    "compress_file",
    "Spline",
    "WignerSeitzAnalysis",
    "AtomicStrain",
    "XYZTrajectory",
    "Trajectory",
    "LindemannParameter",
    "VoidAnalysis",
    "orthogonal_cell",
    "SQS",
    "BondStiffness",
    "unwrap_trajectory",
]
__version__ = "1.0.7a1"
