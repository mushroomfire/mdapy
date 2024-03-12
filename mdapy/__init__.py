# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

__author__ = "mushroomfire aka HerrWu"
__version__ = "0.10.4"
__license__ = "BSD License"

from .main import main
from .ackland_jones_analysis import AcklandJonesAnalysis
from .calculator import Calculator
from .centro_symmetry_parameter import CentroSymmetryParameter
from .cluser_analysis import ClusterAnalysis
from .common_neighbor_analysis import CommonNeighborAnalysis
from .common_neighbor_parameter import CommonNeighborParameter
from .create_polycrystalline import CreatePolycrystalline
from .eam_average import EAMAverage
from .eam_generate import EAMGenerate
from .entropy import AtomicEntropy
from .identify_SFs_TBs import IdentifySFTBinFCC
from .nearest_neighbor import NearestNeighbor
from .lattice_maker import LatticeMaker
from .lindemann_parameter import LindemannParameter
from .mean_squared_displacement import MeanSquaredDisplacement
from .neighbor import Neighbor
from .pair_distribution import PairDistribution
from .plotset import pltset, pltset_old, cm2inch, set_figure
from .polyhedral_template_matching import PolyhedralTemplateMatching
from .potential import EAM
from .phonon import Phonon
from .spatial_binning import SpatialBinning
from .steinhardt_bond_orientation import SteinhardtBondOrientation
from .system import System, MultiSystem
from .temperature import AtomicTemperature
from .timer import timer
from .void_distribution import VoidDistribution
from .voronoi_analysis import VoronoiAnalysis
from .warren_cowley_parameter import WarrenCowleyParameter
from .replicate import Replicate
from .pigz import compress_file
from .tool_function import atomic_numbers, vdw_radii, atomic_masses

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def init(
    arch="cpu",
    cpu_max_num_threads=-1,
    offline_cache=False,
    debug=False,
    device_memory_fraction=0.9,
    kernel_profiler=False,
):
    """Initilize the mdapy calculation. One should call this function after import mdapy.
    This is a simple wrapper function of `taichi.init() <https://docs.taichi-lang.org/api/taichi/#taichi.init>`_.

    Args:
        arch (str, optional): run on CPU or GPU. Defaults to "cpu", choose in 'cpu' and 'gpu'.

        cpu_max_num_threads (int, optional): maximum CPU core to use in calculation. Defaults to -1, indicating using all available CPU cores.

        offline_cache (bool, optional): whether save compile cache. Defaults to False.

        debug (bool, optional): whether use debug mode. Defaults to False.

        device_memory_fraction (float, optional): preallocate available GPU memory fraction. Defaults to 90%.

        kernel_profiler (bool, optional): whether enable profiler. Defaults to False.

    Raises:
        ValueError: Unrecognized arch, please choose in ['cpu', 'gpu'].
    """
    import taichi as ti

    if arch == "cpu":
        if cpu_max_num_threads == -1:
            ti.init(
                arch=ti.cpu,
                offline_cache=offline_cache,
                debug=debug,
                kernel_profiler=kernel_profiler,
                default_fp=ti.f64,
            )
        else:
            ti.init(
                arch=ti.cpu,
                cpu_max_num_threads=cpu_max_num_threads,
                offline_cache=offline_cache,
                debug=debug,
                kernel_profiler=kernel_profiler,
                default_fp=ti.f64,
            )
    elif arch == "gpu":
        ti.init(
            arch=ti.gpu,
            offline_cache=offline_cache,
            device_memory_fraction=device_memory_fraction,
            debug=debug,
            kernel_profiler=kernel_profiler,
            default_fp=ti.f64,
        )
    else:
        raise ValueError("Unrecognized arch, please choose in ['cpu', 'gpu'].")
