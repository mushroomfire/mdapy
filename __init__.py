__author__ = "HerrWu"
__version__ = "0.7.1"

from .src.lattice_maker import LatticeMaker
from .src.system import System, MultiSystem
from .src.neighbor import Neighbor
from .src.temperature import AtomicTemperature
from .src.entropy import AtomicEntropy
from .src.centro_symmetry_parameter import CentroSymmetryParameter
from .src.pair_distribution import PairDistribution
from .src.cluser_analysis import ClusterAnalysis
from .src.common_neighbor_analysis import CommonNeighborAnalysis
from .plot.pltset import pltset, cm2inch

from .src.kdtree import kdtree
from .src.potential import EAM
from .src.calculator import Calculator
from .src.eam_generate import EAMGenerate
from .src.eam_average import EAMAverage
from .src.void_distribution import VoidDistribution
from .src.warren_cowley_parameter import WarrenCowleyParameter
from .src.voronoi_analysis import VoronoiAnalysis

from .src.create_polycrystalline import CreatePolycrystalline
from .src.mean_squared_displacement import MeanSquaredDisplacement
from .src.lindemann_parameter import LindemannParameter

import taichi.profiler as profiler


def init(
    arch="cpu",
    cpu_max_num_threads=-1,
    offline_cache=False,
    packed=False,
    debug=False,
    device_memory_GB=2.0,
    kernel_profiler=False,
):
    """
    arch : str, "cpu" or "gpu", default is "cpu".
    cpu_max_num_threads : int, number of parallel cpu threads, -1 use all theards in your computer.
    debug : bool, default is False.
    offline_cache : bool, defults is False.
    packed: bool, data layout.
    device_memory_GB : float, memory for GPU only, default is 2 GB.
    kernel_profiler : bool, default is False.
    """

    import taichi as ti

    if arch == "cpu":
        if cpu_max_num_threads == -1:
            ti.init(
                arch=ti.cpu,
                offline_cache=offline_cache,
                packed=packed,
                debug=debug,
                kernel_profiler=kernel_profiler,
            )
        else:
            ti.init(
                arch=ti.cpu,
                cpu_max_num_threads=cpu_max_num_threads,
                offline_cache=offline_cache,
                packed=packed,
                debug=debug,
                kernel_profiler=kernel_profiler,
            )
    elif arch == "gpu":
        ti.init(
            arch=ti.gpu,
            offline_cache=offline_cache,
            packed=packed,
            device_memory_GB=device_memory_GB,
            debug=debug,
            kernel_profiler=kernel_profiler,
        )
    else:
        raise ValueError("Unrecognized arch, please choose in ['cpu', 'gpu'].")
