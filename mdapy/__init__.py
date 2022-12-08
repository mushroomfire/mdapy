__author__ = "mushroomfire aka HerrWu"
__version__ = "0.7.4"

from .calculator import Calculator
from .centro_symmetry_parameter import CentroSymmetryParameter
from .cluser_analysis import ClusterAnalysis
from .common_neighbor_analysis import CommonNeighborAnalysis
from .create_polycrystalline import CreatePolycrystalline
from .eam_average import EAMAverage
from .eam_generate import EAMGenerate
from .entropy import AtomicEntropy
from .kdtree import kdtree
from .lattice_maker import LatticeMaker
from .lindemann_parameter import LindemannParameter
from .mean_squared_displacement import MeanSquaredDisplacement
from .neighbor import Neighbor
from .pair_distribution import PairDistribution
from .plotset import pltset
from .plotset import cm2inch
from .potential import EAM
from .spatial_binning import SpatialBinning
from .system import System, MultiSystem
from .temperature import AtomicTemperature
from .timer import timer
from .void_distribution import VoidDistribution
from .voronoi_analysis import VoronoiAnalysis
from .warren_cowley_parameter import WarrenCowleyParameter


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
