__author__ = "HerrWu"
__version__ = "0.6.2"

from .src.lattice_maker import LatticeMaker
from .src.system import System
from .src.neighbor import Neighbor
from .src.temperature import AtomicTemperature
from .src.entropy import AtomicEntropy
from .src.centro_symmetry_parameter import CentroSymmetryParameter
from .src.pair_distribution import PairDistribution
from .src.cluser_analysis import ClusterAnalysis
from .src.common_neighbor_analysis import CommonNeighborAnalysis
from .src.adaptive_common_neighbor_analysis import AdaptiveCommonNeighborAnalysis
from .plot.pltset import pltset, cm2inch

from .src.potential import EAM
from .src.calculator import Calculator

import taichi.profiler as profiler


def init(
    arch="cpu",
    offline_cache=False,
    debug=False,
    device_memory_GB=2.0,
    kernel_profiler=False,
):
    """
    arch : str, "cpu" or "gpu", default is "cpu".
    debuge : bool, default is False.
    offline_cache : bool, defults is False.
    device_memory_GB : float, memory for GPU only, default is 2 GB.
    kernel_profiler : bool, default is False.
    """
    import taichi as ti

    if arch == "cpu":
        ti.init(
            arch=ti.cpu,
            offline_cache=offline_cache,
            debug=debug,
            kernel_profiler=kernel_profiler,
        )
    elif arch == "gpu":
        ti.init(
            arch=ti.gpu,
            offline_cache=offline_cache,
            device_memory_GB=device_memory_GB,
            debug=debug,
            kernel_profiler=kernel_profiler,
        )
    else:
        raise ValueError("Unrecognized arch, please choose in ['cpu', 'gpu'].")
