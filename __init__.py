__author__ = "HerrWu"
__version__ = "0.4.4"

from .src.lattice_maker import LatticeMaker
from .src.system import System
from .src.neighbor import Neighbor
from .src.temperature import AtomicTemperature
from .src.entropy import AtomicEntropy
from .src.centro_symmetry_parameter import CentroSymmetryParameter
from .plot.pltset import pltset, cm2inch


def init(arch="cpu", debug=False, device_memory_GB=2.0, kernel_profiler=False):
    """
    arch : str, "cpu" or "gpu", default is "cpu".
    debuge : bool, default is False.
    device_memory_GB : float, memory for GPU only, default is 2 GB.
    kernel_profiler : bool, default is False.
    """
    import taichi as ti

    if arch == "cpu":
<<<<<<< HEAD
        ti.init(arch=ti.cpu, debug=debug,kernel_profiler=kernel_profiler)
    elif arch == "gpu":
        ti.init(arch=ti.gpu, device_memory_GB=device_memory_GB, debug=debug, kernel_profiler=kernel_profiler)
=======
        ti.init(arch=ti.cpu, debug=debug, kernel_profiler=kernel_profiler)
    elif arch == "gpu":
        ti.init(
            arch=ti.gpu,
            device_memory_GB=device_memory_GB,
            debug=debug,
            kernel_profiler=kernel_profiler,
        )
>>>>>>> 53a39594cb1dbccddca06cf05d1d9e660bf432ff
    else:
        raise ValueError("Unrecognized arch, please choose in ['cpu', 'gpu'].")
