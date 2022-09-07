__author__ = "HerrWu"
__version__ = "0.1.0"

from .src.lattice_maker import LatticeMaker
from .src.system import System
from .src.neighbor import Neighbor
import taichi as ti

ti.init(arch=ti.cpu)
