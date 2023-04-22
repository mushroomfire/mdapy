# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.
# We highly thanks to Dr. Jiayin Lu for the help on parallelism of this module.

import numpy as np
import multiprocessing as mt

if __name__ == "__main__":
    from voronoi import _voronoi_analysis
else:
    import _voronoi_analysis


class VoronoiAnalysis:
    """This class is used to calculate the Voronoi polygon, wchich can be applied to
    estimate the atomic volume. The calculation is conducted by the `voro++ <https://math.lbl.gov/voro++/>`_ package and
    this class only provides a wrapper.

    Args:
        pos (np.ndarray): (:math:`N_p, 3`) particles positions.
        box (np.ndarray): (:math:`3, 2`) system box.
        boundary (list): boundary conditions, 1 is periodic and 0 is free boundary, such as [1, 1, 1].
        num_t (int, optional): threads number to generate Voronoi diagram. If not given, use all avilable threads.

    Outputs:
        - **vol** (np.ndarray) - (:math:`N_p`), atom Voronoi volume.
        - **neighbor_number** (np.ndarray) - (:math:`N_p`), atom Voronoi neighbor number.
        - **cavity_radius** (np.ndarray) - the distance from the particle to the farthest vertex of its Voronoi cell.

    Examples:
        >>> import mdapy as mp

        >>> mp.init()

        >>> FCC = mp.LatticeMaker(4.05, 'FCC', 10, 10, 10) # Create a FCC structure.

        >>> FCC.compute() # Get atom positions.

        >>> avol = mp.VoronoiAnalysis(FCC.pos, FCC.box, [1, 1, 1]) # Initilize the Voronoi class.

        >>> avol.compute() # Calculate the Voronoi volume.

        >>> avol.vol # Check atomic Voronoi volume.

        >>> avol.neighbor_number # Check neighbor number.

        >>> avol.cavity_radius # Check the cavity radius.
    """

    def __init__(self, pos, box, boundary, num_t=None) -> None:
        self.pos = pos
        self.box = box
        self.boundary = boundary
        if num_t is None:
            self.num_t = mt.cpu_count()
        else:
            assert num_t >= 1, "num_t should be a positive integer!"
            self.num_t = int(num_t)

    def compute(self):
        """Do the real Voronoi volume calculation."""
        N = self.pos.shape[0]
        self.vol = np.zeros(N)
        self.neighbor_number = np.zeros(N, dtype=int)
        self.cavity_radius = np.zeros(N)
        _voronoi_analysis.get_voronoi_volume(
            self.pos,
            self.box,
            np.bool_(self.boundary),
            self.vol,
            self.neighbor_number,
            self.cavity_radius,
            self.num_t,
        )


if __name__ == "__main__":
    import taichi as ti
    from lattice_maker import LatticeMaker
    from time import time

    ti.init()

    FCC = LatticeMaker(4.05, "FCC", 100, 100, 50)  # Create a FCC structure.
    FCC.compute()  # Get atom positions.
    # FCC.write_data()
    start = time()
    avol = VoronoiAnalysis(FCC.pos, FCC.box, [1, 1, 1])  # Initilize the Voronoi class.
    avol.compute()  # Calculate the Voronoi volume.
    end = time()

    print(f"Calculate volume time: {end-start} s.")
    print(avol.vol)  # Check atomic Voronoi volume.
    print(avol.neighbor_number)  # Check neighbor number.
    print(avol.cavity_radius)  # Check the cavity radius.
