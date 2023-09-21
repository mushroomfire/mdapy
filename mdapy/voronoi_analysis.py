# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.
# We highly thanks to Dr. Jiayin Lu, Prof. Christipher Rycroft
# and Prof. Emanuel Lazar for the help on parallelism of this module.

import numpy as np
import multiprocessing as mt

if __name__ == "__main__":
    from voronoi import _voronoi_analysis
else:
    import _voronoi_analysis


class VoronoiAnalysis:
    """This class is used to calculate the Voronoi polygon, which can be applied to
    estimate the atomic volume. The calculation is conducted by the `voro++ <https://math.lbl.gov/voro++/>`_ package and
    this class only provides a wrapper. From mdapy v0.8.6, we use extended parallel voro++ to improve the performance, the
    implementation can be found in `An extension to VORO++ for multithreaded computation of Voronoi cells <https://arxiv.org/abs/2209.11606>`_.

    Args:
        pos (np.ndarray): (:math:`N_p, 3`) particles positions.
        box (np.ndarray): (:math:`3, 2`) system box.
        boundary (list): boundary conditions, 1 is periodic and 0 is free boundary, default to [1, 1, 1].
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

    def __init__(self, pos, box, boundary=[1, 1, 1], num_t=None) -> None:
        if pos.dtype != np.float64:
            pos = pos.astype(np.float64)
        self.pos = pos
        self.N = self.pos.shape[0]
        if box.dtype != np.float64:
            box = box.astype(np.float64)
        if box.shape == (4, 3):
            for i in range(3):
                for j in range(3):
                    if i != j:
                        assert box[i, j] == 0, "Do not support triclinic box."
            self.box = np.zeros((3, 2))
            self.box[:, 0] = box[-1]
            self.box[:, 1] = (
                np.array([box[0, 0], box[1, 1], box[2, 2]]) + self.box[:, 0]
            )
        elif box.shape == (3, 2):
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

    FCC = LatticeMaker(4.05, "FCC", 1, 1, 1)  # Create a FCC structure.
    FCC.compute()  # Get atom positions.
    # FCC.write_data()
    start = time()
    avol = VoronoiAnalysis(FCC.pos, FCC.box)  # Initilize the Voronoi class.
    avol.compute()  # Calculate the Voronoi volume.
    end = time()

    print(f"Calculate volume time: {end-start} s.")
    print(avol.vol)  # Check atomic Voronoi volume.
    print(avol.neighbor_number)  # Check neighbor number.
    print(avol.cavity_radius)  # Check the cavity radius.
