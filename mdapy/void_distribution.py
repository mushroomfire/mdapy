# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import taichi as ti
import numpy as np
import polars as pl

try:
    from neighbor import Neighbor
    from cluser_analysis import ClusterAnalysis
    from load_save_data import SaveFile
except Exception:
    from .neighbor import Neighbor
    from .cluser_analysis import ClusterAnalysis
    from .load_save_data import SaveFile


@ti.data_oriented
class VoidDistribution:
    """This class is used to detect the void distribution in solid structure.
    First we divid particles into three-dimensional grid and check the its
    neighbors, if all neighbor grid is empty we treat this grid is void, otherwise it is
    a point defect. Then useing clustering all connected 'void' grid into an entire void.
    Excepting counting the number and volume of voids, this class can also output the
    spatial coordination of void for analyzing the void distribution.

    .. note:: The results are sensitive to the selection of :math:`cell\_length`, which should
      be illustrated if this class is used in your publication.

    Args:
        pos (np.ndarray): (:math:`N_p, 3`) particles positions.
        box (np.ndarray): (:math:`3, 2`) system box.
        cell_length (float): length of cell, larger than lattice constant is okay.
        boundary (list, optional): boundary conditions, 1 is periodic and 0 is free boundary. Defaults to [1, 1, 1].
        out_void (bool, optional): whether outputs void coordination. Defaults to False.
        out_name (str, optional): filename of generated void DUMP file. Defaults to "void.dump".

    Outputs:
        - **void_number** (int) - total number of voids.
        - **void_volume** (float) - total volume of voids

    Examples:
        >>> import mdapy as mp

        >>> mp.init()

        >>> import numpy as np

        >>> FCC = mp.LatticeMaker(4.05, 'FCC', 50, 50, 50) # Create a FCC structure.

        >>> FCC.compute() # Get atom positions.

        Generate four voids.

        >>> pos = FCC.pos.copy()

        >>> pos = pos[np.sum(np.square(pos - np.array([50, 50, 50])), axis=1) > 100]

        >>> pos = pos[np.sum(np.square(pos - np.array([100, 100, 100])), axis=1) > 100]

        >>> pos = pos[np.sum(np.square(pos - np.array([150, 150, 150])), axis=1) > 400]

        >>> pos = pos[np.sum(np.square(pos - np.array([50, 150, 50])), axis=1) > 400]

        >>> void = mp.VoidDistribution(pos, FCC.box, 5., out_void=True) # Initilize void class.

        >>> void.compute() # Calculated the voids and generate a file named void.dump.

        >>> void.void_number # Check the void number, should be 4 here.

        >>> void.void_volume # Check the void volume.

    """

    def __init__(
        self,
        pos,
        box,
        cell_length,
        boundary=[1, 1, 1],
        out_void=False,
        out_name="void.dump",
    ):
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
        self.cell_length = cell_length
        self.boundary = boundary
        self.out_void = out_void
        self.out_name = out_name
        self.origin = ti.Vector([self.box[0, 0], self.box[1, 0], self.box[2, 0]])
        self.ncel = ti.Vector(
            [
                i if i > 3 else 3
                for i in [
                    int(np.floor((self.box[i, 1] - self.box[i, 0]) / self.cell_length))
                    for i in range(self.box.shape[0])
                ]
            ]
        )

    @ti.kernel
    def _fill_cell(
        self,
        pos: ti.types.ndarray(dtype=ti.math.vec3),
        cell_id_list: ti.types.ndarray(),
        id_list: ti.types.ndarray(),
    ):
        for i in range(self.N):
            r_i = pos[i]
            icel, jcel, kcel = ti.floor(
                (r_i - self.origin) / self.cell_length, dtype=ti.i32
            )
            iicel, jjcel, kkcel = icel, jcel, kcel
            if icel < 0:
                iicel = 0
            elif icel > self.ncel[0] - 1:
                iicel = self.ncel[0] - 1
            if jcel < 0:
                jjcel = 0
            elif jcel > self.ncel[1] - 1:
                jjcel = self.ncel[1] - 1
            if kcel < 0:
                kkcel = 0
            elif kcel > self.ncel[2] - 1:
                kkcel = self.ncel[2] - 1
            id_list[iicel, jjcel, kkcel] = 1

        # ti.loop_config(serialize=True)
        for iicel, jjcel, kkcel in ti.ndrange(self.ncel[0], self.ncel[1], self.ncel[2]):
            num = 0
            for i, j, k in ti.ndrange(
                (iicel - 1, iicel + 2), (jjcel - 1, jjcel + 2), (kkcel - 1, kkcel + 2)
            ):
                ii, jj, kk = i, j, k
                if i < 0:
                    ii += self.ncel[0]
                elif i > self.ncel[0] - 1:
                    ii = self.ncel[0] - 1
                if j < 0:
                    jj += self.ncel[1]
                elif j > self.ncel[1] - 1:
                    jj = self.ncel[1] - 1
                if k < 0:
                    kk += self.ncel[2]
                elif k > self.ncel[2] - 1:
                    kk = self.ncel[2] - 1
                num += id_list[ii, jj, kk]
            if num < 26:  # 26 is empty
                cell_id_list[iicel, jjcel, kkcel] = 1  # is void
            # else:
            #     cell_id_list[iicel, jjcel, kkcel] = 0  # is point defect

    def compute(self):
        """Do the real void calculation."""
        cell_id_list = np.zeros(
            (self.ncel[0], self.ncel[1], self.ncel[2]), dtype=np.int32
        )
        id_list = np.zeros((self.ncel[0], self.ncel[1], self.ncel[2]), dtype=np.int32)
        self._fill_cell(self.pos, cell_id_list, id_list)

        if 1 in cell_id_list:
            void_pos = np.argwhere(cell_id_list == 1) * self.cell_length
            void_pos += self.origin.to_numpy()

            cluster = ClusterAnalysis(
                self.cell_length * 1.1,
                pos=void_pos,
                box=self.box,
                boundary=self.boundary,
            )
            cluster.compute()

            self.void_number = cluster.cluster_number
            self.void_volume = void_pos.shape[0] * self.cell_length**3

            if self.out_void:
                void_data = pl.DataFrame(
                    {
                        "id": np.arange(1, void_pos.shape[0] + 1),
                        "type": np.ones(void_pos.shape[0], int),
                        "x": void_pos[:, 0],
                        "y": void_pos[:, 1],
                        "z": void_pos[:, 2],
                        "cluster_id": cluster.particleClusters,
                    }
                )
                SaveFile.write_dump(self.out_name, self.box, self.boundary, void_data)

        else:
            self.void_number = 0
            self.void_volume = 0.0


if __name__ == "__main__":
    import mdapy as mp
    from time import time

    mp.init()

    # ss = mp.System(r"E:\Al+SiC\compress\compress.154000.dump")

    # start = time()
    # void = VoidDistribution(ss.pos, ss.box, 4.0, out_void=True)
    # void.compute()
    # end = time()
    # print(f"Calculate void time: {end-start} s.")
    # print(void.void_number, void.void_volume)

    # print(ss.vol, void.void_volume, ss.vol-void.void_volume)
    from lattice_maker import LatticeMaker

    # from time import time

    # # ti.init(ti.gpu, device_memory_GB=5.0)
    # ti.init(ti.cpu)
    start = time()
    lattice_constant = 4.05
    x, y, z = 50, 50, 50
    FCC = LatticeMaker(lattice_constant, "FCC", x, y, z)
    FCC.compute()
    end = time()
    print(f"Build {FCC.pos.shape[0]} atoms FCC time: {end-start} s.")
    pos = FCC.pos.copy()

    pos = pos[np.sum(np.square(pos - np.array([50, 50, 50])), axis=1) > 100]
    pos = pos[np.sum(np.square(pos - np.array([100, 100, 100])), axis=1) > 100]
    pos = pos[np.sum(np.square(pos - np.array([150, 150, 150])), axis=1) > 400]
    pos = pos[np.sum(np.square(pos - np.array([50, 150, 50])), axis=1) > 400]

    # FCC.pos = pos
    # FCC.N = pos.shape[0]
    # FCC.write_data()
    print("Generate four voids.")

    start = time()
    void = VoidDistribution(pos, FCC.box, lattice_constant + 1.0, out_void=False)
    void.compute()
    end = time()
    print("void number is:", void.void_number)
    print("void volume is:", void.void_volume)
    print("real valume is:", 4 / 3 * np.pi * (2 * 10**3 + 2 * 20**3))
    print(f"Calculate void time: {end-start} s.")
