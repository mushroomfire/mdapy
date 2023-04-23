# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import taichi as ti
import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import csv


@ti.data_oriented
class Replicate:
    """This class used to replicate a position with np.array format.

    Args:
        pos (np.ndarray): (:math:`N_p, 3`), initial position to be replicated.
        box (np.ndarray): (:math:`3, 2`), initial system box.
        x (int, optional): replication number (positive integer) along x axis. Defaults to 1.
        y (int, optional): replication number (positive integer) along y axis. Defaults to 1.
        z (int, optional): replication number (positive integer) along z axis. Defaults to 1.

    Outputs:
        - **box** (np.ndarray) - (:math:`3, 2`), replicated box.
        - **pos** (np.ndarray) - (:math:`N, 3`), replicated particle position.
        - **N** (int) - replicated atom number.

    Examples:
        >>> import mdapy as mp

        >>> mp.init()

        >>> FCC = mp.LatticeMaker(3.615, 'FCC', 10, 10, 10) # Create a FCC structure.

        >>> FCC.compute() # Get atom positions.

        >>> repli = mp.Replicate(FCC.pos, FCC.box, 5, 5, 5) # Initilize a replication class.

        >>> repli.compute() # Get replicated positions.

        >>> repli.pos # Check replicated positions.

        >>> repli.write_data() # Save to DATA file.
    """

    def __init__(self, pos, box, x=1, y=1, z=1):
        self.old_pos = pos
        self.old_box = box
        assert x > 0 and isinstance(x, int), "x should be a positive integer."
        self.x = x
        assert y > 0 and isinstance(y, int), "y should be a positive integer."
        self.y = y
        assert z > 0 and isinstance(z, int), "z should be a positive integer."
        self.z = z
        self.if_computed = False

    @ti.kernel
    def _compute(
        self,
        old_pos: ti.types.ndarray(dtype=ti.math.vec3),
        old_box: ti.types.ndarray(),
        pos: ti.types.ndarray(dtype=ti.math.vec3),
    ):
        box_length = ti.Vector([old_box[i, 1] - old_box[i, 0] for i in range(3)])
        for i, j, k, m in ti.ndrange(self.x, self.y, self.z, old_pos.shape[0]):
            move = box_length * ti.Vector([i, j, k])
            pos[i, j, k, m] = old_pos[m] + move

    def compute(self):
        """Do the real lattice calculation."""
        pos = np.zeros((self.x, self.y, self.z, self.old_pos.shape[0], 3))
        self._compute(self.old_pos, self.old_box, pos)
        self.pos = pos.reshape(-1, 3)
        self.box = self.old_box.copy()
        self.box[:, 1] = self.box[:, 0] + (
            self.old_box[:, 1] - self.old_box[:, 0]
        ) * np.array([self.x, self.y, self.z])
        self.if_computed = True

    @property
    def N(self):
        """The particle number."""
        if not self.if_computed:
            self.compute()
        return self.pos.shape[0]

    def write_data(self, type_list=None, output_name=None):
        """This function writes position into a DATA file.

        Args:
            type_list (np.ndarray, optional): (:math:`N_p`) atom type list. If not given, the atom type is set as 1.

            output_name (str, optional): filename of generated DATA file.
        """
        if not self.if_computed:
            self.compute()

        if output_name is None:
            output_name = f"{self.x}-{self.y}-{self.z}.data"
        if type_list is None:
            type_list = np.ones(self.N, int)
            Ntype = 1
        else:
            assert len(type_list) == self.N
            type_list = np.array(type_list, int)
            Ntype = len(np.unique(type_list))
        df = pd.DataFrame(
            {
                "id": np.arange(1, self.N + 1),
                "type": type_list,
                "x": self.pos[:, 0].astype(np.float32),
                "y": self.pos[:, 1].astype(np.float32),
                "z": self.pos[:, 2].astype(np.float32),
            }
        )
        table = pa.Table.from_pandas(df)
        with pa.OSFile(output_name, "wb") as op:
            op.write("# LAMMPS data file generated by mdapy@HerrWu.\n\n".encode())
            op.write(f"{self.N} atoms\n{Ntype} atom types\n\n".encode())
            for i, j in zip(range(3), ["x", "y", "z"]):
                op.write(f"{self.box[i,0]} {self.box[i,1]} {j}lo {j}hi\n".encode())
            op.write("\n".encode())
            op.write(r"Atoms # atomic".encode())
            op.write("\n\n".encode())
            write_options = csv.WriteOptions(delimiter=" ", include_header=False)
            csv.write_csv(table, op, write_options=write_options)

    def write_dump(self, type_list=None, output_name=None):
        """This function writes position into a DUMP file.

        Args:
            type_list (np.ndarray, optional): (:math:`N_p`) atom type list. If not given, the atom type is set as 1.

            output_name (str, optional): filename of generated DUMP file.
        """
        if not self.if_computed:
            self.compute()

        if output_name is None:
            output_name = f"{self.x}-{self.y}-{self.z}.dump"
        if type_list is None:
            type_list = np.ones(self.N, int)
        else:
            assert len(type_list) == self.N
            type_list = np.array(type_list, int)
        df = pd.DataFrame(
            {
                "id": np.arange(1, self.N + 1),
                "type": type_list,
                "x": self.pos[:, 0].astype(np.float32),
                "y": self.pos[:, 1].astype(np.float32),
                "z": self.pos[:, 2].astype(np.float32),
            }
        )

        table = pa.Table.from_pandas(df)
        with pa.OSFile(output_name, "wb") as op:
            op.write("ITEM: TIMESTEP\n0\n".encode())
            op.write("ITEM: NUMBER OF ATOMS\n".encode())
            op.write(f"{self.N}\n".encode())
            op.write(f"ITEM: BOX BOUNDS pp pp pp\n".encode())
            op.write(f"{self.box[0, 0]} {self.box[0, 1]}\n".encode())
            op.write(f"{self.box[1, 0]} {self.box[1, 1]}\n".encode())
            op.write(f"{self.box[2, 0]} {self.box[2, 1]}\n".encode())
            op.write("ITEM: ATOMS id type x y z\n".encode())
            write_options = csv.WriteOptions(delimiter=" ", include_header=False)
            csv.write_csv(table, op, write_options=write_options)


if __name__ == "__main__":
    ti.init()

    from lattice_maker import LatticeMaker

    fcc = LatticeMaker(3.615, "FCC", 10, 10, 10)
    fcc.compute()
    print(f"build {fcc.N} atoms...")

    repli = Replicate(fcc.pos - 10, fcc.box - 10, 10, 5, 10)
    repli.compute()
    print(f"replicate {repli.x} {repli.y} {repli.z}...")
    print(repli.box)
    # repli.write_data()
    # repli.write_dump()