# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import taichi as ti
import numpy as np


@ti.data_oriented
class LatticeMaker:
    """This class is used to create some standard lattice structure.

    Args:
        lattice_constant (float): lattice constant :math:`a`.
        lattice_type (str): lattice type, seleted in ['FCC', 'BCC', 'HCP', 'GRA']. Here the HCP is ideal structure and the :math:`c/a=1.633`.
        x (int): repeat times along :math:`x` axis.
        y (int): repeat times along :math:`y` axis.
        z (int): repeat times along :math:`z` axis.

    Outputs:
        - **box** (np.ndarray) - (:math:`3, 2`), system box.
        - **pos** (np.ndarray) - (:math:`N_p, 3`), particle position.

    Examples:
        >>> import mdapy as mp

        >>> mp.init()

        >>> FCC = mp.LatticeMaker(3.615, 'FCC', 10, 10, 10) # Create a FCC structure.

        >>> FCC.compute() # Get atom positions.

        >>> FCC.write_data() # Save to DATA file.

        >>> GRA = mp.LatticeMaker(1.42, 'GRA', 10, 20, 1) # Create a graphene structure.

        >>> GRA.write_data(output_name='graphene.data') # Save to graphene.data file.
    """

    def __init__(self, lattice_constant, lattice_type, x, y, z):

        self.lattice_constant = lattice_constant
        self.lattice_type = lattice_type
        self.x = x
        self.y = y
        self.z = z
        self.basis_vector, self.basis_atoms = self._init_global()
        self.box = np.vstack(
            (
                np.zeros(3),
                np.diagonal(
                    self.basis_vector.to_numpy() * np.array([self.x, self.y, self.z])
                ),
            )
        ).T
        self.if_computed = False

    def _init_global(self):
        """
        Define the base vector and base atoms in float64.
        """
        if self.lattice_type == "FCC":
            basis_vector_arr = (
                np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
                * self.lattice_constant
            )
            basis_atoms_arr = (
                np.array(
                    [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]
                )
                * self.lattice_constant
            )
        elif self.lattice_type == "BCC":
            basis_vector_arr = (
                np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
                * self.lattice_constant
            )
            basis_atoms_arr = (
                np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]) * self.lattice_constant
            )
        elif self.lattice_type == "HCP":
            basis_vector_arr = (
                np.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, np.sqrt(3), 0.0],
                        [0.0, 0.0, np.sqrt(8 / 3)],
                    ]
                )
                * self.lattice_constant
            )
            basis_atoms_arr = (
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.5, 0.5 * np.sqrt(3), 0.0],
                        [0.5, np.sqrt(3) * 5 / 6, 0.5 * np.sqrt(8 / 3)],
                        [0.0, 1 / 3 * np.sqrt(3), 0.5 * np.sqrt(8 / 3)],
                    ]
                )
                * self.lattice_constant
            )
        elif self.lattice_type == "GRA":
            basis_vector_arr = (
                np.array(
                    [
                        [3.0, 0.0, 0.0],
                        [0.0, np.sqrt(3), 0.0],
                        [0.0, 0.0, 3.4 / self.lattice_constant],
                    ]
                )
                * self.lattice_constant
            )
            basis_atoms_arr = np.array(
                [[1 / 6, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [2 / 3, 0.5, 0.0]]
            ) * np.array(
                [self.lattice_constant * 3, self.lattice_constant * np.sqrt(3), 0.0]
            )
        else:
            raise ValueError(
                "Unrecgonized Lattice Type, please choose in [FCC, BCC, HCP, GRA]."
            )

        basis_vector = ti.Vector.field(
            basis_vector_arr.shape[1], dtype=ti.f64, shape=(basis_vector_arr.shape[0])
        )
        basis_atoms = ti.Vector.field(
            basis_atoms_arr.shape[1], dtype=ti.f64, shape=(basis_atoms_arr.shape[0])
        )
        basis_vector.from_numpy(basis_vector_arr)
        basis_atoms.from_numpy(basis_atoms_arr)

        return basis_vector, basis_atoms

    @ti.kernel
    def _compute(self, pos: ti.types.ndarray(dtype=ti.math.vec3)):
        """
        Get the position
        """
        # ti.loop_config(serialize=True)
        for i, j, k, h in ti.ndrange(self.x, self.y, self.z, self.basis_atoms.shape[0]):
            basis_origin = ti.Vector(
                [
                    self.basis_vector[m].dot(ti.Vector([i, j, k]))
                    for m in range(self.basis_vector.shape[0])
                ]
            )
            pos[i, j, k, h] = self.basis_atoms[h] + basis_origin

    def compute(self):
        """Do the real lattice calculation."""
        pos = np.zeros(
            (self.x, self.y, self.z, self.basis_atoms.shape[0], 3), dtype=np.float64
        )

        self._compute(pos)
        self.pos = pos.reshape(-1, 3)
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
            output_name = f"{self.lattice_type}-{self.x}-{self.y}-{self.z}.data"
        if type_list is None:
            type_list = [1] * self.N
            Ntype = 1
        else:
            assert len(type_list) == self.N
            Ntype = len(np.unique(type_list))

        with open(output_name, "w") as op:
            op.write("# LAMMPS data file written by mdapy@HerrWu.\n\n")
            op.write(f"{self.N} atoms\n{Ntype} atom types\n\n")
            for i, j in zip(range(3), ["x", "y", "z"]):
                op.write(f"{self.box[i,0]} {self.box[i,1]} {j}lo {j}hi\n")
            op.write("\n")
            op.write(r"Atoms # atomic")
            op.write("\n\n")
            for i in range(self.N):
                op.write(
                    f"{i+1} {type_list[i]} {self.pos[i,0]:.6f} {self.pos[i,1]:.6f} {self.pos[i,2]:.6f}\n"
                )


if __name__ == "__main__":
    ti.init(ti.gpu)
    # FCC = LatticeMaker(1.42, "GRA", 10, 20, 3)
    FCC = LatticeMaker(1.42, "GRA", 10, 10, 10)
    print(FCC.N)
    # FCC.compute()
    # FCC.write_data()
    # print(FCC.basis_atoms.to_numpy())
    # print(FCC.basis_vector.to_numpy())
    # print(FCC.basis_vector.to_numpy() * np.array([FCC.x, FCC.y, FCC.z]))
    print(FCC.box)
    # print(FCC.pos)
    # print(pos.dtype)
    # FCC.write_data()
