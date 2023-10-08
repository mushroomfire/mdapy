# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import taichi as ti
import numpy as np

try:
    from replicate import Replicate
    from load_save_data import SaveFile
except Exception:
    from .replicate import Replicate
    from .load_save_data import SaveFile


@ti.data_oriented
class LatticeMaker:
    """This class is used to create some standard lattice structure.

    Args:
        lattice_constant (float): lattice constant :math:`a`.
        lattice_type (str): lattice type, seleted in ['FCC', 'BCC', 'HCP', 'GRA']. Here the HCP is ideal structure and the :math:`c/a=1.633`.
        x (int): repeat times along :math:`x` axis.
        y (int): repeat times along :math:`y` axis.
        z (int): repeat times along :math:`z` axis.
        basis_vector (np.ndarray): (:math:`4, 3`) or (:math:`3, 2`) repeat vector.
        basis_atoms (np.ndarray): (:math:`N_p, 3`) basis atom positions.
        type_list (np.ndarray): (:math:`N_p`) type for basis atoms.

    Outputs:
        - **box** (np.ndarray) - (:math:`4, 3`), system box.
        - **pos** (np.ndarray) - (:math:`N_p, 3`), particle position.
        - **type_list** (np.ndarray) - (:math:`N_p`), replicated type list.

    Examples:
        >>> import mdapy as mp

        >>> mp.init()

        >>> FCC = mp.LatticeMaker(3.615, 'FCC', 10, 10, 10) # Create a FCC structure.

        >>> FCC.compute() # Get atom positions.

        >>> FCC.write_data() # Save to DATA file.

        >>> GRA = mp.LatticeMaker(1.42, 'GRA', 10, 20, 1) # Create a graphene structure.

        >>> GRA.write_data(output_name='graphene.data') # Save to graphene.data file.
    """

    def __init__(
        self,
        lattice_constant=None,
        lattice_type=None,
        x=1,
        y=1,
        z=1,
        basis_vector=None,
        basis_atoms=None,
        type_list=None,
    ):
        self.lattice_constant = lattice_constant
        self.lattice_type = lattice_type
        self.x = x
        self.y = y
        self.z = z
        if self.lattice_constant is None or self.lattice_type is None:
            assert basis_atoms is not None
            assert basis_vector is not None
            assert basis_vector.shape == (3, 2) or basis_vector.shape == (4, 3)
            assert basis_atoms.shape[1] == 3
            self.basis_vector = basis_vector
            self.basis_atoms = basis_atoms
        else:
            assert self.lattice_type in [
                "FCC",
                "BCC",
                "HCP",
                "GRA",
            ], "Unrecgonized Lattice Type, please choose in [FCC, BCC, HCP, GRA]."
            self.basis_vector, self.basis_atoms = self._get_basis_vector_atoms()
        self.type_list = type_list
        if self.type_list is not None:
            assert len(self.type_list) == self.basis_atoms.shape[0]
        self.if_computed = False

    def _get_basis_vector_atoms(self):
        """
        Define the base vector and base atoms in float64.
        """
        if self.lattice_type == "FCC":
            basis_vector = (
                np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
                * self.lattice_constant
            )
            basis_atoms = (
                np.array(
                    [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]
                )
                * self.lattice_constant
            )
        elif self.lattice_type == "BCC":
            basis_vector = (
                np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
                * self.lattice_constant
            )
            basis_atoms = (
                np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]) * self.lattice_constant
            )
        elif self.lattice_type == "HCP":
            basis_vector = (
                np.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, np.sqrt(3), 0.0],
                        [0.0, 0.0, np.sqrt(8 / 3)],
                    ]
                )
                * self.lattice_constant
            )
            basis_atoms = (
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
            basis_vector = (
                np.array(
                    [
                        [3.0, 0.0, 0.0],
                        [0.0, np.sqrt(3), 0.0],
                        [0.0, 0.0, 3.4 / self.lattice_constant],
                    ]
                )
                * self.lattice_constant
            )
            basis_atoms = np.array(
                [[1 / 6, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [2 / 3, 0.5, 0.0]]
            ) * np.array(
                [self.lattice_constant * 3, self.lattice_constant * np.sqrt(3), 0.0]
            )

        return basis_vector, basis_atoms

    def compute(self):
        """Do the real lattice calculation."""
        repli = Replicate(
            self.basis_atoms, self.basis_vector, self.x, self.y, self.z, self.type_list
        )
        repli.compute()
        self.pos = repli.pos
        self.box = repli.box
        if self.type_list is None:
            self.type_list = np.ones(self.pos.shape[0], int)
        else:
            self.type_list = repli.type_list
        self.if_computed = True

    @property
    def N(self):
        """The particle number."""
        if not self.if_computed:
            self.compute()
        return self.pos.shape[0]

    @property
    def vol(self):
        """The box volume."""
        if not self.if_computed:
            self.compute()
        return np.inner(self.box[0], np.cross(self.box[1], self.box[2]))

    def write_data(self, output_name=None, data_format="atomic"):
        """This function writes position into a DATA file.

        Args:
            output_name (str, optional): filename of generated DATA file.
            data_format (str, optional): data format, selected in ['atomic', 'charge']
        """
        if not self.if_computed:
            self.compute()

        if output_name is None:
            output_name = f"{self.x}-{self.y}-{self.z}.data"

        SaveFile.write_data(
            output_name,
            self.box,
            [1, 1, 1],
            pos=self.pos,
            type_list=self.type_list,
            data_format=data_format,
        )

    def write_dump(self, output_name=None, compress=False):
        """This function writes position into a DUMP file.

        Args:
            output_name (str, optional): filename of generated DUMP file.
            compress (bool, optional): whether compress the DUMP file.
        """
        if not self.if_computed:
            self.compute()

        if output_name is None:
            output_name = f"{self.x}-{self.y}-{self.z}.dump"

        if compress:
            if output_name.split(".")[-1] != "gz":
                output_name += ".gz"

        SaveFile.write_dump(
            output_name,
            self.box,
            [1, 1, 1],
            pos=self.pos,
            type_list=self.type_list,
            compress=compress,
        )


if __name__ == "__main__":
    ti.init(ti.gpu)
    from time import time

    # FCC = LatticeMaker(1.42, "GRA", 10, 20, 3)
    FCC = LatticeMaker(4.05, "FCC", 1, 1, 1)
    FCC.compute()
    print("Atom number is:", FCC.N)
    # start = time()
    # FCC.write_data()
    # print(f"write data time {time()-start} s.")

    # start = time()
    # FCC.write_dump()
    # print(f"write dump time {time()-start} s.")

    # print(FCC.basis_atoms.to_numpy())
    # print(FCC.basis_vector.to_numpy())
    # print(FCC.basis_vector.to_numpy() * np.array([FCC.x, FCC.y, FCC.z]))
    print(FCC.box)
    print(FCC.vol)
    FCC.write_data()
    FCC.write_dump(compress=True)
    # print(FCC.pos)
    # print(pos.dtype)
    # FCC.write_data()
