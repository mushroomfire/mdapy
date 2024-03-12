# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import taichi as ti
import numpy as np
import polars as pl

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
        crystalline_orientation (np.ndarray, optional): (:math:`3, 3`). Crystalline orientation, only support for 'FCC' and 'BCC' lattice. If not given, the orientation if x[1, 0, 0], y[0, 1, 0], z[0, 0, 1].
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

        >>> BCC = mp.LatticeMaker(3.615, 'FCC', 10, 10, 10,
                  crystalline_orientation=np.array([[1, 1, 1], [1, -1, 0], [1, 1, -2]])) # Create a BCC lattice with special orientations.

        >>> BCC.compute() # Get atom positions.

        >>> BCC.write_dump() # Save to dump file.
    """

    def __init__(
        self,
        lattice_constant=None,
        lattice_type=None,
        x=1,
        y=1,
        z=1,
        crystalline_orientation=None,
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
            self.crystalline_orientation = crystalline_orientation
            if self.crystalline_orientation is not None:
                assert self.lattice_type in [
                    "FCC",
                    "BCC",
                ], "crystalline orientation only supports FCC and BCC lattice."
                assert self.crystalline_orientation.shape == (
                    3,
                    3,
                ), "It should be (3 x 3) array."
                assert self._check_orthogonal(), "three vector must be orthogonal"
            self.basis_vector, self.basis_atoms = self._get_basis_vector_atoms()
        self.type_list = type_list
        if self.type_list is not None:
            assert len(self.type_list) == self.basis_atoms.shape[0]
        self.if_computed = False

    def _check_orthogonal(self):
        v1, v2, v3 = (
            self.crystalline_orientation[0],
            self.crystalline_orientation[1],
            self.crystalline_orientation[2],
        )
        if np.dot(v1, v2) == 0 and np.dot(v1, v3) == 0 and np.dot(v2, v3) == 0:
            return True
        else:
            return False

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
        if self.crystalline_orientation is not None:
            n = int(np.abs(self.crystalline_orientation).max() * 5)
            repli = Replicate(basis_atoms, basis_vector, n, n, n)
            repli.compute()
            length = np.linalg.norm(self.crystalline_orientation, axis=1)
            basis_vector = basis_vector * (length / self._get_ref())
            pos = repli.pos.copy()
            # rotate the atoms
            pos = np.matmul(
                pos, (self.crystalline_orientation / np.expand_dims(length, 1)).T
            )
            pos = pos - np.mean(pos, axis=0) + np.diag(basis_vector) / 2
            basis_atoms = pos[
                (pos[:, 0] > -1e-12)
                & (pos[:, 1] > -1e-12)
                & (pos[:, 2] > -1e-12)
                & (pos[:, 0] < basis_vector[0, 0] - 1e-12)
                & (pos[:, 1] < basis_vector[1, 1] - 1e-12)
                & (pos[:, 2] < basis_vector[2, 2] - 1e-12)
            ]
        return basis_vector, basis_atoms

    def _get_ref(self):
        ref = np.ones(3, int)
        for i in range(3):
            num = 0
            for j in self.crystalline_orientation[i]:
                if j % 2:
                    num += 1
            if num == 2 and self.lattice_type == "FCC":
                ref[i] = 2
            if num == 3 and self.lattice_type == "BCC":
                ref[i] = 2
        return ref

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

    def write_xyz(self, output_name=None, type_name=None, classical=False):
        """This function writes position into a xyz file.

        Args:
            output_name (str, optional): filename of generated xyz file. Defaults to None.
            type_name (list, optional): assign the species name. Such as ['Al', 'Cu']. Defaults to None.
            classical (bool, optional): whether save with classical format. Defaults to False.
        """
        if not self.if_computed:
            self.compute()

        if output_name is None:
            output_name = f"{self.x}-{self.y}-{self.z}.xyz"

        data = pl.DataFrame(
            {
                "type": self.type_list,
                "x": self.pos[:, 0],
                "y": self.pos[:, 1],
                "z": self.pos[:, 2],
            }
        )

        if type_name is not None:
            assert len(type_name) == data["type"].unique().shape[0]

            type2name = {i + 1: j for i, j in enumerate(type_name)}

            data = data.with_columns(
                pl.col("type").replace(type2name).alias("type_name")
            ).select("type_name", "x", "y", "z")

        SaveFile.write_xyz(output_name, self.box, data, [1, 1, 1], classical)

    def write_cif(self, output_name=None, type_name=None):
        """This function writes position into a cif file.

        Args:
            output_name (str, optional): filename of generated cif file.
            type_name (list, optional): species name. Such as ['Al', 'Fe'].
        """
        if not self.if_computed:
            self.compute()

        if output_name is None:
            output_name = f"{self.x}-{self.y}-{self.z}.cif"

        if type_name is not None:
            assert len(type_name) == len(np.unique(self.type_list))

        data = pl.DataFrame(
            {
                "type": self.type_list,
                "x": self.pos[:, 0],
                "y": self.pos[:, 1],
                "z": self.pos[:, 2],
            }
        )

        SaveFile.write_cif(output_name, self.box, data, type_name)

    def write_POSCAR(self, output_name=None, type_name=None, reduced_pos=False):
        """This function writes position into a POSCAR file.

        Args:
            output_name (str, optional): filename of generated POSCAR file.
            type_name (list, optional): species name. Such as ['Al', 'Fe'].
            reduced_pos (bool, optional): whether save directed coordination. Defaults to False.
        """
        if not self.if_computed:
            self.compute()

        if output_name is None:
            output_name = f"{self.x}-{self.y}-{self.z}.POSCAR"

        if type_name is not None:
            assert len(type_name) == len(np.unique(self.type_list))

        data = pl.DataFrame(
            {
                "type": self.type_list,
                "x": self.pos[:, 0],
                "y": self.pos[:, 1],
                "z": self.pos[:, 2],
            }
        )

        SaveFile.write_POSCAR(output_name, self.box, data, type_name, reduced_pos)

    def write_data(
        self,
        output_name=None,
        data_format="atomic",
        type_list=None,
        num_type=None,
        type_name=None,
    ):
        """This function writes position into a DATA file.

        Args:
            output_name (str, optional): filename of generated DATA file.
            data_format (str, optional): data format, selected in ['atomic', 'charge'].
            type_list (np.ndarray, optional): one can mannually assign the type_list.
            num_type (int, optional): explictly assign a number of atom type. Defaults to None.
            type_name (list, optional): explictly assign elemantal name list, such as ['Al', 'C']. Defaults to None.
        """
        if not self.if_computed:
            self.compute()

        if output_name is None:
            output_name = f"{self.x}-{self.y}-{self.z}.data"

        if type_list is None:
            type_list = self.type_list
        else:
            assert len(type_list) == self.N

        SaveFile.write_data(
            output_name,
            self.box,
            pos=self.pos,
            type_list=type_list,
            num_type=num_type,
            type_name=type_name,
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
    ti.init(ti.cpu)
    from time import time

    # FCC = LatticeMaker(1.42, "GRA", 10, 20, 3)
    # crystalline_orientation=np.array([[1, 1, 0], [-1, 1, 1], [1, -1, 2]])
    FCC = LatticeMaker(2.8, "HCP", 1, 1, 1)
    # crystalline_orientation=np.array([[1, 1, -2], [1, -1, 0], [1, 1, 1]]),
    FCC.compute()
    print(FCC.box)
    print(FCC.pos)
    # print("Atom number is:", FCC.N)
    # FCC.write_xyz(type_name=["Al", "Cu"])
    # FCC.write_data(type_name=["Al", "Cu", "C"])
    # FCC.write_POSCAR(type_name=["Cu", "Fe"], reduced_pos=True)
    # start = time()
    # FCC.write_data()
    # print(f"write data time {time()-start} s.")

    # start = time()
    # FCC.write_dump()
    # print(f"write dump time {time()-start} s.")

    # print(FCC.basis_atoms.to_numpy())
    # print(FCC.basis_vector.to_numpy())
    # print(FCC.basis_vector.to_numpy() * np.array([FCC.x, FCC.y, FCC.z]))
    # print(FCC.basis_atoms)
    # print(FCC.basis_vector)
    # FCC.write_data(num_type=2)
    # FCC.write_dump(compress=True)
    # print(FCC.pos)
    # print(pos.dtype)
    # FCC.write_data()
