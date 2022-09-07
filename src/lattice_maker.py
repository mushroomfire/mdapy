import taichi as ti
import numpy as np
import sys


@ti.data_oriented
class LatticeMaker:
    def __init__(self, lattice_constant, lattice_type, x, y, z):
        self.lattice_constant = lattice_constant
        self.lattice_type = lattice_type
        self.x = int(x)
        self.y = int(y)
        self.z = int(z)
        self.basis_vector, self.basis_atoms, self.pos = self.init_global()

    def init_global(self):
        """
        定义基矢量,基原子,坐标.
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
                np.array([[3.0, 0.0, 0.0], [0.0, np.sqrt(3), 0.0], [0.0, 0.0, 3.4]])
                * self.lattice_constant
            )
            basis_atoms_arr = np.array(
                [[1 / 6, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [2 / 3, 0.5, 0.0]]
            ) * np.array(
                [self.lattice_constant * 3, self.lattice_constant * np.sqrt(3), 0.0]
            )
        else:
            print("Unrecgonized Lattice Type, please choose in [FCC, BCC, HCP, GRA].")
            sys.exit()

        basis_vector = ti.Vector.field(
            basis_vector_arr.shape[1], dtype=ti.f32, shape=(basis_vector_arr.shape[0])
        )
        basis_atoms = ti.Vector.field(
            basis_atoms_arr.shape[1], dtype=ti.f32, shape=(basis_atoms_arr.shape[0])
        )
        basis_vector.from_numpy(basis_vector_arr)
        basis_atoms.from_numpy(basis_atoms_arr)
        pos = ti.Vector.field(
            3, dtype=ti.f32, shape=(self.x, self.y, self.z, basis_atoms.shape[0])
        )

        return basis_vector, basis_atoms, pos

    @ti.kernel
    def compute(self):
        """
        建立坐标.
        """
        for i, j, k, h in ti.ndrange(self.x, self.y, self.z, self.basis_atoms.shape[0]):
            basis_origin = ti.Vector(
                [
                    self.basis_vector[m].dot(ti.Vector([i, j, k]))
                    for m in range(self.basis_vector.shape[0])
                ]
            )
            self.pos[i, j, k, h] = self.basis_atoms[h] + basis_origin
