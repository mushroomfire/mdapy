# Copyright (c) 2022-2024, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import numpy as np
import polars as pl

try:
    from lammps import lammps
    from tool_function import atomic_masses, atomic_numbers
except Exception:
    from .tool_function import atomic_masses, atomic_numbers


class CellOptimization:

    def __init__(
        self,
        pos,
        box,
        type_list,
        elements_list,
        boundary,
        pair_parameter,
        units="metal",
        atomic_style="atomic",
        extra_args=None,
        conversion_factor=None,
    ):
        """This class provide a interface to optimize the position and box using lammps.

        Args:
            pos (np.ndarray): (:math:`N_p, 3`) particles positions.
            box (np.ndarray): (:math:`4, 3`) system box.
            type_list (np.ndarray): (:math:`N_p`) atom type list.
            elements_list (list[str]): elements to be calculated, such as ['Al', 'Ni'].
            boundary (list): boundary conditions, 1 is periodic and 0 is free boundary.
            pair_parameter (str): including pair_style and pair_coeff, such as "pair_style eam/alloy\npair_coeff * * example/Al_DFT.eam.alloy Al".
            units (str, optional): lammps units, such as metal, real etc. Defaults to "metal".
            atomic_style (str, optional): atomic_style, such as atomic, charge etc. Defaults to "atomic".
            extra_args (str, optional): any lammps commond. Defaults to None.
            conversion_factor (float, optional): units conversion. Make sure converse the length units to A. Defaults to None.
        """
        self.pos = pos
        self.box = box
        self.type_list = type_list
        self.elements_list = elements_list
        self.boundary = boundary
        self.pair_parameter = pair_parameter
        self.units = units
        self.atomic_style = atomic_style
        self.extra_args = extra_args
        self.conversion_factor = conversion_factor

    def to_lammps_box(self, box):
        xlo, ylo, zlo = box[-1]
        xhi, yhi, zhi = (
            xlo + box[0, 0],
            ylo + box[1, 1],
            zlo + box[2, 2],
        )
        xy, xz, yz = box[1, 0], box[2, 0], box[2, 1]
        xlo_bound = xlo + min(0.0, xy, xz, xy + xz)
        xhi_bound = xhi + max(0.0, xy, xz, xy + xz)
        ylo_bound = ylo + min(0.0, yz)
        yhi_bound = yhi + max(0.0, yz)
        zlo_bound = zlo
        zhi_bound = zhi
        return (
            [xlo_bound, ylo_bound, zlo_bound],
            [xhi_bound, yhi_bound, zhi_bound],
            xy,
            xz,
            yz,
        )

    def to_mdapy_box(self, box):
        boxlo, boxhi, xy, yz, xz, _, _ = box
        xlo_bound, ylo_bound, zlo_bound = boxlo
        xhi_bound, yhi_bound, zhi_bound = boxhi
        xlo = xlo_bound - min(0.0, xy, xz, xy + xz)
        xhi = xhi_bound - max(0.0, xy, xz, xy + xz)
        ylo = ylo_bound - min(0.0, yz)
        yhi = yhi_bound - max(0.0, yz)
        zlo = zlo_bound
        zhi = zhi_bound
        return np.array(
            [
                [xhi - xlo, 0, 0],
                [xy, yhi - ylo, 0],
                [xz, yz, zhi - zlo],
                [xlo, ylo, zlo],
            ]
        )

    def compute(self):
        """Do real computation.

        Returns:
            tuple[pl.DataFrame, np.ndarray]: data, box.
        """
        boundary = " ".join(["p" if i == 1 else "s" for i in self.boundary])

        lmp = lammps()
        try:
            lmp.commands_string(f"units {self.units}")
            lmp.commands_string(f"boundary {boundary}")
            lmp.commands_string(f"atom_style {self.atomic_style}")
            num_type = self.type_list.max()
            create_box = f"""
            region 1 block 0 1 0 1 0 1
            create_box {num_type} 1
            """
            lmp.commands_string(create_box)
            lmp.reset_box(*self.to_lammps_box(self.box))
            N = self.pos.shape[0]
            N_lmp = lmp.create_atoms(
                N, np.arange(1, N + 1), self.type_list, self.pos.flatten()
            )
            assert N == N_lmp, "Wrong atom numbers."

            for i, m in enumerate(self.elements_list, start=1):
                mass = atomic_masses[atomic_numbers[m]]
                lmp.commands_string(f"mass {i} {mass}")
            lmp.commands_string(self.pair_parameter)
            if self.extra_args is not None:
                lmp.commands_string(self.extra_args)
            relax = """fix relax all box/relax x 0.0 y 0.0 z 0.0 vmax 0.001
            min_style cg
            minimize 1.0e-9 1.0e-9 1000 10000"""
            lmp.commands_string(relax)

            index = np.array(lmp.numpy.extract_atom("id"))
            pos = np.array(lmp.numpy.extract_atom("x"))
            type_list = np.array(lmp.numpy.extract_atom("type"))
            box = self.to_mdapy_box(lmp.extract_box())
        except Exception as e:
            print(e)
            raise "Cell optimization failed."
        finally:
            lmp.close()

        if self.conversion_factor is not None:
            pos *= self.conversion_factor
            box *= self.conversion_factor

        data = pl.DataFrame(
            {
                "id": index,
                "type": type_list,
                "x": pos[:, 0],
                "y": pos[:, 1],
                "z": pos[:, 2],
            }
        )
        return data, box


if __name__ == "__main__":
    from lattice_maker import LatticeMaker
    from time import time
    from system import System
    import taichi as ti

    ti.init(ti.cpu)
    start = time()
    lattice_constant = 4.048
    x, y, z = 10, 10, 10
    FCC = LatticeMaker(lattice_constant, "FCC", x, y, z)
    FCC.compute()
    end = time()
    print(f"Build {FCC.pos.shape[0]} atoms FCC time: {end-start} s.")

    start = time()
    cpt = CellOptimization(
        FCC.pos,
        FCC.box,
        FCC.type_list,
        ["Al"],
        [1, 1, 1],
        "pair_style eam/alloy\npair_coeff * * example/Al_DFT.eam.alloy Al",
    )
    data, box = cpt.compute()
    end = time()
    print(f"Cell opt time: {end-start} s.")
    print(data)
    print(box)