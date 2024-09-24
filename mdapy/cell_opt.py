# Copyright (c) 2022-2024, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import numpy as np
import polars as pl
import os

try:
    from lammps import lammps
except Exception:
    raise "One should install lammps-python interface to use this function. Chech the installation guide (https://docs.lammps.org/Python_install.html)."
try:
    from tool_function import atomic_masses, atomic_numbers
except Exception:
    from .tool_function import atomic_masses, atomic_numbers


class CellOptimization:
    """This class provide a interface to optimize the position and box using lammps.

    Args:
        pos (np.ndarray): (:math:`N_p, 3`) particles positions.
        box (np.ndarray): (:math:`4, 3`) system box.
        type_list (np.ndarray): (:math:`N_p`) atom type list.
        elements_list (list[str]): elements to be calculated, such as ['Al', 'Ni'].
        boundary (list): boundary conditions, 1 is periodic and 0 is free boundary.
        pair_parameter (str): including pair_style and pair_coeff.
        units (str, optional): lammps units, such as metal, real etc. Defaults to "metal".
        atomic_style (str, optional): atomic_style, such as atomic, charge etc. Defaults to "atomic".
        extra_args (str, optional): any lammps commond. Defaults to None.
        conversion_factor (float, optional): units conversion. Make sure converse the length units to A. Defaults to None.
    """

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

        return (
            [xlo, ylo, zlo],
            [xhi, yhi, zhi],
            xy,
            xz,
            yz,
        )

    def to_mdapy_box(self, box):
        boxlo, boxhi, xy, yz, xz, _, _ = box
        xlo, ylo, zlo = boxlo
        xhi, yhi, zhi = boxhi

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
            num_type = len(self.elements_list)
            create_box = f"""
            region 1 prism 0 1 0 1 0 1 0 0 0
            create_box {num_type} 1
            """
            lmp.commands_string(create_box)
            pos = self.pos
            box = self.box
            if box[0, 1] != 0 or box[0, 2] != 0 or box[1, 2] != 0:
                old_box = box.copy()
                ax = np.linalg.norm(box[0])
                bx = box[1] @ (box[0] / ax)
                by = np.sqrt(np.linalg.norm(box[1]) ** 2 - bx**2)
                cx = box[2] @ (box[0] / ax)
                cy = (box[1] @ box[2] - bx * cx) / by
                cz = np.sqrt(np.linalg.norm(box[2]) ** 2 - cx**2 - cy**2)
                box = np.array([[ax, bx, cx], [0, by, cy], [0, 0, cz]]).T
                rotation = np.linalg.solve(old_box[:-1], box)
                pos = self.pos @ rotation
                box = np.r_[box, box[-1].reshape(1, -1)]
            # lmp.reset_box(*self.to_lammps_box(box))
            lo, hi, xy, xz, yz = self.to_lammps_box(box)
            lmp.commands_string(
                f"change_box all x final {lo[0]} {hi[0]} y final {lo[1]} {hi[1]} z final {lo[2]} {hi[2]} xy final {xy} xz final {xz} yz final {yz}"
            )

            N = self.pos.shape[0]
            N_lmp = lmp.create_atoms(
                N, np.arange(1, N + 1), self.type_list, pos.flatten()
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
            # print(lmp.extract_box())
            box = self.to_mdapy_box(lmp.extract_box())
        except Exception as e:
            lmp.close()
            os.remove("log.lammps")
            raise e

        lmp.close()
        os.remove("log.lammps")

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
    import taichi as ti
    import mdapy as mp

    ti.init(ti.cpu)
    # start = time()
    # lattice_constant = 4.048
    # x, y, z = 10, 10, 10
    # FCC = LatticeMaker(lattice_constant, "FCC", x, y, z)
    # FCC.compute()
    # end = time()
    # print(f"Build {FCC.pos.shape[0]} atoms FCC time: {end-start} s.")
    # FCC = mp.System(r"D:\Study\Gra-Al\potential_test\phonon\alc\Al4C3.lmp")
    # start = time()
    # cpt = CellOptimization(
    #     FCC.pos,
    #     FCC.box,
    #     FCC.data["type"].to_numpy(),
    #     ["Al", "C"],
    #     [1, 1, 1],
    #     r"""pair_style nep
    #     pair_coeff * * D:\Study\Gra-Al\potential_test\phonon\alc\nep.txt Al C""",
    # )
    # data, box = cpt.compute()
    # end = time()
    # # "pair_style eam/alloy\npair_coeff * * example/Al_DFT.eam.alloy Al"
    # print(f"Cell opt time: {end-start} s.")
    # print(data)
    # print(box)
    FCC = mp.System(r"D:\Package\MyPackage\lammps_nep\example\gra.xyz")

    pair_parameter = """
    pair_style nep
    pair_coeff * * D:\\Package\\MyPackage\\lammps_nep\\example\\C_2024_NEP4.txt C
    """
    elements_list = ["C"]
    # relax_gra = system.cell_opt(pair_parameter, elements_list)
    # print(relax_gra)
    cpt = CellOptimization(
        FCC.pos,
        FCC.box,
        FCC.data["type"].to_numpy(),
        ["C"],
        [1, 1, 1],
        pair_parameter,
    )
    data, box = cpt.compute()
    print(data)
    print(box)
    type_dict = {str(i): j for i, j in enumerate(elements_list, start=1)}
    print(type_dict)
    data = data.with_columns(
        type_name=pl.col("type").cast(pl.Utf8).replace_strict(type_dict)
    )
    print(data)
