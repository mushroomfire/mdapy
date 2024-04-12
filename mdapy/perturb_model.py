# Copyright (c) 2022-2024, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import numpy as np
import os

try:
    from .system import System
    from .lattice_maker import LatticeMaker
except Exception:
    from system import System
    from lattice_maker import LatticeMaker


class PerturbModel:

    def __init__(
        self,
        filename=None,
        lattice_type=None,
        lattice_constant=None,
        x=1,
        y=1,
        z=1,
        crystalline_orientation=None,
        scale_list=None,
        pert_num=None,
        pert_box=None,
        pert_atom=None,
        type_list=None,
        type_name=None,
        save_path="res",
        save_type="cp2k",
        fmt=None,
    ) -> None:
        if filename is not None:
            system = System(filename, fmt=fmt)
            system.replicate(x, y, z)
            self.pos = system.pos
            self.box = system.box
            self.type_list = system.data["type"].to_numpy()
        else:
            assert (
                lattice_constant is not None and lattice_type is not None
            ), "If not filename, must provide lattice_constant and lattice_type."
            lat = LatticeMaker(
                lattice_constant,
                lattice_type,
                x,
                y,
                z,
                type_list=type_list,
                crystalline_orientation=crystalline_orientation,
            )
            lat.compute()
            self.pos = lat.pos
            self.box = lat.box
            self.type_list = lat.type_list

        self.scale_list = scale_list
        self.pert_num = pert_num
        self.pert_box = pert_box
        self.pert_atom = pert_atom
        self.type_name = type_name
        self.save_path = save_path
        assert save_type in [
            "cp2k",
            "vasp",
        ], "For cp2k, we save .cif and .xyz; for vasp, we save POSCAR."
        self.save_type = save_type
        self.eye_matrix = np.eye(3)

    def _generate_box_atom(self):
        if self.pert_box is not None:
            pert_box = abs(self.pert_box)
            pert_box_trans = (
                pert_box * 2 * np.random.random_sample((3, 3))
                - pert_box
                + self.eye_matrix
            )
        else:
            pert_box_trans = self.eye_matrix

        if self.pert_atom is not None:
            pert_atom = abs(self.pert_atom)
            pert_atom_trans = (
                pert_atom * 2 * np.random.random_sample(self.pos.shape) - pert_atom
            )
        else:
            pert_atom_trans = 0
        return pert_box_trans, pert_atom_trans

    def compute(self, init_type="init_bulk", direction="z", distance=10.0):
        assert init_type in [
            "init_bulk",
            "init_surf",
        ], "Only support init_bulk and init_surf."

        surf = {"x": 0, "y": 1, "z": 2}
        for scale in self.scale_list:
            os.makedirs(f"{self.save_path}/scale_{scale}", exist_ok=True)
            system = System(
                pos=self.pos * scale,
                box=np.r_[self.box[:-1] * scale, self.box[-1].reshape(1, -1)],
                type_list=self.type_list,
            )
            if init_type == "init_surf":
                system.box[-1, surf[direction]] -= distance
                system.box[surf[direction], surf[direction]] += 2 * distance

            if self.save_type == "cp2k":
                system.write_cif(
                    output_name=f"{self.save_path}/scale_{scale}/0.cif",
                    type_name=self.type_name,
                )
                system.write_xyz(
                    output_name=f"{self.save_path}/scale_{scale}/0.xyz",
                    type_name=self.type_name,
                    classical=True,
                )
            elif self.save_type == "vasp":
                system.write_POSCAR(
                    output_name=f"{self.save_path}/scale_{scale}/0.POSCAR",
                    type_name=self.type_name,
                )
            for j in range(1, int(abs(self.pert_num))):
                pert_box_trans, pert_atom_trans = self._generate_box_atom()
                system = System(
                    pos=(self.pos + pert_atom_trans) * scale,
                    box=np.r_[
                        (self.box[:-1] * scale) @ pert_box_trans,
                        self.box[-1].reshape(1, -1),
                    ],
                    type_list=self.type_list,
                )
                if init_type == "init_surf":
                    system.box[-1, surf[direction]] -= distance
                    system.box[surf[direction], surf[direction]] += 2 * distance
                if self.save_type == "cp2k":
                    system.write_cif(
                        output_name=f"{self.save_path}/scale_{scale}/{j}.cif",
                        type_name=self.type_name,
                    )
                    system.write_xyz(
                        output_name=f"{self.save_path}/scale_{scale}/{j}.xyz",
                        type_name=self.type_name,
                        classical=True,
                    )
                elif self.save_type == "vasp":
                    system.write_POSCAR(
                        output_name=f"{self.save_path}/scale_{scale}/{j}.POSCAR",
                        type_name=self.type_name,
                    )


if __name__ == "__main__":
    import taichi as ti

    ti.init()

    pert = PerturbModel(
        # lattice_type="GRA",
        # lattice_constant=1.42,
        filename=r"D:\Study\Gra-Al\ref\ref\Al4C3.poscar",
        x=2,
        y=2,
        z=1,
        scale_list=[0.8, 1.0, 1.2],
        pert_num=10,
        pert_atom=0.05,
        pert_box=0.05,
        type_name=["Al", "C"],
        save_path="Al4C3",
        fmt="POSCAR",
        save_type="vasp",
    )
    # pert.compute()
    # pert.save_path = "FCC_surf"
    pert.compute("init_surf")
