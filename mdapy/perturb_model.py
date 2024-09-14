# Copyright (c) 2022-2024, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import numpy as np
import os
from tqdm import tqdm

try:
    from .system import System
    from .lattice_maker import LatticeMaker
except Exception:
    from system import System
    from lattice_maker import LatticeMaker


class PerturbModel:
    """This class is used to generate atomic model with small geometry perturb, which is helpful to preparing
    database for deep learning training.

    Args:
        filename (str, optional): filename one wants to perturb. Defaults to None.
        lattice_type (str, optional): lattice type, selected in ['FCC', 'BCC', 'HCP', 'GRA']. This parameter will be ignored if filename is provides. Defaults to None.
        lattice_constant (float, optional): lattice constant. Defaults to None.
        x (int, optional): replicate along x direction. Defaults to 1.
        y (int, optional): replicate along y direction. Defaults to 1.
        z (int, optional): replicate along z direction. Defaults to 1.
        crystalline_orientation (np.ndarray, optional): (:math:`3, 3`). Crystalline orientation, only support for 'FCC' and 'BCC' lattice. If not given, the orientation is x[1, 0, 0], y[0, 1, 0], z[0, 0, 1]. Defaults to None.
        scale_list (list, optional): one can scale system isotropicly, such as [0.9, 1.0, 1.1]. Defaults to None.
        pert_num (int, optional): number of models per-scale, such as 20. Defaults to None.
        pert_box (float, optional): perturb on box, such as 0.03. Defaults to None.
        pert_atom (float, optional): perturb on atom, such as 0.03. Defaults to None.
        type_list (list[int], optional): type list only for generated lattice, such as [1, 1, 1, 2] for FCC lattice. Defaults to None.
        type_name (list[str], optional): element name, such as ['Al', 'Fe']. Defaults to None.
        save_path (str, optional): save path. Defaults to "res".
        save_type (str, optional): selected in ['cp2k', 'vasp']. For 'vasp', POSCAR will be saved. For 'cp2k', cif and xyz will be saved. Defaults to "cp2k".
        fmt (str, optional): this explitly gives the file format for filename, selected in ["data", "lmp", "POSCAR", "xyz", "cif"]. Defaults to None.

    Outputs:
        - Generate a series of perturb model in save_path.

    Examples:
        >>> import mdapy as mp

        >>> mp.init()

        >>> pert = mp.PerturbModel(
                lattice_type="FCC",
                lattice_constant=4.05,
                x=3,
                y=3,
                z=3,
                scale_list=[0.8, 1.0, 1.2],
                pert_num=20,
                pert_atom=0.03,
                pert_box=0.03,
                type_name=["Al"],
                save_path="Al_FCC",
                save_type="cp2k"
            )

        >>> pert.compute() # check results in './Al_FCC'
    """

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
        ], "For cp2k, we save cp2k; for vasp, we save POSCAR."
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
        """Generate perturb models.

        Args:
            init_type (str, optional): selected in ["init_bulk", "init_surf"]. Defaults to "init_bulk".
            direction (str, optional): the direction to generate vacuum layer. This parameter will be ignored for init_bulk. Defaults to "z".
            distance (float, optional): this length of vacuum layer. This parameter will be ignored for init_bulk. Defaults to 10.0.
        """
        assert init_type in [
            "init_bulk",
            "init_surf",
        ], "Only support init_bulk and init_surf."

        surf = {"x": 0, "y": 1, "z": 2}
        bar = tqdm(self.scale_list)
        for scale in bar:
            bar.set_description(f"Saving scale {scale}")
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
                system.write_cp2k(
                    output_name=f"{self.save_path}/scale_{scale}/0.cp2k",
                    type_name=self.type_name,
                )
                # system.write_cif(
                #     output_name=f"{self.save_path}/scale_{scale}/0.cif",
                #     type_name=self.type_name,
                # )
                # system.write_xyz(
                #     output_name=f"{self.save_path}/scale_{scale}/0.xyz",
                #     type_name=self.type_name,
                #     classical=True,
                # )
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
                    system.write_cp2k(
                        output_name=f"{self.save_path}/scale_{scale}/{j}.cp2k",
                        type_name=self.type_name,
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
        save_type="cp2k",
    )
    pert.compute()
    # pert.save_path = "FCC_surf"
    # pert.compute("init_surf")
