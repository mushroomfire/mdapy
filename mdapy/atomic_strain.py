# Copyright (c) 2022-2024, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import taichi as ti
import numpy as np

try:
    from box import init_box, _pbc
except Exception:
    from .box import init_box, _pbc


@ti.data_oriented
class AtomicStrain:
    """This class is used to calculate the atomic shear strain. More details can be found here.
    https://www.ovito.org/docs/current/reference/pipelines/modifiers/atomic_strain.html

    Args:
        ref_pos (np.ndarray): position of the reference configuration.
        ref_box (np.ndarray): box of the reference configuration.
        cur_pos (np.array): position of the current configuration.
        cur_box (np.array): box of the current configuration.
        verlet_list (np.ndarray): neighbor atom index for the reference configuration.
        distance_list (np.ndarray): neighbor atom number for the reference configuration..
        boundary (list, optional): boundary condition. Defaults to [1, 1, 1].
        affi_map (str, optional): selected in ['off', 'ref']. If use to 'ref', the current position will affine to the reference frame. Defaults to 'off'.

    Outputs:
        - **shear_strain** (np.ndarray) - shear strain value per atoms.
    """

    def __init__(
        self,
        ref_pos,
        ref_box,
        cur_pos,
        cur_box,
        verlet_list,
        neighbor_number,
        boundary=[1, 1, 1],
        affi_map="off",
    ) -> None:
        self.ref_pos = ref_pos
        self.ref_box, self.ref_box_inv, self.rec = init_box(ref_box)
        self.cur_box, self.cur_box_inv, _ = init_box(cur_box)
        assert affi_map in ["off", "ref"]
        self.affi_map = affi_map
        if self.affi_map == "ref":
            map_matrix = np.linalg.solve(self.cur_box[:-1], self.ref_box[:-1])
            cur_pos = cur_pos @ map_matrix
        self.cur_pos = cur_pos
        self.verlet_list = verlet_list
        self.neighbor_number = neighbor_number
        self.boundary = ti.Vector([int(i) for i in boundary])

    @ti.kernel
    def _cal_atomic_strain(
        self,
        verlet_list: ti.types.ndarray(),
        neighbor_number: ti.types.ndarray(),
        ref_box: ti.types.ndarray(element_dim=1),
        cur_box: ti.types.ndarray(element_dim=1),
        ref_box_inv: ti.types.ndarray(element_dim=1),
        cur_box_inv: ti.types.ndarray(element_dim=1),
        ref_pos: ti.types.ndarray(element_dim=1),
        cur_pos: ti.types.ndarray(element_dim=1),
        shear_strain: ti.types.ndarray(),
    ):
        N = verlet_list.shape[0]
        identy = ti.Matrix(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dt=ti.f64
        )
        for i in range(N):
            V = ti.Matrix([[0.0] * 3 for _ in range(3)], dt=ti.f64)
            W = V
            for jj in range(neighbor_number[i]):
                j = verlet_list[i, jj]
                delta_ref = _pbc(
                    ref_pos[j] - ref_pos[i], self.boundary, ref_box, ref_box_inv
                )
                delta_cur = _pbc(
                    cur_pos[j] - cur_pos[i], self.boundary, cur_box, cur_box_inv
                )
                for m in ti.static(range(3)):
                    for n in ti.static(range(3)):
                        V[m, n] += delta_ref[n] * delta_ref[m]
                        W[m, n] += delta_ref[n] * delta_cur[m]

            V_inv = V.inverse()
            F = (W @ V_inv).transpose()

            s = (F.transpose() @ F - identy) / 2.0
            xydiff = s[0, 0] - s[1, 1]
            yzdiff = s[1, 1] - s[2, 2]
            xzdiff = s[0, 0] - s[2, 2]
            shearStrain = ti.sqrt(
                s[0, 1] ** 2
                + s[0, 2] ** 2
                + s[1, 2] ** 2
                + (xydiff * xydiff + xzdiff * xzdiff + yzdiff * yzdiff) / 6.0
            )
            shear_strain[i] = shearStrain

    def compute(self):
        """Do the real compute."""
        self.shear_strain = np.zeros(self.neighbor_number.shape[0])
        self._cal_atomic_strain(
            self.verlet_list,
            self.neighbor_number,
            self.ref_box,
            self.cur_box,
            self.ref_box_inv,
            self.cur_box_inv,
            self.ref_pos,
            self.cur_pos,
            self.shear_strain,
        )


if __name__ == "__main__":
    import mdapy as mp

    ti.init()
    from time import time

    ref = mp.System(r"D:\Study\Gra-Al\paper\Fig6\res\al_gra_deform_1e9_x\dump.0.xyz")
    ref.build_neighbor(5.0, max_neigh=70)
    cur = mp.System(
        r"D:\Study\Gra-Al\paper\Fig6\res\al_gra_deform_1e9_x\dump.150000.xyz"
    )

    start = time()
    strain = AtomicStrain(
        ref.pos,
        ref.box,
        cur.pos,
        cur.box,
        ref.verlet_list,
        ref.neighbor_number,
        ref.boundary,
        "ref",
    )
    strain.compute()
    end = time()
    print(f"Atom strain time costs: {end-start} s.")
    print(strain.shear_strain)
