# Copyright (c) 2022-2024, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import numpy as np
import taichi as ti
from math import gcd

try:
    from .box import init_box
    from .replicate import Replicate
    from .tool_function import _wrap_pos
except Exception:
    from box import init_box
    from replicate import Replicate
    from tool_function import _wrap_pos


@ti.data_oriented
class OrthogonalBox:
    """This class try to change the box to rectangular.

    Args:
        pos (np.ndarray): atom position.
        box (np.ndarray): system box.
        type_list (np.ndarray, optional): type list. Defaults to None.

    Outputs:
        - **rec_pos** (np.ndarray) - new position.
        - **rec_box** (np.ndarray) - new box.
        - **rec_type_list** (np.ndarray) - new type_list.
    """

    def __init__(self, pos, box, type_list=None) -> None:
        if pos.dtype != np.float64:
            pos = pos.astype(np.float64)
        self.pos = pos
        self.box, _, _ = init_box(box)
        if type_list is None:
            self.type_list = np.ones(self.pos.shape[0], int)
        else:
            assert len(type_list) == self.pos.shape[0]
            self.type_list = type_list

    @ti.kernel
    def _find_orthogonal_vectors(
        self,
        trans: ti.types.ndarray(),
        box: ti.types.ndarray(element_dim=1),
        N: int,
        delta: float,
    ):
        x = y = z = False
        while not (x and y and z):
            for ii in range(2 * N):
                for jj in range(2 * N):
                    for kk in range(2 * N):
                        i = ii - N
                        j = jj - N
                        k = kk - N
                        vec = i * box[0] + j * box[1] + k * box[2]
                        if not x:
                            if abs(vec[1]) < delta and abs(vec[2]) < delta:
                                x = True
                                trans[0, 0] = i
                                trans[0, 1] = j
                                trans[0, 2] = k
                        if not y:
                            if abs(vec[0]) < delta and abs(vec[2]) < delta:
                                y = True
                                trans[1, 0] = i
                                trans[1, 1] = j
                                trans[1, 2] = k
                        if not z:
                            if abs(vec[0]) < delta and abs(vec[1]) < delta:
                                z = True
                                trans[2, 0] = i
                                trans[2, 1] = j
                                trans[2, 2] = k

    def compute(self, N=10):
        """Do the real computation.

        Args:
            N (int, optional): search limit. If you can't found rectangular box, increase N. Defaults to 10.
        """
        delta = 1e-6
        trans = np.zeros((3, 3), int)
        self._find_orthogonal_vectors(trans, self.box, N, delta)
        for i, j in zip(trans, ["x", "y", "z"]):
            assert (
                sum(i) != 0
            ), f"Not found proper vector along {j} direction. Try to increase N."
        trans = np.array(
            [i // gcd(*i) for i in np.array([i // gcd(*i) for i in trans])]
        )
        rec_box = np.abs(np.dot(trans, self.box[:-1]))
        rec_box[np.abs(rec_box) < delta] = 0.0
        rec_box, inverse_box, _ = init_box(rec_box)
        replicate = [1, 1, 1]
        for i in range(3):
            for j in range(3):
                if trans[i, j] != 0:
                    replicate[i] *= int(abs(trans[i, j]))
        rep = Replicate(self.pos, self.box, *replicate, self.type_list)
        rep.compute()
        rec_pos = rep.pos
        rec_type_list = rep.type_list
        _wrap_pos(rec_pos, rec_box, np.array([1, 1, 1], int), inverse_box)

        self.rec_box = rec_box
        self.rec_pos = rec_pos
        self.rec_type_list = rec_type_list


if __name__ == "__main__":
    ti.init()
    from system import System

    hex_C = System(r"C:\Users\herrwu\Desktop\xyz\MoS2-H.xyz")

    # box = np.array([
    #    [2.52699457, 0.        , 0.        ],
    #    [1.26349729, 2.18844149, 0.        ],
    #    [1.26349729, 0.7294805 , 2.06328243],
    #    [0.        , 0.        , 0.        ]
    #    ])
    # pos = np.array([
    #    [3.79049186, 2.18844149, 1.54746182],
    #    [2.52699457, 1.458961  , 1.03164121]
    #    ])

    rec = OrthogonalBox(hex_C.pos, hex_C.box, hex_C.data["type"].to_numpy())
    rec.compute()
    print("Rectangular box:")
    print(rec.rec_box)
    print("Rectangular pos::")
    print(rec.rec_pos)

    system = System(pos=rec.rec_pos, box=rec.rec_box, type_list=rec.rec_type_list)
    system.write_xyz("MoS2.xyz", type_name=["Mo", "S"])
