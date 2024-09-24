# Copyright (c) 2022-2024, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import taichi as ti
import numpy as np

try:
    from nearest_neighbor import NearestNeighbor
    from replicate import Replicate
    from tool_function import _check_repeat_nearest
    from box import init_box
    from cna import _cna
except Exception:
    from .nearest_neighbor import NearestNeighbor
    from .replicate import Replicate
    from .tool_function import _check_repeat_nearest
    from .box import init_box
    import _cna


@ti.data_oriented
class IdentifyDiamondStructure:
    """This class is used to identify the Diamond structure. The results and algorithm should be the same in Ovito.
    More details can be found in https://www.ovito.org/manual/reference/pipelines/modifiers/identify_diamond.html .

    Args:
        pos (np.ndarray): atom position.
        box (np.ndarray: system box.
        boundary (list, optional): boundary. Defaults to [1, 1, 1].
        verlet_list (np.ndarray, optional): first 12 neighbors is sorted, if not given, use kdtree to obtain it. Defaults to None.

    Outputs:
        - **pattern** (np.ndarray) - structure pattern per atoms.

    The identified structures include:

    - 0 "other",
    - 1 "cubic_diamond",
    - 2 "cubic_diamond_1st_neighbor",
    - 3 "cubic_diamond_2st_neighbor",
    - 4 "hexagonal_diamond",
    - 5 "hexagonal_diamond_1st_neighbor",
    - 6 "hexagonal_diamond_2st_neighbor"

    """

    def __init__(
        self,
        pos,
        box,
        boundary=[1, 1, 1],
        verlet_list=None,
    ) -> None:
        box, _, _ = init_box(box)
        repeat = [1, 1, 1]
        if verlet_list is None:
            repeat = _check_repeat_nearest(pos, box, boundary)
        if pos.dtype != np.float64:
            pos = pos.astype(np.float64)
        self.old_N = None
        if sum(repeat) == 3:
            self.pos = pos
            self.box, self.inverse_box, self.rec = init_box(box)
        else:
            self.old_N = pos.shape[0]
            repli = Replicate(pos, box, *repeat)
            repli.compute()
            self.pos = repli.pos
            self.box, self.inverse_box, self.rec = init_box(repli.box)

        self.N = self.pos.shape[0]
        self.verlet_list = verlet_list
        self.boundary = boundary
        self.structure = [
            "other",
            "cubic_diamond",
            "cubic_diamond_1st_neighbor",
            "cubic_diamond_2st_neighbor",
            "hexagonal_diamond",
            "hexagonal_diamond_1st_neighbor",
            "hexagonal_diamond_2st_neighbor",
        ]

    def compute(
        self,
    ):
        if self.verlet_list is None:
            kdt = NearestNeighbor(self.pos, self.box, self.boundary)
            _, self.verlet_list = kdt.query_nearest_neighbors(4)

        new_verlet_list = np.zeros((self.N, 12), dtype=np.int32)
        self.pattern = np.zeros(self.N, dtype=np.int32)
        _cna._ids(
            self.pos,
            self.box,
            self.inverse_box,
            np.bool_(self.boundary),
            self.verlet_list,
            new_verlet_list,
            self.pattern,
        )

        if self.old_N is not None:
            self.pattern = np.ascontiguousarray(self.pattern[: self.old_N])


if __name__ == "__main__":
    ti.init()
    # import polars as pl
    from system import System
    from time import time

    hex_C = System(r"C:\Users\herrwu\Desktop\xyz\CubicDiamond.xyz")
    # hex_C = System(r'C:\Users\herrwu\Desktop\xyz\HexDiamond.xyz')
    # hex_C = System(r'C:\Users\herrwu\Desktop\xyz\CubicDiamond.xyz')

    start = time()
    ids = IdentifyDiamondStructure(hex_C.pos, hex_C.box, hex_C.boundary)
    ids.compute()
    end = time()

    print(f"Cal IDS time: {end-start} s.")
    for i in range(7):
        print(ids.structure[i], ":", len(ids.pattern[ids.pattern == i]))

    # hex_C.update_data(hex_C.data.with_columns(
    #     pl.lit(ids.pattern).alias('struc')
    # ))
    # hex_C.write_xyz('test.xyz', type_name=['C'])
