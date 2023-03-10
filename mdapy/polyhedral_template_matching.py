# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import numpy as np

try:
    from ptm import _ptm
except Exception:
    import _ptm

try:
    from kdtree import kdtree
except Exception:
    from .kdtree import kdtree


class PolyhedralTemplateMatching:
    def __init__(self, structure, pos, box, boundary, rmsd_threshold, return_verlet):
        self.structure = structure
        structure_list = [
            "fcc",
            "hcp",
            "bcc",
            "ico",
            "sc",
            "dcub",
            "dhex",
            "graphene",
            "all",
            "default",
        ]
        for i in self.structure.split("-"):
            assert (
                i in structure_list
            ), f'Structure should in ["fcc", "hcp", "bcc", "ico", "sc","dcub", "dhex", "graphene", "all", "default"].'
        self.pos = pos
        self.box = box
        self.boundary = boundary
        self.rmsd_threshold = rmsd_threshold
        self.return_verlet = return_verlet

    def compute(self):
        kdt = kdtree(self.pos, self.box, self.boundary)
        _, verlet_list = kdt.query_nearest_neighbors(18)
        ptm_indices = np.zeros_like(verlet_list, int)
        self.output = np.zeros((self.pos.shape[0], 7))

        _ptm.get_ptm(
            self.structure,
            self.pos,
            verlet_list,
            self.box[:, 1] - self.box[:, 0],
            np.array(self.boundary, int),
            self.output,
            self.rmsd_threshold,
            ptm_indices,
        )
        if self.return_verlet:
            self.ptm_indices = ptm_indices


if __name__ == "__main__":
    import taichi as ti

    ti.init()
    from lattice_maker import LatticeMaker
    from time import time

    FCC = LatticeMaker(3.615, "FCC", 10, 10, 10)
    FCC.compute()

    start = time()
    ptm = PolyhedralTemplateMatching("default", FCC.pos, FCC.box, [1, 1, 1], 0.1, True)
    ptm.compute()
    print(f"PTM time cost: {time()-start} s.")
    print(ptm.output[:3])
    print(ptm.ptm_indices[:3])
