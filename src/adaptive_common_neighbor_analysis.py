import taichi as ti
import numpy as np
from mdapy.src.kdtree import kdtree


@ti.data_oriented
class AdaptiveCommonNeighborAnalysis:
    def __init__(self, pos, box, boundary=[1, 1, 1]):

        self.box = ti.Vector.field(box.shape[1], dtype=ti.f64, shape=(box.shape[0]))
        self.box.from_numpy(box)
        self.boundary = ti.Vector(boundary)
        self.pos = pos
        self.N = self.pos.shape[0]
        self.pattern = np.zeros(self.N, dtype=np.int32)
        self.MAXNEAR = 14
        self.MAXCOMMON = 7
        self.structure = ["other", "fcc", "hcp", "bcc", "ico"]

    @ti.func
    def pbc(self, rij):
        for i in ti.static(range(rij.n)):
            if self.boundary[i] == 1:
                box_length = self.box[i][1] - self.box[i][0]
                rij[i] = rij[i] - box_length * ti.round(rij[i] / box_length)
        return rij

    @ti.func
    def get_common_neighbor(
        self,
        i: ti.i32,
        m: ti.i32,
        N: ti.i32,
        verlet_list: ti.types.ndarray(),
        pos: ti.types.ndarray(),
        common: ti.types.ndarray(),
        r: ti.types.ndarray(),
        r_i,
    ) -> ti.i32:

        j = verlet_list[i, m]
        r_j = ti.Vector([pos[j, 0], pos[j, 1], pos[j, 2]])
        ncommon = 0
        for inear in range(N):
            for jnear in range(N):
                inear_index = verlet_list[i, inear]
                jnear_index = verlet_list[j, jnear]
                if inear_index == jnear_index:
                    r_inear = ti.Vector(
                        [
                            pos[inear_index, 0],
                            pos[inear_index, 1],
                            pos[inear_index, 2],
                        ]
                    )
                    r_inear_i = self.pbc(r_inear - r_i)
                    r_inear_j = self.pbc(r_inear - r_j)

                    if r_inear_i.norm() <= r[i] and r_inear_j.norm() <= r[i]:
                        if ncommon < self.MAXCOMMON:
                            common[i, ncommon] = verlet_list[i, inear]
                            ncommon += 1
        return ncommon

    @ti.func
    def get_common_bonds(
        self,
        i: ti.i32,
        ncommon: ti.i32,
        bonds: ti.types.ndarray(),
        common: ti.types.ndarray(),
        pos: ti.types.ndarray(),
        r: ti.types.ndarray(),
    ) -> ti.i32:
        for n in range(ncommon):
            bonds[i, n] = 0

        nbonds = 0
        for jj in range(ncommon - 1):
            j = common[i, jj]
            for kk in range(jj + 1, ncommon):
                k = common[i, kk]
                r_j = ti.Vector([pos[j, 0], pos[j, 1], pos[j, 2]])
                r_k = ti.Vector([pos[k, 0], pos[k, 1], pos[k, 2]])
                rjk = self.pbc(r_j - r_k)
                if rjk.norm() <= r[i]:
                    nbonds += 1
                    bonds[i, jj] += 1
                    bonds[i, kk] += 1
        return nbonds

    @ti.func
    def get_max_min_bonds(self, i: ti.i32, ncommon: ti.i32, bonds: ti.types.ndarray()):
        maxbonds = 0
        minbonds = self.MAXCOMMON
        for n in range(ncommon):
            maxbonds = max(bonds[i, n], maxbonds)
            minbonds = min(bonds[i, n], minbonds)

        return maxbonds, minbonds

    @ti.func
    def is_fcc_hcp_ico(
        self, i: ti.i32, cna: ti.types.ndarray(), pattern: ti.types.ndarray()
    ):
        nfcc = nhcp = nico = 0
        if_struc = False
        for inear in range(12):
            if (
                cna[i, inear, 0] == 4
                and cna[i, inear, 1] == 2
                and cna[i, inear, 2] == 1
                and cna[i, inear, 3] == 1
            ):
                nfcc += 1
            if (
                cna[i, inear, 0] == 4
                and cna[i, inear, 1] == 2
                and cna[i, inear, 2] == 2
                and cna[i, inear, 3] == 0
            ):
                nhcp += 1
            if (
                cna[i, inear, 0] == 5
                and cna[i, inear, 1] == 5
                and cna[i, inear, 2] == 2
                and cna[i, inear, 3] == 2
            ):
                nico += 1
        if nfcc == 12:
            pattern[i] = 1
            if_struc = True
        elif nfcc == 6 and nhcp == 6:
            pattern[i] = 2
            if_struc = True
        elif nico == 12:
            pattern[i] = 4
            if_struc = True
        return if_struc

    @ti.func
    def clear_common_bond_cna(
        self,
        i: ti.i32,
        cna: ti.types.ndarray(),
        common: ti.types.ndarray(),
        bonds: ti.types.ndarray(),
    ):
        for n in range(self.MAXNEAR):
            for col in range(4):
                cna[i, n, col] = 0
        for n in range(self.MAXCOMMON):
            common[i, n] = 0
            bonds[i, n] = 0

    @ti.func
    def is_bcc(self, i: ti.i32, cna: ti.types.ndarray(), pattern: ti.types.ndarray()):
        nbcc4 = nbcc6 = 0
        for inear in range(14):
            if (
                cna[i, inear, 0] == 4
                and cna[i, inear, 1] == 4
                and cna[i, inear, 2] == 2
                and cna[i, inear, 3] == 2
            ):
                nbcc4 += 1
            if (
                cna[i, inear, 0] == 6
                and cna[i, inear, 1] == 6
                and cna[i, inear, 2] == 2
                and cna[i, inear, 3] == 2
            ):
                nbcc6 += 1
        if nbcc4 == 6 and nbcc6 == 8:
            pattern[i] = 3

    @ti.kernel
    def _compute(
        self,
        pos: ti.types.ndarray(),
        verlet_list: ti.types.ndarray(),
        cna: ti.types.ndarray(),
        common: ti.types.ndarray(),
        bonds: ti.types.ndarray(),
        pattern: ti.types.ndarray(),
        r_fcc: ti.types.ndarray(),
        r_bcc: ti.types.ndarray(),
    ):
        # ti.loop_config(serialize=True)
        for i in range(self.N):
            r_i = ti.Vector([pos[i, 0], pos[i, 1], pos[i, 2]])
            for m in range(12):
                ncommon = self.get_common_neighbor(
                    i, m, 12, verlet_list, pos, common, r_fcc, r_i
                )
                cna[i, m, 0] = ncommon
                nbonds = self.get_common_bonds(i, ncommon, bonds, common, pos, r_fcc)
                cna[i, m, 1] = nbonds
                maxbonds, minbonds = self.get_max_min_bonds(i, ncommon, bonds)
                cna[i, m, 2] = maxbonds
                cna[i, m, 3] = minbonds
            if not self.is_fcc_hcp_ico(i, cna, pattern):
                self.clear_common_bond_cna(i, cna, common, bonds)
                for m in range(14):
                    ncommon = self.get_common_neighbor(
                        i, m, 14, verlet_list, pos, common, r_bcc, r_i
                    )
                    cna[i, m, 0] = ncommon
                    nbonds = self.get_common_bonds(
                        i, ncommon, bonds, common, pos, r_bcc
                    )
                    cna[i, m, 1] = nbonds
                    maxbonds, minbonds = self.get_max_min_bonds(i, ncommon, bonds)
                    cna[i, m, 2] = maxbonds
                    cna[i, m, 3] = minbonds
                self.is_bcc(i, cna, pattern)

    def compute(self):

        cna = np.zeros((self.N, self.MAXNEAR, 4), dtype=np.int32)
        common = np.zeros((self.N, self.MAXCOMMON), dtype=np.int64)
        bonds = np.zeros((self.N, self.MAXCOMMON), dtype=np.int32)

        kdt = kdtree(self.pos, self.box, self.boundary)
        distance_list, verlet_list = kdt.query_nearest_neighbors(self.MAXNEAR)

        try:
            import torch

            distance_list_torch = torch.from_numpy(distance_list)
            r_fcc = (
                torch.sum(distance_list_torch[:, :12], dim=1)
                / 12
                * (1.0 + 2.0**0.5)
                / 2.0
            )
            r_bcc = (
                (
                    torch.sum(distance_list_torch[:, :8], dim=1) * 2 / 3**0.5
                    + torch.sum(distance_list_torch[:, 8:14], dim=1)
                )
                / 14
                * (1.0 + 2.0**0.5)
                / 2.0
            )
        except ImportError:
            r_fcc = (
                np.sum(distance_list[:, :12], axis=1) / 12 * (1.0 + 2.0**0.5) / 2.0
            )
            r_bcc = (
                (
                    np.sum(distance_list[:, :8], axis=1) * 2 / 3**0.5
                    + np.sum(distance_list[:, 8:14], axis=1)
                )
                / 14
                * (1.0 + 2.0**0.5)
                / 2.0
            )
        self._compute(
            self.pos,
            verlet_list,
            cna,
            common,
            bonds,
            self.pattern,
            r_fcc,
            r_bcc,
        )


if __name__ == "__main__":

    from lattice_maker import LatticeMaker
    from time import time

    ti.init(ti.cpu, offline_cache=True)  # , device_memory_GB=5.0)
    # ti.init(ti.cpu)
    start = time()
    lattice_constant = 3.0
    x, y, z = 100, 100, 50
    FCC = LatticeMaker(lattice_constant, "FCC", x, y, z)
    FCC.compute()
    end = time()
    print(f"Build {FCC.pos.shape[0]} atoms FCC time: {end-start} s.")
    np.random.seed(10)
    noise = np.random.rand(*FCC.pos.shape)
    FCC.pos += noise / 10
    start = time()
    CNA = AdaptiveCommonNeighborAnalysis(FCC.pos, FCC.box, [1, 1, 1])
    CNA.compute()
    end = time()
    print(f"Cal CNA time: {end-start} s.")
    for i in range(5):
        print(CNA.structure[i], ":", len(CNA.pattern[CNA.pattern == i]))
