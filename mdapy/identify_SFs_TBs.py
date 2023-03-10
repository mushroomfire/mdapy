import taichi as ti
import numpy as np


@ti.data_oriented
class IdentifySFTWinFCC:
    def __init__(self, structure_types, verlet_list) -> None:
        self.structure_types = structure_types
        self.verlet_list = verlet_list

    @ti.func
    def are_stacked(
        self,
        a: int,
        b: int,
        basal_neighbors: ti.types.ndarray(),
        hcp_neighbors: ti.types.ndarray(),
    ) -> int:

        # Iterate over the basal plane neighbors of both 'a' and 'b'.
        res = 1
        for i in range(6):
            for j in range(6):
                if (
                    hcp_neighbors[a, basal_neighbors[i]]
                    == hcp_neighbors[b, basal_neighbors[j]]
                ):
                    # The two HCP atoms 'a' and 'b' share a common neighbor, which is both basal planes.
                    res = 0
                    break
        return res

    @ti.func
    def binary_search(self, arr: ti.types.ndarray(), value) -> int:
        left, right = 0, arr.shape[0] - 1
        mid = 0
        while left <= right:
            mid = (left + right) // 2
            if value < arr[mid]:
                right = mid - 1
            elif value > arr[mid]:
                left = mid + 1
            else:
                break
        return mid

    @ti.kernel
    def _compute(
        self,
        hcp_indices: ti.types.ndarray(),
        hcp_neighbors: ti.types.ndarray(),
        verlet_list: ti.types.ndarray(),
        structure_types: ti.types.ndarray(),
        fault_types: ti.types.ndarray(),
        layer_dir: ti.types.ndarray(),
        basal_neighbors: ti.types.ndarray(),
        outofplane_neighbors: ti.types.ndarray(),
    ):

        for i in range(hcp_indices.shape[0]):
            aindex = hcp_indices[i]
            for j in range(12):
                bindex = verlet_list[i, j]
                if structure_types[bindex] == 2:  # HCP
                    hcp_neighbors[i, j] = self.binary_search(hcp_indices, bindex)
                else:
                    hcp_neighbors[i, j] = ti.i32(-bindex - 1)

        for i in range(hcp_indices.shape[0]):
            aindex = hcp_indices[i]
            n_basal = 0  # Number of HCP neighbors in the same basal plane
            n_positive = 0  # Number of HCP neighbors on one side of the basal plane
            n_negative = 0  # Number of HCP neighbors on opposite side
            n_fcc_positive = 0  # Number of FCC neighbors on one side of the basal plane
            n_fcc_negative = (
                0  # Number of FCC neighbors on opposite side of the basal plane
            )

            for j in range(12):
                if hcp_neighbors[i, j] >= 0:
                    if layer_dir[j] == 0:
                        n_basal += 1
                    elif self.are_stacked(
                        i, hcp_neighbors[i, j], basal_neighbors, hcp_neighbors
                    ):
                        if layer_dir[j] == 1:
                            n_positive += 1
                        else:
                            n_negative += 1
                elif layer_dir[j] != 0:
                    neighbor_type = structure_types[-hcp_neighbors[i, j] - 1]
                    if neighbor_type == 1:  # FCC
                        if layer_dir[j] > 0:
                            n_fcc_positive += 1
                        else:
                            n_fcc_negative += 1

            # Is it an intrinsic stacking fault (two parallel HCP layers, atom has at least one out-of-plane neighbor in one direction)?
            if (n_positive != 0 and n_negative == 0) or (
                n_positive == 0 and n_negative != 0
            ):
                fault_types[aindex] = 2  # isf_type.id
            # Is it a coherent twin boundary (single HCP layer, atom has no out-of-plane HCP neighbors but at least one in-plane neighbor)?
            elif (
                n_basal >= 1
                and n_positive == 0
                and n_negative == 0
                and n_fcc_positive != 0
                and n_fcc_negative != 0
            ):
                fault_types[aindex] = 3  # twin_type.id
            # Is it a multi-layered stacking fault (three or more HCP layers, atom has out-of-plane HCP neighbors on both sides)?
            elif n_positive != 0 and n_negative != 0:
                fault_types[aindex] = 4  # multi_type.id
            # Otherwise, it must be an isolated HCP atom (undetermined planar fault type)
            else:
                fault_types[aindex] = 1  # other_type.id

        # Must serial run here.
        ti.loop_config(serialize=True)
        for i in range(hcp_indices.shape[0]):
            aindex = hcp_indices[i]

            if fault_types[aindex] == 3 or fault_types[aindex] == 1:
                n_isf_neighbors = 0  # Counts the number of neighbors in the basal plane which are intrinsic stacking fault (ISF) atoms
                n_twin_neighbors = 0  # Counts the number of neighbors in the basal plane which are coherent twin boundary (CTB) atoms
                # Visit the 6 neighbors in the basal plane.
                for jj in range(6):
                    j = basal_neighbors[jj]
                    neighbor_index = hcp_neighbors[i, j]
                    # Is the current neighbor an HCP atom?
                    if neighbor_index >= 0:
                        # Check the planar fault type of the neighbor atom.
                        neighbor_index = hcp_indices[neighbor_index]
                        if fault_types[neighbor_index] == 2:
                            n_isf_neighbors += 1
                        elif fault_types[neighbor_index] == 3:
                            n_twin_neighbors += 1
                # If the TB atom is surrounded by ISF atoms only, turn it into an ISF atom as well.
                if n_isf_neighbors != 0 and n_twin_neighbors == 0:
                    fault_types[aindex] = 2  # isf_type.id
                # If the Other atom is surrounded by TB atoms only, turn it into a TB atom as well.
                elif n_isf_neighbors == 0 and n_twin_neighbors != 0:
                    fault_types[aindex] = 3  # twin_type.id
            elif fault_types[aindex] == 4:
                # Visit the 6 out-of-plane neighbors.
                for jj in range(6):
                    j = outofplane_neighbors[jj]
                    neighbor_index = hcp_neighbors[i, j]
                    # Is the current neighbor an HCP atom of type ISF?
                    if (
                        neighbor_index >= 0
                        and fault_types[hcp_indices[neighbor_index]] == 2
                    ):
                        # Turn the neighbor into a multi-layered fault atom.
                        fault_types[hcp_indices[neighbor_index]] = 4

    def compute(
        self,
    ):
        assert 2 in self.structure_types, "Sample mush include HCP atoms."
        hcp_indices = np.where(self.structure_types == 2)[0].astype(int)
        hcp_neighbors = np.zeros((hcp_indices.shape[0], 12), dtype=int)
        self.fault_types = np.zeros_like(self.structure_types)
        layer_dir = np.array([0, 0, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1])
        basal_neighbors = np.array([0, 1, 5, 6, 7, 8])
        outofplane_neighbors = np.array([2, 3, 4, 9, 10, 11])

        self._compute(
            hcp_indices,
            hcp_neighbors,
            self.verlet_list,
            self.structure_types,
            self.fault_types,
            layer_dir,
            basal_neighbors,
            outofplane_neighbors,
        )


if __name__ == "__main__":
    ti.init()
    from time import time
    from system import System
    from polyhedral_template_matching import PolyhedralTemplateMatching

    system = System(
        r"C:\Users\Administrator\Desktop\python\MY_PACKAGE\MyPackage\IdentifySFTB\ISF.dump"
    )

    ptm = PolyhedralTemplateMatching(
        "default", system.pos, system.box, system.boundary, 0.1, True
    )
    ptm.compute()

    start = time()
    structure_type = np.array(ptm.output[:, 0], int)
    verlet_list = np.ascontiguousarray(ptm.ptm_indices[structure_type == 2][:, 1:13])

    SFTW = IdentifySFTWinFCC(structure_type, verlet_list)
    SFTW.compute()
    print(f"SFTW time cost: {time()-start} s.")
    print(SFTW.fault_types)
    print(SFTW.structure_types)
    system.data["fault_types"] = SFTW.fault_types
    system.data["structure_types"] = SFTW.structure_types
    system.write_dump()
