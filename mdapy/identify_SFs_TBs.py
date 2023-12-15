# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import taichi as ti
import numpy as np


@ti.data_oriented
class IdentifySFTBinFCC:
    """This class is used to identify the stacking faults (SFs) and coherent twin boundaries (TBs) in FCC structure based on the `Polyhedral Template Matching (PTM) <https://mdapy.readthedocs.io/en/latest/mdapy.html#module-mdapy.polyhedral_template_matching>`_.
    It can identify the following structure:

    1. 0 = Non-hcp atoms (e.g. perfect fcc or disordered)
    2. 1 = Indeterminate hcp-like (isolated hcp-like atoms, not forming a planar defect)
    3. 2 = Intrinsic stacking fault (two adjacent hcp-like layers)
    4. 3 = Coherent twin boundary (one hcp-like layer)
    5. 4 = Multi-layer stacking fault (three or more adjacent hcp-like layers)
    6. 5 = Extrinsic stacking fault

    .. note:: This class is translated from that `implementation in Ovito <https://www.ovito.org/docs/current/reference/pipelines/modifiers/identify_fcc_planar_faults.html#modifiers-identify-fcc-planar-faults>`_ but optimized to be run parallely.
      And so-called multi-layer stacking faults maybe a combination of intrinsic stacking faults and/or twin boundary which are located on adjacent {111} plane. It can not be distiguished by the current method.

    Args:
        structure_types (np.ndarray): (:math:`N_p`) structure type from ptm method.
        ptm_indices (np.ndarray): (:math:`N_p, 18`) ptm ordered verlet_list for all atoms.

    Outputs:
        - **fault_types** (np.ndarray) - (:math:`N_p`) planar faults types.

    Examples:

        >>> import mdapy as mp

        >>> mp.init()

        >>> system = mp.System(r"./example/ISF.dump") # Read dump

        >>> ptm = mp.PolyhedralTemplateMatching(
                    system.pos, system.box, system.boundary, "default", 0.1, None, True
                    ) # Initilize ptm method.

        >>> ptm.compute() # Compute ptm per atoms.

        >>> structure_type = np.array(ptm.output[:, 0], int) # Obtain ptm structure

        >>> SFTB = mp.IdentifySFTBinFCC(structure_type, ptm.ptm_indices) # Initialize SFTB class

        >>> SFTB.compute() # Compute the planar faults type

        >>> SFTB.fault_types # Check results
    """

    def __init__(self, structure_types, ptm_indices) -> None:
        self.structure_types = structure_types
        self.ptm_indices = np.ascontiguousarray(ptm_indices[:, 1:13])

    @ti.func
    def _are_stacked(
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
    def _binary_search(self, arr: ti.types.ndarray(), value) -> int:
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
        ptm_indices: ti.types.ndarray(),
        structure_types: ti.types.ndarray(),
        fault_types: ti.types.ndarray(),
        layer_dir: ti.types.ndarray(),
        basal_neighbors: ti.types.ndarray(),
        outofplane_neighbors: ti.types.ndarray(),
    ):
        for i in range(hcp_indices.shape[0]):
            aindex = hcp_indices[i]
            for j in range(12):
                bindex = ptm_indices[aindex, j]
                if structure_types[bindex] == 2:  # HCP
                    hcp_neighbors[i, j] = self._binary_search(hcp_indices, bindex)
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
                    elif self._are_stacked(
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

        # Get ESF
        for i in range(hcp_indices.shape[0]):
            aindex = hcp_indices[i]
            if fault_types[aindex] == 3:  # twin_type.id
                for j in range(12):
                    fcc, hcp = 0, 0
                    jindex = ptm_indices[aindex, j]
                    if structure_types[jindex] == 1:  # FCC
                        for k in range(12):
                            kindex = ptm_indices[jindex, k]
                            if structure_types[kindex] == 1:  # FCC
                                fcc += 1
                            elif structure_types[kindex] == 2:  # HCP
                                hcp += 1
                        if (5 <= fcc <= 6) and (5 <= hcp <= 6):
                            fault_types[aindex] = 5  # ESF
                            break

    def compute(
        self,
    ):
        """Do the real computation."""
        if 2 in self.structure_types:
            hcp_indices = np.where(self.structure_types == 2)[0].astype(int)
            hcp_neighbors = np.zeros((hcp_indices.shape[0], 12), dtype=int)
            self.fault_types = np.zeros_like(self.structure_types)
            layer_dir = np.array([0, 0, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1])
            basal_neighbors = np.array([0, 1, 5, 6, 7, 8])
            outofplane_neighbors = np.array([2, 3, 4, 9, 10, 11])

            self._compute(
                hcp_indices,
                hcp_neighbors,
                self.ptm_indices,
                self.structure_types,
                self.fault_types,
                layer_dir,
                basal_neighbors,
                outofplane_neighbors,
            )
        else:
            self.fault_types = np.zeros_like(self.structure_types)


if __name__ == "__main__":
    ti.init()
    from time import time
    from system import System
    from polyhedral_template_matching import PolyhedralTemplateMatching

    system = System(r"./example/ISF.dump")

    ptm = PolyhedralTemplateMatching(
        system.pos, system.box, system.boundary, "default", 0.1, None, True
    )
    ptm.compute()

    start = time()
    structure_type = np.array(ptm.output[:, 0], int)

    SFTB = IdentifySFTBinFCC(structure_type, ptm.ptm_indices)
    SFTB.compute()
    print(f"SFTB time cost: {time()-start} s.")
    print(np.unique(SFTB.fault_types))
    print(np.unique(SFTB.structure_types))
    # system.data["structure_types"] = SFTB.structure_types
    # system.data["fault_types"] = SFTB.fault_types
    # system.write_dump()
