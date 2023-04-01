# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.
# We highly thanks to Dr. Peter M Larsen for the help on parallelism of this module.

import numpy as np

if __name__ == "__main__":
    from ptm import _ptm
    from kdtree import kdtree
else:
    import _ptm
    from .kdtree import kdtree


class PolyhedralTemplateMatching:
    """This class identifies the local structural environment of particles using the Polyhedral Template Matching (PTM) method, which shows greater reliability than e.g. `Common Neighbor Analysis (CNA) <https://mdapy.readthedocs.io/en/latest/mdapy.html#module-mdapy.common_neighbor_analysis>`_. It can identify the following structure:

    1. other = 0
    2. fcc = 1
    3. hcp = 2
    4. bcc = 3
    5. ico (icosahedral) = 4
    6. sc (simple cubic) = 5
    7. dcub (diamond cubic) = 6
    8. dhex (diamond hexagonal) = 7
    9. graphene = 8

    .. hint:: If you use this class in publication, you should cite the original papar:

      `Larsen P M, Schmidt S, Schiøtz J. Robust structural identification via polyhedral template matching[J]. Modelling and Simulation in Materials Science and Engineering, 2016, 24(5): 055007. <10.1088/0965-0393/24/5/055007>`_

    .. note:: The present version is translated from that in `LAMMPS <https://docs.lammps.org/compute_ptm_atom.html>`_, which is fully parallel via openmp from mdapy verison 0.8.3.

    Args:

        pos (np.ndarray): (:math:`N_p, 3`) particles positions.
        box (np.ndarray): (:math:`3, 2`) system box, must be rectangle.
        boundary (list, optional): boundary conditions, 1 is periodic and 0 is free boundary. Defaults to [1, 1, 1].
        structure (str, optional): the structure one want to identify, one can choose from ["fcc","hcp","bcc","ico","sc","dcub","dhex","graphene","all","default"], such as 'fcc-hcp-bcc'. 'default' represents 'fcc-hcp-bcc-ico'. Defaults to 'fcc-hcp-bcc'.
        rmsd_threshold (float, optional): rmsd threshold. Defaults to 0.1.
        verlet_list (np.ndarray, optional): (:math:`N_p, >=18`) verlet_list[i, j] means j atom is a neighbor of i atom if j > -1. Defaults to None.
        return_verlet (bool, optional): whether return ptm_indicis for pre-processing, if you do not need, set it to False. Defaults to False.

    Outputs:

        - **output** (np.ndarray) - (:math:`N_p, 7`) the columns represent ['type', 'rmsd', 'interatomic distance', 'qw', 'qx', 'qy', 'qz'].
        - **ptm_indices** (np.ndarray) - (:math:`N_p, 18`) ptm neighbors.

    Examples:

        >>> import mdapy as mp

        >>> mp.init()

        >>> import numpy as np

        >>> FCC = mp.LatticeMaker(3.615, 'FCC', 10, 10, 10) # Create a FCC structure

        >>> FCC.compute() # Get atom positions

        >>> ptm = mp.PolyhedralTemplateMatching(FCC.pos, FCC.box) # Initilize ptm class.

        >>> ptm.compute() # Compute ptm per atoms.

        >>> ptm.output[:, 0] # Should be 1, that is fcc.
    """

    def __init__(
        self,
        pos,
        box,
        boundary=[1, 1, 1],
        structure="fcc-hcp-bcc",
        rmsd_threshold=0.1,
        verlet_list=None,
        return_verlet=False,
    ):

        self.pos = pos
        self.box = box
        self.boundary = boundary
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
        self.rmsd_threshold = rmsd_threshold
        self.verlet_list = verlet_list
        self.return_verlet = return_verlet

    def compute(self):
        """Do the real ptm computation."""
        verlet_list = self.verlet_list
        if verlet_list is None:
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

    FCC = LatticeMaker(3.615, "BCC", 10, 10, 10)
    FCC.compute()

    start = time()
    ptm = PolyhedralTemplateMatching(FCC.pos, FCC.box, return_verlet=True)
    ptm.compute()
    print(f"PTM time cost: {time()-start} s.")
    print(ptm.output[:3])
    print(ptm.ptm_indices[:3])
