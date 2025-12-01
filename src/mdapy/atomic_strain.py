# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

import numpy as np
import polars as pl
from mdapy import _strain
from mdapy import System
from mdapy.box import Box
from typing import Optional
import mdapy.tool_function as tool


class AtomicStrain:
    """
    Calculate atomic-level shear and volumetric strain tensors.

    This class computes local strain at each atom by comparing neighbor distances
    between a reference (undeformed) configuration and a current (deformed)
    configuration. The method calculates the deformation gradient tensor F and
    derives strain measures from it.

    Parameters
    ----------
    rc : float
        Cutoff radius for neighbor search in Angstroms.
    ref : System
        Reference (undeformed) atomic system.
    affine : bool, default=False
        Whether to apply affine transformation to map current box to reference box.
        If True, removes global box deformation to isolate local atomic strain.
    max_neigh : int, optional
        Maximum number of neighbors to consider. If None, automatically determined.

    Attributes
    ----------
    ref : System
        Reference system with neighbor list built.
    rc : float
        Cutoff radius used for neighbor calculations.
    max_neigh : int or None
        Maximum neighbors parameter.
    affine : bool
        Affine transformation flag.
    repeat : tuple of int
        Replication factors (nx, ny, nz) if reference box is too small.

    Notes
    -----
    The atomic strain is calculated using the following algorithm:

    1. **Build matrices V and W** for each atom i:

       .. math::

           V_{mn} = \\sum_j \\Delta r^{ref}_{jn} \\Delta r^{ref}_{jm}

           W_{mn} = \\sum_j \\Delta r^{ref}_{jn} \\Delta r^{cur}_{jm}

       where the sum is over all neighbors j within cutoff rc, and
       :math:`\\Delta r^{ref}` and :math:`\\Delta r^{cur}` are neighbor
       distance vectors in reference and current configurations.

    2. **Calculate deformation gradient tensor**:

       .. math::

           F = (W \\cdot V^{-1})^T

    3. **Compute strain tensor** (Green-Lagrange strain):

       .. math::

           \\varepsilon = \\frac{1}{2}(F^T F - I)

    4. **Extract strain measures**:

       - **Shear strain** (von Mises equivalent strain):

         .. math::

             \\eta_s = \\sqrt{\\varepsilon_{xy}^2 + \\varepsilon_{xz}^2 +
                       \\varepsilon_{yz}^2 +
                       \\frac{1}{6}[(\\varepsilon_{xx}-\\varepsilon_{yy})^2 +
                       (\\varepsilon_{xx}-\\varepsilon_{zz})^2 +
                       (\\varepsilon_{yy}-\\varepsilon_{zz})^2]}

       - **Volumetric strain** (hydrostatic strain):

         .. math::

             \\eta_v = \\frac{1}{3}(\\varepsilon_{xx} + \\varepsilon_{yy} +
                       \\varepsilon_{zz})

    The affine transformation option is useful when the simulation box itself
    undergoes deformation (e.g., under stress). Setting affine=True removes this
    global deformation to isolate local atomic-level strain.

    References
    ----------
    .. [1] Shimizu, F., Ogata, S., & Li, J. (2007). Theory of shear banding in
           metallic glasses and molecular dynamics calculations. Materials
           Transactions, 48(11), 2923-2927.

    Examples
    --------
    Calculate atomic strain between two configurations:

    >>> from mdapy import System
    >>> # Load reference (undeformed) configuration
    >>> ref_system = System("reference.dump")
    >>> # Load current (deformed) configuration
    >>> cur_system = System("deformed.dump")
    >>> # Initialize strain calculator
    >>> from mdapy.atomic_strain import AtomicStrain
    >>> strain = AtomicStrain(rc=5.0, ref=ref_system)
    >>> # Compute strain
    >>> strain.compute(cur_system)
    >>> # Results are stored in cur_system.data
    >>> print(cur_system.data["shear_strain"])
    >>> print(cur_system.data["volumetric_strain"])

    With affine transformation (removes global box deformation):

    >>> strain = AtomicStrain(rc=5.0, ref=ref_system, affine=True)
    >>> strain.compute(cur_system)
    """

    def __init__(
        self,
        rc: float,
        ref: System,
        affine: bool = False,
        max_neigh: Optional[int] = None,
    ):
        self.ref = ref
        self.rc = rc
        self.max_neigh = max_neigh
        self.ref.build_neighbor(self.rc, self.max_neigh)
        self.affine = affine
        self.repeat = self.ref.box.check_small_box(self.rc)

    def compute(self, current: System):
        """
        Compute atomic strain for the current configuration.

        This method calculates shear and volumetric strain for each atom by
        comparing neighbor distances with the reference configuration. Results
        are added as new columns 'shear_strain' and 'volumetric_strain' to
        the current system's data.

        Parameters
        ----------
        current : System
            Current (deformed) atomic system. Must have the same number of atoms
            as the reference system.

        Raises
        ------
        AssertionError
            If current system has different number of atoms than reference.

        Notes
        -----
        The computed strain values are added directly to current.data:

        - **shear_strain**: Von Mises equivalent shear strain at each atom
        - **volumetric_strain**: Hydrostatic (volumetric) strain at each atom

        Positive volumetric strain indicates expansion, negative indicates
        compression. Shear strain is always non-negative and indicates local
        distortion.

        If affine=True was set during initialization, the current configuration
        is first transformed to remove global box deformation before calculating
        local atomic strain.

        Examples
        --------
        >>> strain = AtomicStrain(rc=5.0, ref=ref_system)
        >>> strain.compute(cur_system)
        >>> # Access results
        >>> shear = cur_system.data["shear_strain"].to_numpy()
        >>> volumetric = cur_system.data["volumetric_strain"].to_numpy()
        >>> print(f"Max shear strain: {shear.max():.4f}")
        >>> print(f"Mean volumetric strain: {volumetric.mean():.4f}")
        """
        assert current.N == self.ref.N

        cur_data, cur_box = current.data, current.box
        if sum(self.repeat) != 3:
            cur_data, cur_box = tool._replicate_pos(
                current.data, current.box, *self.repeat
            )
        ref_data, ref_box = self.ref.data, self.ref.box
        if hasattr(self.ref, "__enlarge_data"):
            ref_data, ref_box = self.ref._enlarge_data, self.ref._enlarge_box

        if self.affine:
            map_matrix = np.linalg.solve(cur_box.box, ref_box.box)
            # pos @ map_matrix
            cur_data = cur_data.select(
                x=pl.col("x") * map_matrix[0, 0]
                + pl.col("y") * map_matrix[1, 0]
                + pl.col("z") * map_matrix[2, 0],
                y=pl.col("x") * map_matrix[0, 1]
                + pl.col("y") * map_matrix[1, 1]
                + pl.col("z") * map_matrix[2, 1],
                z=pl.col("x") * map_matrix[0, 2]
                + pl.col("y") * map_matrix[1, 2]
                + pl.col("z") * map_matrix[2, 2],
            )

            cur_box = Box(ref_box)
        assert cur_data.shape[0] == ref_data.shape[0]
        shear_strain = np.zeros(cur_data.shape[0], float)
        volumetric_strain = np.zeros(cur_data.shape[0], float)
        _strain.cal_atomic_strain(
            self.ref.verlet_list,
            self.ref.neighbor_number,
            ref_box.box,
            cur_box.box,
            ref_box.origin,
            cur_box.origin,
            ref_box.boundary,
            ref_data["x"].to_numpy(allow_copy=False),
            ref_data["y"].to_numpy(allow_copy=False),
            ref_data["z"].to_numpy(allow_copy=False),
            cur_data["x"].to_numpy(allow_copy=False),
            cur_data["y"].to_numpy(allow_copy=False),
            cur_data["z"].to_numpy(allow_copy=False),
            shear_strain,
            volumetric_strain,
        )

        current.update_data(
            current.data.with_columns(
                shear_strain=shear_strain[: current.N],
                volumetric_strain=volumetric_strain[: current.N],
            )
        )


if __name__ == "__main__":
    pass
