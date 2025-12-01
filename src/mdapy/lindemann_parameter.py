# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

from __future__ import annotations
from mdapy import _lindemann
import numpy as np
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes


class LindemannParameter:
    """Calculate the Lindemann index for distinguishing melt processes.

    This class computes the `Lindemann index <https://en.wikipedia.org/wiki/Lindemann_index>`_,
    which is useful for distinguishing the melting process and determining the melting points
    of nano-particles. The Lindemann index is defined as the root-mean-square bond-length
    fluctuation with the following mathematical expression:

    .. math:: \\left\\langle\\sigma_{i}\\right\\rangle=\\frac{1}{N_{p}(N_{p}-1)} \\sum_{j \\neq i} \\frac{\\sqrt{\\left\\langle r_{i j}^{2}\\right\\rangle_t-\\left\\langle r_{i j}\\right\\rangle_t^{2}}}{\\left\\langle r_{i j}\\right\\rangle_t}

    where :math:`N_p` is the number of particles, :math:`r_{ij}` is the distance between
    atom :math:`i` and :math:`j`, and the brackets :math:`\\left\\langle \\right\\rangle_t`
    represent a time average.

    Parameters
    ----------
    pos_list : np.ndarray
        Array of particle positions with shape (:math:`N_f`, :math:`N_p`, 3), where
        :math:`N_f` is the number of frames and :math:`N_p` is the number of particles.

        .. warning::
            **The positions MUST be unwrapped coordinates!** For systems with periodic
            boundary conditions, you must provide unwrapped positions to correctly
            calculate inter-particle distances across time. Wrapped coordinates will
            produce incorrect results due to discontinuities when particles cross
            periodic boundaries.

    only_global : bool, optional
        If True, only compute the global Lindemann index (faster, parallel computation).
        If False, compute both local and global indices (slower, serial computation).
        Default is False.

    Attributes
    ----------
    lindemann_trj : float
        Global Lindemann index for the entire trajectory.
    lindemann_frame : np.ndarray
        Lindemann index per frame with shape (:math:`N_f`,).
        Only available when `only_global=False`.
    lindemann_atom : np.ndarray
        Local Lindemann index per atom with shape (:math:`N_f`, :math:`N_p`).
        Only available when `only_global=False`.

    Notes
    -----
    - This implementation is partly based on the work at
      https://github.com/N720720/lindemann for calculating the Lindemann index.
    - **Memory requirement**: This calculation has high memory requirements. You can
      estimate the required memory using: :math:`2 \\times 8 \\times N_p^2 / 1024^3` GB.
    - **Parallelization**: If only the global Lindemann index is needed, the calculation
      can be performed in parallel. The local Lindemann index runs serially due
      to dependencies between different frames.
    - **Algorithm**: The `Welford method <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford>`_
      is used to update the variance and mean of :math:`r_{ij}` for numerical stability.

    Examples
    --------
    >>> import mdapy as mp
    >>> import numpy as np

    >>> # Generate a random walk trajectory with 200 frames and 1000 particles
    >>> pos_list = np.cumsum(
    ...     np.random.choice([-1.0, 0.0, 1.0], size=(200, 1000, 3)), axis=0
    ... )

    >>> # Calculate only global Lindemann index (faster)
    >>> LDMG = mp.LindemannParameter(pos_list, only_global=True)
    >>> LDMG.compute()
    >>> print(f"Global Lindemann index: {LDMG.lindemann_trj:.6f}")

    >>> # Calculate both local and global Lindemann indices
    >>> LDML = mp.LindemannParameter(pos_list)
    >>> LDML.compute()
    >>> print(f"Global Lindemann index: {LDML.lindemann_trj:.6f}")
    >>> print(f"First 5 frame indices: {LDML.lindemann_frame[:5]}")

    >>> # Verify consistency
    >>> np.isclose(LDML.lindemann_trj, LDMG.lindemann_trj)
    True

    >>> # Plot evolution
    >>> fig, ax = LDML.plot()

    """

    def __init__(self, pos_list: np.ndarray, only_global: bool = False) -> None:
        """Initialize the LindemannParameter calculator.

        Parameters
        ----------
        pos_list : np.ndarray
            Particle positions with shape (Nframes, Natoms, 3).
            Must be unwrapped coordinates for periodic systems.
        only_global : bool, optional
            Whether to compute only the global Lindemann index, by default False.
        """
        self.pos_list = np.ascontiguousarray(pos_list, dtype=np.float64)
        self.only_global = only_global
        self.lindemann_frame = None

    def compute(self) -> None:
        """Perform the Lindemann index calculation.

        This method calls the C++ backend for efficient computation. If `only_global`
        is True, only the global Lindemann index is computed; otherwise, both local
        and global indices are calculated.

        The computation results are stored in the following attributes:

        - `lindemann_trj` : Global Lindemann index
        - `lindemann_frame` : Lindemann index per frame (if `only_global=False`)
        - `lindemann_atom` : Local Lindemann index per atom (if `only_global=False`)

        """
        Nframes, Natoms = self.pos_list.shape[:2]
        pos_mean = np.zeros((Natoms, Natoms), dtype=np.float64)
        pos_variance = np.zeros_like(pos_mean)

        if self.only_global:
            # Compute only global Lindemann index (parallel)
            self.lindemann_trj = _lindemann.compute_global(
                self.pos_list, pos_mean, pos_variance
            )
        else:
            # Compute local and global Lindemann indices (serial)
            self.lindemann_frame = np.zeros(Nframes, dtype=np.float64)
            self.lindemann_atom = np.zeros((Nframes, Natoms), dtype=np.float64)

            _lindemann.compute_all(
                self.pos_list,
                pos_mean,
                pos_variance,
                self.lindemann_frame,
                self.lindemann_atom,
            )

            self.lindemann_trj = self.lindemann_frame[-1]

    def plot(
        self, fig: Optional[Figure] = None, ax: Optional[Axes] = None
    ) -> Tuple[Figure, Axes]:
        """Plot the evolution of Lindemann index per frame.

        Parameters
        ----------
        fig : Figure, optional
            Matplotlib figure object. If None, a new figure will be created.
        ax : Axes, optional
            Matplotlib axes object. If None, a new axes will be created.

        Returns
        -------
        fig : Figure
            The matplotlib figure object.
        ax : Axes
            The matplotlib axes object.

        Raises
        ------
        RuntimeError
            If `compute()` has not been called with `only_global=False`.

        Examples
        --------
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> pos_list = np.random.randn(100, 500, 3).cumsum(axis=0)
        >>> ldm = LindemannParameter(pos_list)
        >>> ldm.compute()
        >>> fig, ax = ldm.plot()
        >>> plt.savefig("lindemann_evolution.png")
        >>> plt.show()

        Notes
        -----
        This method requires that `compute()` has been called with `only_global=False`,
        as it visualizes the frame-by-frame evolution of the Lindemann index.
        """
        if self.lindemann_frame is None:
            raise RuntimeError(
                "No frame data available. Call compute() with only_global=False first."
            )

        if fig is None and ax is None:
            from mdapy.plotset import set_figure

            fig, ax = set_figure()

        ax.plot(self.lindemann_frame, "o-")
        ax.set_xlabel(r"$\mathregular{N_{frames}}$")
        ax.set_ylabel("Lindemann index")

        return fig, ax


if __name__ == "__main__":
    from time import time

    print("Lindemann Parameter Calculation with C++ Backend")
    print("=" * 60)

    Nframe, Nparticles = 200, 500
    pos_list = np.cumsum(
        np.random.choice([-1.0, 0.0, 1.0], size=(Nframe, Nparticles, 3)), axis=0
    )

    # Test global index calculation only
    print(f"\nTest Configuration: {Nframe} frames, {Nparticles} particles")
    print("\n1. Computing global Lindemann index only:")
    start = time()
    LDMG = LindemannParameter(pos_list, only_global=True)
    LDMG.compute()
    end = time()
    print(f"   Lindemann index (global): {LDMG.lindemann_trj:.6f}")
    print(f"   Computation time: {end - start:.3f} seconds")

    # Test local and global index calculation
    print("\n2. Computing local and global Lindemann indices:")
    start = time()
    LDML = LindemannParameter(pos_list)
    LDML.compute()
    end = time()
    print(f"   Lindemann index (global): {LDML.lindemann_trj:.6f}")
    print(f"   Computation time: {end - start:.3f} seconds")

    # Verify results
    print("\n3. Result Verification:")
    is_close = np.isclose(LDMG.lindemann_trj, LDML.lindemann_trj)
    print(f"   Global indices match: {is_close}")
    print("\n   Lindemann index for first 10 frames:")
    print(f"   {LDML.lindemann_frame[:10]}")

    # Plot evolution
    print("\n4. Plotting evolution...")

    import matplotlib.pyplot as plt

    fig, ax = LDML.plot()
    plt.show()
