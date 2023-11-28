# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import taichi as ti
import numpy as np

import matplotlib.pyplot as plt

try:
    from plotset import set_figure
except Exception:
    from .plotset import set_figure


@ti.data_oriented
class LindemannParameter:
    """This class is used to calculate the `Lindemann index <https://en.wikipedia.org/wiki/Lindemann_index>`_,
    which is useful to distinguish the melt process and determine the melting points of nano-particles.
    The Lindemann index is defined as the root-mean-square bond-length fluctuation with following mathematical expression:

    .. math:: \\left\\langle\\sigma_{i}\\right\\rangle=\\frac{1}{N_{p}(N_{p}-1)} \\sum_{j \\neq i} \\frac{\\sqrt{\\left\\langle r_{i j}^{2}\\right\\rangle_t-\\left\\langle r_{i j}\\right\\rangle_t^{2}}}{\\left\\langle r_{i j}\\right\\rangle_t},

    where :math:`N_p` is the particle number, :math:`r_{ij}` is the distance between atom :math:`i` and :math:`j` and brackets :math:`\\left\\langle \\right\\rangle_t`
    represents an time average.

    .. note:: This class is partly referred to a `work <https://github.com/N720720/lindemann>`_ on calculating the Lindemann index.

    .. note:: This calculation is high memory requirement. One can estimate the memory by: :math:`2 * 8 * N_p^2 / 1024^3` GB.

    .. tip:: If only global lindemann index is needed, the class can be calculated in parallel.
      The local Lindemann index only run serially due to the dependencies between different frames.
      Here we use the `Welford method <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford>`_ to
      update the varience and mean of :math:`r_{ij}`.

    Args:
        pos_list (np.ndarray): (:math:`N_f, N_p, 3`), :math:`N_f` frames particle position, which need to be unwrapped for periodic boundary.
        only_global (bool, optional): whether only calculate the global index. Defaults to False.

    Outputs:
        - **lindemann_atom** (np.ndarray) - (:math:`N_f, N_p`), local Lindemann index per atoms.
        - **lindemann_frame** (np.ndarray) - (:math:`N_f`), Lindemann index per frames.
        - **lindemann_trj** (float) - global Lindemann index for an entire trajectory.

    Examples:
        >>> import mdapy as mp

        >>> mp.init()

        >>> import numpy as np

        >>> pos_list = np.cumsum(
            np.random.choice([-1.0, 0.0, 1.0], size=(200, 1000, 3)), axis=0
            ) # Generate a random walk trajectory with 200 frames and 1000 particles.

        >>> LDMG = mp.LindemannParameter(pos_list, only_global=True) # Generate a Lindemann class.

        >>> LDMG.compute() # Only calculate the global Lindemann index, it will be much faster.

        >>> LDML = mp.LindemannParameter(pos_list) # Generate a Lindemann class.

        >>> LDML.compute() # Calculate the global and local Lindemann index.

        >>> np.isclose(LDML.lindemann_trj, LDMG.lindemann_trj) # Should return True.

        >>> LDML.lindemann_frame # Check Lindemann index per frame.

        >>> LDML.plot() # Plot the evolution of Lindemann index per frame.

    """

    def __init__(self, pos_list, only_global=False) -> None:
        self.pos_list = pos_list
        self.only_global = only_global
        self.if_compute = False

    @ti.kernel
    def _compute_global(
        self,
        pos_list: ti.types.ndarray(dtype=ti.math.vec3),
        pos_mean: ti.types.ndarray(),
        pos_variance: ti.types.ndarray(),
    ) -> float:
        Nframes, Natoms = pos_list.shape
        factor = Natoms * (Natoms - 1) / 2
        for i, j in ti.ndrange(Natoms, Natoms):
            if j > i:
                for frame in range(Nframes):
                    rijdis = (pos_list[frame, i] - pos_list[frame, j]).norm()
                    pos_mean[i, j] += rijdis
                    pos_variance[i, j] += rijdis**2

        lin_index = ti.float64(0.0)
        for i, j in ti.ndrange(Natoms, Natoms):
            if j > i:
                rij_squared_mean = pos_variance[i, j] / Nframes
                rij_mean = pos_mean[i, j] / Nframes
                delta = rij_squared_mean - rij_mean**2
                lin_index += ti.sqrt(delta) / rij_mean
        return lin_index / factor

    @ti.kernel
    def _compute_all(
        self,
        pos_list: ti.types.ndarray(dtype=ti.math.vec3),
        pos_mean: ti.types.ndarray(),
        pos_variance: ti.types.ndarray(),
        lindemann_frame: ti.types.ndarray(),
        lindemann_atom: ti.types.ndarray(),
    ):
        Nframes, Natoms = pos_list.shape
        ti.loop_config(serialize=True)  # serial compute
        for frame in range(Nframes):
            for i in range(Natoms):
                for j in range(i + 1, Natoms):
                    rij = pos_list[frame, i] - pos_list[frame, j]
                    rijdis = rij.norm()
                    mean = pos_mean[i, j]
                    var = pos_variance[i, j]
                    delta = rijdis - mean
                    pos_mean[i, j] = mean + delta / (frame + 1)
                    pos_variance[i, j] = var + delta * (rijdis - pos_mean[i, j])
                    pos_mean[j, i] = pos_mean[i, j]
                    pos_variance[j, i] = pos_variance[i, j]

            lindemann_index = ti.float64(0.0)
            for i in range(Natoms):
                for j in range(Natoms):
                    # if i != j:
                    if pos_variance[i, j] > 0:
                        ldm = ti.sqrt(pos_variance[i, j] / (frame + 1)) / pos_mean[i, j]
                        lindemann_index += ldm
                        lindemann_atom[frame, i] += ldm / (Natoms - 1)

            lindemann_index /= Natoms * (Natoms - 1)  # (Natoms-1)/2
            lindemann_frame[frame] = lindemann_index

    def compute(self):
        """Do the real Lindemann index calculation."""
        Nframes, Natoms = self.pos_list.shape[:2]
        pos_mean = np.zeros((Natoms, Natoms))
        pos_variance = np.zeros_like(pos_mean)
        if self.only_global:
            self.lindemann_trj = self._compute_global(
                self.pos_list, pos_mean, pos_variance
            )
        else:
            self.lindemann_frame = np.zeros(Nframes)
            self.lindemann_atom = np.zeros((Nframes, Natoms))
            self._compute_all(
                self.pos_list,
                pos_mean,
                pos_variance,
                self.lindemann_frame,
                self.lindemann_atom,
            )
            self.lindemann_trj = self.lindemann_frame[-1]
            self.if_compute = True

    def plot(self):
        """Plot the evolution of Lindemann index per frame.

        Raises:
            Exception: One should compute lidemann_frame first!

        Returns:
            tuple: (fig, ax) matplotlib figure and axis class.
        """

        if not self.if_compute:
            raise Exception("One should compute lidemann_frame first!")
        fig, ax = set_figure(
            figsize=(10, 7),
            left=0.14,
            bottom=0.16,
            right=0.95,
            top=0.97,
            use_pltset=True,
        )
        plt.plot(self.lindemann_frame, "o-")
        plt.xlabel("$\mathregular{N_{frames}}$")
        plt.ylabel("Lindemann index")
        plt.show()
        return fig, ax


if __name__ == "__main__":
    from time import time

    ti.init(ti.cpu, offline_cache=True)
    Nframe, Nparticles = 200, 1000
    pos_list = np.cumsum(
        np.random.choice([-1.0, 0.0, 1.0], size=(Nframe, Nparticles, 3)), axis=0
    )

    start = time()
    LDMG = LindemannParameter(pos_list, only_global=True)
    LDMG.compute()
    end = time()
    print(f"LDM_trj: {LDMG.lindemann_trj}, LDM costs: {end-start} s.")

    start = time()
    LDML = LindemannParameter(pos_list)
    LDML.compute()
    end = time()
    print(f"LDM_trj: {LDML.lindemann_trj}, LDM costs: {end-start} s.")

    print(
        "Global Lindemann index is close:",
        np.isclose(LDMG.lindemann_trj, LDML.lindemann_trj),
    )
    print(LDML.lindemann_frame[:10])
    LDML.plot()
