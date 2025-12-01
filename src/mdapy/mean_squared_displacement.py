# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Optional, Tuple, Literal

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes

try:
    import pyfftw

    def _fft(x: np.ndarray, n: int, axis: int) -> np.ndarray:
        """FFT wrapper using pyFFTW for faster computation.

        Using pyFFTW can significantly accelerate the MSD calculation for long trajectories.
        """
        a = pyfftw.empty_aligned(x.shape, "complex64")
        a[:] = x
        fft_obj = pyfftw.builders.fft(a, n=n, axis=axis)
        return fft_obj()

    def _ifft(x: np.ndarray, axis: int) -> np.ndarray:
        """IFFT wrapper using pyFFTW for faster computation."""
        a = pyfftw.empty_aligned(x.shape, "complex64")
        a[:] = x
        fft_obj = pyfftw.builders.ifft(a, axis=axis)
        return fft_obj()

except Exception:
    try:
        from scipy.fft import fft as _fft
        from scipy.fft import ifft as _ifft
    except Exception:
        from numpy.fft import fft as _fft
        from numpy.fft import ifft as _ifft


class MeanSquaredDisplacement:
    r"""
    Mean Squared Displacement (MSD) calculator with optional FFT acceleration.

    The MSD quantifies particle motion over time:

    .. math::
        MSD(t) = \langle \lvert \mathbf{r}(t_0 + t) - \mathbf{r}(t_0) \rvert^2 \rangle

    where :math:`\mathbf{r}(t)` is the particle position at time :math:`t` and
    :math:`\langle \cdot \rangle` denotes ensemble averaging.

    Two computation modes are available:

    1. **window** (FFT-accelerated, O(N log N)):
        Efficient for long trajectories, based on the Wiener-Khinchin theorem:

        .. math::
            MSD(m) = S_1(m) - 2 S_2(m)

        with

        .. math::
            S_1(m) = \frac{1}{N-m} \sum_{t=0}^{N-m-1} \Big[ \mathbf{r}^2(t) + \mathbf{r}^2(t+m) \Big]

        .. math::
            S_2(m) = \frac{1}{N-m} \sum_{t=0}^{N-m-1} \mathbf{r}(t) \cdot \mathbf{r}(t+m)

        FFT is used to efficiently compute the autocorrelation :math:`S_2(m)`.

    2. **direct** (O(N)):
        Computes MSD relative to the initial frame:

        .. math::
          :nowrap:

          \begin{eqnarray*}
              MSD(t) =& \dfrac{1}{N} \sum_{i=1}^{N} (r_i(t) - r_i(0))^2 \\
          \end{eqnarray*}

    **Important**:
        - Input positions must be **unwrapped**. Wrapped coordinates across periodic boundaries
          will produce incorrect MSD.
        - For diffusive motion in 3D: :math:`MSD(t) \approx 6 D t` where :math:`D` is the diffusion coefficient.
        - Using `pyFFTW` (if installed) can accelerate FFT computation significantly for large trajectories.

    Attributes
    ----------
    pos_list : np.ndarray
        Particle positions, shape [Nframe, N, 3] (unwrapped).
    mode : str
        Calculation mode ('window' or 'direct').
    particle_msd : Optional[np.ndarray]
        Per-particle MSD, shape [Nframe, N].
    msd : Optional[np.ndarray]
        Ensemble-averaged MSD, shape [Nframe].

    References
    ----------
    [1] Vania Calandrini, Eric Pellegrini, Paolo Calligari, Konrad Hinsen, Gerald R. Kneller.
        "Nmoldyn – interfacing spectroscopic experiments, molecular dynamics simulations
        and models for time correlation functions." École thématique de la Société Française
        de la Neutronique, 12:201–232, 2011.
    [2] pyFFTW: https://github.com/pyFFTW/pyFFTW
    """

    def __init__(
        self, pos_list: np.ndarray, mode: Literal["window", "direct"] = "window"
    ) -> None:
        """Initialize MSD calculator.

        Parameters
        ----------
        pos_list : np.ndarray
            Unwrapped positions, shape [Nframe, N, 3].
        mode : str
            Calculation mode: 'window' or 'direct'.

        Raises
        ------
        AssertionError
            If pos_list shape is invalid or mode is not recognized.
        """
        self.pos_list: np.ndarray = pos_list
        assert len(self.pos_list.shape) == 3, (
            f"pos_list must have shape [Nframe, N, 3], got {self.pos_list.shape}"
        )

        self.mode: str = mode
        assert self.mode in ["window", "direct"], (
            f"mode must be 'window' or 'direct', got '{self.mode}'"
        )

        self.particle_msd: Optional[np.ndarray] = None
        self.msd: Optional[np.ndarray] = None

    def _autocorrFFT(self, x: np.ndarray) -> np.ndarray:
        """Compute autocorrelation using FFT (Wiener-Khinchin theorem)."""
        N = x.shape[0]
        F = _fft(x, n=2 * N, axis=0)
        PSD = F * F.conjugate()
        res = _ifft(PSD, axis=0)[:N].real
        return res / np.arange(N, 0, -1)[:, np.newaxis]

    def compute(self) -> None:
        """Compute MSD for all particles and time frames."""
        if self.mode == "window":
            Nframe = self.pos_list.shape[0]
            D = np.square(self.pos_list).sum(axis=-1)
            D = np.append(D, np.zeros(self.pos_list.shape[:-1]), axis=0)

            Q = 2 * D.sum(axis=0)
            S1 = np.zeros(self.pos_list.shape[:-1])
            for m in range(Nframe):
                Q -= D[m - 1, :] + D[Nframe - m, :]
                S1[m, :] = Q / (Nframe - m)

            S2 = np.sum(
                [
                    self._autocorrFFT(self.pos_list[:, :, i])
                    for i in range(self.pos_list.shape[-1])
                ],
                axis=0,
            )
            self.particle_msd = S1 - 2 * S2

        elif self.mode == "direct":
            self.particle_msd = np.square(self.pos_list - self.pos_list[0]).sum(axis=-1)

        self.msd = self.particle_msd.mean(axis=-1)

    def plot(
        self, fig: Optional[Figure] = None, ax: Optional[Axes] = None
    ) -> Tuple[Figure, Axes]:
        """Plot MSD vs. frame number.

        Parameters
        ----------
        fig : Optional[Figure]
            Existing matplotlib figure.
        ax : Optional[Axes]
            Existing matplotlib axes.

        Returns
        -------
        Tuple[Figure, Axes]
            Figure and axes for further customization.
        """

        if self.msd is None:
            self.compute()

        if fig is None and ax is None:
            from mdapy.plotset import set_figure

            fig, ax = set_figure()

        ax.plot(self.msd, "o-", label=self.mode)
        ax.legend()
        ax.set_xlabel(r"$\mathrm{Frame}$")
        ax.set_ylabel(r"MSD ($\mathrm{\AA^2}$)")

        return fig, ax


if __name__ == "__main__":
    from time import time
    import matplotlib.pyplot as plt

    # Generate random walk trajectory (3D Brownian motion)
    Nframe, Nparticles = 200, 1000
    np.random.seed(1)
    pos_list = np.cumsum(np.random.randn(Nframe, Nparticles, 3), axis=0)

    # Test window mode (FFT-accelerated)
    start = time()
    MSD = MeanSquaredDisplacement(pos_list=pos_list, mode="window")
    MSD.compute()

    # Plot results
    fig, ax = MSD.plot()
    plt.show()
