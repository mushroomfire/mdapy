# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import numpy as np
import matplotlib.pyplot as plt

try:
    from plotset import set_figure
except Exception:
    from .plotset import set_figure

try:
    import pyfftw

    def fft(x, n, axis):
        # FFT wrapper of pyfftw.
        a = pyfftw.empty_aligned(x.shape, "complex64")
        a[:] = x
        fft_object = pyfftw.builders.fft(a, n=n, axis=axis)
        return fft_object()

    def ifft(x, axis):
        # IFFT wrapper of pyfftw.
        a = pyfftw.empty_aligned(x.shape, "complex64")
        a[:] = x
        fft_object = pyfftw.builders.ifft(a, axis=axis)
        return fft_object()

except Exception:
    try:
        from scipy.fft import fft, ifft
    except Exception:
        from numpy.fft import fft, ifft


class MeanSquaredDisplacement:
    """This class is used to calculate the mean squared displacement MSD of system, which can be used to
    reflect the particle diffusion trend and describe the melting process. Generally speaking, MSD is an
    average displacement over all windows of length :math:`m` over the course of the simulation (so-called
    'windows' mode here) and defined
    by:

    .. math:: MSD(m) = \\frac{1}{N_{p}} \\sum_{i=1}^{N_{p}} \\frac{1}{N_{f}-m} \\sum_{k=0}^{N_{f}-m-1} (\\vec{r}_i(k+m) - \\vec{r}_i(k))^2,

    where :math:`r_i(t)` is the position of particle :math:`i` in frame :math:`t`. It is computationally extensive
    while using a fast Fourier transform can remarkably reduce the computation cost as described in `nMoldyn - Interfacing
    spectroscopic experiments, molecular dynamics simulations and models for time correlation functions
    <https://doi.org/10.1051/sfn/201112010>`_ and discussion in `StackOverflow <https://stackoverflow.com/questions/34222272/computing-mean-square-displacement-using-python-and-fft>`_.

    .. note:: One can install `pyfftw <https://github.com/pyFFTW/pyFFTW>`_ to accelerate the calculation,
      otherwise mdapy will use `scipy.fft <https://docs.scipy.org/doc/scipy/reference/fft.html#module-scipy.fft>`_
      to do the Fourier transform.

    Sometimes one only need the following atomic displacement (so-called 'direct' mode here):

    .. math:: MSD(t) = \\dfrac{1}{N_p} \\sum_{i=1}^{N_p} (r_i(t) - r_i(0))^2.

    Args:
        pos_list (np.ndarray): (:math:`N_f, N_p, 3`), :math:`N_f` frames particle position, which need to be unwrapped for periodic boundary.
        mode (str, optional): calculation mode, selected from ['windows', 'direct']. Defaults to "windows".

    Outputs:
        - **msd** (np.ndarray) - (:math:`N_f`), mean squared displacement per frames.
        - **particle_msd** (np.ndarray) - (:math:`N_f, N_p`), mean squared displacement per atom per frames.

    Examples:
        >>> import mdapy as mp

        >>> mp.init()

        >>> import numpy as np

        >>> pos_list = np.cumsum(
            np.random.choice([-1.0, 0.0, 1.0], size=(200, 1000, 3)), axis=0
            ) # Generate a random walk trajectory with 200 frames and 1000 particles.

        >>> MSD = mp.MeanSquaredDisplacement(pos_list, mode="windows") # Initilize MSD class.

        >>> MSD.compute() # Calculate the MSD in 'windows' mode.

        >>> MSD.msd # Check msd.

        >>> MSD.particle_msd.shape # Check msd per particle, should be (200, 1000) here.

        >>> MSD.plot() # Plot the evolution of msd per frame.
    """

    def __init__(self, pos_list, mode="windows"):
        self.pos_list = pos_list
        assert len(self.pos_list.shape) == 3
        self.mode = mode
        assert self.mode in ["windows", "direct"]
        self.if_compute = False

    def _autocorrFFT(self, x):
        N = x.shape[0]
        F = fft(x, n=2 * N, axis=0)  # 2*N because of zero-padding
        PSD = F * F.conjugate()
        res = ifft(PSD, axis=0)
        res = (res[:N]).real  # now we have the autocorrelation in convention B
        n = np.arange(N, 0, -1)  #   N*np.ones(N)-np.arange(0,N) #divide res(m) by (N-m)
        return res / n[:, np.newaxis]  # this is the autocorrelation in convention A

    def compute(self):
        """Do the real MSD calculation."""
        if self.mode == "windows":
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
            self.particle_msd = np.square(self.pos_list - self.pos_list[0, :, :]).sum(
                axis=-1
            )

        self.msd = self.particle_msd.mean(axis=-1)
        self.if_compute = True

    def plot(self):
        """Plot the evolution of MSD per frame.

        Returns:
            tuple: (fig, ax) matplotlib figure and axis object.
        """
        if not self.if_compute:
            self.compute()
        fig, ax = set_figure(
            figsize=(10, 7),
            left=0.18,
            bottom=0.16,
            right=0.95,
            top=0.97,
            use_pltset=True,
        )
        plt.plot(self.msd, "o-", label=self.mode)
        plt.legend()
        plt.xlabel("$\mathregular{N_{frames}}$")
        plt.ylabel("MSD ($\mathregular{\AA^2}$)")
        plt.show()
        return fig, ax


if __name__ == "__main__":
    from time import time

    Nframe, Nparticles = 200, 1000
    pos_list = np.cumsum(
        np.random.choice([-1.0, 1.0], size=(Nframe, Nparticles, 3)), axis=0
    ) * np.sqrt(2.0)
    start = time()
    MSD = MeanSquaredDisplacement(pos_list=pos_list, mode="windows")
    MSD.compute()
    end = time()
    msd_w = MSD.msd
    print(f"windows mode costs: {end-start} s.")
    MSD.plot()
    start = time()
    MSD = MeanSquaredDisplacement(pos_list=pos_list, mode="direct")
    MSD.compute()
    end = time()
    msd_d = MSD.msd
    print(f"direct mode costs: {end-start} s.")
    MSD.plot()
    fig, ax = set_figure(figsize=(9, 7), left=0.19, bottom=0.16, right=0.95, top=0.97)
    plt.plot(msd_w, "o-", label="windows")
    plt.plot(msd_d, "o-", label="direct")
    plt.plot(np.arange(Nframe) * 6, label="theoritical")
    plt.legend()
    plt.xlabel("$\mathregular{N_{frames}}$")
    plt.ylabel("MSD ($\mathregular{\AA^2}$)")
    plt.xlim(0, msd_w.shape[0])
    plt.show()
