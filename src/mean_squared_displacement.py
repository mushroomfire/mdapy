import numpy as np
from mdapy.plot.pltset import pltset, cm2inch
import matplotlib.pyplot as plt

try:
    import pyfftw

    def fft(x, n, axis=0):
        a = pyfftw.empty_aligned(x.shape, "complex64")
        a[:] = x
        fft_object = pyfftw.builders.fft(a, n=n, axis=axis)
        return fft_object()

    def ifft(x, axis=0):
        a = pyfftw.empty_aligned(x.shape, "complex64")
        a[:] = x
        fft_object = pyfftw.builders.ifft(a, axis=axis)
        return fft_object()

except ImportError:
    try:
        from scipy.fftpack import fft, ifft
    except ImportError:
        from numpy.fft import fft, ifft


class MeanSquaredDisplacement:
    """
    ref: https://stackoverflow.com/questions/34222272/computing-mean-square-displacement-using-python-and-fft
    pos_list : np.ndarray, shape(Nframe, Nparticles, 3), one should make sure every particle locate at same row for all frames.
    """

    def __init__(self, pos_list, mode="windows"):
        self.pos_list = pos_list
        assert len(self.pos_list.shape) == 3
        self.mode = mode
        assert self.mode in ["windows", "direct"]
        self.if_compute = False

    def _autocorrFFT(self, x):
        N = x.shape[0]
        F = fft(x, n=2 * N)  # 2*N because of zero-padding
        PSD = F * F.conjugate()
        res = ifft(PSD)
        res = (res[:N]).real  # now we have the autocorrelation in convention B
        n = np.arange(N, 0, -1)  #   N*np.ones(N)-np.arange(0,N) #divide res(m) by (N-m)
        return res / n[:, np.newaxis]  # this is the autocorrelation in convention A

    def compute(self):

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
            self.partical_msd = S1 - 2 * S2
        elif self.mode == "direct":
            self.partical_msd = np.square(self.pos_list - self.pos_list[0, :, :]).sum(
                axis=-1
            )

        self.msd = self.partical_msd.mean(axis=-1)
        self.if_compute = True

    def plot(self):
        pltset()
        if not self.if_compute:
            self.compute()
        fig = plt.figure(figsize=(cm2inch(10), cm2inch(7)), dpi=150)
        plt.subplots_adjust(left=0.16, bottom=0.16, right=0.95, top=0.97)
        plt.plot(self.msd, "o-", label=self.mode)
        plt.legend()
        plt.xlabel("$\mathregular{N_{frames}}$")
        plt.ylabel("MSD ($\mathregular{\AA}$)")
        # plt.xlim(0, self.msd.shape[0])

        ax = plt.gca()
        plt.show()
        return fig, ax


if __name__ == "__main__":
    from time import time

    Nframe, Nparticles = 500, 10000
    pos_list = np.cumsum(
        np.random.choice([-1.0, 0.0, 1.0], size=(Nframe, Nparticles, 3)), axis=0
    )
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
    fig = plt.figure(figsize=(cm2inch(10), cm2inch(7)), dpi=150)
    plt.subplots_adjust(left=0.16, bottom=0.16, right=0.95, top=0.97)
    plt.plot(msd_w, "o-", label="windows")
    plt.plot(msd_d, "o-", label="direct")
    plt.legend()
    plt.xlabel("$\mathregular{N_{frames}}$")
    plt.ylabel("MSD ($\mathregular{\AA}$)")
    plt.xlim(0, msd_w.shape[0])
    plt.show()
