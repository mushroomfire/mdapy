# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import taichi as ti
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    from plotset import pltset, cm2inch
else:
    from .plotset import pltset, cm2inch


@ti.data_oriented
class SpatialBinning:
    """This class is used to divide particles into different bins and operating on each bin.
    One-dimensional to Three-dimensional binning are supported.

    Args:
        pos (np.ndarray): (:math:`N_p, 3`) particles positions.
        direction (str): binning direction, selected in ['x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz']
        vbin (np.ndarray): (:math:`N_p, x`), values to be operated, :math:`x` means arbitrary columns.
        wbin (float, optional): width of each bin. Defaults to 5.0.
        operation (str, optional): operation on each bin, selected from ['mean', 'sum', 'min', 'max']. Defaults to "mean".

    Outputs:
        - **res** (np.ndarray) - binning results, res[:, ..., 0] is the number of atoms each bin.
        - **coor** (dict) - coordination along binning direction, such as coor['x'].

    Examples:
        >>> import mdapy as mp

        >>> mp.init()

        >>> FCC = mp.LatticeMaker(4.05, 'FCC', 100, 50, 50) # Create a FCC structure.

        >>> FCC.compute() # Get atom positions.

        >>> binning = SpatialBinning(
                FCC.pos,
                "xz",
                FCC.pos[:, 0],
                operation="mean",
            ) # Initilize the binning class along 'xz' plane.

        >>> binning.compute() # Do the binnning calculation.

        >>> binning.res # Check the binning results.

        >>> binning.coor['x'] # Check the x coordination.

        >>> binning.plot(bar_label='x coordination')

    """

    def __init__(self, pos, direction, vbin, wbin=5.0, operation="mean") -> None:

        self.pos = pos
        self.N = self.pos.shape[0]
        assert direction in [
            "x",
            "y",
            "z",
            "xy",
            "xz",
            "yz",
            "xyz",
        ], f"unsupported direction {direction}. chosen in ['x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz']"
        self.direction = direction
        assert vbin.shape[0] == self.N, "shpae dismatchs between pos and vbin."
        if vbin.ndim == 1:
            self.vbin = np.ascontiguousarray(vbin[:, np.newaxis])
        else:
            self.vbin = vbin
        self.wbin = wbin
        assert operation in [
            "mean",
            "sum",
            "min",
            "max",
        ], f"unsupport operation {operation}, chosen in ['mean', 'sum', 'min', 'max']"
        self.operation = operation
        self.if_compute = False

    @ti.kernel
    def _Binning_sum(
        self,
        pos: ti.types.ndarray(),
        pos_min: ti.types.ndarray(),
        vbin: ti.types.ndarray(),
        res: ti.types.ndarray(),
    ):

        for i, j in ti.ndrange(self.N, res.shape[-1]):
            cindex = ti.floor((pos[i] - pos_min[0]) / self.wbin, dtype=ti.i32)
            if j == 0:
                res[cindex, 0] += 1.0
            else:
                res[cindex, j] += vbin[i, j - 1]

    @ti.kernel
    def _Binning_mean(
        self,
        pos: ti.types.ndarray(),
        pos_min: ti.types.ndarray(),
        vbin: ti.types.ndarray(),
        res: ti.types.ndarray(),
    ):

        for i, j in ti.ndrange(self.N, res.shape[-1]):
            cindex = ti.floor((pos[i] - pos_min[0]) / self.wbin, dtype=ti.i32)
            if j == 0:
                res[cindex, 0] += 1.0
            else:
                res[cindex, j] += vbin[i, j - 1]

        for I in ti.grouped(res):
            if I[I.n - 1] != 0:  # do not divide number per bin
                J = I
                J[J.n - 1] = 0
                if res[J] > 0:
                    res[I] /= res[J]

    @ti.kernel
    def _Binning_min(
        self,
        pos: ti.types.ndarray(),
        pos_min: ti.types.ndarray(),
        vbin: ti.types.ndarray(),
        res: ti.types.ndarray(),
    ):

        # init res
        for i, j in ti.ndrange(self.N, (1, res.shape[-1])):
            cindex = ti.floor((pos[i] - pos_min[0]) / self.wbin, dtype=ti.i32)
            res[cindex, j] = vbin[i, j - 1]
        # get min
        ti.loop_config(serialize=True)
        for i in range(self.N):
            cindex = ti.floor((pos[i] - pos_min[0]) / self.wbin, dtype=ti.i32)
            res[cindex, 0] += 1.0
            for j in range(1, res.shape[-1]):
                if vbin[i, j - 1] < res[cindex, j]:
                    res[cindex, j] = vbin[i, j - 1]

    @ti.kernel
    def _Binning_max(
        self,
        pos: ti.types.ndarray(),
        pos_min: ti.types.ndarray(),
        vbin: ti.types.ndarray(),
        res: ti.types.ndarray(),
    ):

        # init res
        for i, j in ti.ndrange(self.N, (1, res.shape[-1])):
            cindex = ti.floor((pos[i] - pos_min[0]) / self.wbin, dtype=ti.i32)
            res[cindex, j] = vbin[i, j - 1]
        # get max
        ti.loop_config(serialize=True)
        for i in range(self.N):
            cindex = ti.floor((pos[i] - pos_min[0]) / self.wbin, dtype=ti.i32)
            res[cindex, 0] += 1.0
            for j in range(1, res.shape[-1]):
                if vbin[i, j - 1] > res[cindex, j]:
                    res[cindex, j] = vbin[i, j - 1]

    def compute(self):
        """Do the real binning calculation."""
        xyz2dim = {
            "x": [0],
            "y": [1],
            "z": [2],
            "xy": [0, 1],
            "xz": [0, 2],
            "yz": [1, 2],
            "xyz": [0, 1, 2],
        }

        pos_min = np.min(self.pos, axis=0) - 0.001
        pos_max = np.max(self.pos, axis=0) + 0.001
        pos_delta = pos_max - pos_min
        nbin = np.ceil(pos_delta[xyz2dim[self.direction]] / self.wbin).astype(int)
        self.res = np.zeros((*nbin, self.vbin.shape[1] + 1))
        self.coor = {}
        for i in range(len(self.direction)):
            self.coor[self.direction[i]] = (
                np.arange(self.res.shape[i]) * self.wbin + pos_min[i] + 0.001
            )

        vecarray = ti.types.vector(len(xyz2dim[self.direction]), ti.f64)
        pos_bin = ti.ndarray(dtype=vecarray, shape=(self.N))
        pos_bin.from_numpy(self.pos[:, xyz2dim[self.direction]])
        pos_min_bin = ti.ndarray(dtype=vecarray, shape=(1))
        pos_min_bin.from_numpy(pos_min[xyz2dim[self.direction]][np.newaxis, :])

        if self.operation == "sum":
            self._Binning_sum(
                pos_bin,
                pos_min_bin,
                self.vbin,
                self.res,
            )
        elif self.operation == "mean":
            self._Binning_mean(
                pos_bin,
                pos_min_bin,
                self.vbin,
                self.res,
            )
        elif self.operation == "min":
            self._Binning_min(
                pos_bin,
                pos_min_bin,
                self.vbin,
                self.res,
            )
        elif self.operation == "max":
            self._Binning_max(
                pos_bin,
                pos_min_bin,
                self.vbin,
                self.res,
            )
        self.if_compute = True

    def plot(self, label_list=None, bar_label=None):
        """Plot the binning results for One- and Two-dimensional binning.
        For 2-D binning, only the first column value will be plotted.

        Args:
            label_list (list, optional): value name. Defaults to None.
            bar_label (str, optional): colorbar label for Two-dimensional binning. Defaults to None.

        Raises:
            NotImplementedError: "Three-dimensional binning visualization is not supported yet!"

        Returns:
            tuple: (fig, ax) matplotlib figure and axis class.
        """
        pltset()
        if not self.if_compute:
            self.compute()

        fig = plt.figure(figsize=(cm2inch(10), cm2inch(7)), dpi=150)
        plt.subplots_adjust(bottom=0.18, top=0.97, left=0.15, right=0.92)
        if len(self.direction) == 1:
            if label_list is not None:
                for i in range(1, self.res.shape[1]):
                    plt.plot(
                        self.coor[self.direction],
                        self.res[:, i],
                        "o-",
                        label=label_list[i - 1],
                    )
                plt.legend()
            else:
                for i in range(1, self.res.shape[1]):
                    plt.plot(self.coor[self.direction], self.res[:, i], "o-")

            plt.xlabel(f"Coordination {self.direction}")
            plt.ylabel(f"Some values")
            ax = plt.gca()
            plt.show()
            return fig, ax
        elif len(self.direction) == 2:
            data = np.zeros(self.res.shape[:2])
            for i in range(self.res.shape[0]):
                for j in range(self.res.shape[1]):
                    data[i, j] = self.res[i, j, 1]

            X, Y = np.meshgrid(
                self.coor[self.direction[0]], self.coor[self.direction[1]]
            )
            h = plt.contourf(X, Y, data.T, cmap="GnBu")
            plt.xlabel(f"Coordination {self.direction[0]}")
            plt.ylabel(f"Coordination {self.direction[1]}")

            ax = plt.gca()
            bar = fig.colorbar(h, ax=ax)
            if bar_label is not None:
                bar.set_label(bar_label)
            else:
                bar.set_label("Some value")
            plt.show()
            return fig, ax
        else:
            raise NotImplementedError(
                "Three-dimensional binning visualization is not supported yet!"
            )


if __name__ == "__main__":
    from lattice_maker import LatticeMaker
    from time import time

    ti.init(ti.cpu)
    FCC = LatticeMaker(4.05, "FCC", 100, 50, 50)
    FCC.compute()
    pos = FCC.pos
    pos = pos[(pos[:, 0] < 100) | (pos[:, 0] > 300)]
    start = time()
    binning = SpatialBinning(
        pos,
        "x",
        pos[:, 0] + pos[:, 1],
        operation="max",
    )
    binning.compute()
    end = time()
    print(f"Binning time: {end-start} s.")
    print(binning.res[:, ..., 1].max())
    print(binning.coor["x"])
    # print(binning.coor)
    # binning.plot(label_list=["x"], bar_label="x")
    binning.plot(bar_label="x")
