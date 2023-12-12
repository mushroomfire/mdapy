# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import taichi as ti
import numpy as np
import matplotlib.pyplot as plt

try:
    from plotset import set_figure, pltset, cm2inch
except Exception:
    from .plotset import set_figure, pltset, cm2inch


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
        pos: ti.types.ndarray(element_dim=1),
        pos_min: ti.types.ndarray(element_dim=1),
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
        pos: ti.types.ndarray(element_dim=1),
        pos_min: ti.types.ndarray(element_dim=1),
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
        pos: ti.types.ndarray(element_dim=1),
        pos_min: ti.types.ndarray(element_dim=1),
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
        pos: ti.types.ndarray(element_dim=1),
        pos_min: ti.types.ndarray(element_dim=1),
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
                np.arange(self.res.shape[i]) * self.wbin
                + pos_min[xyz2dim[self.direction[i]][0]]
                + self.wbin / 2
            )
        # Thoes codes will cause memory leak!!!
        # vecarray = ti.types.vector(len(xyz2dim[self.direction]), ti.f64)
        # pos_bin = ti.ndarray(dtype=vecarray, shape=(self.N))
        # pos_bin.from_numpy(self.pos[:, xyz2dim[self.direction]])
        # pos_min_bin = ti.ndarray(dtype=vecarray, shape=(1))
        # pos_min_bin.from_numpy(pos_min[xyz2dim[self.direction]][np.newaxis, :])

        pos_bin = np.ascontiguousarray(self.pos[:, xyz2dim[self.direction]])
        pos_min_bin = np.ascontiguousarray(
            pos_min[xyz2dim[self.direction]][np.newaxis, :]
        )
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

    def plot(self, value_label=None):
        """Plot the binning results multidimensional binning.
        For multi values, only the first value will be plotted.

        Args:
            value_label (str, optional): y-lable for One-dimensional binning, or colorbar label for Two/Three-dimensional binning. Defaults to None.

        Returns:
            tuple: (fig, ax) matplotlib figure and axis class.
        """

        if not self.if_compute:
            self.compute()
        if len(self.direction) in [1, 2]:
            fig, ax = set_figure(
                figsize=(10, 7),
                bottom=0.18,
                top=0.97,
                left=0.18,
                right=0.92,
                use_pltset=True,
            )

        if len(self.direction) == 1:
            x, y = self.coor[self.direction], self.res[:, 1]
            plt.plot(
                x,
                y,
                "o-",
            )
            b, t = plt.ylim()
            plt.fill_between(x, b, y, alpha=0.2)
            plt.ylim(b, t)
            plt.xlabel(f"Coordination {self.direction}")
            if value_label is None:
                plt.ylabel(f"Some values")
            else:
                plt.ylabel(value_label)
            plt.show()
            return fig, ax
        elif len(self.direction) == 2:
            data = self.res[:, :, 1]
            X, Y = np.meshgrid(
                self.coor[self.direction[0]], self.coor[self.direction[1]]
            )
            h = plt.contourf(X, Y, data.T, cmap="GnBu")
            plt.xlabel(f"Coordination {self.direction[0]}")
            plt.ylabel(f"Coordination {self.direction[1]}")

            bar = fig.colorbar(h, ax=ax)
            if value_label is not None:
                bar.set_label(value_label)
            else:
                bar.set_label("Some value")
            plt.show()
            return fig, ax
        else:
            pltset()
            fig = plt.figure(figsize=(cm2inch(14), cm2inch(10)), dpi=150)
            ax = fig.add_subplot(111, projection="3d")
            plt.subplots_adjust(bottom=0.08, top=0.971, left=0.01, right=0.843)
            x, y, z = (
                self.coor[self.direction[0]],
                self.coor[self.direction[1]],
                self.coor[self.direction[2]],
            )
            X, Y, Z = np.meshgrid(x, y, z)
            data = np.zeros(X.shape)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    for k in range(X.shape[2]):
                        data[i, j, k] = self.res[j, i, k, 1]
            # data = self.res[:, :, :, 1].reshape(X.shape)

            kw = {
                "vmin": data.min(),
                "vmax": data.max(),
                "levels": np.linspace(data.min(), data.max(), 50),
                "cmap": "GnBu",
            }

            # Plot contour surfaces
            _ = ax.contourf(
                X[:, :, 0], Y[:, :, 0], data[:, :, 0], zdir="z", offset=Z.max(), **kw
            )

            _ = ax.contourf(
                X[0, :, :], data[0, :, :], Z[0, :, :], zdir="y", offset=Y.min(), **kw
            )
            C = ax.contourf(
                data[:, -1, :], Y[:, -1, :], Z[:, -1, :], zdir="x", offset=X.max(), **kw
            )
            # Set limits of the plot from coord limits
            xmin, xmax = X.min(), X.max()
            ymin, ymax = Y.min(), Y.max()
            zmin, zmax = Z.min(), Z.max()
            ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

            # Set labels and zticks
            ax.set(
                xlabel="X",
                ylabel="Y",
                zlabel="Z",
            )

            # Set zoom and angle view
            ax.view_init(40, -30, 0)
            # ax.set_box_aspect((xmax-xmin, ymax-ymin, zmax-zmin))
            # ax.set_aspect('auto')
            bar = fig.colorbar(C, ax=ax, fraction=0.015, pad=0.15)
            if value_label is not None:
                bar.set_label(value_label)
            else:
                bar.set_label("Some value")

            plt.show()
            return fig, ax


if __name__ == "__main__":
    from lattice_maker import LatticeMaker
    from time import time

    ti.init(ti.cpu)
    FCC = LatticeMaker(4.05, "FCC", 50, 50, 50)
    FCC.compute()
    # val = np.random.random(FCC.N)
    pos = FCC.pos
    # pos = pos[(pos[:, 0] < 100) | (pos[:, 0] > 300)]
    start = time()
    # for i in range(10):
    #     print(i)
    # binning = SpatialBinning(
    #     pos,
    #     "xyz",
    #     val,
    #     operation="mean",
    # )
    # binning.compute()
    binning = SpatialBinning(
        pos,
        "xyz",
        np.cos(pos[:, 0]) ** 2 - np.sin(pos[:, 1]) ** 2,
        wbin=4.06,
        operation="mean",
    )
    binning.compute()
    end = time()
    print(f"Binning time: {end-start} s.")
    print(binning.res[:, ..., 1])
    # print(binning.coor["x"])
    # print(binning.coor)
    # binning.plot(label_list=["x"], bar_label="x")
    # binning.plot(bar_label="x")
    binning.plot("x")
