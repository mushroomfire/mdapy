# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

import polars as pl
import polars.selectors as cs
from mdapy.box import Box
from typing import Union, List, Optional


class SpatialBinning:
    """
    Spatial binning of atomic data in a simulation box.

    This class divides a simulation box into spatial bins along specified
    directions and computes aggregated statistics for atomic properties
    within each bin.

    Parameters
    ----------
    data : pl.DataFrame
        DataFrame containing atomic data with at least 'x', 'y', 'z' columns.
    box : Box
        Simulation box object containing box dimensions.
    direction : str
        Binning direction(s). Supported values: 'x', 'y', 'z', 'xy', 'xz',
        'yz', 'xyz'.
    bin_width : float
        Width of each bin in the same units as box dimensions.
    origin : list of float, optional
        Origin point for binning [x, y, z]. If None, uses box.origin.

    Attributes
    ----------
    bin_volume : float
        Volume of a single bin.

    Notes
    -----
    Currently only supports orthogonal (non-triclinic) boxes.

    """

    def __init__(
        self,
        data: pl.DataFrame,
        box: Box,
        direction: str,
        bin_width: float,
        origin: Optional[List[float]] = None,
    ):
        self.data = data
        self.box = box
        assert not self.box.triclinic, "only support orthogonal box."
        self.direction = direction.lower()
        self.bin_width = bin_width
        if origin is None:
            self.origin = self.box.origin
        else:
            assert len(origin) == 3
            self.origin == origin
        self.bin_volume = self._get_bin_volume()
        self._group = self._get_group()

    def _get_group(self):
        data = self.data.with_columns(
            pl.col("x") - self.origin[0],
            pl.col("y") - self.origin[1],
            pl.col("z") - self.origin[2],
        )
        if self.direction in ["x", "y", "z"]:
            return data.with_columns(
                pl.col(self.direction).alias(f"coor_{self.direction}")
            ).group_by(
                (pl.col(f"coor_{self.direction}") / self.bin_width)
                .cast(pl.Int32)
                .sort()
            )
        elif self.direction in ["xy", "xz", "yz"]:
            return (
                data.with_columns(
                    pl.col(self.direction[0]).alias(f"coor_{self.direction[0]}"),
                    pl.col(self.direction[1]).alias(f"coor_{self.direction[1]}"),
                )
                .group_by(
                    (pl.col(f"coor_{self.direction[0]}") / self.bin_width)
                    .cast(pl.Int32)
                    .sort()
                )
                .agg(
                    (pl.col(f"coor_{self.direction[1]}") / self.bin_width)
                    .cast(pl.Int32)
                    .sort(),
                    *system.data.columns,
                )
                .explode(f"coor_{self.direction[1]}", *system.data.columns)
                .group_by(
                    f"coor_{self.direction[0]}",
                    f"coor_{self.direction[1]}",
                    maintain_order=True,
                )
            )

        elif self.direction == "xyz":
            return (
                data.with_columns(
                    pl.col(self.direction[0]).alias(f"coor_{self.direction[0]}"),
                    pl.col(self.direction[1]).alias(f"coor_{self.direction[1]}"),
                    pl.col(self.direction[2]).alias(f"coor_{self.direction[2]}"),
                )
                .group_by(
                    (pl.col(f"coor_{self.direction[0]}") / self.bin_width)
                    .cast(pl.Int32)
                    .sort()
                )
                .agg(
                    (pl.col(f"coor_{self.direction[1]}") / self.bin_width)
                    .cast(pl.Int32)
                    .sort(),
                    f"coor_{self.direction[2]}",
                    *system.data.columns,
                )
                .explode(
                    f"coor_{self.direction[1]}",
                    f"coor_{self.direction[2]}",
                    *system.data.columns,
                )
                .group_by(f"coor_{self.direction[0]}", f"coor_{self.direction[1]}")
                .agg(
                    (pl.col(f"coor_{self.direction[2]}") / self.bin_width)
                    .cast(pl.Int32)
                    .sort(),
                    *system.data.columns,
                )
                .explode(f"coor_{self.direction[2]}", *system.data.columns)
                .group_by(
                    f"coor_{self.direction[0]}",
                    f"coor_{self.direction[1]}",
                    f"coor_{self.direction[2]}",
                    maintain_order=True,
                )
            )
        else:
            raise ValueError(
                f"Unrecognized direction: {self.direction}, only support ['x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz']"
            )

    def _get_bin_volume(self) -> float:
        """
        Calculate the volume of a single bin based on the binning direction.

        Returns
        -------
        float
            Volume of a single bin in the same units as the box dimensions.

        Notes
        -----
        The bin volume depends on the binning direction:
        - For 1D binning (x, y, or z): bin_width * area_of_perpendicular_plane
        - For 2D binning (xy, xz, or yz): bin_width^2 * perpendicular_length
        - For 3D binning (xyz): bin_width^3

        For orthogonal boxes, the box matrix diagonal elements represent the
        box lengths in each direction.
        """
        box_lengths = [self.box.box[0, 0], self.box.box[1, 1], self.box.box[2, 2]]

        if self.direction in ["x", "y", "z"]:
            # 1D binning: bin_width * perpendicular_area
            idx = {"x": 0, "y": 1, "z": 2}[self.direction]
            perpendicular_area = 1.0
            for i in range(3):
                if i != idx:
                    perpendicular_area *= box_lengths[i]
            return self.bin_width * perpendicular_area

        elif self.direction in ["xy", "xz", "yz"]:
            # 2D binning: bin_width^2 * perpendicular_length
            idx_map = {"xy": 2, "xz": 1, "yz": 0}
            perpendicular_idx = idx_map[self.direction]
            return self.bin_width**2 * box_lengths[perpendicular_idx]

        elif self.direction == "xyz":
            # 3D binning: bin_width^3
            return self.bin_width**3

        else:
            raise ValueError(f"Unrecognized direction: {self.direction}")

    def compute(
        self,
        name: Union[str, List[str]],
        operation: Union[str, List[str]],
        fill_empty: bool = False,
    ) -> pl.DataFrame:
        """
        Compute aggregated statistics for spatial bins.

        Parameters
        ----------
        name : str or list of str
            Column name(s) to aggregate.
        operation : str or list of str
            Aggregation operation(s) to perform. Supported operations:
            'mean', 'sum', 'min', 'max', 'sum/binvol', 'count'.

        Returns
        -------
        pl.DataFrame
            Aggregated results with bin coordinates and statistics.
        """
        if isinstance(name, str):
            name = [name]
        if isinstance(operation, str):
            operation = [operation]
        expr = []
        for n in name:
            assert n in self.data.columns
            assert self.data[n].dtype.is_numeric()
            for op in operation:
                col_name = f"{op}_{n}"
                if op == "sum":
                    expr.append(pl.col(n).sum().alias(col_name))
                elif op == "mean":
                    expr.append(pl.col(n).mean().alias(col_name))
                elif op == "min":
                    expr.append(pl.col(n).min().alias(col_name))
                elif op == "max":
                    expr.append(pl.col(n).max().alias(col_name))
                elif op == "sum/binvol":
                    expr.append((pl.col(n).sum() / self.bin_volume).alias(col_name))
                elif op == "count":
                    expr.append(pl.len().alias(col_name))
                else:
                    raise ValueError(
                        f"Unrecognized operation: {op}, only support ['mean', 'sum', 'min', 'max', 'sum/binvol', 'count']"
                    )

        expr1 = []
        for i in range(len(self.direction)):
            expr1.append((pl.col(f"coor_{self.direction[i]}") + 0.5) * self.bin_width)

        if fill_empty:
            res = self._group.agg(expr)

            direc = f"coor_{self.direction[0]}"
            min_bin, max_bin = res[direc].min(), res[direc].max()
            bin_data = pl.DataFrame(
                {direc: pl.int_range(min_bin, max_bin + 1, 1, eager=True)}
            )
            if len(self.direction) >= 2:
                direc = f"coor_{self.direction[1]}"
                min_bin, max_bin = res[direc].min(), res[direc].max()
                bin_data = bin_data.with_columns(
                    pl.lit(
                        pl.int_range(min_bin, max_bin + 1, 1, eager=True).to_list()
                    ).alias(direc)
                ).explode(direc)
            if len(self.direction) >= 3:
                direc = f"coor_{self.direction[2]}"
                min_bin, max_bin = res[direc].min(), res[direc].max()
                bin_data = bin_data.with_columns(
                    pl.lit(
                        pl.int_range(min_bin, max_bin + 1, 1, eager=True).to_list()
                    ).alias(direc)
                ).explode(direc)

            res = bin_data.join(
                res,
                on=[f"coor_{i}" for i in self.direction],
                how="left",
                maintain_order="left",
            ).with_columns(expr1)
            if "count" in operation:
                res = res.with_columns(cs.starts_with("count").fill_null(0))
        else:
            res = self._group.agg(expr).with_columns(expr1)

        return res


if __name__ == "__main__":
    from mdapy import build_crystal

    system = build_crystal("Cu", "fcc", 3.615, nx=10, ny=10, nz=10)
    system.update_data(system.data.filter((pl.col("x") > 20) | (pl.col("x") < 10)))
    sp = SpatialBinning(system.data, system.box, "xyz", 5.0)
    res = sp.compute(["y", "z"], ["sum", "count"], fill_empty=True)
    print(res.filter(pl.col("count_y") == 0))
