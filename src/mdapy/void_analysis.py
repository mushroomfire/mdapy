# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

from mdapy import _neighbor
from mdapy.system import System
import numpy as np
import polars as pl
from typing import Optional


class VoidAnalysis:
    """
    Detect voids in an atomic system by discretizing the simulation box into cells
    and locating empty cells.

    Parameters
    ----------
    system : System
        The atomic system containing coordinates and box information.
    rc : float
        Cutoff radius that controls the size of the spatial grid cells.

    Attributes
    ----------
    system : System
        Input system.
    rc : float
        Grid cell size parameter.
    void_system : Optional[System]
        A ``System`` containing the centers of detected void regions.
        ``None`` if no voids are found.
    void_number : int
        Number of distinct void clusters.
    void_volume : float
        Estimated total void volume, computed as ``N_void_cells * rc^3``.
    """

    def __init__(self, system: System, rc: float):
        self.system = system
        self.rc = rc
        self.void_system: Optional[System] = None

    def compute(self):
        """
        Perform the void detection.

        1. Compute the number of grid cells ``ncell`` along each dimension based on
           the system thickness and cutoff ``rc``.
        2. Use ``_neighbor._fill_cell_for_void`` to assign each cell a value:
           ``0`` for empty, ``1`` for occupied.
        3. If empty cells exist, compute their geometric centers and create a
           ``System`` object containing void positions.
        4. Perform cluster analysis on these void positions.
        5. Filter clusters that contain only one void cell (noise avoidance).
        6. Renumber clusters and compute final ``void_number`` and ``void_volume``.
        """

        ncell = np.zeros(3, np.int32)
        thickness = self.system.box.get_thickness()
        for i in range(3):
            ncell[i] = max(int(np.floor(thickness[i] / self.rc)), 3)

        cell_id_list = _neighbor._fill_cell_for_void(
            self.system.data["x"].to_numpy(allow_copy=False),
            self.system.data["y"].to_numpy(allow_copy=False),
            self.system.data["z"].to_numpy(allow_copy=False),
            self.system.box.box,
            self.system.box.origin,
            self.system.box.boundary,
            self.rc,
        )

        if 0 in cell_id_list:
            # Compute the geometric center of empty cells
            void_pos = (
                (np.argwhere(cell_id_list == 0) + 0.5) / ncell
            ) @ self.system.box.box + self.system.box.origin

            void_system = System(pos=void_pos, box=self.system.box)

            # Cluster detection among void points
            void_system.cal_cluster_analysis(rc=self.rc * 1.1)
            res = (
                void_system.data.group_by(["cluster_id"])
                .len()
                .filter(pl.col("len") > 1)["cluster_id"]
            )

            if len(res) > 0:
                unique_res = res.unique().sort()
                new_cluster_id = {j: i for i, j in enumerate(unique_res, start=1)}

                void_system.update_data(
                    void_system.data.filter(
                        pl.col("cluster_id").is_in(res)
                    ).with_columns(
                        pl.col("cluster_id").replace_strict(new_cluster_id),
                        pl.lit("X").alias("element"),
                    )
                )

                self.void_system = void_system
                self.void_number = len(unique_res)
                self.void_volume = self.void_system.N * self.rc**3
            else:
                self.void_number = 0
                self.void_volume = 0.0
        else:
            self.void_number = 0
            self.void_volume = 0.0


if __name__ == "__main__":
    pass
