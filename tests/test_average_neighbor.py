import mdapy as mp
from ovito.io import import_file
from ovito.modifiers import ComputePropertyModifier
import numpy as np


class TestAverageNeighbor:
    def average_neighbor(self, filename, average_cutoff):
        pipeline = import_file(filename)
        modifier = ComputePropertyModifier(
            output_property="x_ave",
            operate_on="particles",
            cutoff_radius=average_cutoff,
            expressions=["Position.X / (NumNeighbors + 1)"],
            neighbor_expressions=["Position.X / (NumNeighbors + 1)"],
        )
        pipeline.modifiers.append(modifier)
        data = pipeline.compute()

        system = mp.System(filename)
        system.average_by_neighbor(average_cutoff, "x", include_self=True)

        assert np.allclose(
            data.particles["x_ave"][...],
            system.data["x_ave"].to_numpy(allow_copy=False),
            atol=1e-6,
        )

    def test_box_big_rec(self):
        self.average_neighbor("input_files/rec_box_big.xyz", 4.0)

    def test_box_big_tri(self):
        self.average_neighbor("input_files/tri_box_big.xyz", 4.0)
