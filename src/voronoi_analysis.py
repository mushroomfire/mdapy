# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import numpy as np
from .voronoi import _voronoi_analysis


class VoronoiAnalysis:
    def __init__(self, pos, box, boundary) -> None:
        self.pos = pos
        self.box = box
        self.boundary = boundary

    def compute(self):
        N = self.pos.shape[0]
        self.vol = np.zeros(N)
        self.neighbor_number = np.zeros(N, dtype=int)
        self.cavity_radius = np.zeros(N)
        _voronoi_analysis.get_voronoi_volume(
            self.pos,
            self.box,
            np.bool_(self.boundary),
            self.vol,
            self.neighbor_number,
            self.cavity_radius,
        )
