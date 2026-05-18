# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

import numpy as np
import polars as pl
from mdapy import System
from mdapy.box import Box
from mdapy.render import TachyonRender


def test_render_bond_cpu():
    data = pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0],
            "element": ["Cu", "Zr", "Cu"],
        }
    )
    system = System(data=data, box=Box([6.0, 6.0, 6.0], boundary=[0, 0, 0]))
    system.create_bonds({("Cu", "Cu"): 0.5, ("Cu", "Zr"): 1.1, ("Zr", "Zr"): 1.1})

    ren = TachyonRender(backend="cpu", antialiasing=False, ao=False)
    img = ren.render_system(
        system,
        draw_bond=True,
        draw_box=False,
        bond_radius=0.12,
        bond_color_mode="atom",
        width=160,
        height=120,
    )

    assert isinstance(img, np.ndarray)
    assert img.shape == (120, 160, 4)
    assert img.dtype == np.uint8
