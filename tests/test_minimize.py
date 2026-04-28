# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""FIRE minimization (8 modes) — fixture-driven, no ASE at runtime.

Reference final stress / forces / energies were generated once by running
ASE's FIRE2 (with UnitCellFilter when optimize_cell=True) under the NEP
potential for 5 steps. We compare mdapy's FIRE minimizer to those values.
"""

import numpy as np
import pytest

from mdapy import System
from mdapy.minimizer import FIRE
from mdapy.nep import NEP
from _fixture_helper import load_advanced, input_path


_MODES = [
    # (use_abc, optimize_cell, mask, hydrostatic_strain, constant_volume, scalar_pressure)
    (False, False, None,                 False, False, 0),
    (True,  False, None,                 False, False, 0),
    (False, True,  None,                 False, False, 0),
    (True,  True,  None,                 False, False, 0),
    (False, True,  [1, 0, 0, 0, 0, 0],   False, False, 0),
    (False, True,  None,                 True,  False, 0),
    (False, True,  None,                 False, True,  0),
    (False, True,  None,                 False, False, 1),
]


@pytest.mark.parametrize("idx,params", list(enumerate(_MODES)))
def test_minimize_mode(idx, params):
    use_abc, optimize_cell, mask, hydro, const_v, p = params
    data = load_advanced("minimize")

    system = System(input_path("AlCrNi.xyz"))
    system.calc = NEP(input_path("UNEP-v1.txt"))

    fire = FIRE(
        system,
        use_abc=use_abc,
        optimize_cell=optimize_cell,
        mask=mask,
        hydrostatic_strain=hydro,
        constant_volume=const_v,
        scalar_pressure=p,
    )
    fire.run(steps=int(data["steps"]))

    key = f"mode_{idx}"
    assert np.allclose(system.get_stress(),    data[f"{key}__stress"]),   f"mode {idx}: stress wrong"
    assert np.allclose(system.get_force(),     data[f"{key}__forces"]),   f"mode {idx}: force wrong"
    assert np.allclose(system.get_energies(),  data[f"{key}__energies"]), f"mode {idx}: energy wrong"
