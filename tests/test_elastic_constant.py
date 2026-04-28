# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""Elastic constant — fixture-driven, no pymatgen at runtime.

Reference Voigt tensor was generated once with pymatgen
(`ElasticTensor.from_independent_strains`) on a NEP-relaxed FCC Al
structure; see tests/_generate_fixtures/generate_advanced.py.
"""

import numpy as np

from mdapy.elastic import get_elastic_constant
from mdapy import build_crystal
from mdapy.nep import NEP
from _fixture_helper import load_advanced, input_path


def test_elastic():
    data = load_advanced("elastic_constant")
    system = build_crystal(str(data["symbol"]), str(data["structure"]),
                           float(data["a"]))
    calc = NEP(input_path("UNEP-v1.txt"))
    et_mda = get_elastic_constant(system, calc)
    assert np.allclose(et_mda.voigt, data["voigt"]), "elastic tensor differs"
