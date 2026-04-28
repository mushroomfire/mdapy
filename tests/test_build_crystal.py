# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""build_crystal — fixture-driven, no ASE at runtime."""

import numpy as np

from mdapy import build_crystal
from _fixture_helper import load_advanced


def test_bulk():
    data = load_advanced("build_crystal")

    Cu = build_crystal("Cu", "fcc", a=3.615)
    assert np.allclose(data["bulk__Cu_fcc__box"], Cu.box.box), "FCC bulk box wrong"
    assert np.allclose(data["bulk__Cu_fcc__pos"],
                       Cu.data.select("x", "y", "z").to_numpy()), "FCC bulk pos wrong"

    Fe = build_crystal("Fe", "bcc", a=2.83)
    assert np.allclose(data["bulk__Fe_bcc__box"], Fe.box.box), "BCC bulk box wrong"
    assert np.allclose(data["bulk__Fe_bcc__pos"],
                       Fe.data.select("x", "y", "z").to_numpy()), "BCC bulk pos wrong"

    C = build_crystal("C", "diamond", a=3.6)
    assert np.allclose(data["bulk__C_diamond__box"], C.box.box), "Diamond bulk box wrong"
    assert np.allclose(data["bulk__C_diamond__pos"],
                       C.data.select("x", "y", "z").to_numpy()), "Diamond bulk pos wrong"


def test_miller():
    data = load_advanced("build_crystal")

    Cu = build_crystal("Cu", "fcc", a=3.615,
                       miller1=[1,-1,0], miller2=[1,1,-2], miller3=[1,1,1])
    assert np.allclose(data["miller__Cu_fcc__box"], Cu.box.box), "FCC miller box wrong"
    assert int(data["miller__Cu_fcc__N"]) == Cu.N, "FCC miller atom count wrong"

    Fe = build_crystal("Fe", "bcc", a=2.83,
                       miller1=[1,2,1], miller2=[-1,0,1], miller3=[1,-1,1])
    assert np.allclose(data["miller__Fe_bcc__box"], Fe.box.box), "BCC miller box wrong"
    assert int(data["miller__Fe_bcc__N"]) == Fe.N, "BCC miller atom count wrong"

    C = build_crystal("C", "diamond", a=3.6,
                      miller1=[1,2,1], miller2=[-1,0,1], miller3=[1,-1,1])
    assert np.allclose(data["miller__C_diamond__box"], C.box.box), "Diamond miller box wrong"
    assert int(data["miller__C_diamond__N"]) == C.N, "Diamond miller atom count wrong"
