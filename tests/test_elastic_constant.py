# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""Elastic constant — fixture-driven, no pymatgen at runtime.

Reference Voigt tensor was generated once with pymatgen
(`ElasticTensor.from_independent_strains`) on a NEP-relaxed FCC Al
structure; see tests/_generate_fixtures/generate_advanced.py.
"""

import numpy as np
import pytest

from mdapy.elastic import (
    get_elastic_constant,
    DeformedStructureSet,
    ElasticTensor2D,
    _strain_to_deformation,
)
from mdapy import build_crystal, System
from mdapy.minimizer import FIRE
from mdapy.nep import NEP
from _fixture_helper import load_advanced, input_path


def test_elastic():
    data = load_advanced("elastic_constant")
    system = build_crystal(str(data["symbol"]), str(data["structure"]),
                           float(data["a"]))
    calc = NEP(input_path("UNEP-v1.txt"))
    et_mda = get_elastic_constant(system, calc)
    assert np.allclose(et_mda.voigt, data["voigt"]), "elastic tensor differs"


# ------------------------------------------------------------------
# 2D materials: a hexagonal Al monolayer with UNEP-v1 is a cheap,
# self-contained in-plane isotropic sheet.
# ------------------------------------------------------------------


def _al_monolayer(c=20.0, a=2.80):
    box = np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]])
    system = System(box=box, pos=np.array([[0.0, 0.0, c / 2]]))
    system.set_element(np.array(["Al"]))
    return system


@pytest.fixture(scope="module")
def unep():
    return NEP(input_path("UNEP-v1.txt"))


def test_elastic_2d_vacuum_independent(unep):
    et20 = get_elastic_constant(_al_monolayer(20.0), unep, dim=2)
    et30 = get_elastic_constant(_al_monolayer(30.0), unep, dim=2)
    assert isinstance(et20, ElasticTensor2D)
    assert et20.voigt.shape == (3, 3)
    assert np.allclose(et20.voigt, et30.voigt, rtol=1e-6, atol=1e-6)


def test_elastic_2d_hexagonal_isotropy(unep):
    C = get_elastic_constant(_al_monolayer(), unep, dim=2).voigt
    assert C[0, 0] > 0 and C[0, 1] > 0
    assert np.isclose(C[0, 0], C[1, 1], rtol=1e-2)
    assert np.isclose(C[2, 2], (C[0, 0] - C[0, 1]) / 2, rtol=1e-2)
    assert np.allclose(C[[0, 1], 2], 0.0, atol=1e-2)


def test_elastic_2d_matches_energy_strain(unep):
    """Stress-strain C (N/m) must equal the curvature of energy per area."""
    system = _al_monolayer()
    C = get_elastic_constant(system, unep, dim=2, fmax=1e-6).voigt
    cell = system.box.box.copy()
    area = np.linalg.norm(np.cross(cell[0], cell[1]))
    e0 = system.get_energy()

    def energy_density(strain):
        new_cell = cell @ _strain_to_deformation(strain).T
        s = System(box=new_cell, pos=np.array([[0.0, 0.0, new_cell[2, 2] / 2]]))
        s.set_element(np.array(["Al"]))
        s.calc = unep
        return (s.get_energy() - e0) / area * 16.02176634  # eV/Å^2 -> N/m

    eps = np.linspace(-0.004, 0.004, 9)
    curv = lambda f: 2 * np.polyfit(eps, [energy_density(f(e)) for e in eps], 4)[2]
    c11 = curv(lambda e: np.diag([e, 0.0, 0.0]))
    c11_c12 = curv(lambda e: np.diag([e, e, 0.0])) / 2  # = C11 + C12
    assert np.isclose(C[0, 0], c11, rtol=5e-3)
    assert np.isclose(C[0, 0] + C[0, 1], c11_c12, rtol=5e-3)


def test_elastic_2d_relax_keeps_vacuum(unep):
    system = _al_monolayer(20.0)
    get_elastic_constant(system, unep, dim=2)
    assert np.allclose(system.box.box[2], [0.0, 0.0, 20.0])
    assert np.allclose(system.box.box[:2, 2], 0.0)


def test_deformed_structure_set_2d():
    dss = DeformedStructureSet(_al_monolayer(), dim=2)
    assert len(dss) == 12  # (xx, yy, xy) x 4 strains
    for defo, _ in dss:
        assert np.allclose(defo[2], [0, 0, 1]) and np.allclose(defo[:, 2], [0, 0, 1])


def test_elastic_2d_wrong_orientation(unep):
    # vacuum along x instead of along the third cell vector
    system = System(
        box=np.diag([20.0, 2.8 * np.sqrt(3), 2.8]),
        pos=np.array([[0.0, 0.0, 0.0], [0.0, 1.4 * np.sqrt(3), 1.4]]),
    )
    system.set_element(np.array(["Al", "Al"]))
    with pytest.raises(ValueError, match="vacuum must be along"):
        get_elastic_constant(system, unep, dim=2)


def test_elastic_2d_thin_vacuum_warns(unep):
    with pytest.warns(UserWarning, match="Vacuum thickness"):
        get_elastic_constant(_al_monolayer(c=3.0), unep, dim=2)


def test_elastic_tensor_2d_isotropic_moduli():
    K, G = 200.0, 140.0  # isotropic sheet: C11 = K + G, C12 = K - G, C66 = G
    et = ElasticTensor2D([[K + G, K - G, 0], [K - G, K + G, 0], [0, 0, G]])
    r = et.vrh()
    for key, ref in [("K_V", K), ("K_R", K), ("G_V", G), ("G_R", G)]:
        assert np.isclose(r[key], ref)
    assert np.isclose(r["E"], 4 * K * G / (K + G))
    assert np.isclose(r["nu"], (K - G) / (K + G))
    e = et.engineering_constants()
    assert np.isclose(e["E_x"], r["E"]) and np.isclose(e["nu_xy"], r["nu"])
    assert et.is_stable()
    assert np.allclose(et.to_gpa(2.0), et.voigt * 5.0)
