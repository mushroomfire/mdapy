# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""
MDElastic — finite-T elastic constants on a small CrCoNi SQS using the
Sheng NiCoCr EAM potential. Test runs are deliberately short for speed —
they verify the protocol executes correctly and gives ball-park values,
not converged elastic constants.
"""

import numpy as np
import pytest

# MDElastic depends on the lammps Python module; skip the whole module on
# CI / environments where lammps is not installed.
pytest.importorskip("lammps", reason="lammps Python module required for MDElastic")

import mdapy as mp
from mdapy.md_elastic import MDElastic
from _fixture_helper import input_path


def _build_crconi():
    """Small CrCoNi HEA cell (3x3x3 conventional FCC = 108 atoms)."""
    return mp.build_hea(
        ["Ni", "Co", "Cr"], [1 / 3] * 3, "fcc", a=3.53, nx=3, ny=3, nz=3, random_seed=1
    )


def _common_kwargs():
    return dict(
        pair_style="eam/alloy",
        pair_coeff=f"* * {input_path('NiCoCr.lammps.eam')} Ni Co Cr",
        elements=["Ni", "Co", "Cr"],
        delta=0.02,
        n_equil=2000,
        n_run=2000,
        n_relax=5000,
        timestep=0.001,
        seed=12345,
        quiet=True,
    )


def test_md_elastic_basic_300K():
    hea = _build_crconi()
    mde = MDElastic(hea, temperature=300.0, ensemble="isothermal", **_common_kwargs())
    res = mde.run()

    # Volume ~ 27 fcc units * 4 atoms/unit * (3.53 Å)^3 ~ 1190 Å^3.
    assert 1100 < res.V_eq < 1400, f"V_eq = {res.V_eq:.1f} unexpected"

    # Reference stress should be small after NPT (a few GPa max with short runs).
    assert (
        np.abs(res.stress_ref).max() < 5.0
    ), f"|sigma_ref| = {np.abs(res.stress_ref).max():.2f} GPa too large"

    # Temperature roughly correct (within ~30 K of target).
    assert abs(res.T_actual - 300.0) < 50.0, f"T_actual = {res.T_actual:.1f}"

    # Ball-park CrCoNi (Sheng EAM) at room T:
    # C11 ~ 230-260, C12 ~ 165-200, C44 ~ 80-110 GPa (with 108-atom SQS noise).
    c11, c12, c44 = res.cubic_average()
    assert 180 < c11 < 300, f"C11 = {c11:.1f} GPa out of range"
    assert 130 < c12 < 220, f"C12 = {c12:.1f} GPa out of range"
    assert 60 < c44 < 130, f"C44 = {c44:.1f} GPa out of range"

    # 6x6 should be roughly cubic: off-diagonal coupling terms (e.g. C[1,4])
    # should be small compared to the diagonal block.
    C = res.cij_voigt
    coupling = max(
        abs(C[0, 3]),
        abs(C[0, 4]),
        abs(C[0, 5]),
        abs(C[1, 3]),
        abs(C[1, 4]),
        abs(C[1, 5]),
        abs(C[2, 3]),
        abs(C[2, 4]),
        abs(C[2, 5]),
    )
    assert (
        coupling < 30.0
    ), f"normal-shear coupling block too large for cubic-like SQS: {coupling:.1f} GPa"

    # Symmetric.
    assert np.allclose(C, C.T, atol=1e-8), "Cij should be symmetric"

    # Born stable.
    assert res.born_stable_cubic()

    # VRH dict has the expected keys.
    v = res.vrh()
    for key in ("K_V", "K_R", "K_H", "G_V", "G_R", "G_H", "E", "nu"):
        assert key in v
    assert v["K_H"] > 0 and v["G_H"] > 0


def test_md_elastic_scan_two_temperatures(tmp_path):
    """Scan two temperatures (parallel) and verify DataFrame columns + softening trend."""
    hea = _build_crconi()
    df = MDElastic.scan_parallel(
        hea,
        temperatures=[300.0, 700.0],
        n_workers=2,
        work_dir=str(tmp_path),
        ensemble="isothermal",
        **_common_kwargs(),
    )
    expected = {
        "T",
        "V_eq",
        "T_actual",
        "C11",
        "C12",
        "C44",
        "K_VRH",
        "G_VRH",
        "E_VRH",
        "nu_VRH",
        "stable",
    }
    assert expected.issubset(set(df.columns))
    # Per-component Cij also present.
    for i in range(1, 7):
        for j in range(1, 7):
            assert f"c{i}{j}" in df.columns

    # Volume grows with T.
    assert df["V_eq"][1] > df["V_eq"][0]
    # C44 (and usually C11) softens with T.
    assert df["C44"][1] < df["C44"][0]


def test_md_elastic_input_validation():
    hea = _build_crconi()
    with pytest.raises(ValueError, match="ensemble"):
        MDElastic(
            hea,
            temperature=300.0,
            ensemble="ridiculous",
            **_common_kwargs(),
        )
    with pytest.raises(ValueError, match="n_run"):
        bad = _common_kwargs()
        bad["n_run"] = 50  # < nevery*nrepeat = 100
        MDElastic(hea, temperature=300.0, **bad)
