# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""
QHAElastic — temperature-dependent elastic constants on Al fcc with the
UNEP-v1 NEP potential. Tests both the MD (in-Python calculator) path and
the DFT round-trip (export -> mock VASP outputs -> import).
"""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

# QHAElastic depends on phonopy (and spglib). Skip the whole module on
# environments / CI runners where it is not installed.
pytest.importorskip("phonopy", reason="phonopy required for QHAElastic")
pytest.importorskip("spglib",  reason="spglib required for QHAElastic")

from mdapy import build_crystal, FIRE
from mdapy.qha_elastic import QHAElastic
from mdapy.nep import NEP
from _fixture_helper import input_path


def _relax_al():
    Al = build_crystal("Al", "fcc", 4.05)
    Al.calc = NEP(input_path("UNEP-v1.txt"))
    fy = FIRE(Al, optimize_cell=True)
    assert fy.run(fmax=1e-6, steps=100000, show_process=False), "Al relax failed"
    return Al


def _make_qha(Al, calc=None):
    return QHAElastic(
        Al,
        calc=calc,
        t_min=0,
        t_max=600,
        t_step=200,  # coarse, for test speed
        volume_strains=(-0.06, -0.03, 0.0, 0.03, 0.06),
        strain_values=(-0.02, -0.01, 0.0, 0.01, 0.02),
        supercell=(2, 2, 2),
        mesh=(6, 6, 6),
        symprec=1e-3,
        quiet=True,
    )


def test_md_path_basic():
    """Run the full MD path on Al and check the 0K constants and softening."""
    Al = _relax_al()
    qha = _make_qha(Al, calc=Al.calc)

    assert len(qha.unique_cells) == 65  # 5 V × (1 base + 3 modes × 4 nonzero eps)
    assert len(qha.grid) == 75

    qha.run()
    df = qha.compute()

    expected_cols = {
        "T", "V_eq", "alpha", "B_T",
        "C11_iso", "C12_iso", "C44_iso",
        "C11_adi", "C12_adi", "C44_adi",
        "K_VRH", "G_VRH", "E_VRH", "nu_VRH",
        "stable",
    }
    assert set(df.columns) == expected_cols

    # 0K elastic constants (NEP-Al on UNEP-v1; cubic Born-stable).
    c11_0 = df["C11_iso"][0]
    c12_0 = df["C12_iso"][0]
    c44_0 = df["C44_iso"][0]
    bt_0 = df["B_T"][0]
    assert 100.0 < c11_0 < 130.0, f"C11(0K) = {c11_0:.1f} GPa out of range"
    assert 40.0 < c12_0 < 70.0, f"C12(0K) = {c12_0:.1f} GPa out of range"
    assert 25.0 < c44_0 < 50.0, f"C44(0K) = {c44_0:.1f} GPa out of range"
    assert 65.0 < bt_0 < 90.0, f"B_T(0K) = {bt_0:.1f} GPa out of range"

    # Softening with temperature.
    assert df["C11_iso"][-1] < df["C11_iso"][0], "C11 should decrease with T"
    assert df["C44_iso"][-1] < df["C44_iso"][0], "C44 should decrease with T"
    assert df["B_T"][-1] < df["B_T"][0], "B_T should decrease with T"

    # Thermal expansion: V grows with T.
    assert df["V_eq"][-1] > df["V_eq"][0], "V_eq should grow with T"
    assert df["alpha"][-1] > 0, "alpha should be positive at 600 K"

    # 0K alpha should be near zero (quantum-suppressed).
    assert abs(df["alpha"][0]) < 1e-4

    # Born stability holds throughout the test range.
    assert df["stable"].all()

    # Adiabatic correction sign: Delta(C11) = C11_S - C11_T >= 0.
    delta_c11 = df["C11_adi"] - df["C11_iso"]
    assert (delta_c11 >= -1e-6).all(), "Adiabatic - isothermal C11 must be non-negative"
    # Cubic shear is identical isothermal == adiabatic.
    assert np.allclose(df["C44_adi"].to_numpy(), df["C44_iso"].to_numpy())


def test_dft_roundtrip():
    """Export inputs, fill with mock VASP outputs from the MD calculator,
    re-import and verify the C_ij agree."""
    Al = _relax_al()

    # MD reference run.
    qha_md = _make_qha(Al, calc=Al.calc)
    qha_md.run()
    df_md = qha_md.compute()

    # Capture per-cell results.
    md_cell_results = [
        {"E_static": uc["E_static"], "forces": [f.copy() for f in uc["forces"]]}
        for uc in qha_md.unique_cells
    ]

    # Fresh QHAElastic in DFT mode and export inputs.
    tmp = Path(tempfile.mkdtemp(prefix="qha_dft_"))
    try:
        qha_dft = _make_qha(Al, calc=None)
        qha_dft.export_inputs(tmp)

        # Sanity: manifest exists, all POSCAR-unitcell + disp dirs exist.
        with open(tmp / "manifest.json") as f:
            manifest = json.load(f)
        assert len(manifest["unique_cells"]) == len(qha_dft.unique_cells)
        for entry in manifest["unique_cells"]:
            sub = tmp / entry["path"]
            assert (sub / "static" / "POSCAR").exists()
            for d in range(1, entry["n_disp"] + 1):
                assert (sub / f"disp-{d:03d}" / "POSCAR").exists()

        # Plant fake VASP outputs from the MD results.
        for entry, result in zip(manifest["unique_cells"], md_cell_results):
            sub = tmp / entry["path"]
            _write_oszicar(sub / "static" / "OSZICAR", result["E_static"])
            for d, fc in enumerate(result["forces"], start=1):
                _write_vasprun_forces(sub / f"disp-{d:03d}" / "vasprun.xml", fc)

        qha_dft.import_results(tmp)
        df_dft = qha_dft.compute()
    finally:
        shutil.rmtree(tmp)

    # MD and DFT-roundtrip should agree to OSZICAR precision (<= 1e-3 GPa).
    for col in df_md.columns:
        if not df_md[col].dtype.is_numeric():
            continue
        diff = (df_md[col].to_numpy() - df_dft[col].to_numpy())
        assert np.abs(diff).max() < 1e-3, f"{col} differs by {np.abs(diff).max():.3e}"


# ----------------------------------------------------------
# Helpers — mock VASP outputs
# ----------------------------------------------------------


def _write_oszicar(path: Path, energy: float) -> None:
    with open(path, "w") as f:
        f.write(f"   1 F= {energy:.10E} E0= {energy:.10E} d E =-0.0E+00\n")


def _write_vasprun_forces(path: Path, forces: np.ndarray) -> None:
    lines = [
        "<modeling>",
        "<calculation>",
        '  <varray name="forces">',
    ]
    for fx, fy, fz in forces:
        lines.append(f"    <v>  {fx:.10f}  {fy:.10f}  {fz:.10f} </v>")
    lines.append("  </varray>")
    lines.append('  <energy><i name="e_fr_energy"> 0.0 </i></energy>')
    lines.append("</calculation>")
    lines.append("</modeling>")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def test_input_validation():
    Al = _relax_al()
    # strain_values without 0
    with pytest.raises(ValueError, match="must include 0"):
        QHAElastic(Al, calc=Al.calc, strain_values=(-0.01, 0.01, 0.02))
    # too few volume_strains
    with pytest.raises(ValueError, match="at least 5"):
        QHAElastic(Al, calc=Al.calc, volume_strains=(-0.04, 0.0, 0.04))
