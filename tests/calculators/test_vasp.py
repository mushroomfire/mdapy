# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import sys
from pathlib import Path

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal, random_alloy
from mdapy.calculators import EAM, VASP
from mdapy.errors import CalculationFailed, InvalidParameter, MissingProperty
from mdapy.simulation import minimize

FAKE = f'"{sys.executable}" "{Path(__file__).with_name("fake_vasp.py")}"'


def potcars(folder, *setups):
    """A folder of POTCARs, one folder per setup, each naming itself."""
    for setup in setups:
        (folder / setup).mkdir(parents=True)
        (folder / setup / "POTCAR").write_text(f"  PAW_PBE {setup} 01Jan2000\n")
    return folder


def alloy(seed=1):
    """Copper and nickel at random, the atoms of the two elements mixed."""
    cell = build_crystal("fcc", 3.6, elements="Cu", repeat=(2, 2, 2))
    system = random_alloy(cell, {"Cu": 0.5, "Ni": 0.5}, seed=seed)
    rng = np.random.default_rng(seed)
    return system.with_columns(
        positions=system["positions"] + rng.normal(scale=0.08, size=(32, 3))
    )


def test_a_single_point_as_vasp_would_give_it(tmp_path):
    system = alloy()
    folder = potcars(tmp_path / "potpaw_PBE", "Cu", "Ni_pv")
    vasp = VASP(
        {"encut": 400, "ediff": 1e-6, "lreal": False, "magmom": [1.0, 2.0]},
        kpoints=(3, 3, 3),
        potcars=folder,
        setups={"Ni": "Ni_pv"},
        command=FAKE,
        directory=tmp_path / "run",
    )
    result = vasp.compute(system, ("energy", "forces", "virial"))
    expected = EAM.zhou(["Cu", "Ni"]).compute(system, ("energy", "forces", "virial"))
    # as printed in an OUTCAR, the atoms back in their order
    assert result.energy == pytest.approx(expected.energy, abs=1e-8)
    np.testing.assert_allclose(result.forces, expected.forces, atol=1e-6)
    np.testing.assert_allclose(result.virial, expected.virial, atol=1e-5)
    run = tmp_path / "run"
    incar = (run / "INCAR").read_text().splitlines()
    assert incar == [
        "ENCUT = 400",
        "EDIFF = 1e-06",
        "LREAL = .FALSE.",
        "MAGMOM = 1.0 2.0",
        "NSW = 0",
        "IBRION = -1",
        "ISIF = 2",
    ]
    assert (run / "KPOINTS").read_text() == "mdapy\n0\nGamma\n3 3 3\n0 0 0\n"
    # POTCARs in the order of the POSCAR, which holds each element together
    written = mdapy.read(run / "POSCAR", format="poscar")
    first = written.table["element"].cast(str).to_list()[0]
    order = ["Cu", "Ni_pv"] if first == "Cu" else ["Ni_pv", "Cu"]
    assert (run / "POTCAR").read_text() == "".join(
        f"  PAW_PBE {s} 01Jan2000\n" for s in order
    )
    elements = written.table["element"].cast(str).to_list()
    assert elements == sorted(elements, key=elements.index)  # grouped


def test_settings_from_the_environment(tmp_path, monkeypatch):
    root = tmp_path / "pp"
    potcars(root / "potpaw_PBE", "Cu")
    monkeypatch.setenv("VASP_PP_PATH", str(root))
    monkeypatch.setenv("MDAPY_VASP_COMMAND", FAKE)
    monkeypatch.chdir(tmp_path)
    vasp = VASP()
    assert vasp.potcars == root / "potpaw_PBE"
    assert vasp.command == FAKE and vasp.directory == tmp_path / "vasp"
    assert vasp.cutoff == 0.0 and vasp.elements is None
    assert repr(vasp) == f"VASP(command={FAKE!r}, directory={str(tmp_path / 'vasp')!r})"
    copper = build_crystal("fcc", 3.6, elements="Cu", repeat=(2, 2, 2))
    result = vasp.compute(copper)
    assert result.energy == pytest.approx(
        EAM.zhou(["Cu"]).compute(copper).energy, abs=1e-7
    )
    assert not (tmp_path / "vasp" / "KPOINTS").exists()  # KSPACING in charge
    # the folder of the variable itself, and the commands of ASE and VASP
    monkeypatch.setenv("VASP_PP_PATH", str(root / "potpaw_PBE"))
    monkeypatch.delenv("MDAPY_VASP_COMMAND")
    monkeypatch.setenv("ASE_VASP_COMMAND", "mpirun vasp_std")
    assert VASP().potcars == root / "potpaw_PBE"
    assert VASP().command == "mpirun vasp_std"
    monkeypatch.delenv("ASE_VASP_COMMAND")
    monkeypatch.delenv("VASP_PP_PATH")
    assert VASP().command == "vasp_std" and VASP().potcars is None


def test_restarts_keep_or_drop_the_wave_functions(tmp_path):
    folder = potcars(tmp_path / "pp", "Cu")
    copper = build_crystal("fcc", 3.6, elements="Cu", repeat=(2, 2, 2))
    for restart in (True, False):
        run = tmp_path / f"run{restart}"
        run.mkdir()
        for name in ("WAVECAR", "CHGCAR"):
            (run / name).write_text("old")
        vasp = VASP(potcars=folder, command=FAKE, directory=run, restart=restart)
        vasp.compute(copper)
        assert (run / "WAVECAR").exists() is restart
        assert (run / "CHGCAR").exists() is restart


def test_a_minimization_with_vasp(tmp_path):
    # copper with one atom off its site
    system = build_crystal("fcc", 3.615, elements="Cu", repeat=(2, 2, 2))
    moved = system["positions"].copy()
    moved[0] += [0.15, -0.1, 0.05]
    system = system.with_columns(positions=moved)
    vasp = VASP(
        potcars=potcars(tmp_path / "pp", "Cu"), command=FAKE, directory=tmp_path
    )
    # the same first ten steps as with the potential itself
    relaxed = minimize(system, vasp, max_force=0.01, max_steps=10)
    reference = minimize(system, EAM.zhou(["Cu"]), max_force=0.01, max_steps=10)
    assert relaxed.steps == reference.steps == 10
    np.testing.assert_allclose(
        relaxed.system["positions"], reference.system["positions"], atol=1e-5
    )
    assert relaxed.energy == pytest.approx(reference.energy, abs=1e-7)


def test_failures(tmp_path):
    folder = potcars(tmp_path / "pp", "Cu")
    copper = build_crystal("fcc", 3.6, elements="Cu", repeat=(2, 2, 2))
    run = tmp_path / "run"
    for tag, message in (
        ("FAKE_FAIL", "exit status 3: VERY BAD NEWS"),
        ("FAKE_UNCONVERGED", "no result"),
        ("FAKE_SILENT", "no result"),
    ):
        vasp = VASP({tag: 1}, potcars=folder, command=FAKE, directory=run)
        with pytest.raises(CalculationFailed, match=message):
            vasp.compute(copper)
    with pytest.raises(InvalidParameter, match="no POTCAR for Ni"):
        VASP(potcars=folder, command=FAKE, directory=run).compute(alloy())
    with pytest.raises(InvalidParameter, match="no POTCARs"):
        VASP(potcars=None, command=FAKE, directory=run).compute(copper)
    with pytest.raises(MissingProperty, match="element column"):
        VASP(potcars=folder, command=FAKE, directory=run).compute(
            mdapy.System(copper["positions"], box=copper.box)
        )
    with pytest.raises(InvalidParameter, match="atomic_energies"):
        VASP(potcars=folder, command=FAKE, directory=run).compute(
            copper, ("atomic_energies",)
        )


def test_invalid_parameters():
    with pytest.raises(InvalidParameter, match="ISIF, NSW"):
        VASP({"nsw": 10, "ISIF": 3})
    with pytest.raises(InvalidParameter, match="three integers"):
        VASP(kpoints=(3, 3))
    with pytest.raises(InvalidParameter, match="three integers"):
        VASP(kpoints=(3.0, 3.0, 3.0))
    with pytest.raises(InvalidParameter, match="at least 1"):
        VASP(kpoints=(3, 0, 3))
