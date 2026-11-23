# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The calculator DFT: its results in eV and Å are those of the
self-consistent field in hartree and bohr; rotations, reordering and lattice
translations of the atoms change nothing but the directions of the forces; the
mesh of kspacing as VASP's; spins from the first moments; its errors."""

import os

import numpy as np
import pytest

import mdapy
from mdapy.calculators import DFT
from mdapy.calculators._dft import scf
from mdapy.calculators._dft.calculator import BOHR, HARTREE
from mdapy.calculators._dft.upf import read_upf
from mdapy.errors import CalculationFailed, InvalidParameter, MissingProperty

from dft import espresso

_dft = pytest.importorskip("mdapy_dft_cuda12._dft")
if _dft.device_count() == 0:
    pytest.skip("no GPU", allow_module_level=True)

FRACTIONS = np.array([[0.01, -0.02, 0.0], [0.27, 0.25, 0.23]])


def silicon_cell():
    h = 5.431 / 2
    return np.array([[0, h, h], [h, 0, h], [h, h, 0]])


def silicon(cell=None, fractions=FRACTIONS):
    cell = silicon_cell() if cell is None else cell
    return mdapy.System(
        fractions @ cell, mdapy.Box(cell), elements=["Si"] * len(fractions)
    )


@pytest.fixture(scope="module")
def folder():
    espresso.pseudo("Si_std.upf")
    return os.environ["MDAPY_PSEUDO_DIR"]


def calculator(folder, **options):
    settings = dict(encut=12 * HARTREE, kpoints=(2, 2, 2), smearing="fixed", functional="pbe_pw",
                    tolerance=1e-10, pseudopotentials=folder)  # fmt: skip
    settings.update(options)
    return DFT(**settings)


def test_the_units(folder):
    result = calculator(folder, gpu=0).compute(
        silicon(), ("energy", "forces", "virial")
    )
    pp = read_upf(espresso.pseudo("Si_std.upf"))
    r = scf.run(silicon_cell() / BOHR, FRACTIONS, [0, 0], [pp], 12.0, kgrid=(2, 2, 2),
                functional="pbe_pw", energy_tolerance=1e-10 / HARTREE,
                density_tolerance=1e-10 / HARTREE)  # fmt: skip
    assert result.energy == pytest.approx(r["energy"] * HARTREE, abs=1e-8)
    np.testing.assert_allclose(result.forces, r["forces"] * HARTREE / BOHR, atol=1e-7)
    np.testing.assert_allclose(
        result.virial, -r["strain_derivative"] * HARTREE, atol=1e-6
    )
    # the stress of MDAPY, positive under tension, against QE's, positive under pressure
    assert result.stress.trace() < 0  # this cutoff leaves silicon compressed


def test_symmetries(folder):
    dft = calculator(folder)
    base = dft.compute(silicon())
    rng = np.random.default_rng(3)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotated = dft.compute(silicon(silicon_cell() @ q.T))
    assert rotated.energy == pytest.approx(base.energy, abs=1e-8)
    np.testing.assert_allclose(rotated.forces, base.forces @ q.T, atol=1e-6)
    swapped = dft.compute(silicon(fractions=FRACTIONS[::-1] + [1.0, -2.0, 3.0]))
    assert swapped.energy == pytest.approx(base.energy, abs=1e-8)
    np.testing.assert_allclose(swapped.forces, base.forces[::-1], atol=1e-6)
    # a translation that is not of the FFT grid moves the atoms against it:
    # the energy changes only by the ripple of the grid
    moved = dft.compute(silicon(fractions=FRACTIONS + 0.0123))
    assert moved.energy == pytest.approx(base.energy, abs=1e-4)


def test_kspacing_as_vasp(folder):
    dft = calculator(folder, kpoints=None, kspacing=0.5, tolerance=1e-6)
    dft.compute(silicon())
    b = 2 * np.pi * np.linalg.norm(np.linalg.inv(silicon_cell()).T, axis=1)
    assert dft.last["kpoints"] == tuple(int(np.ceil(x / 0.5)) for x in b)
    provenance = dft.last["provenance"]
    assert (
        provenance["pseudopotentials"]["Si"]
        == read_upf(espresso.pseudo("Si_std.upf")).digest
    )
    assert (
        provenance["kpoints"] == dft.last["kpoints"]
        and provenance["functional"] == "pbe_pw"
    )
    assert provenance["encut"] == 12 * HARTREE and provenance["spins"] == 1


def test_spin(folder):
    espresso.pseudo("Fe_std.upf")
    a = 2.87
    box = mdapy.Box(np.diag([a, a, a]))
    iron = mdapy.System(
        np.array([[0, 0, 0], [0.5, 0.5, 0.5]]) * a, box, elements=["Fe", "Fe"]
    )
    dft = DFT(
        16 * HARTREE, pseudopotentials=folder, kpoints=(2, 2, 2), magmoms={"Fe": 2.5}
    )
    result = dft.compute(iron)
    assert dft.last["magnetization"] > 2.0 and dft.last["absolute_magnetization"] > 2.0
    np.testing.assert_allclose(result.forces, 0.0, atol=1e-4)
    plain = calculator(folder)
    plain.compute(silicon())
    assert "magnetization" not in plain.last and plain.last["iterations"] > 1


def test_the_mixing_with_and_without_spin(folder, monkeypatch):
    """The density mixed by 0.7 without spin, as Quantum ESPRESSO, by 0.4 with
    it, the magnetization by 1.0; a mixing given is kept."""
    seen = []

    class Seen(Exception):
        pass

    def spy(*args, **options):
        seen.append((options["beta"], options["magnetic_beta"]))
        raise Seen

    monkeypatch.setattr(scf, "run", spy)
    for options in (
        {},
        {"magmoms": {"Si": 1.0}},
        {"magmoms": [1.0, 0.0], "mixing": 0.2},
    ):
        with pytest.raises(Seen):
            calculator(folder, **options).compute(silicon())
    assert seen == [(0.7, 1.0), (0.4, 1.0), (0.2, 1.0)]
    with pytest.raises(InvalidParameter, match="0 < mixing <= 1"):
        calculator(folder, mixing=1.5)


def test_errors(folder, tmp_path):
    with pytest.raises(InvalidParameter, match="encut must be positive"):
        DFT(0.0, pseudopotentials=folder)
    with pytest.raises(InvalidParameter, match="smearing must be one of"):
        DFT(300.0, pseudopotentials=folder, smearing="cold")
    with pytest.raises(InvalidParameter, match="kpoints must be three integers"):
        DFT(300.0, pseudopotentials=folder, kpoints=(2, 2))
    with pytest.raises(InvalidParameter, match="is not a folder"):
        DFT(300.0, pseudopotentials=tmp_path / "nowhere")
    with pytest.raises(InvalidParameter, match="periodic along a, b and c"):
        cell = silicon_cell()
        system = mdapy.System(FRACTIONS @ cell, mdapy.Box(cell, pbc=(True, True, False)),
                              elements=["Si", "Si"])  # fmt: skip
        calculator(folder).compute(system)
    with pytest.raises(InvalidParameter, match="one pseudopotential of Xe"):
        xenon = mdapy.System(FRACTIONS @ silicon_cell(), mdapy.Box(silicon_cell()),
                             elements=["Xe", "Xe"])  # fmt: skip
        calculator(folder).compute(xenon)
    with pytest.raises(InvalidParameter, match="magmoms holds 1 moments for 2 atoms"):
        calculator(folder, magmoms=[1.0]).compute(silicon())
    with pytest.raises(ValueError, match="no GPU 99"):
        calculator(folder, gpu=99).compute(silicon())
    with pytest.raises(InvalidParameter, match="gpu must be at least 0"):
        calculator(folder, gpu=-1)
    with pytest.raises(CalculationFailed, match="did not converge in 2 steps"):
        calculator(folder, max_iterations=2).compute(silicon())
    with pytest.raises(MissingProperty, match="element column"):
        calculator(folder).compute(
            mdapy.System(FRACTIONS @ silicon_cell(), mdapy.Box(silicon_cell()))
        )


def test_restart(folder):
    # the forces of a run are good to about the root of its tolerance, which
    # bounds the density residual by its Hartree energy
    dft = calculator(folder, restart=True, tolerance=1e-10)
    cold = calculator(folder, tolerance=1e-10)
    first = dft.compute(silicon())
    assert not dft.last["restarted"]
    moved = silicon(fractions=FRACTIONS + [[0.003, 0.0, -0.002], [0.0, 0.004, 0.001]])
    warm = dft.compute(moved)
    assert dft.last["restarted"]
    steps = dft.last["iterations"]
    reference = cold.compute(moved)
    # the density of the last run takes no more steps than a cold start from
    # the free atoms and their orbitals
    assert steps <= cold.last["iterations"]
    assert warm.energy == pytest.approx(reference.energy, abs=1e-7)
    np.testing.assert_allclose(warm.forces, reference.forces, atol=1e-5)
    assert first.energy != warm.energy
    # another number of atoms starts cold
    bigger = mdapy.System(np.vstack([FRACTIONS, FRACTIONS + 0.5]) @ silicon_cell(),
                          mdapy.Box(2 * silicon_cell()), elements=["Si"] * 4)  # fmt: skip
    dft.compute(bigger)
    assert not dft.last["restarted"]


def test_molecular_dynamics(folder):
    """MDAPY's dynamics with the DFT as its calculator, the steps on the
    processor, each computation from the bands and the density of the last:
    the total energy of NVE holds, and the potential energy of the last step
    is that of a cold start."""
    from mdapy.simulation import MolecularDynamics

    dft = calculator(folder, restart=True)
    with MolecularDynamics(silicon(), dft, timestep=0.002) as md:
        md.velocities(600.0, seed=1)
        md.run(10, ensemble="nve", thermo=1)
        thermo, last = md.thermo, md.system
    assert dft.last["restarted"]
    assert np.ptp(thermo["total_energy"].to_numpy()) < 2e-3
    # without the virial when the dynamics needs no pressure
    from mdapy.calculators._base import WITHOUT_VIRIAL

    assert (
        dft.compute(last, ("energy", "forces", "virial", WITHOUT_VIRIAL)).virial is None
    )
    cold = calculator(folder).compute(last)
    assert thermo["potential_energy"][-1] == pytest.approx(cold.energy, abs=1e-6)


def test_ultrasoft(folder):
    path = espresso.pseudo("pslibrary/Si.pbe-n-rrkjus_psl.1.0.0.UPF")
    dft = DFT(12 * HARTREE, ecutrho=96 * HARTREE, pseudopotentials={"Si": path}, kpoints=(2, 2, 2),
              smearing="fixed", functional="pbe_pw", tolerance=1e-10)  # fmt: skip
    result = dft.compute(silicon())
    r = scf.run(silicon_cell() / BOHR, FRACTIONS, [0, 0], [read_upf(path)], 12.0, kgrid=(2, 2, 2),
                density_cutoff=96.0, functional="pbe_pw", energy_tolerance=1e-10 / HARTREE,
                density_tolerance=1e-10 / HARTREE)  # fmt: skip
    assert result.energy == pytest.approx(r["energy"] * HARTREE, abs=1e-8)
    assert dft.last["provenance"]["ecutrho"] == 96 * HARTREE
    with_virial = dft.compute(silicon(), ("energy", "forces", "virial"))
    np.testing.assert_allclose(
        with_virial.virial, -r["strain_derivative"] * HARTREE, atol=1e-6
    )
    with pytest.raises(InvalidParameter, match="ecutrho must be at least 4 encut"):
        DFT(300.0, ecutrho=600.0, pseudopotentials=folder)


def test_paw_and_its_restart(folder):
    """PAW silicon through the calculator, as the self-consistent field gives
    it; a restart from the last occupations of the projectors too."""
    path = espresso.pseudo("pslibrary/Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    settings = dict(pseudopotentials={"Si": path}, kpoints=(2, 2, 2), smearing="fixed",
                    functional="pbe_pw", tolerance=1e-10)  # fmt: skip
    dft = DFT(12 * HARTREE, restart=True, **settings)
    result = dft.compute(silicon())
    r = scf.run(silicon_cell() / BOHR, FRACTIONS, [0, 0], [read_upf(path)], 12.0, kgrid=(2, 2, 2),
                functional="pbe_pw", energy_tolerance=1e-10 / HARTREE,
                density_tolerance=1e-10 / HARTREE)  # fmt: skip
    assert result.energy == pytest.approx(r["energy"] * HARTREE, abs=1e-7)
    assert dft.last["provenance"]["ecutrho"] is None
    moved = silicon(fractions=FRACTIONS + [[0.003, 0.0, -0.002], [0.0, 0.004, 0.001]])
    warm = dft.compute(moved)
    assert dft.last["restarted"]
    steps = dft.last["iterations"]
    fresh = DFT(12 * HARTREE, **settings)
    cold = fresh.compute(moved)
    assert steps <= fresh.last["iterations"]
    assert warm.energy == pytest.approx(cold.energy, abs=1e-6)
    np.testing.assert_allclose(warm.forces, cold.forces, atol=1e-5)
