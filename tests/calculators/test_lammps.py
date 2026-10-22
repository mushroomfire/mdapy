# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""The LAMMPS calculator against mdapy's EAM. Skipped where LAMMPS is not
installed, except the error without it."""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

import mdapy
from mdapy.calculators import EAM, LAMMPS, minimize
from mdapy.calculators import _lammps as module
from mdapy.calculators.testing import check_calculator
from mdapy.errors import InvalidParameter, MissingProperty

from .eam_files import alloy, write_setfl

ALL = ("energy", "atomic_energies", "forces", "virial", "atomic_virials")
BOXES = {
    "triclinic": [[7.3, 0, 0], [1.0, 7.0, 0], [0.5, -0.8, 7.5]],
    "rotated": [[3.1, 4.2, 0], [-4.0, 3.0, 0.5], [0.3, -0.2, 7.0]],
    "left-handed": [[7.0, 0, 0], [0, 0, 7.2], [0, 7.1, 0]],
}


def test_without_lammps(monkeypatch):
    monkeypatch.setitem(sys.modules, "lammps", None)
    with pytest.raises(ImportError, match="LAMMPS Python module"):
        LAMMPS("pair_style zero 3.0\npair_coeff * *", ["Cu"])


def test_kokkos_is_shut_down_once_at_exit(monkeypatch):
    class Fake:
        def __init__(self, cmdargs):
            self.cmdargs, self.finalized = cmdargs, False

        def finalize(self):
            self.finalized = True

    monkeypatch.setitem(sys.modules, "lammps", type(sys)("lammps"))
    sys.modules["lammps"].lammps = Fake
    registered = []
    monkeypatch.setattr(module.atexit, "register", registered.append)
    monkeypatch.setattr(module, "_kokkos", [])
    plain = module._lammps([])
    first = module._lammps(["-k", "on", "t", "2", "-sf", "kk"])
    second = module._lammps(["-kokkos", "on"])
    assert "-nocite" in plain.cmdargs and registered == [module._finalize_kokkos]
    module._finalize_kokkos()  # at exit, by the last instance with Kokkos
    assert second.finalized and not first.finalized and not plain.finalized


lammps = pytest.importorskip("lammps")


@pytest.fixture
def potentials(tmp_path):
    path = write_setfl(tmp_path / "CuNi.eam.alloy")
    calculator = LAMMPS(
        f"pair_style eam/alloy\npair_coeff * * {path} Cu Ni", ["Cu", "Ni"]
    )
    yield EAM(path), calculator
    calculator.close()


def random_system(box_name, pbc, seed, n=20):
    """Atoms at least 1.6 Å apart, counting periodic images, some outside the
    box."""
    rng = np.random.default_rng(seed)
    box = mdapy.Box(BOXES[box_name], origin=[1.0, -2.0, 0.5], pbc=pbc)
    reach = [range(-1, 2) if p else range(1) for p in box.pbc]
    images = np.array(np.meshgrid(*reach, indexing="ij")).reshape(3, -1).T @ box.matrix
    kept: list = []
    while len(kept) < n:
        x = box.to_cartesian(rng.random(3) * 1.2 - 0.1)
        if all(np.linalg.norm(y + images - x, axis=1).min() >= 1.6 for y in kept):
            kept.append(x)
    return mdapy.System(np.array(kept), box, elements=rng.choice(["Cu", "Ni"], n))


def agree(a, b, atol=1e-9):
    assert a.energy == pytest.approx(b.energy, rel=1e-12, abs=atol)
    for name in ALL[1:]:
        np.testing.assert_allclose(
            getattr(a, name), getattr(b, name), rtol=1e-9, atol=atol
        )


@pytest.mark.parametrize("box_name", BOXES)
@pytest.mark.parametrize(
    "pbc", [True, (True, False, True), (False, True, False), False]
)
def test_agrees_with_eam(potentials, box_name, pbc):
    eam, calculator = potentials
    system = random_system(box_name, pbc, seed=len(box_name))
    agree(calculator.compute(system, ALL), eam.compute(system, ALL))


def test_passes_the_calculator_checks(potentials):
    _, calculator = potentials
    assert calculator.cutoff == pytest.approx(5.0)
    check_calculator(calculator, alloy(seed=5, jitter=0.1))


def test_one_instance_for_the_same_atoms(potentials, monkeypatch):
    eam, calculator = potentials
    started = []
    original = module._lammps
    monkeypatch.setattr(
        module, "_lammps", lambda args: started.append(1) or original(args)
    )
    system = random_system("triclinic", (True, True, False), seed=3)
    rng = np.random.default_rng(4)
    for _ in range(3):
        moved = system.with_positions(
            system.positions + rng.normal(scale=0.05, size=(len(system), 3))
        )
        agree(calculator.compute(moved, ALL), eam.compute(moved, ALL))
    # far along the non-periodic direction, out of the LAMMPS box, and in
    # another box
    away = system.with_positions(system.positions + [0.0, 0.0, 30.0])
    agree(calculator.compute(away, ALL), eam.compute(away, ALL))
    strained = system.with_box(
        mdapy.Box(
            system.box.matrix * 1.05, origin=system.box.origin, pbc=system.box.pbc
        ),
        scale_positions=True,
    )
    agree(calculator.compute(strained, ALL), eam.compute(strained, ALL))
    agree(calculator.compute(system, ALL), eam.compute(system, ALL))
    assert len(started) == 1
    # other atoms, other periodic directions: a new one
    agree(calculator.compute(alloy(seed=6), ALL), eam.compute(alloy(seed=6), ALL))
    periodic = system.with_box(mdapy.Box(system.box.matrix, pbc=True))
    agree(calculator.compute(periodic, ALL), eam.compute(periodic, ALL))
    assert len(started) == 3
    calculator.close()
    agree(calculator.compute(alloy(seed=6), ALL), eam.compute(alloy(seed=6), ALL))
    assert len(started) == 4


def test_empty_systems(potentials):
    _, calculator = potentials
    empty = mdapy.System(np.zeros((0, 3)), mdapy.Box(8.0, pbc=(True, True, False)))
    result = calculator.compute(empty.with_columns(element=[]), ALL)
    assert result.energy == 0.0 and result.forces.shape == (0, 3)
    assert result.atomic_virials.shape == (0, 3, 3)


def test_minimization(potentials, monkeypatch):
    eam, calculator = potentials
    system = alloy(seed=8, repeat=2, jitter=0.15)
    a = minimize(system, calculator, max_force=1e-6, max_steps=2000)
    b = minimize(system, eam, max_force=1e-6, max_steps=2000)
    assert a.converged and b.converged and a.steps == b.steps
    np.testing.assert_allclose(a.system.positions, b.system.positions, atol=1e-8)
    # with the box, in one LAMMPS instance
    calculator.close()
    started = []
    original = module._lammps
    monkeypatch.setattr(
        module, "_lammps", lambda args: started.append(1) or original(args)
    )
    a = minimize(system, calculator, cell="full", max_force=1e-6, max_stress=1e-6)
    b = minimize(system, eam, cell="full", max_force=1e-6, max_stress=1e-6)
    assert a.converged and b.converged and a.steps == b.steps
    assert len(started) == 1
    np.testing.assert_allclose(a.system.box.matrix, b.system.box.matrix, atol=1e-8)
    np.testing.assert_allclose(a.stress, b.stress, atol=1e-6)


def test_zhou_potentials_read_by_lammps(tmp_path):
    potential = EAM.zhou(["Cu", "Ni", "Cr"])
    path = tmp_path / "CuNiCr.eam.alloy"
    potential.write(path)
    calculator = LAMMPS(
        f"pair_style eam/alloy\npair_coeff * * {path} Cu Ni Cr", ["Cu", "Ni", "Cr"]
    )
    rng = np.random.default_rng(9)
    system = alloy(seed=9, repeat=3).with_columns(
        element=rng.choice(["Cu", "Ni", "Cr"], 108)
    )
    agree(calculator.compute(system, ALL), potential.compute(system, ALL))
    # the average atom, named A in the file
    average = potential.average({"Cu": 0.4, "Ni": 0.4, "Cr": 0.2})
    average.write(tmp_path / "average.eam.alloy")
    calculator = LAMMPS(
        f"pair_style eam/alloy\npair_coeff * * {tmp_path / 'average.eam.alloy'} A",
        ["Cu"],
    )
    copper = system.with_columns(element=["Cu"] * 108)
    agree(calculator.compute(copper, ALL), average.compute(system, ALL))
    calculator.close()


def test_cutoff_given_or_from_lammps(tmp_path, monkeypatch):
    path = write_setfl(tmp_path / "CuNi.eam.alloy")
    pair = f"pair_style eam/alloy\npair_coeff * * {path} Cu Ni"
    assert LAMMPS(pair, ["Cu", "Ni"], cutoff=6.5).cutoff == 6.5
    with pytest.raises(InvalidParameter, match="cutoff must be positive"):
        LAMMPS(pair, ["Cu", "Ni"], cutoff=0.0)
    # LAMMPS before 30 Sep 2026 does not report it
    monkeypatch.setattr(lammps.lammps, "extract_global", lambda self, name: None)
    with pytest.raises(InvalidParameter, match="give cutoff="):
        LAMMPS(pair, ["Cu", "Ni"])
    assert LAMMPS(pair, ["Cu", "Ni"], cutoff=5.0).cutoff == 5.0


def test_errors(potentials):
    _, calculator = potentials
    with pytest.raises(InvalidParameter, match="LAMMPS: .*Unrecognized pair style"):
        LAMMPS("pair_style nonsense", ["Cu"])
    with pytest.raises(InvalidParameter, match="chemical symbols"):
        LAMMPS("pair_style zero 3.0\npair_coeff * *", ["Cu", "Xx"])
    with pytest.raises(
        InvalidParameter, match=r"no atom type has the elements \['Al'\]"
    ):
        calculator.compute(alloy(seed=1).with_columns(element=["Al"] * 32))
    with pytest.raises(MissingProperty, match="element column"):
        calculator.compute(mdapy.System(np.zeros((1, 3)), mdapy.Box(5.0)))
    assert "LAMMPS('pair_style eam/alloy" in repr(calculator)


def test_kokkos_shuts_down_at_exit(tmp_path):
    instance = lammps.lammps(cmdargs=["-screen", "none", "-log", "none", "-nocite"])
    kokkos = "KOKKOS" in instance.installed_packages
    instance.close()
    if not kokkos:
        pytest.skip("LAMMPS without Kokkos")
    path = write_setfl(tmp_path / "CuNi.eam.alloy")
    script = textwrap.dedent(
        f"""
        import mdapy
        from mdapy.calculators import LAMMPS
        from tests.calculators.eam_files import alloy
        calculator = LAMMPS(
            "pair_style eam/alloy/kk\\npair_coeff * * {path} Cu Ni",
            ["Cu", "Ni"],
            cmdargs=["-k", "on", "t", "1", "-sf", "kk"],
        )
        print(calculator.compute(alloy(seed=1)).energy)
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    assert float(done.stdout) < 0
