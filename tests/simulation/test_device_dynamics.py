"""The dynamics of a potential on the GPU, its steps called on the atoms of a
stand-in for the GPU module that computes on the processor: the same runs
as the processor's own, and the errors. tests/cuda runs the GPU itself."""

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.calculators import EAM, Calculator, CalculatorResult, LennardJones, _gpu
from mdapy.errors import CalculationFailed, InvalidParameter
from mdapy.simulation import MolecularDynamics

from calculators.gpu_stand_in import stand_in


@pytest.fixture
def gpu(monkeypatch):
    monkeypatch.setattr(_gpu, "cuda", stand_in)


def crystal(repeat=4, tilt=0.0):
    system = build_crystal("fcc", 3.6, elements="Cu", repeat=(repeat,) * 3)
    matrix = system.box.matrix.copy()
    matrix[1, 0] += tilt * matrix[0, 0]
    system = system.with_box(mdapy.Box(matrix, pbc=True), scale_positions=True)
    kinds = np.random.default_rng(0).choice(["Cu", "Ni"], len(system))
    return system.with_columns(element=list(kinds))


def run(system, potential, steps, **options):
    with MolecularDynamics(system, potential, timestep=0.002) as md:
        md.velocities(600, seed=1)
        md.run(steps, thermo=steps // 2, **options)
        return md.system, md.thermo


@pytest.mark.parametrize(
    "options",
    [
        {"ensemble": "nvt", "temperature": 300},
        {"ensemble": "npt", "temperature": 300, "pressure": 1.0, "barostat": "tri"},
    ],
)
def test_the_runs_of_the_processor(gpu, options):
    system = crystal(tilt=0.2)
    a, ta = run(system, EAM.zhou(["Cu", "Ni"], device="cuda"), 40, **options)
    b, tb = run(system, EAM.zhou(["Cu", "Ni"]), 40, **options)
    np.testing.assert_allclose(a.box.matrix, b.box.matrix, atol=1e-10)
    np.testing.assert_allclose(a["positions"], b["positions"], atol=1e-9)
    np.testing.assert_allclose(a["velocities"], b["velocities"], atol=1e-8)
    for column in ("temperature", "potential_energy", "pressure"):
        np.testing.assert_allclose(ta[column], tb[column], rtol=1e-9, atol=1e-9)


def held(system):
    """The crystal turned, so that the frame of the dynamics is not the
    system's, with atoms held along all, one or two of its axes."""
    c, s = np.cos(0.3), np.sin(0.3)
    q = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ np.array(
        [[1, 0, 0], [0, c, -s], [0, s, c]]
    )
    turned = system.with_box(
        mdapy.Box(system.box.matrix @ q.T, pbc=True), scale_positions=True
    )
    movable = np.ones((len(system), 3), dtype=bool)
    movable[:16] = False
    movable[16:48, 0] = False
    movable[48:64, 1:] = False
    return turned.with_columns(movable=movable)


@pytest.mark.parametrize(
    "options",
    [
        {"ensemble": "nvt", "temperature": 300},
        {"ensemble": "npt", "temperature": 300, "pressure": 1.0, "barostat": "tri"},
    ],
)
def test_held_atoms_as_on_the_processor(gpu, options):
    system = held(crystal(tilt=0.2))
    a, ta = run(system, EAM.zhou(["Cu", "Ni"], device="cuda"), 40, **options)
    b, tb = run(system, EAM.zhou(["Cu", "Ni"]), 40, **options)
    np.testing.assert_allclose(a.box.matrix, b.box.matrix, atol=1e-10)
    np.testing.assert_allclose(a["positions"], b["positions"], atol=1e-9)
    np.testing.assert_allclose(a["velocities"], b["velocities"], atol=1e-8)
    np.testing.assert_allclose(ta["temperature"], tb["temperature"], rtol=1e-9)
    assert np.abs(a["velocities"][~system["movable"]]).max() < 1e-12


def test_held_atoms_in_langevin_dynamics(gpu):
    system = held(crystal())
    options = {"ensemble": "nvt", "temperature": 300, "thermostat": "langevin"}
    a, _ = run(system, EAM.zhou(["Cu", "Ni"], device="cuda"), 20, seed=2, **options)
    assert np.abs(a["velocities"][~system["movable"]]).max() < 1e-12
    assert np.abs(a["velocities"][system["movable"]]).max() > 1.0


def test_langevin_dynamics_repeats_with_its_seed(gpu):
    lj = LennardJones({"Cu": (0.4, 2.3), "Ni": (0.5, 2.2)}, 5.0, device="cuda")
    options = {"ensemble": "nvt", "temperature": 300, "thermostat": "langevin"}
    first, _ = run(crystal(), lj, 10, seed=3, **options)
    second, _ = run(crystal(), lj, 10, seed=3, **options)
    np.testing.assert_array_equal(first["positions"], second["positions"])
    other, _ = run(crystal(), lj, 10, **options)  # a seed of its own
    assert not np.array_equal(other["positions"], first["positions"])


class _Elsewhere(Calculator):
    """A calculator of someone else that says it is on the GPU."""

    cutoff = 3.0
    device = "cuda"

    def compute(self, system, properties=("energy", "forces"), *, neighbors=None):
        n = len(system)
        return CalculatorResult(
            energy=0.0, forces=np.zeros((n, 3)), virial=np.zeros((3, 3))
        )


def test_errors(gpu, monkeypatch):
    potential = EAM.zhou(["Cu", "Ni"], device="cuda")
    with pytest.raises(InvalidParameter, match="thicker than twice the cutoff"):
        run(crystal(repeat=3), potential, 2)
    with pytest.raises(InvalidParameter, match="has no molecular dynamics on the GPU"):
        MolecularDynamics(crystal(), _Elsewhere())
    # forces that are not finite move the atoms to positions that are not
    lj = LennardJones({"Cu": (0.4, 2.3), "Ni": (0.5, 2.2)}, 5.0, device="cuda")
    original = lj._on_device.potential.compute

    def broken(forces, energies):
        def compute(atoms, *args):
            original(atoms, *args)
            e, f, w = atoms.result
            atoms.result = (e * energies, f * forces, w)

        return compute

    monkeypatch.setattr(lj._on_device.potential, "compute", broken(np.nan, 1.0))
    with pytest.raises(CalculationFailed, match="not finite"):
        run(crystal(), lj, 5)
    monkeypatch.setattr(lj._on_device.potential, "compute", broken(1.0, np.inf))
    with pytest.raises(CalculationFailed, match="not finite"):
        run(crystal(), lj, 5)
