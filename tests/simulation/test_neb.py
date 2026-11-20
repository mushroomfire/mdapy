# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The nudged elastic band: the saddle point of an analytic surface, the
symmetry of a vacancy hop, the curvatures at its saddle, ASE's band on the
same potential, and its options and errors."""

import io
import time
from functools import cache

import numpy as np
import polars as pl
import pytest

from mdapy.build import build_crystal
from mdapy.calculators import EAM
from mdapy.calculators._base import Calculator, CalculatorResult
from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.errors import CalculationFailed, InvalidParameter, MissingProperty
from mdapy.simulation import NEBResult, minimize, neb
from mdapy.simulation._neb import _idpp

# The surface of Müller and Brown (1979), in eV / 100 over x and y in Å.
_A = np.array([-200.0, -100.0, -170.0, 15.0]) / 100
_a = np.array([-1.0, -1.0, -6.5, 0.7])
_b = np.array([0.0, 0.0, 11.0, 0.6])
_c = np.array([-10.0, -10.0, -6.5, 0.7])
_X = np.array([1.0, 0.0, -0.5, -1.0])
_Y = np.array([0.0, 0.5, 1.5, 1.0])


def muller_brown(x: float, y: float) -> tuple[float, np.ndarray]:
    """The energy and its gradient at (x, y)."""
    dx, dy = x - _X, y - _Y
    terms = _A * np.exp(_a * dx**2 + _b * dx * dy + _c * dy**2)
    gx = (terms * (2 * _a * dx + _b * dy)).sum()
    gy = (terms * (_b * dx + 2 * _c * dy)).sum()
    return float(terms.sum()), np.array([gx, gy])


class MullerBrown(Calculator):
    """One atom on the surface of Müller and Brown, free along z."""

    cutoff = 0.0
    implemented = frozenset({"energy", "forces"})

    def compute(self, system, properties=("energy", "forces"), *, neighbors=None):
        x, y, _ = system["positions"][0]
        energy, gradient = muller_brown(x, y)
        forces = np.array([[-gradient[0], -gradient[1], 0.0]])
        return CalculatorResult(energy=energy, forces=forces)


class Failing(Calculator):
    """Gives a force that is not finite."""

    cutoff = 0.0
    implemented = frozenset({"energy", "forces"})

    def compute(self, system, properties=("energy", "forces"), *, neighbors=None):
        return CalculatorResult(energy=0.0, forces=np.full((len(system), 3), np.nan))


class Energies(Calculator):
    """Computes energies only."""

    cutoff = 0.0
    implemented = frozenset({"energy"})

    def compute(self, system, properties=("energy",), *, neighbors=None):
        return CalculatorResult(energy=0.0)


def atom(x: float, y: float) -> System:
    table = pl.DataFrame(
        {"element": ["Cu"], "positions": [[x, y, 0.0]]},
        schema={"element": pl.String, "positions": pl.Array(pl.Float64, 3)},
    )
    return System(table=table, box=Box(np.eye(3) * 20.0, pbc=False))


def minimum(x: float, y: float) -> System:
    return minimize(atom(x, y), MullerBrown(), max_force=1e-10, max_steps=10000).system


def newton(point: np.ndarray) -> np.ndarray:
    """The stationary point of the surface nearest a point, by Newton's
    method with the Hessian of finite differences."""
    for _ in range(50):
        g = muller_brown(*point)[1]
        h = np.empty((2, 2))
        for k in range(2):
            e = np.zeros(2)
            e[k] = 1e-6
            h[:, k] = (
                muller_brown(*(point + e))[1] - muller_brown(*(point - e))[1]
            ) / 2e-6
        point = point - np.linalg.solve(h, g)
    return point, h


def test_the_climbing_image_finds_the_saddle_point_of_an_analytic_surface():
    initial, final = minimum(0.62, 0.03), minimum(-0.05, 0.47)
    result = neb(
        initial, final, MullerBrown(), images=7, spring=0.5, max_force=1e-8,
        max_steps=20000,
    )  # fmt: skip
    assert isinstance(result, NEBResult) and result.converged
    saddle = result.images[result.highest]["positions"][0, :2]
    exact, hessian = newton(saddle)
    np.testing.assert_allclose(saddle, exact, atol=1e-7)
    np.testing.assert_allclose(exact, [0.212, 0.293], atol=1e-3)  # Müller and Brown
    curvatures = np.linalg.eigvalsh(hessian)
    assert curvatures[0] < 0 < curvatures[1]  # a saddle point of first order
    np.testing.assert_allclose(
        result.barrier, muller_brown(*exact)[0] - result.energies[0], atol=1e-12
    )
    assert result.highest not in (0, len(result.images) - 1)


def test_without_climbing_the_highest_image_is_below_the_saddle():
    initial, final = minimum(0.62, 0.03), minimum(-0.05, 0.47)
    climbing = neb(
        initial, final, MullerBrown(), images=4, max_force=1e-6, max_steps=20000
    )
    plain = neb(
        initial, final, MullerBrown(), images=4, climb=False, max_force=1e-6,
        max_steps=20000,
    )  # fmt: skip
    assert plain.converged and plain.barrier < climbing.barrier - 1e-4
    # the cubic between the images comes closer to the saddle
    _, energies = plain.profile(2000)
    fitted = energies.max() - plain.energies[0]
    assert abs(fitted - climbing.barrier) < abs(plain.barrier - climbing.barrier)


@cache
def vacancy_hop(repeat: int = 3) -> tuple[System, System]:
    """A vacancy in copper with Zhou's EAM, and its neighbor hopped into it."""
    eam = EAM.zhou(["Cu"])
    a = 3.615
    crystal = build_crystal("fcc", a, elements="Cu", repeat=(repeat,) * 3)
    x = crystal["positions"]
    hop = int(np.argmin(np.linalg.norm(x - [a / 2, a / 2, 0.0], axis=1)))
    keep = np.ones(len(crystal), bool)
    keep[0] = False
    moved = x.copy()
    moved[hop] = x[0]
    initial = minimize(crystal.filter(keep), eam, max_force=1e-6).system
    final = crystal.with_columns(positions=moved).filter(keep)
    return initial, minimize(final, eam, max_force=1e-6).system


def test_a_vacancy_hop_is_symmetric_and_its_saddle_is_the_middle():
    initial, final = vacancy_hop()
    eam = EAM.zhou(["Cu"])
    result = neb(initial, final, eam, images=5, max_force=1e-5, max_steps=5000)
    assert result.converged and result.steps > 0 and result.highest == 3
    energies = result.energies - result.energies[0]
    np.testing.assert_allclose(energies, energies[::-1], atol=1e-8)
    assert 0.5 < result.barrier < 0.6
    # the hopping atom halfway, with the atoms around it symmetric
    x = [image["positions"] for image in result.images]
    hop = int(np.argmax(np.abs(x[-1] - x[0]).sum(axis=1)))
    np.testing.assert_allclose(x[3][hop], 0.5 * (x[0][hop] + x[-1][hop]), atol=1e-4)
    np.testing.assert_allclose(result.distances[0], 0.0)
    assert np.all(np.diff(result.distances) > 0)
    np.testing.assert_allclose(
        result.distances, result.distances[-1] - result.distances[::-1], atol=1e-6
    )
    # each image keeps the forces and the energy of the potential
    for image, energy in zip(result.images, result.energies):
        computed = eam.compute(image)
        np.testing.assert_allclose(image["forces"], computed.forces, atol=1e-10)
        assert (
            image.info["energy"] == pytest.approx(energy, abs=1e-10) == computed.energy
        )


def test_the_saddle_of_a_vacancy_hop_has_one_negative_curvature():
    initial, final = vacancy_hop()
    eam = EAM.zhou(["Cu"])
    result = neb(initial, final, eam, images=3, max_force=1e-6, max_steps=5000)
    saddle = result.images[result.highest]
    x = saddle["positions"]
    h, n = 1e-4, x.size
    hessian = np.empty((n, n))
    for k in range(n):
        step = np.zeros(n)
        step[k] = h
        up = eam.compute(
            saddle.with_columns(positions=x + step.reshape(x.shape))
        ).forces
        down = eam.compute(
            saddle.with_columns(positions=x - step.reshape(x.shape))
        ).forces
        hessian[:, k] = -(up - down).ravel() / (2 * h)
    curvatures = np.linalg.eigvalsh(0.5 * (hessian + hessian.T))
    # one negative, three zero for the translations, the rest positive
    assert curvatures[0] < -0.1
    np.testing.assert_allclose(curvatures[1:4], 0.0, atol=1e-4)
    assert curvatures[4] > 1e-3


def test_ase_finds_the_same_band():
    ase_neb = pytest.importorskip("ase.mep").NEB
    from ase.optimize import FIRE

    from mdapy.interop import to_ase, to_ase_calculator

    initial, final = vacancy_hop()
    eam = EAM.zhou(["Cu"])
    ours = neb(
        initial, final, eam, images=3, spring=1.0, max_force=1e-5, max_steps=5000
    )
    band = [to_ase(initial)] + [to_ase(initial) for _ in range(3)] + [to_ase(final)]
    for image in band:
        image.calc = to_ase_calculator(eam)
    theirs = ase_neb(band, k=1.0, climb=True, method="improvedtangent")
    theirs.interpolate(mic=True)
    assert FIRE(theirs, logfile=None).run(fmax=1e-5, steps=20000)
    energies = np.array([image.get_potential_energy() for image in band])
    np.testing.assert_allclose(ours.energies, energies, atol=1e-5)
    assert ours.barrier == pytest.approx(energies.max() - energies[0], abs=1e-8)
    cell = band[0].cell.array
    for image, mine in zip(band, ours.images):
        d = (image.get_positions() - mine["positions"]) @ np.linalg.inv(cell)
        d = (d - np.round(d)) @ cell
        assert np.abs(d).max() < 1e-4


def test_the_band_backwards_is_the_band_reversed():
    initial, final = vacancy_hop()
    eam = EAM.zhou(["Cu"])
    # an odd number of images: with an even one the two in the middle are as
    # high, and the first of them climbs
    forth = neb(initial, final, eam, images=3, max_force=1e-6, max_steps=5000)
    back = neb(final, initial, eam, images=3, max_force=1e-6, max_steps=5000)
    np.testing.assert_allclose(back.energies, forth.energies[::-1], atol=1e-6)
    np.testing.assert_allclose(back.distances[-1], forth.distances[-1], atol=1e-5)


def test_one_calculator_for_each_image_gives_the_same_band():
    initial, final = vacancy_hop()
    one = neb(initial, final, EAM.zhou(["Cu"]), images=4, max_force=1e-4)
    each = neb(
        initial, final, [EAM.zhou(["Cu"]) for _ in range(4)], images=4, max_force=1e-4
    )
    assert each.steps == one.steps
    np.testing.assert_allclose(each.energies, one.energies, rtol=0, atol=1e-10)
    for a, b in zip(each.images, one.images):
        np.testing.assert_allclose(a["positions"], b["positions"], rtol=0, atol=1e-10)
    # images that share a calculator compute one after another
    shared = EAM.zhou(["Cu"])
    pairs = neb(
        initial,
        final,
        [shared, shared, EAM.zhou(["Cu"]), shared],
        images=4,
        max_force=1e-4,
    )
    np.testing.assert_allclose(pairs.energies, one.energies, rtol=0, atol=1e-10)


class Slow(Calculator):
    """A calculator that takes its time, as a program running elsewhere
    does, and notes when it computes."""

    cutoff = EAM.zhou(["Cu"]).cutoff
    implemented = frozenset({"energy", "forces"})

    def __init__(self, times):
        self.eam = EAM.zhou(["Cu"])
        self.times = times

    def compute(self, system, properties=("energy", "forces"), *, neighbors=None):
        start = time.perf_counter()
        time.sleep(0.02)
        result = self.eam.compute(system, properties, neighbors=neighbors)
        self.times.append((start, time.perf_counter()))
        return result


def test_calculators_of_their_own_compute_at_the_same_time():
    initial, final = vacancy_hop()
    times = [[] for _ in range(3)]
    neb(initial, final, [Slow(t) for t in times], images=3, max_steps=2)
    # the first and the last computed the two states first; then, in each
    # evaluation of the band, the three computed together
    times[0], times[-1] = times[0][1:], times[-1][1:]
    assert len({len(t) for t in times}) == 1 and len(times[0]) >= 3
    for k in range(len(times[0])):
        starts = [t[k][0] for t in times]
        ends = [t[k][1] for t in times]
        assert max(starts) < min(ends)
    shared = []
    one = Slow(shared)
    neb(initial, final, one, images=3, max_steps=2)
    # one calculator for all: one after another
    spans = sorted(shared)
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))


def test_a_band_starts_from_the_images_given():
    initial, final = vacancy_hop()
    eam = EAM.zhou(["Cu"])
    done = neb(initial, final, eam, images=3, max_force=1e-4)
    again = neb(initial, final, eam, images=done.images[1:-1], max_force=1e-4)
    assert again.converged and again.steps == 0
    np.testing.assert_allclose(again.energies, done.energies)
    # images given across the periodic boundary are taken back to the band
    box = initial.box.matrix
    shifted = [
        image.with_columns(positions=image["positions"] + box[0])
        for image in done.images[1:-1]
    ]
    np.testing.assert_allclose(
        neb(initial, final, eam, images=shifted, max_force=1e-4).energies, done.energies
    )


def test_the_final_state_across_the_periodic_boundary():
    initial, final = vacancy_hop()
    eam = EAM.zhou(["Cu"])
    x = final["positions"]
    wrapped = final.with_columns(positions=x + final.box.matrix[1] * (x[:, 1:2] < 1.0))
    a = neb(initial, final, eam, images=3, max_force=1e-4)
    b = neb(initial, wrapped, eam, images=3, max_force=1e-4)
    np.testing.assert_allclose(b.energies, a.energies, atol=1e-10)
    np.testing.assert_allclose(b.distances, a.distances, atol=1e-10)


@pytest.mark.parametrize("interpolation", ["linear", "idpp"])
def test_atoms_held_still(interpolation):
    initial, final = vacancy_hop()
    eam = EAM.zhou(["Cu"])
    x = initial["positions"]
    far = np.linalg.norm(x - x.mean(axis=0), axis=1) > 3.0
    movable = np.repeat(~far[:, None], 3, axis=1)
    held = initial.with_columns(movable=movable)
    result = neb(
        held, final, eam, images=3, interpolation=interpolation, max_force=1e-4
    )
    assert result.converged and far.sum() > 10
    start = neb(held, final, eam, images=3, max_steps=0)
    for before, after in zip(start.images, result.images):
        np.testing.assert_array_equal(after["positions"][far], before["positions"][far])


def test_idpp_keeps_the_distances_of_the_states():
    initial, final = vacancy_hop()
    eam = EAM.zhou(["Cu"])
    start = neb(initial, final, eam, images=3, max_steps=0)
    band = np.stack([image["positions"] for image in start.images])
    relaxed = _idpp(band, initial, 1.0, None)
    np.testing.assert_array_equal(relaxed[0], band[0])
    np.testing.assert_array_equal(relaxed[-1], band[-1])
    assert not np.allclose(relaxed, band)
    linear = neb(initial, final, eam, images=3, max_force=1e-5, max_steps=5000)
    idpp = neb(
        initial,
        final,
        eam,
        images=3,
        interpolation="idpp",
        max_force=1e-5,
        max_steps=5000,
    )
    assert idpp.barrier == pytest.approx(linear.barrier, abs=1e-6)


def test_idpp_of_a_dimer_that_turns():
    # a dimer turning by 90 degrees: the straight line squeezes it, the
    # pair potential keeps its length
    table = pl.DataFrame(
        {
            "element": ["Cu", "Cu"],
            "positions": [[10.0, 10.0, 10.0], [12.2, 10.0, 10.0]],
        },
        schema={"element": pl.String, "positions": pl.Array(pl.Float64, 3)},
    )
    first = System(table=table, box=Box(np.eye(3) * 20.0, pbc=False))
    last = first["positions"] + [[0.0, 0.0, 0.0], [-2.2, 2.2, 0.0]]
    band = np.stack([first["positions"], 0.5 * (first["positions"] + last), last])
    turned = _idpp(band, first, 1.0, None)
    length = np.linalg.norm(turned[1][1] - turned[1][0])
    assert np.linalg.norm(band[1][1] - band[1][0]) < 1.6 < 2.1 < length < 2.3


def test_a_log_a_callback_and_max_steps():
    initial, final = vacancy_hop()
    eam = EAM.zhou(["Cu"])
    stream = io.StringIO()
    result = neb(initial, final, eam, images=3, max_force=1e-4, log=stream, log_every=5)
    text = stream.getvalue()
    assert "nudged elastic band" in text and "converged after" in text
    rows = np.loadtxt(io.StringIO(text), ndmin=2)
    assert rows[-1, 0] == result.steps and rows[-1, 3] == result.highest
    assert rows[-1, 2] == pytest.approx(result.barrier, abs=1e-7)
    seen = []

    def stop(step, images, energies, max_force):
        seen.append((step, len(images), energies.shape, max_force))
        return step == 3

    stopped = neb(initial, final, eam, images=3, callback=stop, log=io.StringIO())
    assert stopped.steps == 3 and not stopped.converged
    assert seen[-1][:3] == (3, 5, (5,))
    none = neb(initial, final, eam, images=3, max_steps=0)
    assert none.steps == 0 and not none.converged and none.max_force > 0.1
    stream = io.StringIO()
    neb(initial, final, eam, images=3, max_steps=2, log=stream)
    assert "reached max_steps after 2 steps" in stream.getvalue()


def test_profile_passes_through_the_images():
    initial, final = vacancy_hop()
    result = neb(initial, final, EAM.zhou(["Cu"]), images=3, max_force=1e-4)
    distances, energies = result.profile(points=201)
    assert distances.shape == energies.shape == (201,)
    assert distances[0] == 0.0 and distances[-1] == pytest.approx(result.distances[-1])
    np.testing.assert_allclose(energies[[0, -1]], result.energies[[0, -1]], atol=1e-12)
    np.testing.assert_allclose(energies.max(), result.energies.max(), atol=1e-3)
    for points in (1, 2.5, True):
        with pytest.raises(InvalidParameter, match="points"):
            result.profile(points)


def test_invalid_states_and_options():
    initial, final = vacancy_hop()
    eam = EAM.zhou(["Cu"])
    with pytest.raises(InvalidParameter, match="same atoms"):
        neb(initial, final.filter(np.arange(len(final)) > 0), eam)
    with pytest.raises(InvalidParameter, match="same box"):
        neb(initial, final.with_box(Box(final.box.matrix * 1.01)), eam)
    with pytest.raises(InvalidParameter, match="element"):
        neb(initial, final.with_columns(element=["Ni"] * len(final)), eam)
    for bad in (
        {"images": 0},
        {"images": True},
        {"images": []},
        {"images": [1, 2]},
        {"interpolation": "spline"},
        {"spring": 0.0},
        {"max_force": -1.0},
        {"max_steps": -1},
        {"max_step": 0.0},
        {"timestep": 0.0},
        {"log_every": 0},
    ):
        with pytest.raises(InvalidParameter):
            neb(initial, final, eam, **bad)
    with pytest.raises(InvalidParameter, match="one for each"):
        neb(initial, final, [eam, eam], images=3)
    with pytest.raises(InvalidParameter, match="one for each"):
        neb(initial, final, [eam, eam, "eam"], images=3)
    with pytest.raises(InvalidParameter, match="energies and forces"):
        neb(initial, final, Energies(), images=3)
    with pytest.raises(CalculationFailed, match="not finite"):
        neb(initial, final, Failing(), images=3)
    with pytest.raises(CalculationFailed, match="not finite"):
        neb(initial, final, [eam, Failing(), eam], images=3)
    with pytest.raises(MissingProperty):
        bare = initial.with_columns(positions=initial["positions"])
        neb(
            System(table=bare.table.drop("element"), box=bare.box),
            final,
            MullerBrown(),
            images=1,
        )


def test_the_images_shared_among_processes(monkeypatch, tmp_path):
    """Under MPI each process computes its images, every process holds the
    band: the path of one process, to the last bit."""
    from simulation import processes

    initial, final = vacancy_hop()
    alone = neb(initial, final, EAM.zhou(["Cu"]), images=5, max_force=1e-3)
    for size in (2, 3, 7):  # of seven, two compute no image
        split = processes.run(
            size,
            lambda rank: neb(initial, final, EAM.zhou(["Cu"]), images=5, max_force=1e-3,
                             log=tmp_path / f"neb{rank}.log"),
            monkeypatch,
        )  # fmt: skip
        for result in split:
            assert result.steps == alone.steps and result.converged
            np.testing.assert_array_equal(result.energies, alone.energies)
            for a, b in zip(result.images, alone.images):
                np.testing.assert_array_equal(a["positions"], b["positions"])
        # the main process writes the log
        assert f"{size} processes" in (tmp_path / "neb0.log").read_text()
        assert not (tmp_path / "neb1.log").exists()

    # the callback of one process stops them all
    def stopped(rank):
        def stop(step, band, energies, largest):
            return rank == 1 and step == 3

        return neb(initial, final, EAM.zhou(["Cu"]), images=3, callback=stop).steps

    assert processes.run(2, stopped, monkeypatch) == [3, 3]
