# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The two-phase melting point: the search in closed form, and argon. Skipped
where LAMMPS is not installed."""

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.calculators import LAMMPS
from mdapy.core.threads import CancelToken
from mdapy.errors import Cancelled, InvalidParameter, MissingProperty
from mdapy.simulation import MolecularDynamics, minimize
from mdapy.simulation import _melting as module
from mdapy.simulation._melting import MeltingAttempt, melting_point

pytest.importorskip("lammps")

EPSILON, SIGMA, CUT = 0.0104, 3.4, 8.5
BOLTZMANN = 8.617333262e-5
FAST = dict(timestep=0.005, equilibration=4.0, coexistence=50.0, seed=2)


@pytest.fixture(scope="module")
def potential():
    return LAMMPS(
        f"pair_style lj/cut {CUT}\npair_coeff * * {EPSILON} {SIGMA}\n"
        "pair_modify shift yes",
        ["Ar"],
    )


@pytest.fixture(scope="module")
def argon(potential):
    start = build_crystal("fcc", 5.3, elements="Ar")
    return minimize(start, potential, cell="full", max_stress=1e-4).system


def attempt(start, outcome, temperature=None, fraction=0.5, drift=0.0):
    return MeltingAttempt(
        start=start,
        outcome=outcome,
        temperature=start if temperature is None else temperature,
        fraction=fraction,
        drift=drift,
    )


def test_the_answer_needs_both_phases_and_a_steady_temperature():
    is_answer = module._is_answer
    assert is_answer(attempt(1000, "coexisting", 1005), 0.5, 3.0)
    assert not is_answer(attempt(1000, "melted", 1005), 0.5, 3.0)
    assert not is_answer(attempt(1000, "coexisting", 1005, 0.15), 0.5, 3.0)
    assert not is_answer(attempt(1000, "coexisting", 1005, 0.85), 0.5, 3.0)
    assert not is_answer(attempt(1000, "coexisting", 1005, drift=4.0), 0.5, 3.0)
    # a drift within the noise of the temperature is no drift
    assert is_answer(attempt(1000, "coexisting", 1005, drift=4.0), 3.0, 3.0)


def test_the_next_temperature():
    following = module._next_temperature
    assert following([attempt(1000, "coexisting", 1100)]) == 1100
    # still drifting: on from where it was going
    assert following([attempt(1000, "coexisting", 1005, drift=15)]) == 1020
    # only one side known
    assert following([attempt(1000, "melted")]) == pytest.approx(850)
    assert following([attempt(1000, "frozen")]) == pytest.approx(1150)
    assert following(
        [attempt(1000, "melted"), attempt(850, "melted")]
    ) == pytest.approx(722.5)
    # both: halfway
    assert following(
        [attempt(1000, "melted"), attempt(850, "frozen")]
    ) == pytest.approx(925)
    assert following(
        [attempt(1000, "melted"), attempt(850, "frozen"), attempt(925, "melted")]
    ) == pytest.approx(887.5)


def run_with(outcomes, seen):
    """A stand-in for one run, with outcomes in turn."""

    def one(start, potential, temperature, *args):
        seen.append(temperature)
        outcome, mean, fraction = outcomes[len(seen) - 1]
        return module._Run(
            attempts=(MeltingAttempt(temperature, outcome, mean, fraction, 0.0),),
            error=1.0,
            time=np.zeros(2),
            temperatures=np.full(2, mean),
            pressures=np.zeros(2),
            fractions=np.array([fraction]),
            fraction_time=np.zeros(1),
            system=start,
        )

    return one


def test_the_search_ends_at_two_phases(monkeypatch, argon, potential):
    seen = []
    outcomes = [
        ("melted", 200.0, 0.0),
        ("frozen", 40.0, 1.0),
        ("coexisting", 83.0, 0.1),
        ("coexisting", 82.0, 0.5),
    ]
    monkeypatch.setattr(module, "_one_run", run_with(outcomes, seen))
    result = melting_point(argon, potential, 100.0, atoms=64, tolerance=2.0)
    assert seen == [100.0, 85.0, pytest.approx(92.5), 83.0]
    assert result.temperature == 82.0 and result.error == 1.0
    assert len(result.attempts) == 4
    assert not result.temperatures.flags.writeable


def test_a_search_that_does_not_end_says_so(monkeypatch, argon, potential):
    seen = []
    monkeypatch.setattr(
        module, "_one_run", run_with([("melted", 500.0, 0.0)] * 3, seen)
    )
    with pytest.raises(InvalidParameter, match="no melting point in 3 runs"):
        melting_point(argon, potential, 100.0, atoms=64, max_runs=3)


def test_the_supercell():
    crystal = build_crystal("fcc", 4.0, elements="Ar")
    big = module._supercell(crystal, 4000)
    a, b, c = big.box.thickness
    assert a == pytest.approx(b)
    assert c == pytest.approx(3 * a, rel=0.2)
    assert 2500 < len(big) < 6000
    assert round(c / 4.0) % 2 == 0


@pytest.mark.parametrize(
    "structure, a, neighbors",
    [("fcc", 3.6, 12), ("hcp", 2.5, 12), ("bcc", 2.9, 14), ("diamond", 5.4, 4)],
)
def test_the_first_shell(structure, a, neighbors):
    crystal = build_crystal(structure, a, elements="Ar", repeat=(4, 4, 4))
    assert module._shell(crystal) == neighbors


def test_the_solid_like_atoms_tell_the_phases_apart(potential, argon):
    solid = mdapy.transform.replicate(argon, 4)
    neighbors = module._shell(solid)
    with MolecularDynamics(solid, potential, timestep=0.005) as md:
        md.velocities(100, seed=1)
        md.run(400, ensemble="nvt", temperature=50, thermo=400)
        cold = md.system
        md.velocities(900, seed=2)
        md.run(1000, ensemble="nvt", temperature=450, thermo=1000)
        hot = md.system
    assert module._solid_like(cold, neighbors).mean() > 0.9
    assert module._solid_like(hot, neighbors).mean() < 0.1


def run_at(potential, argon, temperature, **options):
    atoms = mdapy.transform.replicate(argon, (4, 4, 8))
    counter = module._Progress(1, None, None)
    settings = dict(
        pressure=0.0,
        barostat_damping=None,
        timestep=0.005,
        equilibrium=800,
        coexistence=4000,
        neighbors=12,
        tolerance=2.0,
        seed=1,
        counter=counter,
    )
    return module._one_run(
        atoms, potential, temperature=temperature, **settings | options
    )


def test_the_run_of_two_phases_at_a_melting_temperature(potential, argon):
    """One run started at about the melting point of the model keeps both
    phases, and the fractions of solid atoms are those of two phases."""
    expected = 0.62 * EPSILON / BOLTZMANN  # T* of the truncated and shifted model
    run = run_at(potential, argon, expected)
    last = run.attempts[-1]
    assert last.outcome == "coexisting"
    assert 0.1 < last.fraction < 0.9
    assert run.system.table["solid"].sum() == pytest.approx(
        last.fraction * len(run.system), abs=1
    )
    assert last.temperature == pytest.approx(expected, rel=0.15)
    assert len(run.time) == len(run.temperatures) == len(run.pressures)
    assert abs(run.pressures.mean()) < 0.1  # GPa


def test_a_run_far_too_hot_melts_at_once(potential, argon):
    run = run_at(potential, argon, 200.0)
    assert [a.outcome for a in run.attempts] == ["melted"]


def test_a_run_too_cold_freezes(potential, argon):
    run = run_at(potential, argon, 30.0, coexistence=20000)
    assert run.attempts[-1].outcome == "frozen"


def test_a_run_too_hot_melts(potential, argon):
    run = run_at(potential, argon, 95.0, coexistence=20000)
    assert run.attempts[-1].outcome == "melted"


def test_from_guesses_on_both_sides(potential, argon):
    """Guesses far from the melting point end in one place, within the noise
    of a small box and short runs."""
    found = [
        melting_point(
            argon, potential, guess, repeat=(5, 5, 10), tolerance=2.0, **FAST
        ).temperature
        for guess in (60.0, 110.0)
    ]
    assert found[0] == pytest.approx(found[1], rel=0.12)
    expected = 0.62 * EPSILON / BOLTZMANN
    assert np.mean(found) == pytest.approx(expected, rel=0.15)


def test_progress_and_cancel(potential, argon, monkeypatch):
    seen = []
    token = CancelToken()

    def stop(done, total):
        seen.append((done, total))
        token.cancel()

    with pytest.raises(Cancelled):
        melting_point(
            argon,
            potential,
            70.0,
            repeat=(2, 2, 2),
            timestep=0.005,
            equilibration=0.5,
            coexistence=1.0,
            progress=stop,
            cancel=token,
        )
    assert seen and seen[0][0] <= seen[0][1]


def test_a_half_that_does_not_melt_is_refused(monkeypatch, potential, argon):
    monkeypatch.setattr(module, "_travel", lambda start, end, first: 0.0)
    with pytest.raises(InvalidParameter, match="does not melt"):
        melting_point(
            argon,
            potential,
            70.0,
            repeat=(2, 2, 2),
            timestep=0.005,
            equilibration=0.5,
            coexistence=1.0,
        )


def test_invalid_input(potential, argon):
    for wrong, match in (
        (dict(timestep=0.0), "positive"),
        (dict(equilibration=-1.0), "positive"),
        (dict(coexistence=0.0), "positive"),
        (dict(tolerance=0.0), "positive"),
        (dict(atoms=0), "positive"),
        (dict(max_runs=0), "positive"),
        (dict(seed=0), "positive"),
        (dict(repeat=(1, 1)), "repeat"),
        (dict(repeat=(1, 1, 0)), "repeat"),
        (dict(repeat=(1.5, 1, 1)), "repeat"),
    ):
        with pytest.raises(InvalidParameter, match=match):
            melting_point(argon, potential, 80.0, **wrong)
    with pytest.raises(InvalidParameter, match="temperature"):
        melting_point(argon, potential, 0.0)
    with pytest.raises(InvalidParameter, match="LAMMPS"):
        melting_point(argon, object(), 80.0)
    with pytest.raises(MissingProperty, match="element"):
        melting_point(argon.drop("element"), potential, 80.0)
    with pytest.raises(InvalidParameter, match="periodic"):
        melting_point(
            argon.with_box(mdapy.Box(argon.box.matrix, pbc=(True, True, False))),
            potential,
            80.0,
        )
