# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Phase diagrams against closed forms, and against pymatgen where installed."""

from types import SimpleNamespace

import numpy as np
import pytest

from mdapy import units
from mdapy.errors import InvalidParameter
from mdapy.simulation import binary_phase_diagram, pressure_temperature_diagram
from mdapy.simulation._phase_diagram import _lower_hull, _mixing

BOLTZMANN = units.BOLTZMANN


def solution(temperatures, ends, interaction=0.0, fractions=np.linspace(0, 1, 11)):
    """Free energies of a regular solution at these fractions; ends gives the two
    pure elements as functions of the temperature."""
    t = np.asarray(temperatures, dtype=float)
    return {
        float(x): (
            t,
            (1 - x) * ends[0](t)
            + x * ends[1](t)
            + BOLTZMANN * t * _mixing(np.array(x))
            + interaction * x * (1 - x),
        )
        for x in fractions
    }


def zero(t):
    return np.zeros_like(t)


def test_the_lower_hull():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([0.0, 1.0, -1.0, 2.0, 0.0])
    assert _lower_hull(x, y) == [0, 2, 4]


def test_the_miscibility_gap_of_a_regular_solution():
    """Two phases of one solution, where ln(x / (1 - x)) = W (2 x - 1) / kT."""
    interaction = 0.1  # eV
    critical = interaction / (2 * BOLTZMANN)
    below = 0.7 * critical
    t = np.array([below])
    diagram = binary_phase_diagram(
        {"fcc": solution(t, (zero, zero), interaction)}, t, degree=6, points=2001
    )
    assert len(diagram.tie_lines) == 1
    line = diagram.tie_lines[0]
    assert line.first == line.second == "fcc"
    assert line.first_fraction == pytest.approx(1 - line.second_fraction, abs=1e-3)
    x = line.second_fraction
    assert np.log(x / (1 - x)) == pytest.approx(
        interaction * (2 * x - 1) / (BOLTZMANN * below), rel=0.02
    )
    # above the critical temperature, one phase everywhere
    above = np.array([1.2 * critical])
    one = binary_phase_diagram(
        {"fcc": solution(above, (zero, zero), interaction)}, above, degree=6
    )
    assert one.tie_lines == ()
    assert (one.stable == 0).all()


def test_the_lens_of_two_ideal_solutions():
    """Solid and liquid of ideal solutions: the closed form of a lens."""
    melt = (1300.0, 1700.0)
    heat = (0.13, 0.17)

    def liquid_end(k):
        return lambda t: heat[k] * (1 - t / melt[k])

    temperatures = np.array([1400.0, 1500.0, 1600.0])
    diagram = binary_phase_diagram(
        {
            "solid": solution(temperatures, (zero, zero)),
            "liquid": solution(temperatures, (liquid_end(0), liquid_end(1))),
        },
        temperatures,
        elements=("Cu", "Ni"),
        points=2001,
    )
    assert len(diagram.tie_lines) == len(temperatures)
    for line, t in zip(diagram.tie_lines, temperatures):
        a = np.exp(-liquid_end(0)(t) / (BOLTZMANN * t))
        b = np.exp(-liquid_end(1)(t) / (BOLTZMANN * t))
        solid = (1 - a) / (b - a)
        assert {line.first, line.second} == {"solid", "liquid"}
        assert (
            line.first == "liquid"
        )  # the liquid has less of the element melting higher
        assert line.second_fraction == pytest.approx(solid, abs=2e-3)
        assert line.first_fraction == pytest.approx(b * solid, abs=2e-3)
    assert diagram.elements == ("Cu", "Ni")
    assert diagram.phases_at(1500.0, 0.0) == ("liquid",)
    assert diagram.phases_at(1500.0, 1.0) == ("solid",)
    middle = diagram.tie_lines[1]
    between = 0.5 * (middle.first_fraction + middle.second_fraction)
    assert diagram.phases_at(1500.0, between) == ("liquid", "solid")
    assert (
        diagram.stable[1, diagram.fractions < middle.first_fraction - 1e-3] == 1
    ).all()
    assert (
        diagram.stable[1, diagram.fractions > middle.second_fraction + 1e-3] == 0
    ).all()
    assert (
        diagram.stable[
            1,
            (diagram.fractions > middle.first_fraction + 1e-3)
            & (diagram.fractions < middle.second_fraction - 1e-3),
        ]
        == -1
    ).all()
    with pytest.raises(InvalidParameter, match="fraction"):
        diagram.phases_at(1500.0, 1.5)


def test_a_compound_between_two_elements():
    t = np.array([500.0])
    level = -0.5
    diagram = binary_phase_diagram(
        {
            "A": {0.0: (t, np.array([0.0]))},
            "B": {1.0: (t, np.array([0.0]))},
            "AB": {0.5: (t, np.array([level]))},
        },
        t,
    )
    assert [
        (c.first, c.first_fraction, c.second, c.second_fraction)
        for c in diagram.tie_lines
    ] == [
        ("A", 0.0, "AB", 0.5),
        ("AB", 0.5, "B", 1.0),
    ]
    assert diagram.phases_at(500.0, 0.25) == ("A", "AB")
    assert diagram.phases_at(500.0, 0.5) in (("A", "AB"), ("AB", "B"))
    middle = len(diagram.fractions) // 2
    assert diagram.phases[diagram.stable[0, middle]] == "AB"
    assert (np.delete(diagram.stable[0, 1:-1], middle - 1) == -1).all()


def test_past_the_temperatures_of_a_free_energy():
    """Beyond its temperatures a free energy goes on along a straight line."""
    t = np.linspace(1000.0, 1200.0, 21)
    both = {0.0: (t, -1e-3 * t), 1.0: (t, -1e-3 * t)}
    wanted = np.array([900.0, 1100.0, 1300.0])
    other = {0.0: (wanted, np.zeros(3)), 1.0: (wanted, np.zeros(3))}
    diagram = binary_phase_diagram({"a": both, "b": other}, wanted)
    assert diagram.stable[:, 0].tolist() == [0, 0, 0]
    assert diagram.temperatures.tolist() == wanted.tolist()


def test_against_pymatgen():
    """pymatgen finds the same stable points from the same energies, at one
    temperature, as a phase diagram of formation energies."""
    pytest.importorskip("pymatgen.core")
    composition = pytest.importorskip("pymatgen.core.composition")
    pd = pytest.importorskip("pymatgen.analysis.phase_diagram")
    t = 400.0
    fractions = np.linspace(0, 1, 201)
    interaction = 0.1
    energies = BOLTZMANN * t * _mixing(fractions) + interaction * fractions * (
        1 - fractions
    )
    entries = [
        pd.PDEntry(composition.Composition({"Cu": 1 - x, "Ni": x}), float(e))
        for x, e in zip(fractions, energies)
        if 0 < x < 1
    ]
    entries += [
        pd.PDEntry(composition.Composition({"Cu": 1}), 0.0),
        pd.PDEntry(composition.Composition({"Ni": 1}), 0.0),
    ]
    diagram = pd.PhaseDiagram(entries)
    stable = sorted(
        e.composition.get_atomic_fraction("Ni") for e in diagram.stable_entries
    )
    jumps = np.diff(stable)
    k = int(jumps.argmax())
    ours = binary_phase_diagram(
        {
            "solution": {
                float(x): (np.array([t]), np.array([e]))
                for x, e in zip(fractions[::10], energies[::10])
            }
        },
        [t],
        degree=8,
        points=201,
    )
    line = ours.tie_lines[0]
    assert line.first_fraction == pytest.approx(stable[k], abs=0.01)
    assert line.second_fraction == pytest.approx(stable[k + 1], abs=0.01)


def test_invalid_input():
    t = np.array([500.0])
    good = {"a": {0.0: (t, t), 1.0: (t, t)}}
    with pytest.raises(InvalidParameter, match="needs phases"):
        binary_phase_diagram({}, t)
    with pytest.raises(InvalidParameter, match="no free energies"):
        binary_phase_diagram({"a": {}}, t)
    with pytest.raises(InvalidParameter, match="from 0 to 1"):
        binary_phase_diagram({"a": {2.0: (t, t)}}, t)
    with pytest.raises(InvalidParameter, match="temperatures"):
        binary_phase_diagram(good, [0.0])
    with pytest.raises(InvalidParameter, match="points"):
        binary_phase_diagram(good, t, points=2)
    with pytest.raises(InvalidParameter, match="degree"):
        binary_phase_diagram(good, t, degree=-1)
    with pytest.raises(InvalidParameter, match="two elements"):
        binary_phase_diagram(good, t, elements=("A",))
    with pytest.raises(InvalidParameter, match="one value per temperature"):
        binary_phase_diagram({"a": {0.0: (t, np.zeros(2)), 1.0: (t, t)}}, t)
    with pytest.raises(InvalidParameter, match="reaches the fractions"):
        binary_phase_diagram({"a": {0.2: (t, t), 0.8: (t, t)}}, t)


def linear(energy, entropy, volume, pressure, temperatures):
    """A phase whose free energy is energy - entropy T + volume P, in eV and Å³."""
    t = np.asarray(temperatures, dtype=float)
    work = volume * pressure * units.GPA_TO_EV_PER_A3
    return SimpleNamespace(
        pressure=pressure,
        temperatures=t,
        free_energy=energy + work - entropy * t,
        enthalpy=np.full_like(t, energy + work),
        volume=np.full_like(t, volume),
    )


PRESSURES = (0.0, 2.0, 5.0, 10.0)
RANGE = np.arange(800.0, 2401.0, 50.0)


def test_a_melting_line_in_closed_form():
    """The boundary of two phases with constant entropies and volumes, and the
    Clausius-Clapeyron relation with its own numbers."""
    solid = (-3.5, 1e-3, 12.0)
    liquid = (-3.35, 1.1e-3, 12.6)
    diagram = pressure_temperature_diagram(
        {
            "solid": [linear(*solid, p, RANGE) for p in PRESSURES],
            "liquid": [linear(*liquid, p, RANGE[::-1]) for p in PRESSURES],
        }
    )
    assert diagram.phases == ("solid", "liquid")
    np.testing.assert_array_equal(diagram.pressures, PRESSURES)
    (boundary,) = diagram.boundaries
    assert (boundary.first, boundary.second) == ("solid", "liquid")
    heat = (
        liquid[0]
        - solid[0]
        + (liquid[2] - solid[2]) * diagram.pressures * (units.GPA_TO_EV_PER_A3)
    )
    expected = heat / (liquid[1] - solid[1])
    np.testing.assert_allclose(boundary.temperatures, expected, rtol=1e-9)
    np.testing.assert_allclose(boundary.latent_heat, heat, rtol=1e-9)
    np.testing.assert_allclose(boundary.volume, liquid[2] - solid[2])
    # the slope of the line itself
    measured = np.gradient(boundary.temperatures, boundary.pressures)
    np.testing.assert_allclose(boundary.slope, measured, rtol=1e-9)
    assert diagram.phase_at(5.0, 900.0) == "solid"
    assert diagram.phase_at(5.0, 2300.0) == "liquid"


def test_three_phases_of_which_two_are_solid():
    low = (-3.5, 0.9e-3, 12.0)
    dense = (-3.6, 0.8e-3, 11.0)
    liquid = (-3.0, 1.2e-3, 12.6)
    diagram = pressure_temperature_diagram(
        {
            "low": [linear(*low, p, RANGE) for p in PRESSURES],
            "dense": [linear(*dense, p, RANGE) for p in PRESSURES],
            "liquid": [linear(*liquid, p, RANGE) for p in PRESSURES],
        }
    )
    pairs = {(b.first, b.second) for b in diagram.boundaries}
    assert pairs == {("dense", "low"), ("low", "liquid")}
    assert diagram.phase_at(10.0, 1000.0) == "dense"
    assert diagram.phase_at(10.0, 1700.0) == "low"
    assert diagram.phase_at(10.0, 2000.0) == "liquid"
    for b in diagram.boundaries:
        one = {"low": low, "dense": dense, "liquid": liquid}[b.first]
        two = {"low": low, "dense": dense, "liquid": liquid}[b.second]
        heat = (
            two[0] - one[0] + (two[2] - one[2]) * b.pressures * (units.GPA_TO_EV_PER_A3)
        )
        np.testing.assert_allclose(b.temperatures, heat / (two[1] - one[1]), rtol=1e-9)


def test_a_phase_that_ends_is_no_boundary():
    """A crystal known only up to where it stayed a crystal, still the lowest."""
    solid = linear(-3.5, 1e-3, 12.0, 0.0, np.arange(800.0, 1401.0, 50.0))
    liquid = linear(-3.0, 1.1e-3, 12.6, 0.0, RANGE)
    diagram = pressure_temperature_diagram({"solid": [solid], "liquid": [liquid]})
    assert diagram.boundaries == ()
    assert diagram.phase_at(0.0, 1000.0) == "solid"
    assert diagram.phase_at(0.0, 2000.0) == "liquid"


def test_where_nothing_is_known():
    one = linear(-3.5, 1e-3, 12.0, 0.0, np.arange(800.0, 1001.0, 50.0))
    two = linear(-3.0, 1.1e-3, 12.6, 0.0, np.arange(1500.0, 1701.0, 50.0))
    diagram = pressure_temperature_diagram({"one": [one], "two": [two]})
    assert diagram.phase_at(0.0, 1250.0) is None and diagram.boundaries == ()


@pytest.mark.parametrize(
    "phases, options, message",
    [
        ({"one": []}, {}, "at least two phases"),
        ({"one": [], "two": []}, {}, "no free energies"),
        (
            {"one": [SimpleNamespace(pressure=None)], "two": []},
            {},
            "has no pressure",
        ),
        (
            {"one": [linear(0, 0, 1, 0.0, RANGE)] * 2, "two": []},
            {},
            "two free energies of one",
        ),
        (
            {"one": [linear(0, 0, 1, 0.0, RANGE)], "two": []},
            {"points": 1},
            "points must be",
        ),
    ],
)
def test_invalid_pressure_temperature(phases, options, message):
    with pytest.raises(InvalidParameter, match=message):
        pressure_temperature_diagram(phases, **options)
