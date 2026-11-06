# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from types import SimpleNamespace

import numpy as np
import pytest

import mdapy
from mdapy.analysis import (
    BondAngleDistribution,
    MeanSquaredDisplacement,
    RadialDistributionFunction,
    StructureFactor,
    WarrenCowley,
)
from mdapy.build import build_crystal, random_alloy
from mdapy.errors import InvalidParameter
from mdapy.identify import CommonNeighborAnalysis
from mdapy.simulation import StackingFaultResult, pressure_temperature_diagram

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mdapy.visualization import plot  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_style_figure_and_save(tmp_path):
    with plot.style():
        assert plt.rcParams["font.size"] == 7 and not plt.rcParams["axes.spines.top"]
        fig, axes = plot.figure("double", nrows=1, ncols=2)
        assert fig.get_size_inches()[0] == pytest.approx(180 / 25.4)
        assert fig.get_size_inches()[1] == pytest.approx(0.75 * 180 / 25.4 / 2)
        plot.panel_label(axes[0], "a")
        assert axes[0].texts[0].get_text() == "a"
        files = plot.save(fig, tmp_path / "figure.png")
    assert [f.suffix for f in files] == [".pdf", ".svg", ".png"]
    assert all(f.stat().st_size > 0 for f in files)
    assert plot.figure(60.0, 40.0)[0].get_size_inches()[1] == pytest.approx(40 / 25.4)
    assert plot.figure()[0].get_size_inches()[0] == pytest.approx(88 / 25.4)
    with pytest.raises(InvalidParameter, match="width must be"):
        plot.figure("triple")


def test_parity():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=200)
    predicted = reference + 0.01
    ax = plot.parity(reference, predicted, label="test", unit="meV", scale=1000)
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("reference", "predicted")
    assert ax.get_legend().get_texts()[0].get_text() == "test: RMSE 10 meV"
    dense = plot.parity(reference, predicted, density=True)
    assert dense.get_legend().get_texts()[0].get_text().startswith("RMSE 0.01")
    empty = plot.parity([np.nan], [np.nan])
    assert "nan" in empty.get_legend().get_texts()[0].get_text()
    with pytest.raises(InvalidParameter, match="same size"):
        plot.parity([1.0], [1.0, 2.0])


def test_scatter():
    """A point for each atom with finite values, at most max_points of them,
    the same each time."""
    crystal = build_crystal("fcc", 3.6, elements="Cu", repeat=(4, 4, 4))
    value = np.arange(len(crystal), dtype=float)
    value[3] = np.nan
    system = crystal.with_columns(value=value)
    ax = plot.scatter(system, "positions.z", "value * 2")
    points = ax.collections[0].get_offsets()
    keep = np.arange(len(system)) != 3
    np.testing.assert_allclose(points[:, 0], system["positions.z"][keep])
    np.testing.assert_allclose(points[:, 1], 2 * value[keep])
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("positions.z", "value * 2")
    few = plot.scatter(system, "positions.x", "value", max_points=10, color="k")
    drawn = few.collections[0].get_offsets()
    assert len(drawn) == 10 and np.isfinite(drawn).all()
    np.testing.assert_array_equal(
        plot.scatter(system, "positions.x", "value", max_points=10)
        .collections[0]
        .get_offsets(),
        drawn,
    )
    some = plot.scatter(system, "positions.x", "value", atoms=value < 5)
    np.testing.assert_allclose(some.collections[0].get_offsets()[:, 1], [0, 1, 2, 4])
    with pytest.raises(InvalidParameter, match="positive"):
        plot.scatter(system, "positions.x", "value", max_points=0)
    with pytest.raises(InvalidParameter, match="one component"):
        plot.scatter(system, "positions", "value")


def test_results():
    crystal = random_alloy(
        build_crystal("fcc", 3.6, repeat=(4, 4, 4)), {"Cu": 1, "Ni": 1}, seed=1
    )
    rdf = RadialDistributionFunction(cutoff=6.0, bins=60).compute(crystal)
    assert len(plot.rdf(rdf).lines) == 2
    assert len(plot.rdf(rdf, partial=True).get_legend().get_texts()) == 3
    sq = StructureFactor(q_max=4.0, bins=40).compute(crystal)
    assert len(plot.structure_factor(sq).lines) == 1
    angles = BondAngleDistribution(cutoff=3.0, bins=36).compute(crystal)
    assert plot.bond_angles(angles).get_xlim() == (0, 180)
    walk = np.cumsum(
        np.random.default_rng(2).normal(scale=0.1, size=(40, 20, 3)), axis=0
    )
    frames = [mdapy.System(x, mdapy.Box(30.0), elements="Ar") for x in walk]
    displacement = MeanSquaredDisplacement().compute(frames)
    ax = plot.msd(displacement, timestep=0.1, fit=(1, 20))
    assert ax.get_legend().get_texts()[1].get_text().startswith("D = ")
    assert len(plot.msd(displacement).lines) == 1
    curve = StackingFaultResult(
        np.linspace(0, 1, 5), np.zeros((5, 3)), np.array([0, 0.1, 0.03, 0.1, 0.0]), 10.0
    )
    assert plot.stacking_fault(curve).lines[0].get_ydata()[1] == pytest.approx(100.0)
    line = plot.stacking_fault(curve, label="fcc").lines[0]
    assert line.get_label() == "fcc" and line.get_marker() == "o"
    structures = CommonNeighborAnalysis().compute(crystal)
    bars = plot.structure_counts(structures)
    assert len(bars.patches) == len(structures.names)


def test_phonons():
    from mdapy.lattice_dynamics import phonons, primitive_cell

    from calculators.test_testing import Harmonic

    a = 3.6
    cell = primitive_cell(build_crystal("fcc", a, elements="Ar"))
    result = phonons(cell, Harmonic(k=0.7, r0=a / np.sqrt(2)), supercell=3)
    bands = result.band_structure("fcc", points=5)
    ax = plot.band_structure(bands)
    # a line for each branch on each of the six straight pieces of the path,
    # with the marks of the seven points and of zero
    assert len(ax.lines) == 3 * 6 + 7 + 1
    assert [t.get_text() for t in ax.get_xticklabels()] == [x for _, x in bands.labels]
    one = plot.band_structure(bands, color="k")
    assert all(line.get_color() == "k" for line in one.lines[:6])
    dos = result.dos(6, sigma=0.2, projected=True)
    assert len(plot.phonon_dos(dos).lines) == 1
    assert len(plot.phonon_dos(dos, projected=True).get_legend().get_texts()) == 2


@pytest.mark.parametrize("mode", [True, False])
def test_style_keeps_interactive_mode(mode):
    """A notebook turns interactive mode on when its first figure loads the
    backend; if that was inside the style, later figures must still show."""
    import matplotlib.pyplot as plt

    before = plt.isinteractive()
    try:
        plt.ioff()
        with plot.style():
            plt.ion() if mode else plt.ioff()
            assert plt.rcParams["font.size"] == plot.STYLE["font.size"]
        assert plt.isinteractive() is mode
        assert (
            plt.rcParams["font.size"] != plot.STYLE["font.size"]
        )  # the rest goes back
        plt.ioff()
        with pytest.raises(ZeroDivisionError), plot.style():
            plt.ion() if mode else plt.ioff()
            1 / 0
        assert plt.isinteractive() is mode
    finally:
        plt.ion() if before else plt.ioff()


def test_style_shows_the_figures_of_a_notebook_in_the_style(monkeypatch):
    """A notebook draws after the cell: its figures show while the style holds."""
    import matplotlib.pyplot as plt

    sizes = []
    monkeypatch.setattr(plt, "show", lambda: sizes.append(plt.rcParams["font.size"]))
    with plot.style():
        pass
    assert sizes == []  # another backend: nothing shows by itself
    monkeypatch.setattr(
        plt, "get_backend", lambda: "module://matplotlib_inline.backend_inline"
    )
    with plot.style():
        pass
    assert sizes == [plot.STYLE["font.size"]]
    with pytest.raises(ZeroDivisionError), plot.style():
        1 / 0
    assert len(sizes) == 1  # not after an error


def test_labels_are_upright():
    assert plot.STYLE["mathtext.default"] == "regular"
    system = build_crystal("fcc", 3.6, elements="Cu", repeat=(3, 3, 3))
    with plot.style():
        fig, ax = plot.figure("single")
        plot.rdf(RadialDistributionFunction(cutoff=4.0).compute(system), ax=ax)
        labels = ax.get_xlabel() + ax.get_ylabel()
        plt.close(fig)
    assert "$" not in labels


def test_phase_diagram():
    from mdapy.simulation import binary_phase_diagram

    t = np.linspace(1000.0, 1700.0, 15)
    k = 8.617e-5
    fractions = np.linspace(0, 1, 5)

    def mixing(x):
        return k * t * (x * np.log(max(x, 1e-12)) + (1 - x) * np.log(max(1 - x, 1e-12)))

    solid = {float(x): (t, mixing(x)) for x in fractions}
    liquid = {
        float(x): (
            t,
            mixing(x) + (1 - x) * 0.13 * (1 - t / 1300) + x * 0.17 * (1 - t / 1600),
        )
        for x in fractions
    }
    diagram = binary_phase_diagram(
        {"solid": solid, "liquid": liquid}, t, elements=("Cu", "Ni")
    )
    ax = plot.phase_diagram(diagram)
    assert ax.get_xlabel() == "fraction of Ni" and ax.get_ylabel() == "temperature (K)"
    assert len(ax.lines) >= 2
    assert {text.get_text() for text in ax.texts} == {"solid", "liquid"}
    bare = plot.phase_diagram(diagram, labels=False)
    assert not bare.texts


def test_pressure_temperature():
    ranges = np.arange(800.0, 2001.0, 50.0)

    def phase(energy, entropy, volume, pressure):
        work = volume * pressure * 0.00624
        return SimpleNamespace(
            pressure=pressure,
            temperatures=ranges,
            free_energy=energy + work - entropy * ranges,
            enthalpy=np.full_like(ranges, energy + work),
            volume=np.full_like(ranges, volume),
        )

    diagram = pressure_temperature_diagram(
        {
            "solid": [phase(-3.5, 1e-3, 12.0, p) for p in (0.0, 5.0)],
            "liquid": [phase(-3.35, 1.1e-3, 12.6, p) for p in (0.0, 5.0)],
        }
    )
    with plot.style():
        fig, ax = plot.figure("single")
        plot.pressure_temperature(diagram, ax=ax)
        names = {text.get_text() for text in ax.texts}
        assert names == {"solid", "liquid"} and len(ax.lines) == 1
        assert "$" not in ax.get_xlabel() + ax.get_ylabel()
        plot.pressure_temperature(diagram, ax=ax, labels=False)
        plt.close(fig)


def _cells(ax):
    """The text written at each cell of a matrix."""
    return {(t.get_position()): t.get_text() for t in ax.texts}


def test_warren_cowley_of_a_known_order():
    # L1_2 Cu3Au among the 12 nearest: Au has only Cu around it
    crystal = build_crystal("l1_2", 3.75, elements=["Cu", "Au"], repeat=(4, 4, 4))
    result = WarrenCowley(num_neighbors=12).compute(crystal)
    assert result.elements == ("Cu", "Au")
    np.testing.assert_allclose(result.alpha, [[1 / 9, -1 / 3], [-1 / 3, 1]], atol=1e-12)
    with plot.style():
        ax = plot.warren_cowley(result)
        assert _cells(ax) == {
            (0, 0): "0.11",
            (0, 1): "-0.33",
            (1, 0): "-0.33",
            (1, 1): "1.00",
        }
        assert [t.get_text() for t in ax.get_xticklabels()] == ["Cu", "Au"]
        assert [t.get_text() for t in ax.get_yticklabels()] == ["Cu", "Au"]
        assert ax.get_xlabel() == "Central element"
        assert ax.get_ylabel() == "Neighboring element"
        bar = ax.figure.axes[-1]
        assert bar.get_ylabel() == "Warren–Cowley α"
        np.testing.assert_allclose(bar.get_yticks(), [-2, 0, 1])
        plt.close(ax.figure)


def test_warren_cowley_puts_each_pair_at_its_place():
    # row a, column b: the central element a along x, its neighbor b along y
    result = SimpleNamespace(
        elements=("A", "B", "C"),
        alpha=np.array([[0.1, 0.2, 0.3], [0.4, -0.001, 0.6], [0.7, np.nan, -1.5]]),
    )
    fig, ax = plt.subplots()
    assert plot.warren_cowley(result, ax=ax, vmin=-3, vmax=2, cmap="viridis") is ax
    cells = _cells(ax)
    assert cells[(0, 1)] == "0.20" and cells[(1, 0)] == "0.40"
    assert cells[(0, 2)] == "0.30" and cells[(2, 0)] == "0.70"
    assert cells[(1, 1)] == "0.00"  # no -0.00
    assert cells[(2, 1)] == "–"  # no atoms to tell
    assert cells[(2, 2)] == "-1.50"
    image = ax.images[0]
    assert image.get_cmap().name == "viridis" and image.origin == "lower"
    assert image.get_clim() == (-3, 2)
    assert image.get_array()[1, 0] == 0.2  # y, then x
    plt.close(fig)


def test_warren_cowley_is_readable_on_dark_cells():
    result = SimpleNamespace(
        elements=("A", "B"), alpha=np.array([[1.0, -2.0], [-2.0, 0.0]])
    )
    fig, ax = plt.subplots()
    plot.warren_cowley(result, ax=ax)
    colors = {t.get_position(): t.get_color() for t in ax.texts}
    assert colors[(0, 0)] == "w" and colors[(0, 1)] == "k" and colors[(1, 1)] == "k"
    plt.close(fig)


def test_warren_cowley_of_shells():
    crystal = build_crystal("cscl", 2.95, elements=["Cu", "Zn"], repeat=(4, 4, 4))
    result = WarrenCowley(num_neighbors=(8, 6)).compute(crystal)
    fig, ax = plt.subplots()
    plot.warren_cowley(result, ax=ax, shell=1)
    assert _cells(ax) == {
        (0, 0): "-1.00",
        (0, 1): "1.00",
        (1, 0): "1.00",
        (1, 1): "-1.00",
    }
    plt.close(fig)
    with pytest.raises(InvalidParameter, match="of 2 shells: choose one"):
        plot.warren_cowley(result)
    for shell in (2, -1):
        with pytest.raises(InvalidParameter, match="is not among 2 shells"):
            plot.warren_cowley(result, shell=shell)
    one = WarrenCowley(num_neighbors=8).compute(crystal)
    with pytest.raises(InvalidParameter, match="of one shell"):
        plot.warren_cowley(one, shell=0)
