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


def test_style_options():
    """The options change the text, the lines and the colors, settings of
    matplotlib by name take precedence, and each change ends with the context."""
    from matplotlib.colors import to_hex

    with plot.style():
        assert plt.rcParams["lines.linewidth"] == plot.STYLE["lines.linewidth"]
        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        assert [to_hex(c) for c in cycle] == list(plot.COLORS)
    with plot.style(font_size=14, line_width=2.5, colors=["red", "#123456"]):
        assert plt.rcParams["font.size"] == 14
        assert plt.rcParams["axes.labelsize"] == 14
        assert plt.rcParams["xtick.labelsize"] == pytest.approx(12)
        assert plt.rcParams["legend.fontsize"] == pytest.approx(12)
        assert plt.rcParams["lines.linewidth"] == 2.5
        assert plt.rcParams["axes.spines.top"] is False  # the rest of STYLE stays
        fig, ax = plot.figure()
        lines = [ax.plot([0, 1], [k, k])[0] for k in range(3)]
        assert [to_hex(line.get_color()) for line in lines] == [
            "#ff0000",
            "#123456",
            "#ff0000",
        ]
        assert lines[0].get_linewidth() == 2.5
    with plot.style(colors="green", settings={"axes.grid": True, "font.size": 9}):
        assert plt.rcParams["axes.grid"] is True
        assert plt.rcParams["font.size"] == 9
        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        assert cycle == ["green"]
    with plot.style(font_size=12, settings={"font.size": 10}):
        assert plt.rcParams["font.size"] == 10  # settings win
        assert plt.rcParams["axes.labelsize"] == 12
    assert plt.rcParams["font.size"] != 14 and not plt.rcParams["axes.grid"]
    for options, message in (
        ({"font_size": 0}, "font_size must be positive"),
        ({"line_width": -1.0}, "line_width must be positive"),
        ({"colors": []}, "at least one color"),
        ({"colors": ["notacolor"]}, "matplotlib"),
        ({"settings": {"axes.gridx": True}}, "matplotlib"),
        ({"settings": {"font.size": "huge"}}, "matplotlib"),
        ({"colors": 3}, "a color or a sequence of colors"),
        ({"settings": "axes.grid"}, "settings must map"),
        ({"settings": [1, 2]}, "settings must map"),
    ):
        with pytest.raises(InvalidParameter, match=message), plot.style(**options):
            pass  # pragma: no cover - the context does not start
    assert plt.rcParams["font.size"] != 14


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


def elastic_result(c11=168.6, c12=121.6, c44=75.4):
    from mdapy.lattice_dynamics import ElasticResult

    c = np.zeros((6, 6))
    c[:3, :3] = c12
    np.fill_diagonal(c, [c11, c11, c11, c44, c44, c44])
    return ElasticResult(c)


def surface_of(ax, monkeypatch):
    """What elastic_surface gives plot_surface: the points and their colors."""
    calls = []
    draw = ax.plot_surface

    def record(*args, **options):
        calls.append((args, options))
        return draw(*args, **options)

    monkeypatch.setattr(ax, "plot_surface", record)
    return calls


def test_elastic_surface_lies_at_the_moduli(monkeypatch):
    result = elastic_result()
    with plot.style():
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        calls = surface_of(ax, monkeypatch)
        assert plot.elastic_surface(result, ax=ax, points=20) is ax
    ((x, y, z), options) = calls[0]
    assert x.shape == (20, 40)
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    young = result.young_modulus(points)
    np.testing.assert_allclose(np.linalg.norm(points, axis=1), young, rtol=1e-12)
    low, high = result.young_modulus([[1, 0, 0], [1, 1, 1]])
    expected = matplotlib.colormaps["viridis"]((young - young.min()) / np.ptp(young))
    np.testing.assert_allclose(options["facecolors"].reshape(-1, 4), expected)
    assert young.min() == pytest.approx(low) and young.max() <= high
    reach = np.abs(young).max()
    assert ax.get_xlim() == pytest.approx((-reach, reach))
    assert ax.get_zlabel() == "z (GPa)"
    bar = fig.axes[-1]
    assert bar.get_ylabel() == "Young's modulus (GPa)"
    assert bar.get_ylim() == pytest.approx((young.min(), young.max()))


def test_elastic_surface_of_the_compressibility(monkeypatch):
    result = elastic_result(240.0, 140.0, 40.0)
    ax = plot.elastic_surface(
        result, "linear_compressibility", points=8, colormap="magma"
    )
    assert ax.name == "3d"
    assert ax.get_zlabel() == "z (1/GPa)"
    assert ax.figure.axes[-1].get_ylabel() == "linear compressibility (1/GPa)"
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    calls = surface_of(ax, monkeypatch)
    plot.elastic_surface(result, "linear_compressibility", ax=ax, points=8)
    points = np.stack(calls[0][0], axis=-1).reshape(-1, 3)
    np.testing.assert_allclose(
        np.linalg.norm(points, axis=1),
        result.linear_compressibility(points),
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    "options, message",
    [
        ({"quantity": "bulk_modulus"}, "quantity"),
        ({"quantity": "shear_modulus"}, "extreme"),
        ({"quantity": "poisson_ratio", "extreme": "mean"}, "extreme"),
        ({"extreme": "min"}, "no extreme"),
        ({"points": 2}, "points"),
        ({"colormap": "no such map"}, "colormap"),
    ],
)
def test_elastic_surface_errors(options, message):
    with pytest.raises(InvalidParameter, match=message):
        plot.elastic_surface(elastic_result(), **options)


def test_elastic_surface_needs_axes_in_3d():
    _, ax = plot.figure()
    with pytest.raises(InvalidParameter, match="3D"):
        plot.elastic_surface(elastic_result(), ax=ax)


def test_panel_label_on_axes_in_3d():
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    plot.panel_label(ax, "a")
    (label,) = ax.texts
    assert label.get_text() == "a" and label.get_transform() is ax.transAxes


def test_the_extremes_across_a_direction():
    from mdapy.visualization.plot import _elastic_values

    c11, c12, c44 = 168.6, 121.6, 75.4
    result = elastic_result(c11, c12, c44)
    axes = unit = np.array([[0, 0, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=float)
    unit = axes / np.linalg.norm(axes, axis=1, keepdims=True)
    low = _elastic_values(result, "shear_modulus", "min", unit)
    high = _elastic_values(result, "shear_modulus", "max", unit)
    # the same across a cube axis and across [111]: C44 and the slip shear
    slip = 3 * c44 * (c11 - c12) / (4 * c44 + c11 - c12)
    np.testing.assert_allclose(low[:2], [c44, slip], rtol=1e-12)
    np.testing.assert_allclose(high[:2], [c44, slip], rtol=1e-12)
    # across [110], from C' to C44
    assert low[2] == pytest.approx((c11 - c12) / 2, rel=1e-12)
    assert high[2] == pytest.approx(c44, rel=1e-12)
    low = _elastic_values(result, "poisson_ratio", "min", unit)
    high = _elastic_values(result, "poisson_ratio", "max", unit)
    assert low[2] == pytest.approx(
        result.poisson_ratio([1, 1, 0], [1, -1, 0]), rel=1e-12
    )
    assert high[2] == pytest.approx(
        result.poisson_ratio([1, 1, 0], [0, 0, 1]), rel=1e-12
    )
    assert low[3] == pytest.approx(high[3]) and low[3] == pytest.approx(
        result.poisson_ratio([1, 0, 0], [0, 1, 0])
    )


@pytest.mark.parametrize("quantity", ["shear_modulus", "poisson_ratio"])
def test_the_extremes_against_every_direction_across(quantity):
    from lattice_dynamics.test_elastic_anisotropy import triclinic

    from mdapy.lattice_dynamics import ElasticResult
    from mdapy.visualization.plot import _across, _elastic_values

    result = ElasticResult(triclinic(17))
    n = np.random.default_rng(18).normal(size=(30, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    first, second = _across(n)
    np.testing.assert_allclose((first * n).sum(1), 0, atol=1e-14)
    np.testing.assert_allclose((second * n).sum(1), 0, atol=1e-14)
    np.testing.assert_allclose((first * second).sum(1), 0, atol=1e-14)
    angles = np.linspace(0, np.pi, 4001)
    sampled = []
    for k in range(len(n)):
        m = np.cos(angles)[:, None] * first[k] + np.sin(angles)[:, None] * second[k]
        a = np.repeat(n[k : k + 1], len(m), axis=0)
        if quantity == "shear_modulus":
            sampled.append(result.shear_modulus(normal=a, direction=m))
        else:
            sampled.append(result.poisson_ratio(a, m))
    sampled = np.array(sampled)
    low = _elastic_values(result, quantity, "min", n)
    high = _elastic_values(result, quantity, "max", n)
    assert np.all(low <= sampled.min(axis=1) + 1e-12)
    assert np.all(high >= sampled.max(axis=1) - 1e-12)
    span = sampled.max(axis=1) - sampled.min(axis=1)
    # the samples miss the extremes by the square of half their spacing
    size = np.abs(sampled).max(axis=1)
    np.testing.assert_array_less(np.abs(low - sampled.min(axis=1)), 1e-5 * size)
    np.testing.assert_array_less(np.abs(high - sampled.max(axis=1)), 1e-5 * size)
    assert span.min() > 0


def test_elastic_surface_of_the_smallest_poisson_ratio(monkeypatch):
    result = elastic_result()
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    calls = surface_of(ax, monkeypatch)
    plot.elastic_surface(result, "poisson_ratio", ax=ax, extreme="min", points=12)
    points = np.stack(calls[0][0], axis=-1).reshape(-1, 3)
    from mdapy.visualization.plot import _elastic_values

    values = _elastic_values(
        result, "poisson_ratio", "min", points / np.linalg.norm(points, axis=1)[:, None]
    )
    np.testing.assert_allclose(
        np.linalg.norm(points, axis=1), np.abs(values), rtol=1e-12
    )
    assert values.min() < 0
    assert ax.get_zlabel() == "z"
    assert fig.axes[-1].get_ylabel() == "smallest Poisson ratio"
    ax = plot.elastic_surface(result, "shear_modulus", extreme="max", points=6)
    assert ax.figure.axes[-1].get_ylabel() == "largest shear modulus (GPa)"


def test_elastic_section_in_the_cube_plane():
    result = elastic_result()
    ax = plot.elastic_section(result, points=8, label="Cu")
    line, *guides = ax.lines
    assert len(guides) == 2  # the axes through the origin
    x, y = line.get_data()
    angles = np.linspace(0, 2 * np.pi, 9)
    directions = np.stack([np.cos(angles), np.sin(angles), np.zeros(9)], axis=1)
    np.testing.assert_allclose(
        np.hypot(x, y), result.young_modulus(directions), rtol=1e-12
    )
    np.testing.assert_allclose(x / np.hypot(x, y), np.cos(angles), atol=1e-12)
    np.testing.assert_allclose(y / np.hypot(x, y), np.sin(angles), atol=1e-12)
    assert line.get_label() == "Cu" and line.get_linestyle() == "-"
    assert ax.get_xlabel() == "[1 0 0] (GPa)" and ax.get_ylabel() == "[0 1 0] (GPa)"


@pytest.mark.parametrize(
    "normal, first, second",
    [
        ((1, 1, 0), "[0 0 1]", "[1 -1 0]"),
        ((1, 1, 1), "[2 -1 -1]", "[0 1 -1]"),
        ((0, 0, -2), "[1 0 0]", "[0 -1 0]"),
        ((0.3, 0.2, 0.9), "[-0.07 0.98 -0.20]", "[-3 0 1]"),
    ],
)
def test_elastic_section_names_the_axes_of_its_plane(normal, first, second):
    ax = plot.elastic_section(
        elastic_result(), "linear_compressibility", normal, points=6
    )
    assert ax.get_xlabel() == first + " (1/GPa)"
    assert ax.get_ylabel() == second + " (1/GPa)"


def test_elastic_section_dashes_negative_values():
    result = elastic_result()
    _, ax = plot.figure()
    plot.elastic_section(
        result, "poisson_ratio", ax=ax, extreme="min", color="C1", label="min"
    )
    solid, dashed = ax.lines[:2]
    assert solid.get_linestyle() == "-" and dashed.get_linestyle() == "--"
    assert solid.get_label() == "min" and dashed.get_label().startswith("_")
    assert solid.get_color() == dashed.get_color()
    assert ax.get_xlabel() == "[1 0 0]"
    x, y = dashed.get_data()
    # dashed around [110], where a stretch makes [1-10] grow
    angle = np.degrees(np.arctan2(y, x))[np.isfinite(x)]
    assert np.any(np.isclose(angle % 90, 45))
    negative = SimpleNamespace(young_modulus=lambda d: -np.ones(len(d)))
    _, ax = plot.figure()
    plot.elastic_section(negative, ax=ax, points=4, label="all")
    line, *guides = ax.lines
    assert len(guides) == 2
    assert line.get_linestyle() == "--" and line.get_label() == "all"


@pytest.mark.parametrize(
    "options, message",
    [
        ({"normal": (0, 0, 0)}, "normal"),
        ({"normal": (1, 0)}, "normal"),
        ({"normal": (np.inf, 0, 1)}, "normal"),
        ({"points": 2}, "points"),
        ({"quantity": "poisson_ratio"}, "extreme"),
    ],
)
def test_elastic_section_errors(options, message):
    with pytest.raises(InvalidParameter, match=message):
        plot.elastic_section(elastic_result(), **options)
