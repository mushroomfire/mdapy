# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import numpy as np
import pytest

import mdapy
from mdapy.analysis import (
    BondAngleDistribution,
    MeanSquaredDisplacement,
    RadialDistributionFunction,
    StructureFactor,
)
from mdapy.build import build_crystal, random_alloy
from mdapy.errors import InvalidParameter
from mdapy.identify import CommonNeighborAnalysis
from mdapy.simulation import StackingFaultResult

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
    assert ax.get_legend().get_texts()[0].get_text() == "test: RMSE 10 meV"
    dense = plot.parity(reference, predicted, density=True)
    assert dense.get_legend().get_texts()[0].get_text().startswith("RMSE 0.01")
    empty = plot.parity([np.nan], [np.nan])
    assert "nan" in empty.get_legend().get_texts()[0].get_text()
    with pytest.raises(InvalidParameter, match="same size"):
        plot.parity([1.0], [1.0, 2.0])


def test_results():
    crystal = random_alloy(
        build_crystal("fcc", 3.6, repeat=(4, 4, 4)), {"Cu": 1, "Ni": 1}, seed=1
    )
    rdf = RadialDistributionFunction(r_max=6.0, bins=60).compute(crystal)
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
