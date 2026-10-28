# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The elastic strain against that of OVITO 3.17.1, atom by atom, on
dislocations, stacking faults and deformed crystals of the five lattices.
Skipped where OVITO is not installed."""

import numpy as np
import pytest

import mdapy
from mdapy.analysis import ElasticStrain
from mdapy.build import build_crystal, dislocation_quadrupole

ovito = pytest.importorskip("ovito")
from ovito.modifiers import ElasticStrainModifier  # noqa: E402

from dislocation_models import (  # noqa: E402
    QUADRUPOLES,
    crossing_loops,
    cubic_loops,
    hexagonal,
)
from ovito_reference import ovito_compute  # noqa: E402

LATTICE = {
    "fcc": ElasticStrainModifier.Lattice.FCC,
    "hcp": ElasticStrainModifier.Lattice.HCP,
    "bcc": ElasticStrainModifier.Lattice.BCC,
    "cubic_diamond": ElasticStrainModifier.Lattice.CubicDiamond,
    "hexagonal_diamond": ElasticStrainModifier.Lattice.HexagonalDiamond,
}
VOIGT = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]


def compare(system, crystal, a, c=None, atol=1e-12):
    """The same structures, strain tensors and volumetric strains as OVITO,
    whose c/a is its axial ratio."""
    ours = ElasticStrain(crystal, a, c).compute(system)
    options = {} if c is None else {"axial_ratio": c / a}
    data = ovito_compute(
        system,
        ElasticStrainModifier(
            input_crystal_structure=LATTICE[crystal], lattice_constant=a, **options
        ),
    )
    particles = data.particles
    np.testing.assert_array_equal(
        ours.structure, np.asarray(particles["Structure Type"])
    )
    theirs = np.asarray(particles["Elastic Strain"])
    for k, (i, j) in enumerate(VOIGT):
        np.testing.assert_allclose(
            ours.strain_tensor[:, i, j], theirs[:, k], rtol=0, atol=atol
        )
        np.testing.assert_allclose(
            ours.strain_tensor[:, j, i], theirs[:, k], rtol=0, atol=atol
        )
    np.testing.assert_allclose(
        ours.volumetric, np.asarray(particles["Volumetric Strain"]), rtol=0, atol=atol
    )
    return ours


@pytest.mark.parametrize("name", list(QUADRUPOLES))
def test_quadrupoles(name):
    """Straight dislocations, some dissociated with a stacking fault between
    the partials, whose atoms have no strain."""
    crystal, args, options = QUADRUPOLES[name]
    ours = compare(dislocation_quadrupole(*args, **options), crystal, args[1])
    assert np.abs(ours.volumetric).max() > 1e-3


@pytest.mark.parametrize("name", ["perfect fcc loop", "shockley loop", "bcc loop"])
def test_loops(name):
    crystal, system = cubic_loops()[name]
    compare(system, crystal, 3.615 if crystal == "fcc" else 2.87)


def test_crossing_loops():
    compare(crossing_loops(), "fcc", 3.615)


@pytest.mark.parametrize(
    "crystal, structure, a, element, kind",
    [
        ("hcp", "hcp", 3.2, "Mg", "screw"),
        ("hcp", "hcp", 3.2, "Mg", "edge"),
        ("hexagonal_diamond", "lonsdaleite", 2.51, "C", "screw"),
        ("hexagonal_diamond", "lonsdaleite", 2.51, "C", "edge"),
    ],
)
def test_hexagonal_dislocations(crystal, structure, a, element, kind):
    """OVITO keeps the transitions between clusters in single precision,
    which its strain of the atoms of clusters it joins across a fault
    carries."""
    compare(hexagonal(structure, a, element, kind), crystal, a, atol=2e-7)


@pytest.mark.parametrize(
    "crystal, structure, a, c",
    [
        ("fcc", "fcc", 3.615, None),
        ("bcc", "bcc", 2.865, None),
        ("cubic_diamond", "diamond", 5.431, None),
        ("hcp", "hcp", 2.95, 2.95 * 1.58),
        ("hcp", "hcp", 2.95, 2.95 * 1.86),
        ("hexagonal_diamond", "lonsdaleite", 2.51, 4.2),
    ],
)
def test_deformed_and_shaken(crystal, structure, a, c):
    """A crystal along general directions in a deformed triclinic box, its
    atoms shaken; analyzed with other constants than those it was built
    with."""
    orientation = None
    if c is None:
        orientation = [(1, 2, 3), (1, 1, -1), (-5, 4, -1)]
    cell = build_crystal(structure, a, c=c, orientation=orientation, repeat=(4, 4, 4))
    rng = np.random.default_rng(3)
    gradient = np.eye(3) + 0.01 * rng.standard_normal((3, 3))
    box = mdapy.Box(cell.box.matrix @ gradient.T, pbc=True)
    moved = cell.with_box(box, scale_positions=True)
    moved = moved.with_columns(
        positions=moved["positions"] + rng.normal(0, 0.02 * a, (len(cell), 3))
    )
    compare(moved, crystal, 1.002 * a, None if c is None else 0.997 * c)
