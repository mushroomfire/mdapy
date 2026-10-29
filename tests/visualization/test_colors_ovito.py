# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Colors and selections of atoms against OVITO, live."""

import numpy as np
import pytest

from mdapy.build import build_crystal
from mdapy.visualization import assign_color, color_by_type, color_coding

ovito = pytest.importorskip("ovito")
from ovito.modifiers import (  # noqa: E402
    AssignColorModifier,
    ColorByTypeModifier,
    ColorCodingModifier,
    ExpressionSelectionModifier,
    SelectTypeModifier,
)

from ovito_reference import ovito_compute  # noqa: E402

GRADIENTS = {
    "rainbow": "Rainbow",
    "viridis": "Viridis",
    "magma": "Magma",
    "jet": "Jet",
    "hot": "Hot",
    "blue_white_red": "BlueWhiteRed",
    "gray": "Grayscale",
}


def alloy():
    system = build_crystal("fcc", 3.6, elements="Cu", repeat=(4, 4, 4))
    rng = np.random.default_rng(12)
    return system.with_columns(
        element=rng.choice(["Cu", "Ni", "Al"], size=len(system)),
        positions=system["positions"] + rng.normal(0, 0.2, (len(system), 3)),
    )


@pytest.mark.parametrize("name", GRADIENTS)
def test_color_coding(name):
    """OVITO's gradients over the range of the values and over one given."""
    system = alloy()
    gradient = getattr(ColorCodingModifier, GRADIENTS[name])()
    z = np.asarray(system["positions.z"])
    data = ovito_compute(
        system,
        ColorCodingModifier(
            property="Position.Z",
            gradient=gradient,
            start_value=z.min(),
            end_value=z.max(),
        ),
    )
    ours = color_coding(system, "positions.z", name)
    np.testing.assert_allclose(
        ours["color"], np.asarray(data.particles.colors), atol=1e-6
    )
    data = ovito_compute(
        system,
        ColorCodingModifier(
            property="Position.X", gradient=gradient, start_value=2.0, end_value=8.0
        ),
    )
    ours = color_coding(system, "positions.x", name, (2.0, 8.0))
    np.testing.assert_allclose(
        ours["color"], np.asarray(data.particles.colors), atol=1e-6
    )


def test_color_coding_and_assign_color_of_a_selection():
    """The atoms selected colored, the others as they were: OVITO's colors
    of the elements are its own, so the others are compared by their
    color staying put."""
    system = alloy()
    chosen = system.where("positions.y > 7")
    data = ovito_compute(
        system,
        ExpressionSelectionModifier(expression="Position.Y > 7"),
        ColorCodingModifier(
            property="Position.Y",
            gradient=ColorCodingModifier.Jet(),
            start_value=7.0,
            end_value=14.0,
            only_selected=True,
        ),
    )
    ours = color_coding(system, "positions.y", "jet", (7.0, 14.0), atoms=chosen)
    np.testing.assert_allclose(
        ours["color"][chosen], np.asarray(data.particles.colors)[chosen], atol=1e-6
    )
    data = ovito_compute(
        system,
        ExpressionSelectionModifier(expression="Position.Y > 7"),
        AssignColorModifier(color=(0.2, 0.4, 0.9)),
    )
    ours = assign_color(system, (0.2, 0.4, 0.9), atoms=chosen)
    np.testing.assert_allclose(
        ours["color"][chosen], np.asarray(data.particles.colors)[chosen], atol=1e-6
    )
    np.testing.assert_allclose(
        ours["color"][~chosen],
        color_coding(system, "positions.y", atoms=[])["color"][~chosen],
    )


def test_color_by_type_groups_the_atoms_as_ovito():
    """OVITO colors by the colors of its types and mdapy by its own, so
    the atoms of one color must be the same."""
    system = alloy()
    data = ovito_compute(system, ColorByTypeModifier(property="Particle Type"))
    theirs = np.unique(np.asarray(data.particles.colors), axis=0, return_inverse=True)[
        1
    ]
    ours = np.unique(
        color_by_type(system, "element")["color"], axis=0, return_inverse=True
    )[1]
    # one color for each element in both, the same atoms in each
    pairs = set(zip(ours.ravel().tolist(), theirs.ravel().tolist()))
    assert len(pairs) == len(set(ours.ravel())) == len(set(theirs.ravel())) == 3


def test_selections():
    """System.where against OVITO's select type and expression selection."""
    system = alloy()
    data = ovito_compute(
        system, SelectTypeModifier(property="Particle Type", types={"Ni", "Al"})
    )
    np.testing.assert_array_equal(
        system.where("element in ('Ni', 'Al')"),
        np.asarray(data.particles.selection) != 0,
    )
    data = ovito_compute(
        system,
        ExpressionSelectionModifier(expression="Position.X > 5 && Position.Z < 8"),
    )
    np.testing.assert_array_equal(
        system.where("positions.x > 5 and positions.z < 8"),
        np.asarray(data.particles.selection) != 0,
    )
