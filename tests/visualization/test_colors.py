# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.errors import InvalidParameter
from mdapy.visualization import (
    COLORMAPS,
    LABEL_COLORS,
    assign_color,
    atom_colors,
    atom_radii,
    color_by_type,
    color_coding,
    colormap,
    element_colors,
    element_radii,
    render,
)


def alloy():
    system = build_crystal("fcc", 3.6, elements="Cu", repeat=(3, 3, 3))
    rng = np.random.default_rng(9)
    return system.with_columns(
        element=rng.choice(["Cu", "Ni", "Al"], size=len(system)),
        value=rng.normal(size=len(system)),
        kind=rng.integers(0, 15, size=len(system)),
    )


@pytest.mark.parametrize("name", COLORMAPS)
def test_colormaps_end_to_end(name):
    """Values beyond the range take the colors of its ends; colors stay in
    0 to 1."""
    colors = colormap([-1.0, 0.0, 0.5, 1.0, 2.0], name, (0, 1))
    np.testing.assert_allclose(colors[0], colors[1])
    np.testing.assert_allclose(colors[4], colors[3])
    assert colors.shape == (5, 3) and colors.min() >= 0 and colors.max() <= 1


def test_known_colors_of_the_maps():
    t = [0.0, 0.25, 0.5, 0.75, 1.0]
    np.testing.assert_allclose(
        colormap(t, "rainbow", (0, 1))[[0, -1]], [[0.2, 0, 1], [1, 0, 0]], atol=1e-12
    )
    np.testing.assert_allclose(colormap(t, "blue_white_red", (0, 1))[2], [1, 1, 1])
    np.testing.assert_allclose(
        colormap(t, "hot", (0, 1))[[0, -1]], [[0, 0, 0], [1, 1, 1]]
    )
    np.testing.assert_allclose(colormap(t, "jet", (0, 1))[2], [0.5, 1, 0.5])
    np.testing.assert_allclose(colormap(t, "gray", (0, 1))[1], [0.25] * 3)
    np.testing.assert_allclose(
        colormap([0.0], "magma", (0, 1))[0], [0.001462, 0.000466, 0.013866]
    )


def test_color_by_type():
    system = alloy()
    colored = color_by_type(system, "kind")
    kinds = np.asarray(system["kind"])
    order = np.unique(kinds)
    for rank, value in enumerate(order):
        np.testing.assert_allclose(
            colored["color"][kinds == value],
            LABEL_COLORS[rank % 12][None].repeat((kinds == value).sum(), 0),
        )
    by_element = color_by_type(system, "element")
    # Al, Cu, Ni in alphabetical order
    np.testing.assert_allclose(
        by_element["color"][np.asarray(system["element"]) == "Ni"][0], LABEL_COLORS[2]
    )
    some = color_by_type(system, "kind", atoms=[0, 1])
    np.testing.assert_allclose(
        some["color"][2:], element_colors(np.asarray(system["element"]))[2:]
    )
    with pytest.raises(InvalidParameter, match="components"):
        color_by_type(system, "positions")


def test_color_coding():
    system = alloy()
    values = np.asarray(system["value"])
    colored = color_coding(system, "value", "viridis")
    np.testing.assert_allclose(colored["color"], colormap(values, "viridis"))
    ranged = color_coding(system, "value * 2", value_range=(-1, 1))
    np.testing.assert_allclose(
        ranged["color"], colormap(2 * values, "rainbow", (-1, 1))
    )
    chosen = values > 0
    some = color_coding(system, "value", atoms=chosen)
    low, high = values[chosen].min(), values[chosen].max()
    np.testing.assert_allclose(
        some["color"][chosen], colormap(values[chosen], "rainbow", (low, high))
    )
    np.testing.assert_allclose(some["color"][~chosen], atom_colors(system)[~chosen])
    with pytest.raises(InvalidParameter, match="not numbers"):
        color_coding(system, "element")
    with pytest.raises(InvalidParameter, match="components"):
        color_coding(system, "positions")


def test_assign_color_and_the_color_column():
    system = alloy()
    red = assign_color(system, (1.0, 0.0, 0.0), atoms=system.where(element="Ni"))
    nickel = np.asarray(system["element"]) == "Ni"
    np.testing.assert_allclose(red["color"][nickel], [[1, 0, 0]] * nickel.sum())
    # the color column is what the atoms look like from now on
    np.testing.assert_allclose(atom_colors(red), red["color"])
    blue = assign_color(red, "#0000ff", atoms=~nickel)
    np.testing.assert_allclose(blue["color"][nickel], [[1, 0, 0]] * nickel.sum())
    np.testing.assert_allclose(blue["color"][~nickel], [[0, 0, 1]] * (~nickel).sum())
    np.testing.assert_allclose(
        render(blue, width=40, height=40),
        render(system, color=blue["color"], width=40, height=40),
    )


def test_colors_and_radii_of_elements_given():
    system = alloy()
    names = np.asarray(system["element"])
    colors = atom_colors(system, elements={"Cu": "#000000", "Ni": (0.0, 1.0, 0.0)})
    np.testing.assert_allclose(colors[names == "Cu"][0], [0, 0, 0])
    np.testing.assert_allclose(colors[names == "Ni"][0], [0, 1, 0])
    np.testing.assert_allclose(colors[names == "Al"][0], element_colors(["Al"])[0])
    radii = atom_radii(system, elements={"Al": 2.0}, scale=0.5)
    np.testing.assert_allclose(radii[names == "Al"], 1.0)
    np.testing.assert_allclose(radii[names == "Cu"], 0.5 * element_radii(["Cu"])[0])
    np.testing.assert_allclose(atom_radii(system, "vdw"), element_radii(names, "vdw"))
    sized = system.with_columns(radius=np.full(len(system), 0.7))
    np.testing.assert_allclose(atom_radii(sized, scale=2), 1.4)
    np.testing.assert_allclose(
        render(sized, width=40, height=40),
        render(system, radius=0.7, width=40, height=40),
    )
    plain = mdapy.System(np.zeros((2, 3)), mdapy.Box(5.0))
    np.testing.assert_allclose(atom_radii(plain), element_radii(["X", "X"]))
