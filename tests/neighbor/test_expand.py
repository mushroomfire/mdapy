# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Selections grown by neighbors: as OVITO's expand selection."""

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.errors import InvalidParameter, MissingProperty
from mdapy.neighbor import NeighborList, expand_selection


@pytest.fixture(scope="module")
def shaken():
    """Copper in a triclinic box, shaken so that the nearest neighbors of
    each atom are told apart, with bonds between neighbors closer than
    2.7 Å."""
    crystal = build_crystal("fcc", 3.615, elements="Cu", repeat=(4, 4, 4))
    box = mdapy.Box(crystal.box.matrix + [[0, 0, 0], [2.0, 0, 0], [0, 0, 0]])
    rng = np.random.default_rng(3)
    moved = crystal.with_box(box, scale_positions=True)
    moved = moved.with_columns(
        positions=moved["positions"] + rng.normal(0, 0.08, (len(moved), 3))
    )
    nl = NeighborList.build(moved, 2.7)
    i = np.repeat(np.arange(len(moved)), np.diff(nl.offsets))
    pairs = np.c_[i, nl.j][i < nl.j]
    return moved.with_topology(mdapy.Topology(bonds=pairs))


SELECTED = [0, 17, 100]


@pytest.mark.parametrize(
    "options",
    [
        {"cutoff": 2.9},
        {"cutoff": 2.9, "iterations": 2},
        {"num_neighbors": 4},
        {"num_neighbors": 4, "iterations": 3},
        {"bonds": True, "iterations": 2},
    ],
)
def test_as_ovito(shaken, options):
    pytest.importorskip("ovito")
    from ovito.modifiers import ExpandSelectionModifier
    from ovito.pipeline import Pipeline

    from ovito_reference import ovito_source

    ours = expand_selection(shaken, SELECTED, **options)
    source = ovito_source(shaken)
    particles = source.data.particles_
    selection = np.zeros(len(shaken), dtype=np.int32)
    selection[SELECTED] = 1
    particles.create_property("Selection", data=selection)
    if options.get("bonds"):
        pairs = np.asarray(shaken.topology.bonds["atoms"].to_list())
        bonds = particles.create_bonds(count=len(pairs))
        bonds.create_property("Topology", data=pairs)
        mode = ExpandSelectionModifier.ExpansionMode.Bonded
    elif "cutoff" in options:
        mode = ExpandSelectionModifier.ExpansionMode.Cutoff
    else:
        mode = ExpandSelectionModifier.ExpansionMode.Nearest
    modifier = ExpandSelectionModifier(
        mode=mode, iterations=options.get("iterations", 1)
    )
    if "cutoff" in options:
        modifier.cutoff = options["cutoff"]
    if "num_neighbors" in options:
        modifier.num_neighbors = options["num_neighbors"]
    pipeline = Pipeline(source=source)
    pipeline.modifiers.append(modifier)
    data = pipeline.compute()
    np.testing.assert_array_equal(ours, np.asarray(data.particles.selection) != 0)
    assert ours.sum() > len(SELECTED)


def test_shells_of_a_perfect_crystal():
    """In fcc the first shell has 12 atoms and the second step reaches 42
    more; no step keeps the selection; a mask selects as indices do."""
    crystal = build_crystal("fcc", 3.615, elements="Cu", repeat=(4, 4, 4))
    assert expand_selection(crystal, [0], cutoff=3.0).sum() == 13
    assert expand_selection(crystal, [0], cutoff=3.0, iterations=2).sum() == 55
    np.testing.assert_array_equal(
        expand_selection(crystal, [5], cutoff=3.0, iterations=0),
        np.arange(len(crystal)) == 5,
    )
    mask = np.arange(len(crystal)) == 5
    np.testing.assert_array_equal(
        expand_selection(crystal, mask, num_neighbors=12),
        expand_selection(crystal, [5], cutoff=3.0),
    )
    # it stops growing when the whole crystal is selected
    assert expand_selection(crystal, [0], cutoff=3.0, iterations=100).all()


@pytest.mark.parametrize(
    "options, error, message",
    [
        ({}, InvalidParameter, "exactly one"),
        ({"cutoff": 3.0, "bonds": True}, InvalidParameter, "exactly one"),
        ({"cutoff": 3.0, "iterations": -1}, InvalidParameter, "not be negative"),
        ({"cutoff": 3.0, "iterations": 1.5}, InvalidParameter, "an integer"),
        ({"bonds": True}, MissingProperty, "no bonds"),
        ({"cutoff": -1.0}, InvalidParameter, "cutoff"),
    ],
)
def test_errors(options, error, message):
    crystal = build_crystal("fcc", 3.615, elements="Cu", repeat=(2, 2, 2))
    with pytest.raises(error, match=message):
        expand_selection(crystal, [0], **options)
