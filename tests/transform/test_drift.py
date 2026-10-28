# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest

import mdapy
from mdapy.elements import atomic_masses
from mdapy.errors import InvalidParameter
from mdapy.transform import remove_drift

BOX = mdapy.Box([[12.0, 0, 0], [2.0, 11.0, 0], [-1.0, 1.5, 13.0]])


def drifting(flags=True, ids=True, masses=False, step=1.0, seed=3):
    """30 atoms of two elements walking with a common drift through a
    periodic triclinic box over 8 frames, wrapped; with ids, each frame
    lists them in another order. Also the unwrapped walk."""
    rng = np.random.default_rng(seed)
    start = BOX.to_cartesian(rng.random((30, 3)))
    steps = step * (rng.normal(scale=1.5, size=(8, 30, 3)) + [2.0, -1.0, 0.7])
    walk = start + np.cumsum(steps, axis=0)
    elements = ["Cu"] * 20 + ["Ar"] * 10
    frames = []
    for positions in walk:
        wrapped, image = BOX.wrap(positions)
        frame = mdapy.System(wrapped, BOX, elements=elements)
        if flags:
            frame = frame.with_columns(image=image)
        if masses:
            frame = frame.with_columns(mass=np.linspace(1.0, 4.0, 30))
        if ids:
            frame = frame.with_columns(id=np.arange(1, 31))
            frame = frame.with_table(frame.table[rng.permutation(30)])
        frames.append(frame)
    return frames, walk


def unwrapped(frame):
    """The positions of a frame by its ids, unwrapped by its image flags."""
    order = np.argsort(frame["id"])
    return (frame["positions"] + frame["image"] @ frame.box.matrix)[order]


@pytest.mark.parametrize("reference", [0, 3, -1])
def test_the_center_of_mass_stays(reference):
    """Every frame moved by one translation, after which the center of mass
    of the unwrapped walk is that of the reference frame."""
    frames, walk = drifting()
    corrected = remove_drift(frames, reference)
    assert len(corrected) == len(frames)
    assert corrected.shifts.shape == (8, 3) and not corrected.shifts.flags.writeable
    m = atomic_masses(["Cu"] * 20 + ["Ar"] * 10)
    center = m @ walk[reference] / m.sum()
    shifts = []
    for before, after in zip(frames, corrected):
        shift = after["positions"] - before["positions"]
        np.testing.assert_allclose(shift, shift[:1].repeat(30, axis=0), atol=1e-12)
        np.testing.assert_allclose(shift[0], corrected.shifts[len(shifts)], atol=1e-12)
        shifts.append(shift[0])
        np.testing.assert_allclose(m @ unwrapped(after) / m.sum(), center, atol=1e-10)
        assert after.table.drop("positions").equals(before.table.drop("positions"))
        assert after.box == before.box


def test_masses_and_atoms():
    """The mass column before the elements; equal weights without either;
    a subset of the atoms of the first frame."""
    frames, walk = drifting(masses=True)
    m = np.linspace(1.0, 4.0, 30)
    for after, positions in zip(remove_drift(frames), walk):
        np.testing.assert_allclose(
            m @ unwrapped(after) / m.sum(), m @ walk[0] / m.sum()
        )
    bare = [frame.with_table(frame.table.drop("element", "mass")) for frame in frames]
    for after in remove_drift(bare, 2):
        np.testing.assert_allclose(unwrapped(after).mean(axis=0), walk[2].mean(axis=0))
    first = np.asarray(frames[0]["id"]) - 1  # the atom of the walk in each row
    chosen = np.flatnonzero(first < 5)
    for after in remove_drift(bare, atoms=chosen):
        np.testing.assert_allclose(
            unwrapped(after)[:5].mean(axis=0), walk[0][:5].mean(axis=0)
        )
    mask = np.zeros(30, dtype=bool)
    mask[chosen] = True
    for a, b in zip(remove_drift(bare, atoms=mask), remove_drift(bare, atoms=chosen)):
        np.testing.assert_array_equal(a["positions"], b["positions"])


def test_without_unwrapping():
    """The center of mass of the positions as they are."""
    frames, _ = drifting(ids=False)
    m = atomic_masses(["Cu"] * 20 + ["Ar"] * 10)
    center = m @ frames[0]["positions"] / m.sum()
    for after in remove_drift(frames, unwrap=False):
        np.testing.assert_allclose(m @ after["positions"] / m.sum(), center, atol=1e-10)


def test_followed_without_image_flags():
    """Small steps followed by the nearest image give the same as the image
    flags."""
    flagged, _ = drifting(ids=False, step=0.2)
    bare = [frame.with_table(frame.table.drop("image")) for frame in flagged]
    for a, b in zip(remove_drift(bare, 4), remove_drift(flagged, 4)):
        np.testing.assert_allclose(a["positions"], b["positions"], atol=1e-10)


def test_a_trajectory_and_slices(tmp_path):
    """A trajectory read lazily gives the same frames as a list; slices are
    the corrected frames they select."""
    frames, _ = drifting()
    mdapy.write(tmp_path / "walk.dump", frames)
    with mdapy.read_trajectory(tmp_path / "walk.dump") as trajectory:
        lazy = remove_drift(trajectory, 2)
        eager = remove_drift(frames, 2)
        for a, b in zip(lazy, eager):
            np.testing.assert_allclose(a["positions"], b["positions"], atol=1e-6)
        part = lazy[1:6:2]
        assert len(part) == 3
        np.testing.assert_array_equal(part.shifts, lazy.shifts[1:6:2])
        for a, k in zip(part, (1, 3, 5)):
            np.testing.assert_array_equal(a["positions"], lazy[k]["positions"])
        np.testing.assert_array_equal(
            lazy[np.int64(-1)]["positions"], lazy[7]["positions"]
        )
        with pytest.raises(IndexError):
            lazy[8]


def test_errors():
    frames, _ = drifting()
    with pytest.raises(InvalidParameter, match="no frames"):
        remove_drift([])
    with pytest.raises(InvalidParameter, match="reference frame 8"):
        remove_drift(frames, 8)
    with pytest.raises(InvalidParameter, match="no mass"):
        remove_drift(frames, atoms=[])
    massless = [frame.with_columns(mass=np.zeros(30)) for frame in frames]
    with pytest.raises(InvalidParameter, match="no mass"):
        remove_drift(massless)
    with pytest.raises(InvalidParameter, match="frame 1 has 29 atoms"):
        remove_drift([frames[0], mdapy.System(frames[1]["positions"][:29], BOX)])
    with pytest.raises(InvalidParameter, match="indices"):
        remove_drift(frames, atoms=[30])
