# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest

import mdapy
from mdapy.analysis import REDUCTIONS, SpatialBinning
from mdapy.errors import InvalidParameter, MissingProperty


def scattered(n=400, pbc=True, seed=0):
    rng = np.random.default_rng(seed)
    box = mdapy.Box(
        [[10.0, 0, 0], [2.0, 9.0, 0], [1.0, 1.0, 8.0]], origin=[1, 2, 3], pbc=pbc
    )
    positions = box.to_cartesian(rng.uniform(-0.3, 1.3, (n, 3)))
    system = mdapy.System(positions, box, elements="Cu")
    return system.with_columns(charge=rng.normal(size=n))


def brute(system, prop, axes, bins, reduction):
    """Each bin by itself, from the fractional coordinates."""
    values = np.asarray(system[prop], dtype=float).reshape(len(system), -1)
    f = system.box.to_fractional(system["positions"])[:, axes]
    pbc = system.box.pbc[list(axes)]
    f[:, pbc] %= 1.0
    inside = ((f >= 0) & (f < 1)).all(axis=1)
    index = np.minimum((f * bins).astype(int), np.array(bins) - 1)
    out = np.full(tuple(bins) + (values.shape[1],), np.nan)
    counts = np.zeros(tuple(bins), dtype=int)
    for cell in np.ndindex(*bins):
        mine = inside & (index == cell).all(axis=1)
        counts[cell] = mine.sum()
        v = values[mine]
        if reduction == "sum":
            out[cell] = v.sum(axis=0)
        elif reduction == "density":
            out[cell] = v.sum(axis=0) / (system.box.volume / np.prod(bins))
        elif mine.any():
            out[cell] = getattr(np, reduction)(v, axis=0)
    return out, counts


# the reductions of any column; those about the centers of mass below
PLAIN = [r for r in REDUCTIONS if r not in ("mass_mean", "temperature", "stress")]


@pytest.mark.parametrize("reduction", PLAIN)
@pytest.mark.parametrize(
    "axes, bins", [((2,), (7,)), ((0, 1), (3, 5)), ((2, 0, 1), (2, 3, 4))]
)
@pytest.mark.parametrize("pbc", [True, (True, False, False)])
def test_against_each_bin(reduction, axes, bins, pbc):
    system = scattered(pbc=pbc)
    result = SpatialBinning("charge", bins, axes=axes, reduction=reduction).compute(
        system
    )
    expected, counts = brute(system, "charge", axes, bins, reduction)
    np.testing.assert_allclose(
        result.values, expected[..., 0], rtol=1e-12, equal_nan=True
    )
    np.testing.assert_array_equal(result.counts, counts)
    if not np.all(pbc):
        assert counts.sum() < len(system)  # the atoms outside the open directions
    for axis, n, centers in zip(axes, bins, result.centers):
        length = np.linalg.norm(system.box.matrix[axis])
        np.testing.assert_allclose(centers, (np.arange(n) + 0.5) * length / n)


def test_components_and_expressions():
    system = scattered()
    vector = SpatialBinning("positions", 4, axes="x").compute(system)
    assert vector.values.shape == (4, 3)
    z = SpatialBinning("positions.z", 4, axes="x").compute(system)
    np.testing.assert_allclose(vector.values[:, 2], z.values)
    doubled = SpatialBinning(
        "2 * charge", (3, 2), axes=("y", "z"), reduction="sum"
    ).compute(system)
    once = SpatialBinning("charge", 3, axes=("y", "z"), reduction="sum").compute(system)
    assert once.values.shape == (3, 3)
    np.testing.assert_allclose(doubled.values.shape, (3, 2))
    total = SpatialBinning("charge", 5, reduction="sum").compute(system)
    assert total.values.sum() == pytest.approx(system["charge"].sum())
    assert (
        repr(SpatialBinning("charge", 5))
        == "SpatialBinning('charge', (5,), axes=(2,), reduction='mean')"
    )


def test_empty():
    system = scattered().filter("charge > 100")
    result = SpatialBinning("charge", 3).compute(system)
    assert np.isnan(result.values).all() and not result.counts.any()
    assert (
        SpatialBinning("charge", 3, reduction="min").compute(system).values != 0
    ).all()


@pytest.mark.parametrize(
    "args, options, message",
    [
        (("charge", 5), {"axes": "w"}, "axis"),
        (("charge", 5), {"axes": (0, 0)}, "differ"),
        (("charge", 5), {"axes": (0, 1, 2, 0)}, "one to three"),
        (("charge", (5, 4)), {"axes": "z"}, "bins"),
        (("charge", 0), {}, "bins"),
        (("charge", 5), {"reduction": "median"}, "reduction"),
    ],
)
def test_errors(args, options, message):
    with pytest.raises(InvalidParameter, match=message):
        SpatialBinning(*args, **options)


def moving_crystal(seed=0):
    """Copper and nickel in 4 slabs along z, each moving as a whole at its own
    velocity, with thermal velocities about it."""
    from mdapy.build import build_crystal

    rng = np.random.default_rng(seed)
    crystal = build_crystal("fcc", 3.6, elements="Cu", repeat=(4, 4, 8))
    crystal = crystal.with_columns(element=list(rng.choice(["Cu", "Ni"], len(crystal))))
    slab = np.minimum(
        (crystal.box.to_fractional(crystal["positions"])[:, 2] * 4).astype(int), 3
    )
    drift = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, -1.0], [0.0, 5.0, 0.0], [-3.0, 1.0, 4.0]]
    )
    thermal = rng.normal(0, 3.0, (len(crystal), 3))
    return crystal.with_columns(velocities=drift[slab] + thermal), slab, thermal, drift


def test_the_mass_weighted_mean_is_the_velocity_of_the_center_of_mass():
    from mdapy.elements import atomic_masses

    system, slab, _, _ = moving_crystal()
    found = SpatialBinning("velocities", 4, reduction="mass_mean").compute(system)
    m = atomic_masses(list(system["element"]))
    v = system["velocities"]
    for b in range(4):
        expected = (m[slab == b, None] * v[slab == b]).sum(0) / m[slab == b].sum()
        np.testing.assert_allclose(found.values[b], expected, rtol=1e-12)
    # the masses of a mass column, as those of the elements
    massive = system.with_columns(mass=m).drop("element")
    again = SpatialBinning("velocities", 4, reduction="mass_mean").compute(massive)
    np.testing.assert_allclose(again.values, found.values, rtol=1e-14)


def test_the_temperature_leaves_out_the_motion_of_each_bin():
    from mdapy import units
    from mdapy.elements import atomic_masses

    system, slab, thermal, drift = moving_crystal()
    found = SpatialBinning("velocities", 4, reduction="temperature").compute(system)
    still = system.with_columns(velocities=thermal)
    at_rest = SpatialBinning("velocities", 4, reduction="temperature").compute(still)
    np.testing.assert_allclose(found.values, at_rest.values, rtol=1e-10)
    m = atomic_masses(list(system["element"]))
    for b in range(4):
        v = system["velocities"][slab == b]
        mb = m[slab == b]
        u = v - (mb[:, None] * v).sum(0) / mb.sum()
        expected = (
            (mb * (u * u).sum(1)).sum()
            * units.AMU_A2_PER_PS2_TO_EV
            / (3 * (len(mb) - 1) * units.BOLTZMANN)
        )
        assert found.values[b] == pytest.approx(expected, rel=1e-12)
    # one atom has no temperature about its own center of mass
    one = SpatialBinning("velocities", 64, reduction="temperature").compute(system)
    assert np.isnan(one.values[one.counts < 2]).all()


def test_the_stress_of_a_bin():
    """The stress of one bin is that of the box: the virial of the potential
    and the kinetic stress about the center of mass; the motion of a bin
    as a whole is no stress."""
    from mdapy import units
    from mdapy.calculators import EAM

    system, slab, thermal, _ = moving_crystal()
    potential = EAM.zhou(["Cu", "Ni"])
    result = potential.compute(system, ("virial", "atomic_virials"))
    system = result.attach(system)
    whole = SpatialBinning("atomic_virials", 1, reduction="stress").compute(
        system.with_columns(velocities=np.zeros((len(system), 3)))
    )
    np.testing.assert_allclose(whole.values[0], result.stress, rtol=1e-10, atol=1e-10)
    moving = SpatialBinning("atomic_virials", 4, reduction="stress").compute(system)
    still = SpatialBinning("atomic_virials", 4, reduction="stress").compute(
        system.with_columns(velocities=thermal)
    )
    # the thermal velocities about their own center of mass: nearly the same
    assert moving.values.shape == (4, 3, 3)
    np.testing.assert_allclose(moving.values, still.values, rtol=0, atol=0.05)
    m = mdapy.elements.atomic_masses(list(system["element"]))
    v = system["velocities"][slab == 1]
    u = v - (m[slab == 1, None] * v).sum(0) / m[slab == 1].sum()
    w = system["atomic_virials"][slab == 1].sum(0)
    kinetic = np.einsum("i,ij,ik->jk", m[slab == 1], u, u) * units.AMU_A2_PER_PS2_TO_EV
    volume = system.box.volume / 4
    expected = -(w + kinetic) / volume * units.EV_PER_A3_TO_GPA
    np.testing.assert_allclose(moving.values[1], expected, rtol=1e-12)


def test_the_moving_reductions_need_their_columns():
    system, *_ = moving_crystal()
    with pytest.raises(InvalidParameter, match="velocities"):
        SpatialBinning("positions.z", 4, reduction="temperature").compute(system)
    with pytest.raises(InvalidParameter, match="virials"):
        SpatialBinning("velocities", 4, reduction="stress").compute(system)
    with pytest.raises(MissingProperty, match="masses"):
        SpatialBinning("velocities", 4, reduction="mass_mean").compute(
            system.drop("element")
        )
    virials = np.zeros((len(system), 3, 3))
    with pytest.raises(MissingProperty, match="velocities"):
        SpatialBinning("atomic_virials", 4, reduction="stress").compute(
            system.drop("velocities").with_columns(atomic_virials=virials)
        )


@pytest.mark.parametrize("reduction", PLAIN)
def test_bounds_bin_a_fixed_range(reduction):
    """Bounds along the third vector, open, in a tilted box away from the
    origin: the bins of a brute force over the place of each atom along it
    from (0, 0, 0), with the atoms outside the box but inside the bounds."""
    system = scattered(pbc=(True, True, False))
    lo, hi, n = -2.0, 11.0, 6
    result = SpatialBinning(
        "charge", n, axes=2, reduction=reduction, bounds=[(lo, hi)]
    ).compute(system)
    box = system.box
    length = np.linalg.norm(box.matrix[2])
    place = (box.to_fractional(system["positions"])[:, 2]
             - box.to_fractional(np.zeros((1, 3)))[0, 2]) * length  # fmt: skip
    charge = system["charge"]
    width = (hi - lo) / n
    expected = np.full(n, np.nan)
    counts = np.zeros(n, dtype=int)
    for k in range(n):
        mine = (place >= lo + k * width) & (place < lo + (k + 1) * width)
        counts[k] = mine.sum()
        v = charge[mine]
        if reduction == "sum":
            expected[k] = v.sum()
        elif reduction == "density":
            expected[k] = v.sum() / (box.volume / length * width)
        elif mine.any():
            expected[k] = getattr(np, reduction)(v)
    np.testing.assert_array_equal(result.counts, counts)
    np.testing.assert_allclose(result.values, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.centers[0], lo + (np.arange(n) + 0.5) * width)
    # the box holds fewer: its own bins leave the atoms outside it out
    assert counts.sum() > SpatialBinning("charge", n).compute(system).counts.sum()


def test_bounds_stay_as_the_box_grows():
    """The same atoms in a box grown along its open vector, as molecular
    dynamics grows it: the same bins and values with bounds."""
    rng = np.random.default_rng(3)
    box = mdapy.Box(np.diag([10.0, 10.0, 20.0]), pbc=(True, True, False))
    positions = box.to_cartesian(rng.uniform(0, 1, (300, 3)))
    positions[:20, 2] += 12.0  # flown beyond the top
    system = mdapy.System(positions, box, elements="Cu").with_columns(
        velocities=rng.normal(size=(300, 3)), charge=rng.normal(size=300)
    )
    grown = system.with_box(
        mdapy.Box(np.diag([10.0, 10.0, 33.0]), origin=[0, 0, -1.0], pbc=box.pbc)
    )
    for reduction in ("mean", "density", "temperature"):
        column = "velocities" if reduction == "temperature" else "charge"
        binning = SpatialBinning(column, 8, reduction=reduction, bounds=[(-1.0, 33.0)])
        a, b = binning.compute(system), binning.compute(grown)
        np.testing.assert_array_equal(a.counts, b.counts)
        np.testing.assert_allclose(a.values, b.values, rtol=1e-12, equal_nan=True)
        assert a.counts.sum() == 300
    # bounds that are the box give its own bins
    plain = SpatialBinning("charge", 5, axes=(0, 2)).compute(grown)
    bounded = SpatialBinning("charge", 5, axes=(0, 2), bounds=[None, (-1.0, 32.0)])
    np.testing.assert_allclose(
        bounded.compute(grown).values, plain.values, rtol=1e-12, equal_nan=True
    )


@pytest.mark.parametrize(
    "bounds, message",
    [
        ([(0.0, 1.0), (0.0, 1.0)], "one range or None for each axis"),
        ([(1.0, 0.0)], "lo < hi"),
        ([(0.0, np.inf)], "finite"),
        ([(0.0,)], r"\(lo, hi\) or None"),
        ([1.0], r"\(lo, hi\) or None"),
    ],
)
def test_bounds_errors(bounds, message):
    with pytest.raises(InvalidParameter, match=message):
        SpatialBinning("charge", 4, bounds=bounds)


def test_bounds_in_the_repr():
    assert "bounds=((0.0, 5.0),)" in repr(SpatialBinning("charge", 4, bounds=[(0, 5)]))
    assert "bounds" not in repr(SpatialBinning("charge", 4))
