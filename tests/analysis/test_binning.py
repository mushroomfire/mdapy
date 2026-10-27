# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest

import mdapy
from mdapy.analysis import REDUCTIONS, SpatialBinning
from mdapy.errors import InvalidParameter


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


@pytest.mark.parametrize("reduction", REDUCTIONS)
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
