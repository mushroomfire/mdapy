# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Reductions, histograms, bond lengths, time series and the columns
carried between frames against OVITO, live."""

import numpy as np
import pytest

import mdapy
from mdapy import transform
from mdapy.analysis import (
    OPERATIONS,
    BondLengthDistribution,
    FrameDifference,
    Histogram,
    TimeSeries,
    reduce,
)

ovito = pytest.importorskip("ovito")
from ovito.io import import_file  # noqa: E402
from ovito.modifiers import (  # noqa: E402
    BondLengthDistributionModifier,
    CreateBondsModifier,
    DifferenceBetweenFramesModifier,
    ExpressionSelectionModifier,
    FreezePropertyModifier,
    HistogramModifier,
    ReducePropertyModifier,
    SmoothTrajectoryModifier,
    TimeSeriesModifier,
)

from ovito_reference import ovito_compute, ovito_source  # noqa: E402

OVITO_OPERATIONS = {
    "mean": "Mean",
    "sum": "Sum",
    "min": "Min",
    "max": "Max",
    "median": "Median",
    "variance": "Variance",
    "std": "StdDev",
    "nonzero": "Nonzero",
    "argmin": "ArgMin",
    "argmax": "ArgMax",
}


def alloy():
    """A random alloy of 300 atoms in a triclinic box."""
    box = mdapy.Box([[14.0, 0, 0], [2.0, 13.0, 0], [-1.0, 1.0, 15.0]])
    rng = np.random.default_rng(11)
    elements = rng.choice(["Cu", "Ni", "Al"], size=300)
    return mdapy.System(box.to_cartesian(rng.random((300, 3))), box, elements=elements)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_reductions(operation):
    """Of the coordinates as a whole and of one of them."""
    system = alloy()
    for column, ovito_name in (
        ("positions.y", "Position.Y"),
        ("positions", "Position"),
    ):
        data = ovito_compute(
            system,
            ReducePropertyModifier(
                property=ovito_name,
                reduction_operation=getattr(
                    ReducePropertyModifier.Operation, OVITO_OPERATIONS[operation]
                ),
            ),
        )
        found = {
            k: v for k, v in data.attributes.items() if k.startswith("ReduceProperty")
        }
        ours = np.atleast_1d(reduce(system, column, operation))
        theirs = [found[k] for k in sorted(found)]
        np.testing.assert_allclose(ours, theirs, rtol=1e-12)


def test_histograms():
    """Automatic and given ranges, of all atoms and of those selected."""
    system = alloy()
    data = ovito_compute(system, HistogramModifier(property="Position.Z", bin_count=37))
    table = data.tables["histogram[Position.Z]"]
    ours = Histogram("positions.z", bins=37).compute(system)
    np.testing.assert_allclose(ours.center, table.xy()[:, 0], rtol=1e-12)
    np.testing.assert_array_equal(ours.counts, table.xy()[:, 1])
    selected = system.where("positions.x > 6")
    data = ovito_compute(
        system,
        ExpressionSelectionModifier(expression="Position.X > 6"),
        HistogramModifier(
            property="Position.Z",
            bin_count=10,
            fix_xrange=True,
            xrange_start=2.0,
            xrange_end=12.0,
            only_selected=True,
        ),
    )
    table = data.tables["histogram[Position.Z]"]
    ours = Histogram("positions.z", bins=10, range=(2, 12)).compute(
        system, atoms=selected
    )
    np.testing.assert_allclose(ours.center, table.xy()[:, 0], rtol=1e-12)
    np.testing.assert_array_equal(ours.counts, table.xy()[:, 1])


def test_bond_lengths():
    """The bonds OVITO creates, the same as create_bonds, measured by both,
    in total and for each pair of elements."""
    system = alloy()
    from ovito.pipeline import Pipeline

    pipeline = Pipeline(source=ovito_source(system))
    pipeline.modifiers.append(CreateBondsModifier(cutoff=3.2))
    pipeline.modifiers.append(
        BondLengthDistributionModifier(
            bins=80,
            length_cutoff=3.5,
            partition=BondLengthDistributionModifier.Partition.ByParticleType,
        )
    )
    data = pipeline.compute()
    pairs = np.asarray(data.particles.bonds.topology)
    bonded = transform.create_bonds(system, 3.2)
    # the same bonds as OVITO's create bonds
    ours = np.asarray(bonded.topology.bonds["atoms"].to_list())
    assert set(map(tuple, ours)) == {(min(a, b), max(a, b)) for a, b in pairs}
    bonded = system.with_topology(mdapy.Topology(bonds=pairs))
    ours = BondLengthDistribution(bins=80, cutoff=3.5).compute(bonded)
    table = data.tables["bond-length-distr"]
    np.testing.assert_allclose(ours.length, table.xy()[:, 0], rtol=1e-12)
    names = [c for c in table.y.component_names]
    for k, name in enumerate(names):
        a, b = name.split("-")
        np.testing.assert_array_equal(ours.pair(a, b), np.asarray(table.y)[:, k])
    np.testing.assert_array_equal(ours.counts, np.asarray(table.y).sum(axis=1))


def walk(path, count=9, seed=2):
    """Atoms of a random walk through a periodic box with a value each,
    written as a dump, each frame in another order."""
    box = mdapy.Box([[12.0, 0, 0], [1.0, 11.0, 0], [0.0, -1.0, 13.0]])
    rng = np.random.default_rng(seed)
    start = box.to_cartesian(rng.random((30, 3)))
    steps = np.cumsum(rng.normal(scale=0.8, size=(count, 30, 3)), axis=0)
    frames = []
    for t in range(count):
        wrapped, _ = box.wrap(start + steps[t])
        frame = mdapy.System(wrapped, box, elements="Ar").with_columns(
            id=np.arange(1, 31), value=rng.normal(size=30)
        )
        frame = frame.with_table(frame.table[rng.permutation(30)])
        frames.append(frame.with_info(timestep=10 * t))
    mdapy.write(path, frames)
    return frames


def by_id(frame, column):
    ids = np.asarray(frame["id"])
    return np.asarray(frame[column])[np.argsort(ids)]


def test_time_series(tmp_path):
    frames = walk(tmp_path / "walk.dump")
    pipeline = import_file(str(tmp_path / "walk.dump"))
    pipeline.modifiers.append(
        ReducePropertyModifier(
            property="value",
            reduction_operation=ReducePropertyModifier.Operation.StdDev,
        )
    )
    pipeline.modifiers.append(
        TimeSeriesModifier(operate_on="ReduceProperty.particles.StdDev.value")
    )
    table = pipeline.compute().tables["time-series"]
    ours = TimeSeries("value", "std").compute(frames)
    np.testing.assert_allclose(ours.values, table.xy()[:, 1], rtol=1e-12)
    np.testing.assert_array_equal(ours.frame, table.xy()[:, 0])


def test_difference_frozen_and_smoothed(tmp_path):
    """The positions and the value smoothed over five frames, the value
    minus that of frame 2 and the value of frame 2, through the smoothing:
    the atoms matched by id."""
    frames = walk(tmp_path / "walk.dump")
    pipeline = import_file(str(tmp_path / "walk.dump"), sort_particles=True)
    pipeline.modifiers.append(SmoothTrajectoryModifier(window_size=5))
    pipeline.modifiers.append(
        DifferenceBetweenFramesModifier(
            operate_on="particles:particles/value", reference_frame=2
        )
    )
    pipeline.modifiers.append(
        FreezePropertyModifier(
            source_property="value", destination_property="value_frozen", freeze_at=2
        )
    )
    smoothed = transform.smooth_trajectory(frames, 5)
    box = frames[0].box
    for k in (0, 4, 8):
        data = pipeline.compute(k)
        change = smoothed[k].with_columns(
            FrameDifference("value").compute(smoothed[k], smoothed[2])
        )
        np.testing.assert_allclose(
            by_id(change, "value_change"),
            np.asarray(data.particles["value (Difference)"]),
            rtol=1e-12,
            atol=1e-12,
        )
        frozen = transform.freeze(smoothed[k], smoothed[2], "value")
        np.testing.assert_allclose(
            by_id(frozen, "value_frozen"),
            np.asarray(data.particles["value_frozen"]),
            rtol=1e-12,
        )
        # the same points up to lattice vectors
        step = by_id(smoothed[k], "positions") - np.asarray(data.particles.positions)
        np.testing.assert_allclose(box.minimum_image(step), 0, atol=1e-9)
