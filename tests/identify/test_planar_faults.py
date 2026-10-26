# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Planar faults of fcc crystals, against OVITO's Identify fcc planar faults
atom by atom where OVITO is installed."""

import dataclasses

import numpy as np
import pytest

from mdapy.build import build_crystal, edge_dislocation, slip, twin_lamellae
from mdapy.calculators import EAM
from mdapy.errors import InvalidParameter
from mdapy.identify import PlanarFaults, PolyhedralTemplateMatching
from mdapy.simulation import minimize
from mdapy.transform import replicate

A = 3.615
ORIENTATION = [(1, -1, 0), (1, 1, -2), (1, 1, 1)]  # z along [111]
SPACING = A / np.sqrt(3)
SHOCKLEY = np.array([0, A / np.sqrt(6), 0])  # a/6 [11-2]


def faulted(planes, jitter=0.0, seed=0):
    """fcc Cu with z along [111], slipped by a Shockley partial above the
    atomic planes given by number."""
    system = replicate(
        build_crystal("fcc", A, elements="Cu", orientation=ORIENTATION), (6, 4, 8)
    )
    height = system.box.matrix[2, 2]
    for plane in planes:
        system = slip(system, SHOCKLEY, height=(plane + 0.25) * SPACING / height)
    if jitter:
        noise = np.random.default_rng(seed).normal(0, jitter, (len(system), 3))
        system = system.with_columns(positions=system["positions"] + noise)
    return system


def dislocation():
    """A relaxed edge dislocation of Cu, split into two Shockley partials."""
    cell = build_crystal("fcc", A, elements="Cu", orientation=ORIENTATION)
    system = edge_dislocation(cell, (30, 1, 12))
    return minimize(system, EAM.zhou(["Cu"]), max_force=1e-3, max_steps=5000).system


CASES = {
    "intrinsic": lambda: faulted([12]),
    "twin": lambda: faulted([12, 13]),
    "three layers": lambda: faulted([12, 14]),
    "several": lambda: faulted([4, 5, 6, 12, 16, 17]),
    "hot": lambda: faulted([5, 10, 11, 18], jitter=0.15, seed=1),
    "hotter": lambda: faulted([5, 10, 11, 18, 21, 22, 23], jitter=0.18, seed=1),
    "melting": lambda: faulted([5, 10, 11, 18, 21, 22, 23], jitter=0.2, seed=1),
    "lamellae": lambda: replicate(
        twin_lamellae(
            build_crystal("fcc", A, elements="Cu", orientation=ORIENTATION), 4
        ),
        (6, 4, 2),
    ),
    "dislocation": dislocation,
}


def test_faults_of_slips():
    counts = PlanarFaults().compute(faulted([12])).counts()
    assert (
        counts["intrinsic stacking fault"] == 2 * 48 and counts["non-hcp"] == 1152 - 96
    )
    counts = PlanarFaults().compute(faulted([12, 13])).counts()
    assert counts["coherent twin boundary"] == 2 * 48
    lamellae = PlanarFaults().compute(CASES["lamellae"]())
    assert lamellae.counts()["coherent twin boundary"] == 4 * 48
    # a twin boundary: one layer of atoms, sqrt(3)/2 d^2 each
    area = lamellae.areas()["coherent twin boundary"]
    box = CASES["lamellae"]().box.matrix
    assert area == pytest.approx(4 * box[0, 0] * box[1, 1])
    assert lamellae.columns()["planar_fault"] is lamellae.fault


def test_extrinsic_stacking_faults():
    # two Shockley partials on neighboring planes: two planes of hcp atoms
    # with one of fcc between, two twin boundaries but for the option
    extrinsic = PlanarFaults(extrinsic=True)
    result = extrinsic.compute(faulted([12, 13]))
    assert result.counts()["extrinsic stacking fault"] == 96
    assert result.counts()["coherent twin boundary"] == 0
    box = faulted([12, 13]).box.matrix
    assert result.areas()["extrinsic stacking fault"] == pytest.approx(
        box[0, 0] * box[1, 1]
    )
    # beside an intrinsic fault and a twin of three planes, which stay
    counts = extrinsic.compute(faulted([4, 5, 12])).counts()
    assert counts["extrinsic stacking fault"] == 96
    assert counts["intrinsic stacking fault"] == 96
    counts = extrinsic.compute(faulted([4, 5, 12, 13, 14])).counts()
    assert counts["extrinsic stacking fault"] == counts["coherent twin boundary"] == 96
    for planes in ([12], [12, 14], [12, 13, 14]):
        system = faulted(planes)
        np.testing.assert_array_equal(
            extrinsic.compute(system).fault, PlanarFaults().compute(system).fault
        )
    lamellae = extrinsic.compute(CASES["lamellae"]())
    assert lamellae.counts()["extrinsic stacking fault"] == 0


def test_given_template_matching():
    system = faulted([4, 5, 12])
    ptm = PolyhedralTemplateMatching(template_neighbors=True).compute(system)
    np.testing.assert_array_equal(
        PlanarFaults().compute(system, ptm=ptm).fault,
        PlanarFaults().compute(system).fault,
    )
    without = PolyhedralTemplateMatching().compute(system)
    with pytest.raises(InvalidParameter, match="template_neighbors=True"):
        PlanarFaults().compute(system, ptm=without)
    with pytest.raises(InvalidParameter, match="template_neighbors=True"):
        PlanarFaults().compute(faulted([3]).gather(np.arange(100)), ptm=ptm)
    broken = dataclasses.replace(
        ptm, neighbors=np.full_like(ptm.neighbors, len(system))
    )
    with pytest.raises(InvalidParameter, match="outside the atoms"):
        PlanarFaults().compute(system, ptm=broken)


def test_invalid_arguments():
    with pytest.raises(InvalidParameter, match="fcc and hcp"):
        PlanarFaults(("fcc", "bcc"))
    with pytest.raises(InvalidParameter, match="fcc and hcp"):
        PlanarFaults("hcp")
    with pytest.raises(InvalidParameter, match="rmsd_cutoff"):
        PlanarFaults(rmsd_cutoff=-1)
    assert repr(PlanarFaults("fcc hcp".split())) == (
        "PlanarFaults(structures=('fcc', 'hcp'), rmsd_cutoff=0.1)"
    )
    assert repr(PlanarFaults(extrinsic=True)).endswith(
        "rmsd_cutoff=0.1, extrinsic=True)"
    )


@pytest.mark.parametrize("case", CASES)
def test_against_ovito(case):
    pytest.importorskip("ovito")
    from ovito.modifiers import (
        IdentifyFCCPlanarFaultsModifier,
        PolyhedralTemplateMatchingModifier,
    )

    from ovito_reference import ovito_compute

    system = CASES[case]()
    ours = PlanarFaults().compute(system)
    ptm = PolyhedralTemplateMatchingModifier(
        output_interatomic_distance=True, output_orientation=True
    )
    for t in ptm.structures:
        t.enabled = t.id in [
            PolyhedralTemplateMatching.NAMES.index(s) for s in ("fcc", "hcp", "bcc")
        ]
    data = ovito_compute(system, ptm, IdentifyFCCPlanarFaultsModifier())
    np.testing.assert_array_equal(ours.fault, data.particles["Planar Fault Type"])
    table = data.tables["planar_faults"]
    np.testing.assert_array_equal(
        [ours.counts()[name] for name in PlanarFaults.NAMES[1:5]], table.y[1:]
    )
    areas = ours.areas()
    np.testing.assert_allclose(
        [areas["intrinsic stacking fault"], areas["coherent twin boundary"]],
        table["Estimated Area"][2:4],
        rtol=1e-12,
    )
    assert areas["extrinsic stacking fault"] == 0
