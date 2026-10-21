# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import numpy as np
import pytest
from lattices import crystal

import mdapy
from mdapy.analysis import WignerSeitz
from mdapy.errors import InvalidParameter

A = 3.6


def perfect(repeat=(4, 4, 4)):
    return crystal("fcc", A, repeat)


def test_thermal_motion_is_no_defect():
    reference = perfect()
    rng = np.random.default_rng(0)
    current = reference.with_positions(
        reference.positions + rng.normal(scale=0.15, size=(len(reference), 3))
    )
    result = WignerSeitz(reference).compute(current)
    np.testing.assert_array_equal(result.site, np.arange(len(reference)))
    assert result.vacancy_count == result.interstitial_count == 0
    assert (current.with_columns(result).table["occupancy"].to_numpy() == 1).all()


def test_frenkel_pairs_vacancies_and_interstitials():
    reference = perfect()
    positions = reference.positions.copy()
    positions[0] = [5.4, 5.4, 5.3]  # near an octahedral site, off its center
    frenkel = WignerSeitz(reference).compute(reference.with_positions(positions))
    assert frenkel.vacancy_count == 1 and frenkel.interstitial_count == 1
    np.testing.assert_array_equal(frenkel.vacancies, [0])
    assert frenkel.occupancy[frenkel.interstitials[0]] == 2
    fewer = reference.select(np.arange(5, len(reference)))  # five vacancies
    added = mdapy.System(
        np.vstack([reference.positions, [[5.4, 5.4, 5.3], [9.0, 9.0, 8.9]]]),
        reference.box,
        elements="Cu",
    )
    for current in (fewer, added):
        result = WignerSeitz(reference).compute(current)
        # every atom is on a site: more atoms means more interstitials than vacancies
        assert (
            len(current) - len(reference)
            == result.interstitial_count - result.vacancy_count
        )
    assert WignerSeitz(reference).compute(fewer).vacancy_count == 5
    assert WignerSeitz(reference).compute(added).interstitial_count == 2


def test_affine_mapping_of_a_stretched_box():
    reference = perfect((8, 8, 8))
    stretch = np.diag([1.1, 1.0, 1.0])
    current = reference.with_box(
        mdapy.Box(reference.box.matrix @ stretch)
    ).with_positions(reference.positions @ stretch)
    assert (
        WignerSeitz(reference).compute(current).vacancy_count > 0
    )  # the far atoms drift
    mapped = WignerSeitz(reference, affine=True).compute(current)
    np.testing.assert_array_equal(mapped.site, np.arange(len(reference)))


def test_nearest_sites_by_brute_force():
    rng = np.random.default_rng(1)
    box = mdapy.Box(
        [[20.0, 0, 0], [3.0, 18.0, 0], [-2.0, 1.0, 22.0]], pbc=(True, True, False)
    )
    sites = mdapy.System(box.to_cartesian(rng.random((300, 3))), box, elements="Cu")
    points = mdapy.System(
        box.to_cartesian(rng.random((1000, 3)) * [1, 1, 1.4] - [0, 0, 0.2]),
        box,
        elements="Cu",
    )
    result = WignerSeitz(sites).compute(points)
    images = (
        np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [0], indexing="ij"))
        .reshape(3, -1)
        .T
        @ box.matrix
    )
    d = np.linalg.norm(
        points.positions[:, None, None]
        - sites.positions[None, :, None]
        - images[None, None],
        axis=3,
    )
    np.testing.assert_array_equal(result.site, d.min(axis=2).argmin(axis=1))
    with mdapy.num_threads(3):
        np.testing.assert_array_equal(
            WignerSeitz(sites).compute(points).site, result.site
        )


def test_errors():
    reference = perfect((2, 2, 2))
    with pytest.raises(InvalidParameter, match="no sites"):
        WignerSeitz(mdapy.System(np.zeros((0, 3)), mdapy.Box(5.0), elements=[]))
    with pytest.raises(InvalidParameter, match="periodicity"):
        WignerSeitz(reference).compute(
            reference.with_box(mdapy.Box(reference.box.matrix, pbc=False))
        )
    assert repr(WignerSeitz(reference)) == "WignerSeitz(sites=32, affine=False)"
