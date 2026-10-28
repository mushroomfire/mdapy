# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest

from mdapy.build import build_crystal
from mdapy.calculators import NEP
from mdapy.errors import InvalidParameter
from mdapy.mlip import (
    farthest_point_sampling,
    perturb,
    principal_components,
    structure_descriptors,
)

from calculators.nep_files import write_nep

# ------------------------------------------------------------ choosing frames


def test_principal_components():
    rng = np.random.default_rng(4)
    points = rng.normal(size=(200, 5)) @ np.diag([5.0, 3.0, 1.0, 0.5, 0.1]) + 7.0
    pca = principal_components(points, n=3)
    # the same as the singular value decomposition of the centered points
    centered = points - points.mean(axis=0)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    np.testing.assert_allclose(pca.variance, s[:3] ** 2 / 199, rtol=1e-10)
    np.testing.assert_allclose(np.abs(pca.components), np.abs(vt[:3]), atol=1e-10)
    np.testing.assert_allclose(pca.ratio, s[:3] ** 2 / (s**2).sum(), rtol=1e-10)
    np.testing.assert_allclose(pca.projection, centered @ pca.components.T, atol=1e-10)
    np.testing.assert_allclose(
        pca.transform(points[:5]), pca.projection[:5], atol=1e-10
    )
    largest = np.argmax(np.abs(pca.components), axis=1)
    assert (pca.components[np.arange(3), largest] > 0).all()
    flat = principal_components(np.ones((4, 2)), n=1)
    np.testing.assert_array_equal(flat.ratio, [0.0])
    with pytest.raises(InvalidParameter, match="two points or more"):
        principal_components(np.ones((1, 3)))
    with pytest.raises(InvalidParameter, match="n must be from 1 to 5"):
        principal_components(points, n=6)


def greedy(points, count, start, selected=None):
    """Farthest point sampling by brute force."""
    chosen = [] if selected is None else list(selected)
    out = []
    candidates = list(range(len(points)))
    for k in range(count):
        if k == 0 and start is not None:
            best = start
        else:
            pool = [points[c] for c in chosen] + [points[o] for o in out]
            best = max(
                candidates,
                key=lambda i: min(np.linalg.norm(points[i] - p) for p in pool),
            )
        out.append(best)
    return out


def test_farthest_point_sampling():
    rng = np.random.default_rng(5)
    points = rng.random((60, 3))
    np.testing.assert_array_equal(
        farthest_point_sampling(points, 10), greedy(points, 10, 0)
    )
    np.testing.assert_array_equal(
        farthest_point_sampling(points, 5, start=7), greedy(points, 5, 7)
    )
    old = rng.random((4, 3))
    ours = farthest_point_sampling(points, 6, start=None, selected=old)
    pool = np.vstack([old])
    expected = []
    for _ in range(6):
        distance = np.min(np.linalg.norm(points[:, None] - pool[None], axis=2), axis=1)
        expected.append(int(np.argmax(distance)))
        pool = np.vstack([pool, points[expected[-1]]])
    np.testing.assert_array_equal(ours, expected)
    assert len(farthest_point_sampling(points, 0)) == 0
    for arguments, message in [
        ({"count": 61}, "count must be"),
        ({"count": 2, "start": 60}, "start must be"),
        ({"count": 2, "start": None}, "only with selected"),
    ]:
        with pytest.raises(InvalidParameter, match=message):
            farthest_point_sampling(points, **arguments)
    with pytest.raises(InvalidParameter, match=r"\(M, D\)"):
        farthest_point_sampling(np.ones(5), 1)


def test_structure_descriptors(tmp_path):
    nep = NEP(write_nep(tmp_path / "nep.txt", elements=("Cu", "Ni")))
    crystal = build_crystal("fcc", 3.6, elements=("Cu", "Ni"), repeat=(2, 2, 2))
    copies = perturb(crystal, 3, seed=1)
    descriptors = structure_descriptors(copies, nep)
    assert descriptors.shape == (3, nep.descriptors(crystal).shape[1])
    np.testing.assert_allclose(descriptors[1], nep.descriptors(copies[1]).mean(axis=0))


def test_perturb():
    crystal = build_crystal("fcc", 3.6, elements="Cu", repeat=(4, 4, 4)).with_columns(
        forces=np.ones((256, 3))
    )
    crystal = crystal.with_info(energy=-1.0)
    copies = perturb(crystal, 4, displacement=0.1, strain=0.02, seed=2)
    assert len(copies) == 4 and "forces" not in copies[0].table.columns
    assert "energy" not in copies[0].info
    for copy in copies:
        deformation = np.linalg.solve(crystal.box.matrix, copy.box.matrix).T
        np.testing.assert_allclose(deformation, deformation.T, atol=1e-12)
        assert np.abs(deformation - np.eye(3)).max() <= 0.02
        step = copy["positions"] - crystal["positions"] @ deformation.T
        assert np.sqrt((step**2).sum(axis=1).mean()) == pytest.approx(0.1, rel=0.1)
    const = perturb(crystal, 1, displacement=0.2, strain=0.0, style="const", seed=3)[0]
    np.testing.assert_allclose(
        np.linalg.norm(const["positions"] - crystal["positions"], axis=1), 0.2
    )
    ball = perturb(crystal, 1, displacement=0.2, strain=0.0, style="uniform", seed=3)[0]
    lengths = np.linalg.norm(ball["positions"] - crystal["positions"], axis=1)
    assert lengths.max() <= 0.2 and lengths.mean() == pytest.approx(
        0.15, rel=0.1
    )  # 3/4 of the radius
    again = perturb(crystal, 4, displacement=0.1, strain=0.02, seed=2)
    np.testing.assert_array_equal(again[3]["positions"], copies[3]["positions"])
    for arguments, message in [
        ({"count": -1}, "count must be"),
        ({"count": 1, "displacement": -1.0}, "must not be negative"),
        ({"count": 1, "style": "gauss"}, "style must be"),
    ]:
        with pytest.raises(InvalidParameter, match=message):
            perturb(crystal, **arguments)
