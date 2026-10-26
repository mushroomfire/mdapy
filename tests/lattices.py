# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Perfect crystals and clusters for the tests."""

import numpy as np

import mdapy

BASES = {
    "fcc": ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]),
    "bcc": ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 0, 0], [0.5, 0.5, 0.5]]),
    "hcp": (
        [[1, 0, 0], [-0.5, np.sqrt(3) / 2, 0], [0, 0, np.sqrt(8 / 3)]],
        [[0, 0, 0], [1 / 3, 2 / 3, 0.5]],
    ),
    "lonsdaleite": (
        [[1, 0, 0], [-0.5, np.sqrt(3) / 2, 0], [0, 0, np.sqrt(8 / 3)]],
        [[1 / 3, 2 / 3, 0], [2 / 3, 1 / 3, 0.5], [1 / 3, 2 / 3, 3 / 8], [2 / 3, 1 / 3, 7 / 8]],
    ),
    "diamond": (
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
         [0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]],
    ),
}  # fmt: skip


def crystal(kind, a=1.0, repeat=(3, 3, 3), element="Cu", jitter=0.0, seed=0):
    """A periodic crystal of lattice constant a."""
    cell, basis = BASES[kind]
    cell = np.array(cell, dtype=float) * a
    counts = np.array(repeat)
    grid = np.stack(
        np.meshgrid(*[np.arange(c) for c in counts], indexing="ij"), -1
    ).reshape(-1, 3)
    fractional = (grid[:, None, :] + np.array(basis)[None]).reshape(-1, 3)
    positions = fractional @ cell
    if jitter:
        positions = positions + np.random.default_rng(seed).normal(
            scale=jitter, size=positions.shape
        )
    return mdapy.System(positions, mdapy.Box(cell * counts[:, None]), elements=element)


def icosahedron(r=2.5, element="Cu"):
    """A centered 13-atom icosahedral cluster, not periodic."""
    phi = (1 + np.sqrt(5)) / 2
    corners = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            corners += [[0, s1, s2 * phi], [s1, s2 * phi, 0], [s2 * phi, 0, s1]]
    corners = np.array(corners) / np.sqrt(1 + phi**2) * r
    positions = np.vstack([[0, 0, 0], corners]) + 10.0
    return mdapy.System(positions, mdapy.Box(20.0, pbc=False), elements=element)


def same_crystal(ours, theirs, tolerance=1e-8):
    """Whether two systems have the same box and the same atoms, element by
    element, up to a common translation and the order of the atoms."""
    if not np.allclose(ours.box.matrix, theirs.box.matrix, atol=tolerance):
        return False
    if len(ours) != len(theirs):
        return False
    kinds = ours.table["element"].to_list()
    others = theirs.table["element"].to_list()
    box = ours.box
    x = box.to_fractional(ours["positions"])
    y = box.to_fractional(theirs["positions"])
    target = others[0]
    for k in [k for k, e in enumerate(kinds) if e == target]:
        shift = y[0] - x[k]
        d = (x + shift)[:, None, :] - y[None, :, :]
        d -= np.round(d)
        close = np.linalg.norm(d @ box.matrix, axis=2) < tolerance
        match = close.argmax(axis=1)
        if (
            close.any(axis=1).all()
            and len(set(match.tolist())) == len(x)
            and all(kinds[i] == others[j] for i, j in enumerate(match))
        ):
            return True
    return False
