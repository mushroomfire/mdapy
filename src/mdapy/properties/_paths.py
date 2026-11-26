# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Paths through the Brillouin zone, for the band structures of phonons and
of electrons: the standard path of a crystal by its symmetry or of a lattice
by its name, or sections given point by point, and the wave vectors along
them."""

from typing import Any

import numpy as np

from mdapy.core.system import System
from mdapy.errors import InvalidParameter

# Special points of the standard paths in Cartesian coordinates, in units of
# 2π/a for the cubic lattices with a along x, y and z, and of 2π/a in the
# plane and 2π/c along z for the hexagonal lattice with a along x
# (Setyawan and Curtarolo 2010; for fcc the path of seekpath, Hinuma et
# al. 2017).
_S3 = np.sqrt(3)
_PATHS = {
    "sc": (
        (("Γ", "X", "M", "Γ", "R", "X"), ("M", "R")),
        {"Γ": (0, 0, 0), "X": (0, 0.5, 0), "M": (0.5, 0.5, 0), "R": (0.5, 0.5, 0.5)},
    ),
    "fcc": (
        (("Γ", "X", "U"), ("K", "Γ", "L", "W", "X")),
        {
            "Γ": (0, 0, 0),
            "X": (0, 1, 0),
            "U": (0.25, 1, 0.25),
            "K": (0.75, 0.75, 0),
            "L": (0.5, 0.5, 0.5),
            "W": (0.5, 1, 0),
        },
    ),
    "bcc": (
        (("Γ", "H", "N", "Γ", "P", "H"), ("P", "N")),
        {"Γ": (0, 0, 0), "H": (0, 1, 0), "N": (0.5, 0.5, 0), "P": (0.5, 0.5, 0.5)},
    ),
    "hexagonal": (
        (("Γ", "M", "K", "Γ", "A", "L", "H", "A"), ("L", "M"), ("K", "H")),
        {
            "Γ": (0, 0, 0),
            "M": (0.5, _S3 / 6, 0),
            "K": (1 / 3, _S3 / 3, 0),
            "A": (0, 0, 0.5),
            "L": (0.5, _S3 / 6, 0.5),
            "H": (1 / 3, _S3 / 3, 0.5),
        },
    ),
}


def standard_path(
    matrix: np.ndarray, primitive: np.ndarray, lattice: str
) -> list[list[tuple[str, np.ndarray]]]:
    """The sections of the standard path of a lattice, its points in units of
    the reciprocal lattice vectors of the cell of rows matrix (Å), whose
    primitive cell has the rows primitive."""
    if lattice not in _PATHS:
        raise InvalidParameter(
            f"path must be one of {tuple(_PATHS)} or a list of segments, "
            f"not {lattice!r}"
        )
    sections, points = _PATHS[lattice]
    # the lattice constants, from the translations of the crystal
    combinations = np.stack(
        np.meshgrid(*[np.arange(-3, 4)] * 3, indexing="ij"), -1
    ).reshape(-1, 3)
    translations = combinations @ primitive
    lengths = np.linalg.norm(translations, axis=1)
    nonzero = lengths > 1e-9
    shortest = lengths[nonzero].min()
    if lattice == "hexagonal":
        flat = nonzero & (np.abs(translations[:, 2]) < 1e-6 * shortest)
        upright = nonzero & (
            np.linalg.norm(translations[:, :2], axis=1) < 1e-6 * shortest
        )
        if not flat.any() or not upright.any():
            raise InvalidParameter("the cell is not a hexagonal lattice with c along z")
        a, c = lengths[flat].min(), lengths[upright].min()
        expected = np.array([[a, 0, 0], [-a / 2, a * _S3 / 2, 0], [0, 0, c]])
        scale = 2 * np.pi * np.array([1 / a, 1 / a, 1 / c])
    else:
        a = shortest * {"sc": 1.0, "fcc": np.sqrt(2), "bcc": 2 / _S3}[lattice]
        expected = {
            "sc": a * np.eye(3),
            "fcc": a / 2 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
            "bcc": a / 2 * np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]),
        }[lattice]
        scale = np.full(3, 2 * np.pi / a)
    integers = expected @ np.linalg.inv(primitive)
    if np.abs(integers - np.round(integers)).max() > 1e-6 or not np.isclose(
        abs(np.linalg.det(np.round(integers))), 1
    ):
        raise InvalidParameter(
            f"the cell is not a {lattice} lattice with its axes along x, y and z"
        )
    to_cell = matrix.T / (2 * np.pi)
    return [
        [(label, (np.array(points[label]) * scale) @ to_cell) for label in section]
        for section in sections
    ]


def auto_path(cell: System, kinds: np.ndarray) -> list[list[tuple[str, np.ndarray]]]:
    """The sections of the standard path of a crystal by its symmetry, from
    seekpath, its points in units of the reciprocal lattice vectors of the
    cell; kinds, an integer for each atom, the same for atoms of one kind."""
    try:
        import seekpath
    except ImportError:
        raise ImportError('path="auto" needs seekpath: pip install seekpath') from None
    found = seekpath.get_path(
        (
            cell.box.matrix,
            cell.box.to_fractional(cell["positions"]),
            (np.asarray(kinds) + 1).tolist(),
        )
    )
    # its points in its own standard cell, turned back into this one
    reciprocal = np.array(found["reciprocal_primitive_lattice"])
    turn = np.array(found["rotation_matrix"])
    to_cell = cell.box.matrix.T / (2 * np.pi)
    points = {
        label: (np.array(q) @ reciprocal) @ turn @ to_cell
        for label, q in found["point_coords"].items()
    }
    sections: list[list[str]] = []
    for start, end in found["path"]:
        if sections and sections[-1][-1] == start:
            sections[-1].append(end)
        else:
            sections.append([start, end])
    name = {"GAMMA": "Γ"}
    return [
        [(name.get(label, label), points[label]) for label in section]
        for section in sections
    ]


def given_path(path: Any) -> list[list[tuple[str, np.ndarray]]]:
    """The sections of a path given as sequences of (label, wave vector),
    checked."""
    try:
        sections = [
            [(str(label), np.asarray(q, dtype=float)) for label, q in section]
            for section in path
        ]
    except (TypeError, ValueError):
        raise InvalidParameter(
            "path must be sections of (label, wave vector) points"
        ) from None
    if any(len(s) < 2 or any(q.shape != (3,) for _, q in s) for s in sections):
        raise InvalidParameter(
            "each section of a path needs two or more wave vectors of 3 values"
        )
    return sections


def sample(
    sections: list, points: int, reciprocal: np.ndarray
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[float, str], ...], list[int]]:
    """The wave vectors (P, 3) of points on each line of the sections, its
    ends included, their distances (P,) along the path in 1/Å with the
    reciprocal lattice vectors as rows of reciprocal, the distance and the
    label of each special point (two that meet at a break of the path share
    one such as "U|K"), and the index of the first wave vector of each
    section."""
    if int(points) < 2:
        raise InvalidParameter(f"points must be at least 2, not {points!r}")
    t = np.linspace(0, 1, int(points))[:, None]
    wave_vectors, distances, labels, starts = [], [], [], []
    reached = 0.0
    for section in sections:
        starts.append(sum(len(w) for w in wave_vectors))
        if labels:
            labels[-1] = (reached, f"{labels[-1][1]}|{section[0][0]}")
        else:
            labels.append((reached, section[0][0]))
        for (_, start), (label, end) in zip(section, section[1:]):
            line = start + t * (end - start)
            length = np.linalg.norm((end - start) @ reciprocal)
            wave_vectors.append(line)
            distances.append(reached + t[:, 0] * length)
            reached += length
            labels.append((reached, label))
    return (
        np.concatenate(wave_vectors),
        np.concatenate(distances),
        tuple((float(d), label) for d, label in labels),
        starts,
    )
