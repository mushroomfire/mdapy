# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal, edge_dislocation, insert_dislocation
from mdapy.errors import InvalidParameter
from mdapy.identify import CommonNeighborAnalysis

from atomsk_reference import ATOMSK, atomsk_apply

A_CU, A_FE = 3.615, 2.87
FCC = [(1, -1, 0), (1, 1, -2), (1, 1, 1)]  # x along b, z normal to the glide plane


def fcc_slab():
    return build_crystal(
        "fcc", A_CU, elements="Cu", orientation=FCC, repeat=(16, 2, 10)
    )


def bcc_screw_cell():
    return build_crystal(
        "bcc",
        A_FE,
        elements="Fe",
        orientation=[(1, 1, -2), (-1, 1, 0), (1, 1, 1)],
        repeat=(6, 10, 2),
    )


def bcc_edge_cell():
    return build_crystal(
        "bcc",
        A_FE,
        elements="Fe",
        orientation=[(1, 1, 1), (-1, 0, 1), (1, -2, 1)],
        repeat=(14, 6, 2),
    )


# atomsk takes the position in the plane in the order of the axes after the
# line: (x, y) for z, (z, x) for y, (y, z) for x
ORDER = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}


@pytest.mark.skipif(ATOMSK is None, reason="atomsk is not installed")
@pytest.mark.parametrize(
    "kind, line, normal, sign",
    [
        ("screw", "z", "y", 1),
        ("screw", "z", "x", -1),
        ("edge", "y", "z", 1),
        ("edge", "y", "z", -1),
        ("edge", "z", "y", 1),
    ],
)
def test_as_atomsk(tmp_path, kind, line, normal, sign):
    """The same displacements as atomsk -dislocation, to rounding."""
    if kind == "screw":
        cell, b = bcc_screw_cell(), A_FE * np.sqrt(3) / 2
    elif line == "y":
        cell, b = fcc_slab(), A_CU / np.sqrt(2)
    else:
        cell, b = bcc_edge_cell(), A_FE * np.sqrt(3) / 2
    lengths = cell.box.matrix.diagonal()
    center = lengths / 2 + [0.11, 0.13, 0.17]
    a1 = 3 - "xyz".index(line) - "xyz".index(normal)
    burgers = np.zeros(3)
    burgers["xyz".index(line) if kind == "screw" else a1] = sign * b
    ours = insert_dislocation(
        cell, burgers, line=line, normal=normal, center=center, poisson=0.3
    )
    p1, p2 = (center[k] for k in ORDER[line])
    options = ["-disloc", p1, p2, kind, line, normal, sign * b]
    if kind == "edge":
        options.append(0.3)
    theirs = atomsk_apply(tmp_path, cell, *options)
    np.testing.assert_allclose(ours.positions, theirs.positions, atol=1e-9)


def test_mixed_is_screw_and_edge():
    cell = fcc_slab()
    b = A_CU / np.sqrt(2)
    center = cell.box.matrix.diagonal() / 2 + 0.1
    screw = insert_dislocation(cell, [0, b / 2, 0], line="y", normal="z", center=center)
    both = insert_dislocation(
        screw, [b * np.sqrt(3) / 2, 0, 0], line="y", normal="z", center=center
    )
    mixed = insert_dislocation(
        cell, [b * np.sqrt(3) / 2, b / 2, 0], line="y", normal="z", center=center
    )
    np.testing.assert_allclose(mixed.positions, both.positions, atol=1e-12)


def test_screw_displacement():
    """Around the line the screw displacement grows by b once."""
    cell = bcc_screw_cell()
    b = A_FE * np.sqrt(3) / 2
    center = cell.box.matrix.diagonal() / 2 + [0.11, 0.13, 0.0]
    moved = insert_dislocation(cell, [0, 0, b], center=center)
    u = moved.positions[:, 2] - cell.positions[:, 2]
    x = cell.positions - center
    np.testing.assert_allclose(
        u, b / (2 * np.pi) * np.arctan2(x[:, 1], x[:, 0]), atol=1e-12
    )
    assert np.ptp(u) == pytest.approx(b, rel=0.05)
    assert np.abs(moved.positions[:, :2] - cell.positions[:, :2]).max() == 0


def test_two_crystals():
    cell = build_crystal("fcc", A_CU, elements="Cu", orientation=FCC)
    system = edge_dislocation(cell, (12, 2, 8))
    per = len(cell)
    assert len(system) == per * 2 * (13 * 4 + 12 * 4)
    length = cell.box.matrix.diagonal()
    np.testing.assert_allclose(
        system.box.matrix.diagonal(), [12.5 * length[0], 2 * length[1], 8 * length[2]]
    )
    assert list(system.box.pbc) == [True, True, False]
    # far from the line along z, both crystals are fcc
    structures = CommonNeighborAnalysis().compute(system).structure
    z = system.positions[:, 2]
    middle = 4 * length[2]
    far = (np.abs(z - middle) > 4.0) & (z > 3.0) & (z < system.box.matrix[2, 2] - 3.0)
    assert (structures[far] == 1).mean() > 0.95
    with pytest.raises(InvalidParameter, match="orthogonal periodic"):
        edge_dislocation(
            cell.with_box(mdapy.Box(cell.box.matrix, pbc=(True, True, False))),
            (2, 2, 2),
        )
    with pytest.raises(InvalidParameter, match="repeat needs"):
        edge_dislocation(cell, (2, 2, 1))


def test_two_crystals_relax_into_partials():
    """With an EAM potential, the edge dislocation of fcc copper splits into
    two Shockley partials around a ribbon of stacking fault."""
    from mdapy.calculators import EAM
    from mdapy.simulation import minimize

    cell = build_crystal("fcc", A_CU, elements="Cu", orientation=FCC)
    system = edge_dislocation(cell, (30, 1, 12))
    result = minimize(system, EAM.zhou(["Cu"]), max_force=1e-3, max_steps=5000)
    assert result.converged
    structures = CommonNeighborAnalysis().compute(result.system).structure
    hcp = result.system.positions[structures == 2]
    # two neighboring {111} planes, a/sqrt(3) apart, some nanometers wide
    layers = np.unique(np.round(hcp[:, 2] / (A_CU / np.sqrt(3))))
    assert len(layers) == 2 and np.ptp(layers) == 1
    assert np.ptp(hcp[:, 0]) > 15.0


def test_errors():
    cell = fcc_slab()
    b = A_CU / np.sqrt(2)
    for arguments, message in [
        ({"line": "z", "normal": "z"}, "normal must differ"),
        ({"burgers": [0, 0, b]}, "lie in the plane of cut"),
        ({"burgers": [b, 0]}, "three components"),
        ({"poisson": 0.5}, "poisson must be"),
        ({"center": cell.positions[0]}, "lies on the dislocation line"),
        ({"line": "w"}, "axis must be"),
    ]:
        options = {"burgers": [b, 0, 0], "line": "y", "normal": "z"} | arguments
        with pytest.raises(InvalidParameter, match=message):
            insert_dislocation(cell, **options)
