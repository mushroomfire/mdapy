# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Bond stiffness against bond length: springs whose force constants are
exactly those of the model come back exactly; a pair potential gives its
second derivative along each bond and its first over the length across it;
the force constants of the springs are those of the formula; the errors."""

import dataclasses

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.calculators import Calculator, CalculatorResult
from mdapy.errors import InvalidParameter
from mdapy.lattice_dynamics import (
    BondStiffness,
    bond_stiffness,
    displaced_supercells,
)
from mdapy.lattice_dynamics._force_constants import _cell_of
from mdapy.lattice_dynamics._phonons import _lattice_points, _supercell

# springs of three pairs in two shells, along and across: (shell, pair) ->
# coefficients of 1, r
TRUTH = BondStiffness(
    pairs=(
        ("Cu", "Cu", 0),
        ("Cu", "Cu", 1),
        ("Cu", "Ni", 0),
        ("Cu", "Ni", 1),
        ("Ni", "Ni", 0),
        ("Ni", "Ni", 1),
    ),
    shells=np.array([3.0, 4.0]),
    degree=1,
    longitudinal=np.array(
        [[6.0, -1.5], [0.8, -0.2], [7.0, -1.8], [0.9, -0.22], [8.0, -2.0], [1.0, -0.25]]
    ),
    transverse=np.array(
        [
            [-0.5, 0.15],
            [-0.1, 0.02],
            [-0.6, 0.2],
            [-0.12, 0.03],
            [-0.4, 0.1],
            [-0.08, 0.015],
        ]
    ),
    lengths=np.zeros((6, 2)),
    error=0.0,
)


def full(constants):
    """(N, N, 3, 3) the force constants between every pair of atoms of the
    supercell, from those of the atoms of the cell."""
    cell, matrix = constants.cell, constants.supercell
    n = len(cell)
    points = _lattice_points(matrix)
    apart = _cell_of(points[None, :, :] - points[:, None, :], matrix)
    out = np.empty((n * len(points), n * len(points), 3, 3))
    for c in range(len(points)):
        columns = (apart[c][:, None] * n + np.arange(n)).reshape(-1)
        out[c * n : (c + 1) * n] = constants.second[:, columns]
    return out


def harmonic(cell, supercell, model, seed):
    """Random displacements of the supercell, with forces exactly those of
    the springs of the model."""
    constants = model.force_constants(cell, supercell)
    everything = full(constants)
    big = _supercell(cell, constants.supercell)
    frames = []
    for frame in displaced_supercells(
        cell, supercell=supercell, method="random", seed=seed
    ):
        u = big.box.minimum_image(frame["positions"] - big["positions"])
        frames.append(
            frame.with_columns(forces=-np.einsum("ijab,jb->ia", everything, u))
        )
    return cell, frames


def strained(cell, factor):
    return cell.with_box(cell.box.matrix * factor).with_columns(
        positions=cell["positions"] * factor
    )


@pytest.fixture(scope="module")
def crystals():
    """fcc Cu, fcc Ni and Cu3Ni of L1_2, each at three sizes, and a small
    random alloy, for Cu-Ni pairs in the second shell."""
    cells = [
        build_crystal("fcc", 3.55, elements="Cu"),
        build_crystal("fcc", 3.5, elements="Ni"),
        build_crystal("l1_2", 3.52, elements=("Cu", "Ni")),
    ]
    out = [strained(c, f) for c in cells for f in (0.98, 1.0, 1.02)]
    alloy = build_crystal("fcc", 3.52, elements="Cu", repeat=2)
    alloy = alloy.with_columns(element=["Cu", "Ni"] * 16)
    return out, [strained(alloy, f) for f in (0.98, 1.0, 1.02)]


def test_springs_of_the_model_come_back_exactly(crystals):
    cells, alloy = crystals
    data = [harmonic(c, 3, TRUTH, k) for k, c in enumerate(cells)]
    data += [harmonic(a, 2, TRUTH, 9 + k) for k, a in enumerate(alloy)]
    fit = bond_stiffness(data, cutoff=(3.0, 4.0))
    assert fit.pairs == TRUTH.pairs
    np.testing.assert_array_equal(fit.shells, [3.0, 4.0])
    assert fit.degree == 1 and fit.error < 1e-12
    np.testing.assert_allclose(fit.longitudinal, TRUTH.longitudinal, rtol=0, atol=1e-10)
    np.testing.assert_allclose(fit.transverse, TRUTH.transverse, rtol=0, atol=1e-10)
    # the range of the lengths of the Cu-Cu bonds: L1_2 and the alloy at their
    # smallest, Cu at its largest
    np.testing.assert_allclose(
        fit.lengths[0], [3.52 * 0.98 / np.sqrt(2), 3.55 * 1.02 / np.sqrt(2)]
    )
    assert repr(fit) == (
        "BondStiffness(Cu-Cu, Cu-Ni, Ni-Ni; 2 shells up to 4 Å, degree 1)"
    )
    # a degree of 2 gives the same springs, its last coefficient zero
    second = bond_stiffness(data, cutoff=(3.0, 4.0), degree=2)
    np.testing.assert_allclose(
        second.longitudinal[:, :2], TRUTH.longitudinal, atol=1e-8
    )
    np.testing.assert_allclose(second.longitudinal[:, 2], 0, atol=1e-8)


class Polynomial(Calculator):
    """E = sum over pairs closer than 3 Å of C r² + D r³: the stiffness
    along a bond is the second derivative, 2C + 6D r, across it the first
    over the length, 2C + 3D r."""

    C, D = -1.5, 0.6
    cutoff = 3.0

    def compute(self, system, properties=("energy", "forces"), *, neighbors=None):
        self._check_properties(properties)
        pairs = mdapy.NeighborList.build(system, self.cutoff)
        near = pairs.distances < self.cutoff
        i, r, d = pairs.i[near], pairs.vectors[near], pairs.distances[near]
        energies = np.zeros(len(system))
        np.add.at(energies, i, 0.5 * (self.C * d**2 + self.D * d**3))
        slope = 2 * self.C * d + 3 * self.D * d**2
        forces = np.zeros((len(system), 3))
        np.add.at(forces, i, (slope / d)[:, None] * r)
        return CalculatorResult(
            energy=float(energies.sum()),
            atomic_energies=energies,
            forces=forces,
            volume=system.box.volume,
        )


def test_a_pair_potential_gives_its_derivatives():
    """Displacements of plus and minus a pattern of 0.005 Å: the even terms of
    the forces leave the fit; the stiffnesses are those of the derivatives of
    the potential, to the cube of the displacement."""
    potential = Polynomial()
    data = []
    for factor in (0.98, 1.0, 1.02):
        cell = strained(build_crystal("fcc", 3.6, elements="Cu"), factor)
        frames = displaced_supercells(
            cell, supercell=3, method="random", displacement=0.005, seed=1
        )
        frames = [
            f.with_columns(forces=potential.compute(f, ("forces",)).forces)
            for f in frames
        ]
        data.append((cell, frames))
    fit = bond_stiffness(data, cutoff=3.0)
    C, D = Polynomial.C, Polynomial.D
    np.testing.assert_allclose(fit.longitudinal[0], [2 * C, 6 * D], rtol=1e-4)
    np.testing.assert_allclose(fit.transverse[0], [2 * C, 3 * D], rtol=1e-4)
    along, across = fit.stiffness("Cu", "Cu", [2.5, 2.6])
    np.testing.assert_allclose(along, 2 * C + 6 * D * np.array([2.5, 2.6]), rtol=1e-4)
    np.testing.assert_allclose(across, 2 * C + 3 * D * np.array([2.5, 2.6]), rtol=1e-4)
    assert fit.error < 1e-3


def test_the_force_constants_of_the_springs():
    """Along and across each bond, and the atom with itself minus the sum
    over its bonds; zero beyond the last shell."""
    cell = build_crystal("l1_2", 3.52, elements=("Cu", "Ni"))
    constants = TRUTH.force_constants(cell, 3)
    assert constants.cutoff == 4.0 and constants.parameters is None
    big = _supercell(cell, constants.supercell)
    vectors = big.box.minimum_image(
        (big["positions"][None] - cell["positions"][:, None]).reshape(-1, 3)
    ).reshape(len(cell), len(big), 3)
    elements = big.table["element"].cast(str).to_numpy()
    for a in range(len(cell)):
        for j in range(len(big)):
            r = np.linalg.norm(vectors[a, j])
            block = constants.second[a, j]
            if r < 1e-9:
                np.testing.assert_allclose(
                    block, -np.delete(constants.second[a], j, axis=0).sum(axis=0)
                )
            elif r > 4.0 + 1e-4:
                assert (block == 0).all()
            else:
                e = vectors[a, j] / r
                along, across = TRUTH.stiffness(elements[a], elements[j], r)
                expected = -(
                    along * np.outer(e, e) + across * (np.eye(3) - np.outer(e, e))
                )
                np.testing.assert_allclose(block, expected, atol=1e-14)
    # the sum rule and the symmetry of the pairs hold by construction
    np.testing.assert_allclose(constants.second.sum(axis=1), 0, atol=1e-12)
    everything = full(constants)
    np.testing.assert_allclose(everything, everything.transpose(1, 0, 3, 2), atol=1e-14)


def test_the_results_cannot_change():
    with pytest.raises(dataclasses.FrozenInstanceError):
        TRUTH.degree = 2
    for array in (TRUTH.shells, TRUTH.longitudinal, TRUTH.transverse, TRUTH.lengths):
        with pytest.raises(ValueError):
            array[0] = 0
    one = dataclasses.replace(
        TRUTH,
        pairs=TRUTH.pairs[:1],
        shells=np.array([3.0]),
        longitudinal=np.zeros((1, 2)),
        transverse=np.zeros((1, 2)),
        lengths=np.zeros((1, 2)),
    )
    assert repr(one) == "BondStiffness(Cu-Cu; 1 shell up to 3 Å, degree 1)"


def test_errors(crystals):
    cells, alloy = crystals
    data = [harmonic(c, 3, TRUTH, k) for k, c in enumerate(cells[:3])]
    with pytest.raises(InvalidParameter, match="crystals with frames"):
        bond_stiffness([], cutoff=3.0)
    with pytest.raises(InvalidParameter, match="frames of each crystal"):
        bond_stiffness([(cells[0], [])], cutoff=3.0)
    with pytest.raises(InvalidParameter, match="cells with atoms"):
        bond_stiffness(
            [(cells[0].filter(np.zeros(len(cells[0]), dtype=bool)), data[0][1])],
            cutoff=3.0,
        )
    bare = mdapy.System(cells[0]["positions"], cells[0].box)
    with pytest.raises(InvalidParameter, match="element column"):
        bond_stiffness([(bare, data[0][1])], cutoff=3.0)
    for degree in (-1, 1.5, True):
        with pytest.raises(InvalidParameter, match="degree"):
            bond_stiffness(data, cutoff=3.0, degree=degree)
    for cutoff in (0.0, (3.0, 2.0), [], np.nan):
        with pytest.raises(InvalidParameter, match="cutoff must be"):
            bond_stiffness(data, cutoff=cutoff)
    with pytest.raises(
        InvalidParameter, match="last shell must be positive and at most"
    ):
        bond_stiffness(data, cutoff=6.0)
    with pytest.raises(InvalidParameter, match="no bonds of 2 Å or shorter"):
        bond_stiffness(data, cutoff=2.0)
    # one size of fcc Cu: one length, too few for a line
    with pytest.raises(InvalidParameter, match="Cu-Cu bonds of shell 0 have 1 length"):
        bond_stiffness(data[1:2], cutoff=3.0)
    # frames of atoms at rest tell nothing
    cell, frames = data[1]
    still = _supercell(cell, np.diag([3, 3, 3])).with_columns(
        forces=np.zeros((27 * len(cell), 3))
    )
    with pytest.raises(InvalidParameter, match="determine 0 of the 2"):
        bond_stiffness([(cell, [still])], cutoff=3.0, degree=0)
    # the springs hold no Cu-Ni bonds of the second shell
    cu = dataclasses.replace(
        TRUTH,
        pairs=TRUTH.pairs[:3],
        longitudinal=TRUTH.longitudinal[:3],
        transverse=TRUTH.transverse[:3],
        lengths=TRUTH.lengths[:3],
    )
    with pytest.raises(
        InvalidParameter, match="no stiffness of Cu-Ni bonds in shell 1"
    ):
        cu.force_constants(alloy[0], 2)
    with pytest.raises(InvalidParameter, match="no longer than 4 Å"):
        TRUTH.stiffness("Cu", "Cu", 4.5)
    with pytest.raises(InvalidParameter, match="no longer than 4 Å"):
        TRUTH.stiffness("Cu", "Cu", 0.0)
    with pytest.raises(InvalidParameter, match="last shell must be positive"):
        TRUTH.force_constants(cells[0], 1)
