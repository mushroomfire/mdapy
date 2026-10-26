# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest

import mdapy
from mdapy import units
from mdapy.build import build_crystal, random_alloy
from mdapy.calculators import EAM
from mdapy.elements import atomic_masses
from mdapy.errors import FileFormatError, InvalidParameter
from mdapy.lattice_dynamics import (
    BandStructure,
    elastic_constants,
    phonons,
    primitive_cell,
    read_band_structure,
)
from mdapy.lattice_dynamics._phonons import _displaced_forces, _symmetrized
from mdapy.simulation import minimize
from mdapy.transform import replicate

from calculators.eam_files import write_setfl
from calculators.test_testing import Harmonic


def thz(eigenvalues):
    """Frequencies in THz of eigenvalues in eV/(Å² amu)."""
    return np.sqrt(np.asarray(eigenvalues) / units.AMU_A2_PER_PS2_TO_EV) / (2 * np.pi)


def springs_fcc(a=3.6):
    """fcc argon with springs at rest between nearest neighbors only."""
    return build_crystal("fcc", a, elements="Ar"), Harmonic(k=0.7, r0=a / np.sqrt(2))


@pytest.fixture(scope="module")
def copper(tmp_path_factory):
    """fcc Cu of the synthetic potential at zero stress: its potential and
    conventional cell."""
    potential = EAM(write_setfl(tmp_path_factory.mktemp("eam") / "CuNi.eam.alloy"))
    result = minimize(
        build_crystal("fcc", 3.29, elements="Cu"),
        potential,
        cell="full",
        max_force=1e-10,
        max_stress=1e-8,
        max_steps=10000,
    )
    assert result.converged
    return potential, result.system


def test_simple_cubic_springs():
    # ω² = 2k/m (1 - cos q_α a) for each direction α
    a, k = 2.5, 1.3
    cell = build_crystal("sc", a, elements="Ar")
    result = phonons(cell, Harmonic(k=k, r0=a), displacement=1e-5)
    np.testing.assert_array_equal(result.supercell, 3 * np.eye(3))
    q = np.random.default_rng(0).random((20, 3)) - 0.5
    mass = atomic_masses(["Ar"])[0]
    expected = np.sort(thz(2 * k / mass * (1 - np.cos(2 * np.pi * q))), axis=1)
    np.testing.assert_allclose(result.frequencies(q), expected, atol=1e-8)
    np.testing.assert_allclose(result.frequencies(q[0]), expected[0], atol=1e-8)
    # the polarizations are the axes
    frequencies, vectors = result.modes(q[:3])
    assert vectors.shape == (3, 3, 1, 3)
    np.testing.assert_allclose(np.abs(vectors[:, :, 0, :]).max(axis=2), 1.0)
    single, vector = result.modes(q[0])
    np.testing.assert_allclose(single, frequencies[0])
    assert vector.shape == (3, 1, 3)


def test_force_constants_are_symmetric_and_translations_cost_nothing():
    cell, springs = springs_fcc()
    result = phonons(cell, springs, supercell=2)
    constants = result.force_constants
    assert constants.shape == (4, 32, 3, 3) and not constants.flags.writeable
    np.testing.assert_allclose(constants.sum(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(result.frequencies([0, 0, 0])[:3], 0.0, atol=1e-6)
    # the pair (a, b in cell L) is the pair (b, a in cell -L)
    np.testing.assert_allclose(
        constants[0, 4 + 1], constants[1, 4 * 7 + 0].T, atol=1e-12
    )


def test_symmetrizing():
    cell, springs = springs_fcc()
    counts = 2 * np.eye(3, dtype=int)
    raw = _displaced_forces(cell, springs, counts, 1e-5)
    # force constants that are already symmetric stay as they are
    np.testing.assert_allclose(_symmetrized(raw, counts), raw, atol=1e-8)
    # noise is made symmetric, and translations cost nothing
    noisy = _symmetrized(
        raw + np.random.default_rng(1).normal(scale=1e-3, size=raw.shape), counts
    )
    np.testing.assert_allclose(noisy.sum(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(_symmetrized(noisy, counts), noisy, atol=1e-6)


def test_conventional_and_primitive_cells_vibrate_alike():
    # the 12 modes of the conventional cell at q are the 3 of the primitive
    # cell at q and at q plus three reciprocal vectors of the cubic lattice
    a = 3.6
    conventional, springs = springs_fcc(a)
    primitive = primitive_cell(conventional)
    assert len(primitive) == 1
    big = phonons(conventional, springs, supercell=2)
    small = phonons(primitive, springs, supercell=4)
    rng = np.random.default_rng(2)
    for q_cart in rng.random((5, 3)) * 2 * np.pi / a:
        folded = [
            q_cart + 2 * np.pi / a * np.array(g)
            for g in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
        ]
        expected = np.sort(
            np.concatenate(
                [
                    small.frequencies(q @ primitive.box.matrix.T / (2 * np.pi))
                    for q in folded
                ]
            )
        )
        np.testing.assert_allclose(
            big.frequencies(q_cart @ conventional.box.matrix.T / (2 * np.pi)),
            expected,
            atol=1e-6,
        )
    # and so, as the meshes grow, are their thermal properties per atom
    np.testing.assert_allclose(
        big.thermal([100, 300], mesh=10).free_energy,
        small.thermal([100, 300], mesh=20).free_energy,
        rtol=1e-3,
    )


def test_density_of_states():
    cell, springs = springs_fcc()
    result = phonons(primitive_cell(cell), springs)
    tetrahedra = result.dos(16)
    gaussians = result.dos(16, sigma=0.05, projected=True)
    for dos in (tetrahedra, gaussians):
        assert np.trapezoid(dos.dos, dos.frequencies) == pytest.approx(3, rel=2e-3)
        assert not dos.dos.flags.writeable
    assert tetrahedra.projected is None
    np.testing.assert_allclose(gaussians.projected[:, 0], gaussians.dos)
    # the mean square frequency: the trace of the dynamical matrix over the
    # mesh, the same by both methods within the broadening
    grid = np.linspace(0, 1.2 * tetrahedra.frequencies.max(), 3001)
    moments = [
        np.trapezoid(result.dos(16, frequencies=grid, sigma=s).dos * grid**2, grid)
        for s in (None, 0.02)
    ]
    assert moments[0] == pytest.approx(moments[1], rel=2e-3)
    conventional = phonons(cell, springs, supercell=2)
    for sigma in (None, 0.1):
        dos = conventional.dos(4, sigma=sigma, projected=True)
        np.testing.assert_allclose(dos.projected.mean(axis=1), dos.dos, rtol=1e-10)
    # the four atoms are alike: summed over each set of modes of one
    # frequency, which the tetrahedra take apart
    for part in dos.projected.T:
        np.testing.assert_allclose(part, dos.projected[:, 0], rtol=1e-8)


def test_thermal_properties():
    cell, springs = springs_fcc()
    result = phonons(primitive_cell(cell), springs)
    temperatures = np.array([0.0, 10.0, 100.0, 300.0, 1e5])
    thermal = result.thermal(temperatures, mesh=12)
    assert not thermal.free_energy.flags.writeable
    zero_point = thermal.energy[0]
    assert thermal.free_energy[0] == zero_point > 0
    assert thermal.entropy[0] == thermal.heat_capacity[0] == 0
    # F = E - T S
    np.testing.assert_allclose(
        thermal.free_energy, thermal.energy - temperatures * thermal.entropy, atol=1e-12
    )
    # S = -dF/dT and C = dE/dT, by central differences
    h = 1e-3
    for t in (100.0, 300.0):
        around = result.thermal([t - h, t + h], mesh=12)
        k = list(temperatures).index(t)
        assert -np.diff(around.free_energy)[0] / (2 * h) == pytest.approx(
            thermal.entropy[k], rel=1e-6
        )
        assert np.diff(around.energy)[0] / (2 * h) == pytest.approx(
            thermal.heat_capacity[k], rel=1e-6
        )
    # the classical limit: 3 k_B per atom, less the three translations
    modes = 3 * 12**3
    assert thermal.heat_capacity[-1] == pytest.approx(
        3 * units.BOLTZMANN * (modes - 3) / modes, rel=1e-6
    )


def test_band_structures():
    cell, springs = springs_fcc()
    result = phonons(primitive_cell(cell), springs)
    bands = result.band_structure("fcc", points=11)
    labels = [label for _, label in bands.labels]
    assert labels == ["Γ", "X", "U|K", "Γ", "L", "W", "X"]
    assert bands.q.shape == (6 * 11, 3) and bands.frequencies.shape == (66, 3)
    assert np.all(np.diff(bands.distances) >= 0)
    # Γ–X is 2π/a long, in Cartesian 1/Å
    assert bands.labels[1][0] == pytest.approx(2 * np.pi / 3.6)
    np.testing.assert_allclose(bands.frequencies[0], 0.0, atol=1e-6)
    # the same along a path given by hand: X is (0, 1, 0) 2π/a in Cartesian
    x = (
        np.array([0, 2 * np.pi / 3.6, 0])
        @ primitive_cell(cell).box.matrix.T
        / (2 * np.pi)
    )
    by_hand = result.band_structure([[("Γ", [0, 0, 0]), ("X", x)]], points=11)
    np.testing.assert_allclose(by_hand.frequencies, bands.frequencies[:11], atol=1e-12)
    # the conventional cell along the same path has the folded modes too
    folded = phonons(cell, springs, supercell=2).band_structure("fcc", points=11)
    assert folded.frequencies.shape == (66, 12)
    for line in range(len(bands.frequencies)):
        assert np.isin(
            np.round(bands.frequencies[line], 6), np.round(folded.frequencies[line], 6)
        ).all()


@pytest.mark.parametrize(
    "structure, a, lattice, elements",
    [
        ("sc", 2.5, "sc", "Ar"),
        ("bcc", 3.0, "bcc", "Ar"),
        ("fcc", 3.6, "fcc", "Ar"),
        ("rocksalt", 5.0, "fcc", ("Na", "Cl")),
        ("hcp", 3.0, "hexagonal", "Ar"),
    ],
)
def test_standard_paths(structure, a, lattice, elements):
    cell = build_crystal(structure, a, elements=elements)
    result = phonons(cell, Harmonic(k=0.5, r0=2.9), supercell=2)
    bands = result.band_structure(lattice, points=3)
    assert bands.frequencies.shape[1] == 3 * len(cell)
    others = {"sc", "bcc", "fcc", "hexagonal"} - {lattice}
    if lattice == "fcc":  # also as the primitive cell
        primitive = phonons(primitive_cell(cell), Harmonic(k=0.5, r0=2.9), supercell=2)
        assert len(primitive.band_structure("fcc", points=3).distances) == 18
    for other in others:
        with pytest.raises(InvalidParameter, match="is not a"):
            result.band_structure(other)


def test_connected_branches_cross():
    # simple cubic springs: three branches polarized along x, y and z, two
    # of which cross halfway along the first line
    a, k = 2.5, 1.3
    cell = build_crystal("sc", a, elements="Ar")
    result = phonons(cell, Harmonic(k=k, r0=a), supercell=3, displacement=1e-5)
    path = [[("A", [0.1, 0.4, 0.2]), ("B", [0.4, 0.1, 0.2]), ("C", [0.4, 0.4, 0.4])], [("D", [0, 0, 0]), ("E", [0.5, 0.3, 0.1])]]  # fmt: skip
    mass = atomic_masses(["Ar"])[0]
    for connected, close in ((True, True), (False, False)):
        bands = result.band_structure(path, points=21, connected=connected)
        q = bands.q
        branches = thz(2 * k / mass * (1 - np.cos(2 * np.pi * q)))  # (P, 3), x y z
        first = np.argsort(branches[0])
        # through the crossing and the point where it is sampled
        error = np.abs(bands.frequencies[:42] - branches[:42, first]).max()
        assert (error < 1e-6) == close
        # the next section starts ascending, from Γ where all three meet
        np.testing.assert_allclose(bands.frequencies[42], 0, atol=1e-6)
        np.testing.assert_array_equal(np.argsort(bands.frequencies[43]), np.arange(3))


@pytest.mark.parametrize(
    "structure, a, lattice",
    [("fcc", 3.6, "fcc"), ("bcc", 3.0, "bcc"), ("hcp", 3.0, "hexagonal")],
)
def test_automatic_paths(structure, a, lattice):
    pytest.importorskip("seekpath")
    cell = primitive_cell(build_crystal(structure, a, elements="Ar"))
    springs = Harmonic(k=0.5, r0=2.9)
    result = phonons(cell, springs, supercell=3)
    auto = result.band_structure("auto", points=5)
    named = result.band_structure(lattice, points=5)
    if lattice == "fcc":  # the same path
        assert [x for _, x in auto.labels] == [x for _, x in named.labels]
        np.testing.assert_allclose(auto.frequencies, named.frequencies, atol=1e-9)
    # every special point is a point of the Brillouin zone of the named path
    reached = {label: q for label, q in zip(auto.labels, auto.q[::5])}
    assert auto.labels[0][1] == "Γ" and len(reached) > 3
    # the same along the path of the crystal turned at random
    rotation = np.linalg.qr(np.random.default_rng(2).normal(size=(3, 3)))[0]
    rotation *= np.sign(np.linalg.det(rotation))
    turned = cell.with_box(mdapy.Box(cell.box.matrix @ rotation.T)).with_columns(
        positions=cell["positions"] @ rotation.T
    )
    again = phonons(turned, springs, supercell=3).band_structure("auto", points=5)
    np.testing.assert_allclose(again.distances, auto.distances, atol=1e-9)
    np.testing.assert_allclose(again.frequencies, auto.frequencies, atol=1e-4)


def test_automatic_path_needs_seekpath(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "seekpath", None)
    cell, springs = springs_fcc()
    with pytest.raises(ImportError, match="seekpath"):
        phonons(primitive_cell(cell), springs, supercell=2).band_structure("auto")


def test_rotated_cell_has_no_standard_path():
    cell = build_crystal(
        "fcc", 3.6, elements="Ar", orientation=[(1, 1, 0), (-1, 1, 0), (0, 0, 1)]
    )
    result = phonons(cell, Harmonic(k=0.5, r0=3.6 / np.sqrt(2)), supercell=1)
    with pytest.raises(InvalidParameter, match="axes along x, y and z"):
        result.band_structure("fcc")
    # no lattice vector along z
    tilted = mdapy.System(
        [[0.0, 0.0, 0.0]],
        mdapy.Box([[3.0, 0, 0], [1.0, 3.0, 0], [0.7, 0.4, 3.1]]),
        elements="Ar",
    )
    result = phonons(tilted, Harmonic(k=0.5, r0=3.0), supercell=2)
    with pytest.raises(InvalidParameter, match="hexagonal lattice with c along z"):
        result.band_structure("hexagonal")


def test_eam_copper_long_waves_follow_the_elastic_constants(copper):
    # at long wavelengths along [100], v = sqrt(C11 / ρ) and sqrt(C44 / ρ)
    potential, conventional = copper
    a = conventional.box.matrix[0, 0]
    constants = elastic_constants(conventional, potential).constants
    density = len(conventional) * atomic_masses(["Cu"])[0] / conventional.box.volume
    cell = primitive_cell(conventional)
    result = phonons(cell, potential, supercell=6)
    length = 0.01 * 2 * np.pi / a
    q = np.array([length, 0, 0]) @ cell.box.matrix.T / (2 * np.pi)
    frequencies = result.frequencies(q)
    speeds = np.sqrt(
        np.array([constants[3, 3], constants[3, 3], constants[0, 0]])
        * units.GPA_TO_EV_PER_A3
        / density
        / units.AMU_A2_PER_PS2_TO_EV
    )
    np.testing.assert_allclose(frequencies, speeds * length / (2 * np.pi), rtol=2e-3)


def test_default_supercell(copper):
    potential, conventional = copper
    result = phonons(conventional, potential)
    # 2 × 5 Å of cutoff over the 3.3 Å of the cube
    np.testing.assert_array_equal(result.supercell, 4 * np.eye(3))
    assert repr(result) == "Phonons(4 atoms, supercell of 64 cells)"
    assert not result.supercell.flags.writeable
    # of the primitive cell, the same cube: 4 × 4 × 4 cubic cells
    primitive = phonons(primitive_cell(conventional), potential)
    box = primitive.supercell @ primitive_cell(conventional).box.matrix
    np.testing.assert_allclose(np.abs(box), 4 * conventional.box.matrix, atol=1e-9)
    q = np.random.default_rng(3).random((10, 3)) - 0.5
    to_cube = (
        np.linalg.inv(primitive_cell(conventional).box.matrix).T
        @ conventional.box.matrix.T
    )
    folded = result.frequencies(q @ to_cube)
    for ours, theirs in zip(primitive.frequencies(q), folded):
        assert np.abs(ours[:, None] - theirs[None]).min(axis=1).max() < 1e-6


def test_supercell_matrices():
    # the cubic supercell of the primitive cell of fcc, as a matrix, and the
    # repeats of the cubic cell: the same force constants, the same modes
    cell, springs = springs_fcc()
    primitive = primitive_cell(cell)
    matrix = np.round(2 * cell.box.matrix @ np.linalg.inv(primitive.box.matrix)).astype(
        int
    )
    by_matrix = phonons(primitive, springs, supercell=matrix)
    by_repeats = phonons(cell, springs, supercell=2)
    assert repr(by_matrix) == "Phonons(1 atom, supercell of 32 cells)"
    np.testing.assert_allclose(by_matrix.force_constants.sum(axis=1), 0, atol=1e-12)
    rng = np.random.default_rng(4)
    for q_cart in rng.random((5, 3)) * 2 * np.pi / 3.6:
        ours = by_matrix.frequencies(q_cart @ primitive.box.matrix.T / (2 * np.pi))
        theirs = by_repeats.frequencies(q_cart @ cell.box.matrix.T / (2 * np.pi))
        assert np.isin(np.round(ours, 6), np.round(theirs, 6)).all()
    # the cell itself first: the displaced atoms are those of the cell
    from mdapy.lattice_dynamics._phonons import _lattice_points

    points = _lattice_points(matrix)
    assert len(points) == 32 and (points[0] == 0).all()
    # a cell thick enough is its own supercell
    big = phonons(replicate(cell, (2, 2, 2)), springs)
    np.testing.assert_array_equal(big.supercell, np.eye(3))


def test_masses_from_a_column():
    cell, springs = springs_fcc()
    heavy = cell.with_columns(mass=np.full(len(cell), 4 * atomic_masses(["Ar"])[0]))
    q = [0.1, 0.2, 0.3]
    np.testing.assert_allclose(
        phonons(heavy, springs, supercell=2).frequencies(q),
        phonons(cell, springs, supercell=2).frequencies(q) / 2,
        rtol=1e-12,
    )


def test_unstable_crystals():
    # springs under compression: transverse modes are imaginary
    cell = build_crystal("sc", 2.5, elements="Ar")
    result = phonons(cell, Harmonic(k=1.0, r0=3.0), supercell=3)
    lowest = result.frequencies([0, 0.25, 0]).min()
    assert lowest < 0
    # the density of states shows them as negative frequencies
    for sigma in (None, 0.1):
        dos = result.dos(6, sigma=sigma)
        assert dos.frequencies[0] < lowest and dos.dos[dos.frequencies < 0].max() > 0
        assert np.trapezoid(dos.dos, dos.frequencies) == pytest.approx(3, rel=0.05)
    # the thermal properties have no meaning
    with pytest.raises(InvalidParameter, match="imaginary frequencies"):
        result.thermal([300], mesh=4)


def test_primitive_cells():
    for structure, elements, atoms in [
        ("fcc", "Cu", 1),
        ("bcc", "Fe", 1),
        ("diamond", "Si", 2),
        ("hcp", "Mg", 2),
        ("rocksalt", ("Na", "Cl"), 2),
        ("perovskite", ("Sr", "Ti", "O"), 5),
    ]:
        crystal = build_crystal(structure, 3.6, elements=elements, repeat=(2, 3, 2))
        cell = primitive_cell(crystal)
        assert len(cell) == atoms
        assert cell.box.volume == pytest.approx(
            crystal.box.volume * atoms / len(crystal)
        )
        assert np.linalg.det(cell.box.matrix) > 0
        np.testing.assert_allclose(cell.box.origin, crystal.box.origin)
        names = crystal.table["element"].to_list()
        share = [
            e for e in set(names) for _ in range(names.count(e) * atoms // len(crystal))
        ]
        assert sorted(cell.table["element"].to_list()) == sorted(share)
        # the cell repeated is the crystal: each atom of the crystal is an
        # atom of the cell moved by a lattice vector of the cell
        fractional = cell.box.to_fractional(
            crystal["positions"][:, None, :] - cell["positions"][None]
        )
        whole = np.abs(fractional - np.round(fractional)).max(axis=2) < 1e-9
        assert whole.any(axis=1).all()
    # an already primitive cell, and a random alloy, are returned as they are
    single = primitive_cell(build_crystal("fcc", 3.6, elements="Cu"))
    assert primitive_cell(single) is single
    alloy = random_alloy(
        build_crystal("fcc", 3.6, repeat=(2, 2, 2)), {"Cu": 0.5, "Ni": 0.5}, seed=3
    )
    assert primitive_cell(alloy) is alloy
    # within the tolerance only
    jitter = np.random.default_rng(4).normal(scale=1e-3, size=(32, 3))
    shaken = build_crystal("fcc", 3.6, elements="Cu", repeat=(2, 2, 2))
    shaken = shaken.with_columns(positions=shaken["positions"] + jitter)
    assert len(primitive_cell(shaken)) == 32
    assert len(primitive_cell(shaken, tolerance=0.01)) == 1
    # atoms without elements, types or masses are all alike
    bare = build_crystal("fcc", 3.6, repeat=(2, 2, 2)).drop("element")
    assert len(primitive_cell(bare)) == 1
    # types count as much as elements
    typed = build_crystal("fcc", 3.6, repeat=(2, 2, 2)).with_columns(
        type=np.arange(32) % 2 + 1
    )
    assert len(primitive_cell(typed)) > 1


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda c, s: phonons(c.with_box(mdapy.Box(3.6, pbc=False)), s), "periodic"),
        (lambda c, s: phonons(c, s, displacement=0.0), "displacement must be positive"),
        (lambda c, s: phonons(c, s, supercell=(1, 0, 1)), "supercell must be"),
        (lambda c, s: phonons(c, s, supercell=1.5), "supercell must be"),
        (
            lambda c, s: phonons(c, s, supercell=[[1, 0, 0], [0, 1, 0], [0, 0, 0]]),
            "positive determinant",
        ),
        (lambda c, s: phonons(c, s, supercell=np.eye(3) * 1.5), "must hold integers"),
        (lambda c, s: phonons(c.drop("element"), s), "mass"),
        (
            lambda c, s: primitive_cell(c.with_box(mdapy.Box(3.6, pbc=False))),
            "periodic",
        ),
        (lambda c, s: phonons(c, s, supercell=1).frequencies([1, 2]), "wave vector"),
        (lambda c, s: phonons(c, s, supercell=1).dos(0), "mesh must be"),
        (lambda c, s: phonons(c, s, supercell=1).dos(2, sigma=0), "sigma"),
        (lambda c, s: phonons(c, s, supercell=1).thermal([-1]), "not be negative"),
        (lambda c, s: phonons(c, s, supercell=1).band_structure("bct"), "path must be"),
        (
            lambda c, s: phonons(c, s, supercell=1).band_structure([[("Γ", 0)]]),
            "two or more",
        ),
        (lambda c, s: phonons(c, s, supercell=1).band_structure([("Γ",)]), "sections"),
        (
            lambda c, s: phonons(c, s, supercell=1).band_structure("fcc", points=1),
            "points",
        ),
    ],
)
def test_invalid_arguments(call, message):
    cell, springs = springs_fcc()
    with pytest.raises(InvalidParameter, match=message):
        call(cell, springs)


def test_calculator_without_forces():
    class EnergyOnly(Harmonic):
        implemented = frozenset({"energy"})

    cell, _ = springs_fcc()
    with pytest.raises(InvalidParameter, match="does not compute forces"):
        phonons(cell, EnergyOnly())


def test_band_dat_files(tmp_path):
    # phonopy's band.dat: the ends of the lines without 2π, then each branch
    # line by line, a blank line after each line and another after each branch
    cell, springs = springs_fcc()
    result = phonons(primitive_cell(cell), springs, supercell=3)
    bands = result.band_structure("fcc", points=4)
    path = tmp_path / "band.dat"
    bands.write(path)
    text = path.read_text().split("\n")
    assert text[0] == "# End points of segments: "
    ends = np.array(text[1].lstrip("#").split(), dtype=float)
    np.testing.assert_allclose(
        ends, [d / (2 * np.pi) for d, _ in bands.labels], atol=1e-8
    )
    assert (
        text[2 : 2 + 4] and text[6] == "" and text[7].split()[0] == text[5].split()[0]
    )
    assert text.count("") == 3 * (6 + 1) + 1  # a blank line after each line and branch
    again = read_band_structure(path)
    np.testing.assert_allclose(again.distances, bands.distances, atol=1e-5)
    np.testing.assert_allclose(again.frequencies, bands.frequencies, atol=1e-6)
    assert np.isnan(again.q).all() and [x for _, x in again.labels] == [""] * 7
    # written again, the same file
    copy = tmp_path / "copy.dat"
    again.write(copy)
    assert copy.read_text() == path.read_text()
    # labels that are not the ends of the lines: the ends from the distances
    path_by_hand = [[("A", [0, 0, 0]), ("B", [0.5, 0, 0]), ("C", [0.5, 0.5, 0])]]
    odd = result.band_structure(path_by_hand, points=3)
    odd = BandStructure(odd.q, odd.distances, odd.frequencies, odd.labels[:1])
    odd.write(tmp_path / "odd.dat")
    assert len(read_band_structure(tmp_path / "odd.dat").labels) == 3
    # a file without its last blank lines
    (tmp_path / "short.dat").write_text(path.read_text().rstrip("\n") + "\n")
    np.testing.assert_array_equal(
        read_band_structure(tmp_path / "short.dat").frequencies, again.frequencies
    )
    (tmp_path / "bad.dat").write_text("# End points\n# a b\n")
    with pytest.raises(FileFormatError, match="not a band.dat"):
        read_band_structure(tmp_path / "bad.dat")
