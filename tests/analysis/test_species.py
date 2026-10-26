# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import numpy as np
import pytest

import mdapy
from mdapy.analysis import ChemicalSpecies
from mdapy.elements import covalent_radii, vdw_radii
from mdapy.errors import InvalidParameter, MissingProperty

# molecules as (element, x, y, z) in Å
WATER = [("O", 0, 0, 0.119), ("H", 0, 0.763, -0.477), ("H", 0, -0.763, -0.477)]
CARBON_DIOXIDE = [("C", 0, 0, 0), ("O", 0, 0, 1.16), ("O", 0, 0, -1.16)]
METHANE = [("C", 0, 0, 0)] + [
    ("H", *(0.629 * np.array(v)))
    for v in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))
]
AMMONIA = [("N", 0, 0, 0.117), ("H", 0, 0.94, -0.273), ("H", 0.814, -0.47, -0.273), ("H", -0.814, -0.47, -0.273)]  # fmt: skip
MOLECULES = {"H2O": WATER, "CO2": CARBON_DIOXIDE, "CH4": METHANE, "H3N": AMMONIA}


def gas(counts, length=24.0, seed=0, spacing=6.0):
    """Molecules at random places and orientations in a periodic box, some
    across its faces."""
    rng = np.random.default_rng(seed)
    positions, elements, centers = [], [], []
    for name, count in counts.items():
        for _ in range(count):
            atoms = MOLECULES[name]
            local = np.array([a[1:] for a in atoms], dtype=float)
            rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
            while True:
                center = rng.random(3) * length
                apart = [
                    c - center - length * np.round((c - center) / length)
                    for c in centers
                ]
                if all(np.linalg.norm(d) > spacing for d in apart):
                    break
            centers.append(center)
            positions.append(local @ rotation.T + center)
            elements += [a[0] for a in atoms]
    box = mdapy.Box(length)
    wrapped, _ = box.wrap(np.concatenate(positions))
    return mdapy.System(wrapped, box, elements=elements)


def brute_force(system, lengths):
    """Molecule partition by union-find over all pairs, minimum images."""
    x, n = system["positions"], len(system)
    names = system.table["element"].cast(str).to_list()
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    length = system.box.matrix[0, 0]
    for i in range(n):
        d = x[i + 1 :] - x[i]
        d -= length * np.round(d / length)
        r = np.linalg.norm(d, axis=1)
        for k in np.flatnonzero(r > 0):
            j = i + 1 + k
            pair = (
                (names[i], names[j])
                if (names[i], names[j]) in lengths
                else (names[j], names[i])
            )
            if r[k] < lengths.get(pair, 0):
                parent[root(i)] = root(j)
    return np.array([root(i) for i in range(n)])


def same_partition(a, b):
    pairs = set(zip(a.tolist(), b.tolist()))
    return len(pairs) == len(set(a.tolist())) == len(set(b.tolist()))


def test_a_gas_of_molecules():
    counts = {"H2O": 20, "CO2": 10, "CH4": 8, "H3N": 5}
    system = gas(counts)
    result = ChemicalSpecies().compute(system)
    table = result.counts()
    assert dict(zip(table["formula"], table["count"])) == counts
    assert table["formula"].to_list() == [
        "H2O",
        "CO2",
        "CH4",
        "H3N",
    ]  # most common first
    assert len(result.formulas) == len(result.sizes) == 43
    assert result.molecule.min() == 1 and np.all(np.diff(result.sizes) <= 0)
    assert np.bincount(result.molecule)[1:].tolist() == result.sizes.tolist()
    assert result.columns()["molecule"] is result.molecule
    # the van der Waals rule of VMD and OVITO finds the same molecules
    vdw = ChemicalSpecies("vdw").compute(system)
    assert same_partition(vdw.molecule, result.molecule)


@pytest.mark.parametrize("seed", [1, 2])
def test_random_atoms_against_brute_force(seed):
    # a dense random mixture: bonds of every length
    rng = np.random.default_rng(seed)
    length = 9.0
    elements = rng.choice(["H", "C", "N", "O"], 120).tolist()
    system = mdapy.System(
        rng.random((120, 3)) * length, mdapy.Box(length), elements=elements
    )
    for species in (
        ChemicalSpecies(),
        ChemicalSpecies("vdw", scale=0.5, tolerance=0.1),
        ChemicalSpecies("vdw", cutoffs={("H", "H"): 1.0}),
        ChemicalSpecies({"H": 0.4, "C": 0.8, "N": 0.75, "O": 0.7}, tolerance=0.2),
        ChemicalSpecies("vdw"),
        ChemicalSpecies(cutoffs={("O", "H"): 0.0, ("C", "C"): 2.0}),
    ):
        result = species.compute(system)
        reference = brute_force(system, species.bond_lengths(["H", "C", "N", "O"]))
        assert same_partition(result.molecule, reference)


def test_against_ase():
    pytest.importorskip("scipy")
    ase_neighbors = pytest.importorskip("ase.neighborlist")
    from scipy.sparse.csgraph import connected_components

    from mdapy.interop import to_ase

    rng = np.random.default_rng(3)
    elements = rng.choice(["H", "C", "O"], 150).tolist()
    system = mdapy.System(
        rng.random((150, 3)) * 10.0, mdapy.Box(10.0), elements=elements
    )
    atoms = to_ase(system)
    # ASE bonds atoms closer than the sum of their cutoffs
    cutoffs = ase_neighbors.natural_cutoffs(atoms, mult=1.0)
    cutoffs = [c + 0.45 / 2 for c in cutoffs]
    nl = ase_neighbors.NeighborList(
        cutoffs, skin=0.0, self_interaction=False, bothways=True
    )
    nl.update(atoms)
    _, theirs = connected_components(nl.get_connectivity_matrix(sparse=True))
    ours = ChemicalSpecies().compute(system)
    assert same_partition(ours.molecule, theirs)
    # and the same Hill formulas
    for m in range(1, 6):
        atoms_of = atoms[np.flatnonzero(ours.molecule == m)]
        assert ours.formulas[m - 1] == atoms_of.get_chemical_formula(mode="hill")


def test_hill_formulas():
    species = ChemicalSpecies()
    for elements, formula in [
        (["O", "H", "H"], "H2O"),
        (["N", "H", "H", "H"], "H3N"),
        (["H", "O", "C", "C", "H", "H", "H", "H", "H"], "C2H6O"),
        (["Cl", "Na"], "ClNa"),
        (["C", "O", "O"], "CO2"),
    ]:
        around = np.linspace(-0.6, 0.6, len(elements) - 1)
        positions = [[0.0, 0.0, 0.0]] + [[0.9, y, 0.0] for y in around]
        system = mdapy.System(positions, mdapy.Box(20.0, pbc=False), elements=elements)
        bonded = ChemicalSpecies(
            cutoffs={(a, b): 3.0 for a in elements for b in elements}
        )
        assert bonded.compute(system).formulas.tolist() == [formula]
    # atoms never bonded stand alone
    lone = mdapy.System([[0, 0, 0], [5, 5, 5]], mdapy.Box(20.0), elements=["Ar", "Ar"])
    result = species.compute(lone)
    assert result.formulas.tolist() == ["Ar", "Ar"] and result.sizes.tolist() == [1, 1]
    alone = ChemicalSpecies(cutoffs={("Ar", "Ar"): 0}).compute(lone)
    assert alone.molecule.tolist() == [1, 2] and alone.counts()["count"].to_list() == [
        2
    ]


def test_frames():
    first = gas({"H2O": 6, "CO2": 2}, seed=4)
    second = gas({"H2O": 4, "CH4": 3}, seed=5)
    table = ChemicalSpecies().count([first, second, first])
    assert table.columns == ["frame", "H2O", "CO2", "CH4"]  # by total count
    assert table["frame"].to_list() == [0, 1, 2]
    assert table["H2O"].to_list() == [6, 4, 6]
    assert table["CO2"].to_list() == [2, 0, 2] and table["CH4"].to_list() == [0, 3, 0]


def test_radii():
    np.testing.assert_allclose(
        covalent_radii(["H", "C", "O", "Cu"]), [0.31, 0.76, 0.66, 1.32]
    )
    np.testing.assert_allclose(vdw_radii([1, 6, 8]), [1.2, 1.77, 1.5])
    with pytest.raises(InvalidParameter, match="no covalent radius for X"):
        covalent_radii(["X"])
    with pytest.raises(InvalidParameter, match="no van der Waals radius for Rn"):
        vdw_radii(["Rn"])
    with pytest.raises(InvalidParameter, match="no element 200"):
        vdw_radii([200])
    lengths = ChemicalSpecies().bond_lengths(["O", "H"])
    assert lengths == {
        ("O", "O"): pytest.approx(2 * 0.66 + 0.45),
        ("O", "H"): pytest.approx(0.66 + 0.31 + 0.45),
        ("H", "H"): pytest.approx(2 * 0.31 + 0.45),
    }
    # van der Waals radii: 0.6 of their sum, two hydrogen atoms never bonded
    lengths = ChemicalSpecies("vdw").bond_lengths(["O", "H"])
    assert lengths == {
        ("O", "O"): pytest.approx(0.6 * 2 * 1.5),
        ("O", "H"): pytest.approx(0.6 * (1.5 + 1.2)),
        ("H", "H"): 0.0,
    }
    lengths = ChemicalSpecies("vdw", cutoffs={("H", "H"): 0.9}).bond_lengths(["H"])
    assert lengths == {("H", "H"): 0.9}
    # radii by element: closer than their sum
    lengths = ChemicalSpecies({"H": 0.4, "O": 0.7}).bond_lengths(["O", "H"])
    assert lengths[("O", "H")] == pytest.approx(1.1)


def test_invalid_arguments():
    for arguments, message in [
        ({"radii": "ionic"}, "radii must be"),
        ({"radii": {"Xx": 1.0}}, "unknown element"),
        ({"radii": {"H": 0.0}}, "must be positive"),
        ({"scale": 0}, "scale must be positive"),
        ({"tolerance": -0.1}, "tolerance must not be negative"),
        ({"cutoffs": {("H", "Qq"): 1.0}}, "unknown element"),
        ({"cutoffs": {("H", "O"): -1.0}}, "must not be negative"),
    ]:
        with pytest.raises(InvalidParameter, match=message):
            ChemicalSpecies(**arguments)
    system = gas({"H2O": 2})
    with pytest.raises(MissingProperty, match="element"):
        ChemicalSpecies().compute(system.drop("element"))
    with pytest.raises(InvalidParameter, match="no radius given for"):
        ChemicalSpecies({"H": 0.3}).compute(system)
    assert repr(ChemicalSpecies({"H": 0.3})) == (
        "ChemicalSpecies(radii='by element', scale=1.0, tolerance=0.0)"
    )
    assert repr(ChemicalSpecies("vdw")) == (
        "ChemicalSpecies(radii='vdw', scale=0.6, tolerance=0.0)"
    )
