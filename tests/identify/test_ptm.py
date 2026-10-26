# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Polyhedral template matching: analytic structures, distances, strains,
orientations and orderings, and OVITO 3.17.0 atom by atom."""

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.errors import InvalidParameter, MissingProperty
from mdapy.identify import PolyhedralTemplateMatching, PTMResult

from lattices import icosahedron

EVERY = PolyhedralTemplateMatching.NAMES[1:]

# structure, lattice constant, elements, the type PTM finds and the
# distance between nearest neighbors
CRYSTALS = {
    "fcc": (3.6, "Cu", "fcc", 3.6 / np.sqrt(2)),
    "hcp": (2.5, "Mg", "hcp", 2.5),
    "bcc": (2.87, "Fe", "bcc", 2.87 * np.sqrt(3) / 2),
    "sc": (2.5, "Po", "sc", 2.5),
    "diamond": (5.43, "Si", "cubic_diamond", 5.43 * np.sqrt(3) / 4),
    "lonsdaleite": (2.51, "C", "hexagonal_diamond", 2.51 * np.sqrt(3 / 8)),
    "graphene": (2.46, "C", "graphene", 2.46 / np.sqrt(3)),
}


def make(kind, a, elements, repeat=(4, 4, 4), jitter=0.0, seed=0):
    extra = {"c": 10.0} if kind == "graphene" else {}
    if kind == "graphene":
        repeat = (repeat[0], repeat[1], 1)
    system = build_crystal(kind, a, elements=elements, repeat=repeat, **extra)
    if jitter:
        rng = np.random.default_rng(seed)
        noise = rng.normal(scale=jitter, size=system.positions.shape)
        if kind == "graphene":
            noise[:, 2] = 0.0
        system = system.with_positions(system.positions + noise)
    return system


def rotation(quaternion):
    """The rotation matrix of a quaternion (x, y, z, w)."""
    x, y, z, w = quaternion / np.linalg.norm(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


# ---------------------------------------------------------------- analytic


@pytest.mark.parametrize("kind", list(CRYSTALS))
def test_perfect_crystals(kind):
    a, element, expected, distance = CRYSTALS[kind]
    system = make(kind, a, element)
    result = PolyhedralTemplateMatching(EVERY).compute(system)
    assert isinstance(result, PTMResult)
    assert result.counts()[expected] == len(system)
    np.testing.assert_allclose(result.rmsd, 0.0, atol=1e-6)
    np.testing.assert_allclose(result.interatomic_distance, distance, rtol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(result.orientation, axis=1), 1.0)
    assert result.deformation_gradient is None and result.ordering is None
    # only the structures asked for
    others = tuple(name for name in EVERY if name != expected)
    assert PolyhedralTemplateMatching(others).compute(system).counts()[expected] == 0


def test_an_icosahedron():
    cluster = icosahedron()
    result = PolyhedralTemplateMatching(EVERY).compute(cluster)
    assert result.structure[0] == PolyhedralTemplateMatching.NAMES.index("ico")
    assert (result.structure[1:] == 0).all()
    assert result.rmsd[0] == pytest.approx(0.0, abs=1e-6)
    assert result.interatomic_distance[0] > 0
    # ico is not looked for by default
    assert PolyhedralTemplateMatching().compute(cluster).structure[0] == 0


def test_a_homogeneous_strain():
    """The deformation gradient of every atom is the strain, at the scale of
    the fitted interatomic distance."""
    system = make("fcc", 3.6, "Cu", repeat=(3, 3, 3))
    strain = np.array([[1.02, 0.01, 0.0], [0.0, 0.99, 0.015], [0.0, 0.0, 1.01]])
    box = mdapy.Box(system.box.matrix @ strain.T)
    strained = system.with_box(box).with_positions(system.positions @ strain.T)
    result = PolyhedralTemplateMatching(deformation_gradient=True).compute(strained)
    assert result.counts()["fcc"] == len(system)
    scaled = result.deformation_gradient * result.interatomic_distance[:, None, None]
    np.testing.assert_allclose(
        scaled, np.broadcast_to(strain * 3.6 / np.sqrt(2), scaled.shape), atol=1e-9
    )
    # unstrained: the identity
    plain = PolyhedralTemplateMatching(deformation_gradient=True).compute(system)
    np.testing.assert_allclose(
        plain.deformation_gradient,
        np.broadcast_to(np.eye(3), (len(system), 3, 3)),
        atol=1e-9,
    )


def test_orientations_of_a_rotated_crystal():
    """The rotation of each atom takes the cubic axes onto the directions
    the crystal was built along, up to the symmetry of the cube."""
    directions = [(1, -1, 0), (1, 1, -2), (1, 1, 1)]
    system = build_crystal(
        "fcc", 3.6, elements="Cu", orientation=directions, repeat=(2, 2, 1)
    )
    result = PolyhedralTemplateMatching().compute(system)
    assert result.counts()["fcc"] == len(system)
    for quaternion in result.orientation:
        rotated = rotation(quaternion)
        for axis, direction in zip(np.eye(3), directions):
            crystal = rotated.T @ axis  # the lab axis in the frame of the crystal
            unit = np.array(direction) / np.linalg.norm(direction)
            np.testing.assert_allclose(
                np.sort(np.abs(crystal)), np.sort(np.abs(unit)), atol=1e-8
            )
    # the standard orientation
    cubic = PolyhedralTemplateMatching().compute(
        make("fcc", 3.6, "Cu", repeat=(2, 2, 2))
    )
    np.testing.assert_allclose(
        np.abs(cubic.orientation), np.broadcast_to([0, 0, 0, 1.0], (32, 4)), atol=1e-8
    )


def test_the_same_after_a_rotation_of_the_system():
    system = make("bcc", 2.87, "Fe", repeat=(4, 4, 4), jitter=0.05, seed=3)
    q, _ = np.linalg.qr(np.random.default_rng(3).normal(size=(3, 3)))
    q[:, 0] *= np.sign(np.linalg.det(q))
    turned = mdapy.System(
        system.positions @ q.T, mdapy.Box(system.box.matrix @ q.T), elements="Fe"
    )
    first = PolyhedralTemplateMatching(EVERY).compute(system)
    second = PolyhedralTemplateMatching(EVERY).compute(turned)
    np.testing.assert_array_equal(first.structure, second.structure)
    np.testing.assert_allclose(first.rmsd, second.rmsd, atol=1e-9)
    np.testing.assert_allclose(
        first.interatomic_distance, second.interatomic_distance, atol=1e-9
    )


@pytest.mark.parametrize(
    "kind, a, elements, expected",
    [
        ("cscl", 3.0, ("Cs", "Cl"), {"B2": 1.0}),
        ("zincblende", 5.43, ("Ga", "As"), {"zincblende_wurtzite": 1.0}),
        ("wurtzite", 3.19, ("Ga", "N"), {"zincblende_wurtzite": 1.0}),
        ("graphene", 2.5, ("B", "N"), {"boron_nitride": 1.0}),
        ("l1_2", 3.75, ("Cu", "Au"), {"L12_A": 0.75, "L12_B": 0.25}),
        ("fcc", 3.6, "Cu", {"pure": 1.0}),
    ],
)
def test_chemical_ordering(kind, a, elements, expected):
    system = make(kind, a, elements)
    result = PolyhedralTemplateMatching(EVERY, ordering=True).compute(system)
    names = PolyhedralTemplateMatching.ORDERING_NAMES
    assert result.ordering_names == names
    found = np.bincount(result.ordering, minlength=len(names)) / len(system)
    np.testing.assert_allclose(found, [expected.get(name, 0.0) for name in names])
    if kind == "l1_2":  # the A sites are those of the majority
        majority = system.table["element"].cast(str).to_numpy() == "Cu"
        assert (result.ordering[majority] == names.index("L12_A")).all()


def test_l10_ordering():
    """Layers of two elements alternating along z."""
    system = make("fcc", 3.8, "Cu")
    layer = np.round(system.positions[:, 2] / 1.9).astype(int) % 2
    alloy = system.with_columns(element=np.where(layer == 0, "Cu", "Au"))
    result = PolyhedralTemplateMatching(ordering=True).compute(alloy)
    assert (
        result.ordering == PolyhedralTemplateMatching.ORDERING_NAMES.index("L10")
    ).all()


def test_the_rmsd_cutoff():
    system = make("fcc", 3.6, "Cu", jitter=0.12, seed=1)
    loose = PolyhedralTemplateMatching(rmsd_cutoff=0).compute(system)
    assert loose.counts()["fcc"] > 0.9 * len(system)
    strict = PolyhedralTemplateMatching(rmsd_cutoff=0.05).compute(system)
    above = loose.rmsd > 0.05
    assert above.any() and (~above).any()
    np.testing.assert_array_equal(strict.structure, np.where(above, 0, loose.structure))
    # as OVITO: the other results stay those of the best template
    np.testing.assert_array_equal(strict.rmsd, loose.rmsd)
    np.testing.assert_array_equal(strict.orientation, loose.orientation)


def test_template_neighbors_and_correspondences():
    result = PolyhedralTemplateMatching(EVERY, template_neighbors=True).compute(
        make("diamond", 5.43, "Si", repeat=(3, 3, 3), jitter=0.02)
    )
    assert result.neighbors.shape == (216, 16)
    np.testing.assert_allclose(
        result.interatomic_distance, 5.43 * np.sqrt(3) / 4, atol=0.05
    )
    # 4 nearest, then the 12 next of a diamond structure
    assert (result.neighbors >= 0).all()
    system = make("bcc", 2.87, "Fe", repeat=(3, 3, 3))
    bcc = PolyhedralTemplateMatching(template_neighbors=True).compute(system)
    assert (bcc.neighbors[:, :14] >= 0).all() and (bcc.neighbors[:, 14:] == -1).all()
    nl = mdapy.NeighborList.build(system, num_neighbors=14)
    for i in range(len(system)):
        js = nl.j[nl.offsets[i] : nl.offsets[i + 1]]
        assert set(bcc.neighbors[i, :14]) == set(js)
    assert (bcc.correspondences > 0).all()
    # no match, no neighbors
    gas = mdapy.System(np.random.default_rng(0).random((40, 3)) * 20, mdapy.Box(20.0))
    none = PolyhedralTemplateMatching(template_neighbors=True, rmsd_cutoff=0).compute(
        gas
    )
    unmatched = none.structure == 0
    assert unmatched.any()
    assert (none.neighbors[unmatched] == -1).all()
    assert (none.rmsd[unmatched] == 0).all()
    assert (none.correspondences[unmatched] == 0).all()


def test_neighbor_lists_threads_and_small_systems():
    system = make("hcp", 2.5, "Mg", jitter=0.05, seed=4)
    analysis = PolyhedralTemplateMatching(EVERY, deformation_gradient=True)
    reference = analysis.compute(system)
    given = analysis.compute(system, neighbors=mdapy.NeighborList.build(system, 5.0))
    for name in ("structure", "rmsd", "orientation", "deformation_gradient"):
        np.testing.assert_array_equal(getattr(given, name), getattr(reference, name))
    with mdapy.num_threads(3):
        threaded = analysis.compute(system)
    np.testing.assert_array_equal(threaded.rmsd, reference.rmsd)
    with pytest.raises(InvalidParameter, match="fewer than 18"):
        analysis.compute(
            system, neighbors=mdapy.NeighborList.build(system, num_neighbors=12)
        )
    # clusters of fewer atoms than the neighbors looked for, and none
    for count in (0, 1, 7, 19):
        cluster = system.gather(np.arange(count))
        cluster = cluster.with_box(mdapy.Box(cluster.box.matrix, pbc=False))
        assert len(analysis.compute(cluster).structure) == count


def test_columns_errors_and_repr():
    system = make("l1_2", 3.75, ("Cu", "Au"), repeat=(2, 2, 2))
    result = PolyhedralTemplateMatching(
        deformation_gradient=True, ordering=True
    ).compute(system)
    labeled = system.with_columns(result)
    for name in (
        "structure",
        "rmsd",
        "interatomic_distance",
        "orientation",
        "deformation_gradient",
        "ordering",
    ):
        assert name in labeled.table.columns
    assert labeled.table["orientation"].dtype.shape == (4,)
    assert set(PolyhedralTemplateMatching().compute(system).columns()) == {
        "structure",
        "rmsd",
        "interatomic_distance",
        "orientation",
    }
    with pytest.raises(InvalidParameter, match="unknown structure"):
        PolyhedralTemplateMatching(("fcc", "other"))
    with pytest.raises(InvalidParameter, match="at least one"):
        PolyhedralTemplateMatching(())
    with pytest.raises(InvalidParameter, match="rmsd_cutoff"):
        PolyhedralTemplateMatching(rmsd_cutoff=-0.1)
    with pytest.raises(MissingProperty, match="element column"):
        PolyhedralTemplateMatching(ordering=True).compute(
            mdapy.System(system.positions, system.box)
        )
    assert PolyhedralTemplateMatching("bcc").structures == ("bcc",)
    assert repr(PolyhedralTemplateMatching()) == (
        "PolyhedralTemplateMatching(structures=('fcc', 'hcp', 'bcc'), rmsd_cutoff=0.1)"
    )


# ---------------------------------------------------------------- OVITO


def ovito_ptm(system, structures, ordering=False):
    pytest.importorskip("ovito")
    from ovito.modifiers import PolyhedralTemplateMatchingModifier

    from ovito_reference import ovito_compute

    modifier = PolyhedralTemplateMatchingModifier(
        output_rmsd=True,
        output_interatomic_distance=True,
        output_orientation=True,
        output_deformation_gradient=True,
        output_ordering=ordering,
    )
    for t in modifier.structures:
        t.enabled = t.id in [
            PolyhedralTemplateMatching.NAMES.index(s) for s in structures
        ]
    return ovito_compute(system, modifier).particles


def same_as_ovito(system, structures, ordering=False):
    theirs = ovito_ptm(system, structures, ordering)
    ours = PolyhedralTemplateMatching(
        structures, deformation_gradient=True, ordering=ordering
    ).compute(system)
    np.testing.assert_array_equal(ours.structure, np.asarray(theirs["Structure Type"]))
    np.testing.assert_allclose(ours.rmsd, np.asarray(theirs["RMSD"]), atol=1e-12)
    np.testing.assert_allclose(
        ours.interatomic_distance,
        np.asarray(theirs["Interatomic Distance"]),
        atol=1e-12,
    )
    # OVITO keeps orientations in single precision, matrices by column
    np.testing.assert_allclose(
        ours.orientation, np.asarray(theirs["Orientation"]), atol=1e-6
    )
    np.testing.assert_allclose(
        ours.deformation_gradient.transpose(0, 2, 1).reshape(-1, 9),
        np.asarray(theirs["Elastic Deformation Gradient"]),
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        ours.correspondences, np.asarray(theirs["Correspondences"]).astype(np.uint64)
    )
    if ordering:
        np.testing.assert_array_equal(
            ours.ordering, np.asarray(theirs["Ordering Type"])
        )
    return ours


@pytest.mark.parametrize("kind", list(CRYSTALS))
@pytest.mark.parametrize("box", ["periodic", "rotated", "cluster"])
def test_like_ovito(kind, box):
    a, element, expected, _ = CRYSTALS[kind]
    jitter = 0.03 if kind in ("diamond", "lonsdaleite", "graphene") else 0.08
    system = make(kind, a, element, jitter=jitter, seed=7)
    if box == "rotated":
        q, _ = np.linalg.qr(np.random.default_rng(7).normal(size=(3, 3)))
        q[:, 0] *= np.sign(np.linalg.det(q))
        system = mdapy.System(
            system.positions @ q.T + 1.5,
            mdapy.Box(
                system.box.matrix @ q.T, origin=[1.5, 1.5, 1.5], pbc=system.box.pbc
            ),
            elements=system.table["element"].to_list(),
        )
    elif box == "cluster":
        system = system.with_box(mdapy.Box(system.box.matrix, pbc=False))
    ours = same_as_ovito(system, EVERY)
    assert ours.counts()[expected] > 0


def test_a_polycrystal_like_ovito():
    from mdapy.build import build_polycrystal

    copper = build_crystal("fcc", 3.615, elements="Cu")
    perfect = build_polycrystal(copper, mdapy.Box(40.0), 5, seed=2)
    # neighbors at exactly the same distance, as at the boundaries of perfect
    # grains, may come in another order than in OVITO, which changes what
    # atoms that match nothing match best; their types are the same
    np.testing.assert_array_equal(
        PolyhedralTemplateMatching(("fcc", "hcp", "bcc", "ico"))
        .compute(perfect)
        .structure,
        np.asarray(ovito_ptm(perfect, ("fcc", "hcp", "bcc", "ico"))["Structure Type"]),
    )
    rng = np.random.default_rng(2)
    grains = perfect.with_positions(
        perfect.positions + rng.normal(scale=0.01, size=perfect.positions.shape)
    )
    ours = same_as_ovito(grains, ("fcc", "hcp", "bcc", "ico"))
    assert ours.counts()["fcc"] > 0.4 * len(grains)
    # the default structures and RMSD cutoff
    theirs = ovito_ptm(grains, ("fcc", "hcp", "bcc"))
    np.testing.assert_array_equal(
        PolyhedralTemplateMatching().compute(grains).structure,
        np.where(
            np.asarray(theirs["RMSD"]) > 0.1, 0, np.asarray(theirs["Structure Type"])
        ),
    )


@pytest.mark.parametrize(
    "kind, a, elements",
    [
        ("l1_2", 3.75, ("Cu", "Au")),
        ("cscl", 3.0, ("Cs", "Cl")),
        ("zincblende", 5.43, ("Ga", "As")),
        ("wurtzite", 3.19, ("Ga", "N")),
        ("graphene", 2.5, ("B", "N")),
    ],
)
def test_ordering_like_ovito(kind, a, elements):
    system = make(kind, a, elements, jitter=0.03, seed=5)
    same_as_ovito(system, EVERY, ordering=True)
