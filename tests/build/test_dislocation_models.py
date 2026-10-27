# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest

import mdapy
import mdapy.build._dislocation_models as models
from mdapy.build import (
    DISLOCATIONS,
    Dislocation,
    build_crystal,
    dislocation_cylinder,
    dislocation_quadrupole,
    insert_dislocation,
)
from mdapy.build._anisotropic import StraightDislocation, rotated, stiffness
from mdapy.calculators import EAM
from mdapy.errors import InvalidParameter
from mdapy.identify import CommonNeighborAnalysis, DiamondStructure
from mdapy.simulation import minimize

from atomsk_reference import ATOMSK, atomsk_apply

FE = (2.865, (243.3, 145.0, 116.1), "Fe")
CU = (3.615, (168.4, 121.4, 75.4), "Cu")
SI = (5.431, (165.6, 63.9, 79.5), "Si")
MATERIAL = {"bcc": FE, "fcc": CU, "diamond": SI}


def frame(*directions):
    axes = np.array(directions, dtype=float)
    return axes / np.linalg.norm(axes, axis=1)[:, None]


def perfect(system, structure):
    """Whether each atom is in the perfect crystal."""
    if structure == "diamond":
        return np.asarray(DiamondStructure().compute(system).structure) == 1
    kind = {"fcc": 1, "bcc": 3}[structure]
    return np.asarray(CommonNeighborAnalysis().compute(system).structure) == kind


# ---------------------------------------------------------------- the field


@pytest.mark.skipif(ATOMSK is None, reason="atomsk is not installed")
@pytest.mark.parametrize("material", [FE, CU], ids=["Fe", "Cu"])
@pytest.mark.parametrize(
    "line, normal, burgers",
    [
        ("z", "y", (0.0, 0.0, 2.48)),
        ("z", "y", (1.2, 0.0, 0.8)),
        ("z", "y", (-2.0, 0.0, 0.0)),
    ],
)
def test_anisotropic_as_atomsk(tmp_path, material, line, normal, burgers):
    """The displacements of atomsk with an elastic tensor and the opposite
    Burgers vector (atomsk follows Hirth and Lothe), to rounding, for the
    line along z and the normal along y (for other axes atomsk takes other
    signs; test_any_line_axis and test_isotropic_limit cover them)."""
    a, elastic, element = material
    orientation = [(1, 1, -2), (-1, 1, 0), (1, 1, 1)]
    cell = build_crystal(
        "bcc", a, elements=element, orientation=orientation, repeat=(5, 8, 2)
    )
    c11, c12, c44 = elastic
    tensor = (
        f"elastic Voigt\n{c11} {c11} {c11}\n{c12} {c12} {c12}\n{c44} {c44} {c44}\n"
        "orientation\n11-2\n-110\n111\n"
    )
    center = cell.box.matrix.diagonal() / 2 + [0.31, 0.37, 0.23]
    ours = insert_dislocation(
        cell,
        burgers,
        line=line,
        normal=normal,
        center=center,
        elastic=elastic,
        orientation=orientation,
    )
    # atomsk: the position in the plane after the line, the Burgers vector
    # along the third axis, the normal and the line
    a3, a2 = "xyz".index(line), "xyz".index(normal)
    a1 = 3 - a2 - a3
    p1, p2 = center[(a3 + 1) % 3], center[(a3 + 2) % 3]
    b = -np.asarray(burgers)
    options = ["-prop", "ctensor.txt", "-disloc", p1, p2, "mixed", line, normal]
    theirs = atomsk_apply(
        tmp_path, cell, *options, b[a1], b[a2], b[a3], files={"ctensor.txt": tensor}
    )
    np.testing.assert_allclose(ours["positions"], theirs["positions"], atol=1e-9)


FIELDS = [
    ("Fe screw", FE[1], frame((1, 1, -2), (-1, 1, 0), (1, 1, 1)), (0.0, 0.0, 2.48)),
    ("Fe mixed", FE[1], frame((-1, -1, -2), (-1, 1, 0), (1, 1, -1)), (2.0, 0.0, 1.5)),
    ("Cu climb", CU[1], frame((1, -1, 0), (1, 1, 1), (-1, -1, 2)), (2.5, 0.7, 0.3)),
    ("isotropic", (300.0, 100.0, 100.0), np.eye(3), (2.0, 0.7, 0.3)),
]


def stress(field, tensor, x1, x2, burgers, h=1e-5):
    """The stress at points, by central differences, each away from the
    cut."""
    cut = np.where(x1 < 0, 0.0, np.pi)
    gradient = np.zeros((len(x1), 3, 3))
    for j, (d1, d2) in enumerate([(h, 0.0), (0.0, h)]):
        for k in range(len(x1)):
            up = field(np.array([x1[k] + d1]), np.array([x2[k] + d2]), burgers, cut[k])
            down = field(
                np.array([x1[k] - d1]), np.array([x2[k] - d2]), burgers, cut[k]
            )
            gradient[k, :, j] = (up[0] - down[0]) / (2 * h)
    return np.einsum("ijkl,nkl->nij", tensor, gradient)


@pytest.mark.parametrize(
    "name, elastic, axes, burgers", FIELDS, ids=[f[0] for f in FIELDS]
)
def test_jump_equilibrium_and_force(name, elastic, axes, burgers):
    tensor = rotated(stiffness(elastic), axes)
    field = StraightDislocation(tensor)
    assert field.isotropic == (name == "isotropic")
    b = np.asarray(burgers)
    # across the cut, from below to above: the Burgers vector
    x1 = -np.array([0.5, 2.0, 7.0])
    above = field(x1, np.full(3, 1e-10), b)
    below = field(x1, np.full(3, -1e-10), b)
    np.testing.assert_allclose(above - below, np.tile(b, (3, 1)), atol=1e-8)
    # a cut along another direction moves the jump only
    points = np.array([3.0, -2.0]), np.array([0.5, 1.5])  # on two sides of 0.3
    turned = field(*points, b, cut=0.3)
    principal = field(*points, b)
    np.testing.assert_allclose(turned[0], principal[0], atol=1e-12)
    np.testing.assert_allclose(turned[1], principal[1] - b, atol=1e-12)
    # equilibrium: the divergence of the stress vanishes
    rng = np.random.default_rng(0)
    p1, p2 = rng.uniform(1, 3, (2, 4))
    h = 1e-3
    divergence = (
        stress(field, tensor, p1 + h, p2, b)[:, :, 0]
        - stress(field, tensor, p1 - h, p2, b)[:, :, 0]
        + stress(field, tensor, p1, p2 + h, b)[:, :, 1]
        - stress(field, tensor, p1, p2 - h, b)[:, :, 1]
    ) / (2 * h)
    scale = np.abs(stress(field, tensor, p1, p2, b)).max()
    assert np.abs(divergence).max() < 1e-4 * scale
    # no force on the line: the traction around a circle adds up to zero,
    # and the stress falls off as 1/r
    t = np.linspace(0, 2 * np.pi, 361)[:-1]
    ring = stress(field, tensor, 2 * np.cos(t), 2 * np.sin(t), b)
    normal = np.column_stack([np.cos(t), np.sin(t), np.zeros_like(t)])
    force = np.einsum("nij,nj->i", ring, normal) * (2 * np.pi * 2 / len(t))
    assert np.abs(force).max() < 1e-6 * np.abs(ring).max()
    far = stress(field, tensor, 4 * np.cos(t[:6]), 4 * np.sin(t[:6]), b)
    np.testing.assert_allclose(ring[:6], 2 * far, rtol=1e-5, atol=1e-6 * scale)


def test_isotropic_limit():
    """An isotropic tensor gives the classic fields, and a slightly
    anisotropic one nearly the same."""
    cell = build_crystal("bcc", 2.865, elements="Fe", repeat=(6, 6, 2))
    center = cell.box.matrix.diagonal() / 2 + [0.21, 0.17, 0.0]
    for line, normal in [("z", "y"), ("z", "x"), ("y", "x"), ("x", "z")]:
        b = np.zeros(3)
        b["xyz".index(line)] = 2.48
        options = dict(line=line, normal=normal, center=center)
        classic = insert_dislocation(cell, b, **options)
        elastic = insert_dislocation(cell, b, elastic=(300, 100, 100), **options)
        np.testing.assert_allclose(
            elastic["positions"], classic["positions"], atol=1e-12
        )
    x1, x2 = np.random.default_rng(1).uniform(-5, 5, (2, 40))
    b = np.array([2.0, 0.0, 0.3])
    iso = StraightDislocation(rotated(stiffness((300, 100, 100)), np.eye(3)))
    near = StraightDislocation(rotated(stiffness((300, 100, 100.05)), np.eye(3)))
    assert iso.isotropic and not near.isotropic
    difference = iso(x1, x2, b) - near(x1, x2, b)
    assert np.abs(difference - difference.mean(axis=0)).max() < 1e-4
    # the edge part: the classic displacements, nu = C12 / (C11 + C12)
    nu = 0.25
    r2 = x1**2 + x2**2
    u1 = 2.0 / (2 * np.pi) * (np.arctan2(x2, x1) + x1 * x2 / (2 * (1 - nu) * r2))
    np.testing.assert_allclose(iso(x1, x2, b)[:, 0], u1, atol=1e-12)


def test_any_line_axis():
    """A line along y, normal x: the same as along z, normal y, in the box
    turned so."""
    orientation = [(1, 1, -2), (-1, 1, 0), (1, 1, 1)]
    cell = build_crystal(
        "bcc", 2.865, elements="Fe", orientation=orientation, repeat=(5, 8, 2)
    )
    center = cell.box.matrix.diagonal() / 2 + [0.31, 0.37, 0.23]
    b = np.array([0.0, 0.9, 1.5])
    options = dict(elastic=FE[1])
    ours = insert_dislocation(
        cell, b, line="y", normal="x", center=center, orientation=orientation, **options
    )
    turn = [2, 0, 1]  # x' = z, y' = x, z' = y
    turned = mdapy.System(
        cell["positions"][:, turn],
        mdapy.Box(np.diag(cell.box.matrix.diagonal()[turn])),
        elements="Fe",
    )
    other = insert_dislocation(
        turned,
        b[turn],
        center=center[turn],
        orientation=[orientation[k] for k in turn],
        **options,
    )
    back = np.empty_like(other["positions"])
    back[:, turn] = other["positions"]
    np.testing.assert_allclose(ours["positions"], back, atol=1e-12)


def test_invalid_elastic_constants():
    cell = build_crystal("bcc", 2.865, elements="Fe", repeat=(4, 4, 1))
    cell = cell.with_columns(positions=cell["positions"] + 0.1)
    for elastic, message in [
        ((100.0, 200.0, 50.0), "positive definite"),
        (np.ones((2, 2)), "shape"),
        (np.triu(np.ones((6, 6))) + np.eye(6), "symmetric"),
    ]:
        with pytest.raises(InvalidParameter, match=message):
            insert_dislocation(cell, [0, 0, 2.48], elastic=elastic)
    with pytest.raises(InvalidParameter, match="orthogonal"):
        insert_dislocation(
            cell,
            [0, 0, 2.48],
            elastic=FE[1],
            orientation=[(1, 0, 0), (1, 1, 0), (0, 0, 1)],
        )
    with pytest.raises(InvalidParameter, match="three directions"):
        insert_dislocation(cell, [0, 0, 2.48], elastic=FE[1], orientation=[(1, 0, 0)])


# ---------------------------------------------------------------- the models


def circuit(system, center, radius, structure, a):
    """The sum of the ideal crystal vectors of the steps of a loop of atoms
    around center, counterclockwise: the Burgers vector inside, as the
    dislocation analysis finds it along +z."""
    xy = system["positions"][:, :2] - center
    r = np.linalg.norm(xy, axis=1)
    loop = np.flatnonzero(np.abs(r - radius) < 1.5)
    loop = loop[np.argsort(np.arctan2(xy[loop, 1], xy[loop, 0]))]
    steps = system["positions"][np.roll(loop, -1)] - system["positions"][loop]
    lz = system.box.matrix[2, 2]
    steps[:, 2] -= np.round(steps[:, 2] / lz) * lz
    # every vector between atoms of the perfect crystal as long as a step
    longest = np.linalg.norm(steps, axis=1).max() + 1.0
    repeat = int(np.ceil(2 * longest / a)) + 1
    p = build_crystal(structure, a, repeat=(repeat,) * 3)["positions"]
    p = p[np.linalg.norm(p - p.mean(axis=0), axis=1) < longest]
    ideal = (p[:, None] - p[None]).reshape(-1, 3)
    return steps, ideal[np.linalg.norm(ideal, axis=1) < longest]


def closure(steps, ideal, rotation):
    """The steps rounded to the nearest vectors of the perfect crystal, in
    the axes of the cubic cell: what they add up to beyond the steps
    themselves, which close the loop but for whole periods of the box."""
    cubic = steps @ rotation
    nearest = np.argmin(
        np.linalg.norm(cubic[:, None, :] - ideal[None, :, :], axis=2), axis=1
    )
    assert np.linalg.norm(cubic - ideal[nearest], axis=1).max() < 0.5
    return ideal[nearest].sum(axis=0) - cubic.sum(axis=0)


@pytest.mark.parametrize("name", sorted(DISLOCATIONS))
def test_cylinder(name):
    spec = DISLOCATIONS[name]
    a, elastic, element = MATERIAL[spec.structure]
    system = dislocation_cylinder(
        name, a, elastic, 22.0, elements=element, fixed=5.0, length=3
    )
    assert system.box.pbc.tolist() == [False, False, True]
    middle = system.box.origin[:2] + system.box.matrix.diagonal()[:2] / 2
    r = np.linalg.norm(system["positions"][:, :2] - middle, axis=1)
    assert r.max() < 22.0 + 2.0  # displaced by at most a little
    movable = system["movable"]
    assert (movable == movable[:, :1]).all()
    shell = ~movable[:, 0]
    assert r[shell].min() > 22.0 - 5.0 - 2.0 and r[~shell].max() < 22.0 - 5.0 + 2.0
    # the crystal is perfect away from the core, save the stacking fault
    # a partial bounds on its -x side
    ring = (r > 9) & (r < 15)
    good = perfect(system, spec.structure)
    if spec.partial:
        x = system["positions"][:, 0] - middle[0]
        y = system["positions"][:, 1] - middle[1]
        fault = (x < 0) & (np.abs(y) < 3.5)
        assert good[ring & ~fault].all() and not good[ring & fault].all()
        structure = np.asarray(DiamondStructure().compute(system).structure)
        assert (structure[ring & fault] != 0).all()  # hexagonal, not broken
    else:
        assert good[ring].all()
    if spec.partial:
        return  # a circuit around a partial crosses its stacking fault
    # the Burgers vector, from a circuit around the line
    steps, ideal = circuit(system, middle, 12.0, spec.structure, a)
    axes = np.array(spec.orientation, dtype=float)
    rotation = axes / np.linalg.norm(axes, axis=1)[:, None]
    np.testing.assert_allclose(
        closure(steps, ideal, rotation) / a, spec.burgers, atol=1e-9
    )


def test_cylinder_length_and_elements():
    one = dislocation_cylinder("bcc screw", *FE[:2], 15.0, elements="Fe")
    three = dislocation_cylinder("bcc screw", *FE[:2], 15.0, elements="Fe", length=3)
    assert len(three) == 3 * len(one)
    assert three.box.matrix[2, 2] == pytest.approx(3 * one.box.matrix[2, 2])
    assert set(three.table["element"].to_list()) == {"Fe"}
    zinc = dislocation_cylinder("diamond 60", SI[0], SI[1], 15.0, elements=("Zn", "S"))
    assert set(zinc.table["element"].to_list()) == {"Zn", "S"}
    loose = dislocation_cylinder("fcc edge", *CU[:2], 15.0, fixed=0.0)
    assert loose["movable"].all()


@pytest.mark.parametrize(
    "name", ["fcc screw", "fcc edge", "diamond 60", "diamond screw"]
)
def test_dissociated_cylinder(name):
    """Two Shockley partials around a ribbon of intrinsic stacking fault."""
    spec = DISLOCATIONS[name]
    a, elastic, element = MATERIAL[spec.structure]
    width = 6 * models._Setup(spec, a, elastic, element).spacing
    system = dislocation_cylinder(
        name, a, elastic, 24.0, elements=element, dissociation=width + 0.4, length=3
    )
    middle = system.box.origin[:2] + system.box.matrix.diagonal()[:2] / 2
    x, y = (system["positions"][:, :2] - middle).T
    if spec.structure == "fcc":
        fault = np.asarray(CommonNeighborAnalysis().compute(system).structure) == 2
    else:
        fault = np.asarray(DiamondStructure().compute(system).structure) == 2
    # in the two layers on the glide plane between the partials
    assert fault.sum() >= 4
    assert np.abs(x[fault]).max() < width / 2 + 1.5 and np.abs(y[fault]).max() < 3.5
    steps, ideal = circuit(system, middle, 16.0, spec.structure, a)
    axes = np.array(spec.orientation, dtype=float)
    rotation = axes / np.linalg.norm(axes, axis=1)[:, None]
    np.testing.assert_allclose(
        closure(steps, ideal, rotation) / a, spec.burgers, atol=1e-9
    )


def test_fcc_partials_relax_apart():
    """Relaxed with a copper potential, the partials of a dissociated edge
    dislocation stay apart around their stacking fault; split the other
    way round, the fault would be extrinsic to fcc and costly."""
    copper = EAM.zhou(["Cu"])
    system = dislocation_cylinder(
        "fcc edge", *CU[:2], 30.0, elements="Cu", dissociation=15.0, length=2
    )
    swapped = models._partials

    def reversed_partials(spec, sign):
        left, right = swapped(spec, sign)
        return [right, left]

    good = copper.compute(system).energy
    models._partials = reversed_partials
    try:
        bad = copper.compute(
            dislocation_cylinder(
                "fcc edge", *CU[:2], 30.0, elements="Cu", dissociation=15.0, length=2
            )
        ).energy
    finally:
        models._partials = swapped
    assert bad > good + 5.0
    relaxed = minimize(system, copper, max_force=1e-3, max_steps=3000).system
    hcp = np.asarray(CommonNeighborAnalysis().compute(relaxed).structure) == 2
    x = relaxed["positions"][hcp, 0]
    assert hcp.sum() > 20 and np.ptp(x) > 10.0


def test_bcc_screw_sits_in_its_easy_core():
    iron = EAM.zhou(["Fe"])
    spec = DISLOCATIONS["bcc screw"]
    energies = []
    for core in (spec.core, (0.0, 2 / 3)):
        other = Dislocation(spec.structure, spec.orientation, spec.burgers, core=core)
        system = dislocation_cylinder(other, *FE[:2], 30.0, elements="Fe")
        energies.append(minimize(system, iron, max_force=1e-4, max_steps=3000).energy)
    assert energies[0] < energies[1] - 0.05


QUADRUPOLES = [
    "bcc screw",
    "bcc edge",
    "bcc mixed",
    "bcc <100>{110} edge",
    "fcc edge",
    "diamond 60",
]


@pytest.mark.parametrize("name", QUADRUPOLES)
def test_quadrupole(name):
    spec = DISLOCATIONS[name]
    a, elastic, element = MATERIAL[spec.structure]
    setup = models._Setup(spec, a, elastic, element)
    repeat = tuple(2 * int(round(n / 2)) for n in 44 / setup.size[:2]) + (1,)
    system = dislocation_quadrupole(name, a, elastic, repeat, elements=element)
    full = build_crystal(
        spec.structure, a, elements=element, orientation=spec.orientation, repeat=repeat
    )
    assert system.box.pbc.all()
    # the slab the cut sweeps across the planes: its volume of atoms
    b = a * setup.frame @ np.asarray(spec.burgers)
    lx, ly, lz = full.box.matrix.diagonal()
    slab = (
        abs(b[0]) * (repeat[1] // 2) * setup.size[1] * lz * len(full) / (lx * ly * lz)
    )
    assert len(full) - len(system) == round(slab)
    # the box tilted by the plastic strain of the dipole
    tilt = system.box.matrix - full.box.matrix
    assert np.abs(tilt[2]).max() == 0 and np.abs(tilt[:2, :2]).max() < np.abs(b).max()
    np.testing.assert_allclose(np.abs(tilt[:2]).sum(axis=0) > 0, np.abs(b) > 0)
    # perfect crystal, across the periodic faces too, away from the cores
    counts = np.array(repeat[:2])
    half = (counts // 2) * setup.size[:2] * [1, -1 if b[0] > 0 else 1]
    cores = [
        setup.site(1) + (counts // 4) * setup.size[:2],
        setup.site(-1) + (counts // 4) * setup.size[:2] + half,
    ]
    inverse = np.linalg.inv(system.box.matrix[:2, :2])
    xy = system["positions"][:, :2]
    distance = []
    for core in cores:
        f = (xy - core) @ inverse
        f -= np.round(f)
        distance.append(np.linalg.norm(f @ system.box.matrix[:2, :2], axis=1))
    near = np.min(distance, axis=0) < 6.0
    good = perfect(system, spec.structure)
    assert good[~near].all() and not good[near].all()


def test_quadrupole_relaxes_without_shear():
    """The tilt carries the plastic strain of the dipole: relaxed at fixed
    box, the screw dipole keeps its cores and leaves no shear stress; with
    the tilt the other way round, a shear stress is left."""
    iron = EAM.zhou(["Fe"])
    system = dislocation_quadrupole("bcc screw", *FE[:2], (6, 10, 3), elements="Fe")
    relaxed = minimize(system, iron, max_force=1e-4, max_steps=3000)
    shear = np.abs(relaxed.stress[[0, 0, 1], [1, 2, 2]]).max()
    assert shear < 0.03
    assert not perfect(relaxed.system, "bcc").all()
    plain = build_crystal(
        "bcc",
        FE[0],
        orientation=DISLOCATIONS["bcc screw"].orientation,
        repeat=(6, 10, 3),
    ).box.matrix
    reversed_box = 2 * plain - system.box.matrix
    other = minimize(
        system.with_box(type(system.box)(reversed_box, origin=system.box.origin)),
        iron,
        max_force=1e-4,
        max_steps=3000,
    )
    assert np.abs(other.stress[[0, 0, 1], [1, 2, 2]]).max() > 3 * shear + 0.05
    assert other.energy > relaxed.energy


def test_quadrupole_images_converge(monkeypatch):
    systems = []
    for images in (10, 20):
        monkeypatch.setattr(models, "_IMAGES", images)
        systems.append(dislocation_quadrupole("bcc edge", *FE[:2], (16, 10, 1)))
    difference = systems[0]["positions"] - systems[1]["positions"]
    assert np.abs(difference).max() < 3e-3
    np.testing.assert_allclose(systems[0].box.matrix, systems[1].box.matrix)


def test_dissociated_quadrupole():
    system = dislocation_quadrupole(
        "fcc edge", *CU[:2], (16, 10, 1), elements="Cu", dissociation=10.0
    )
    hcp = np.asarray(CommonNeighborAnalysis().compute(system).structure) == 2
    assert hcp.sum() >= 8  # two ribbons of stacking fault


def test_errors():
    spec = DISLOCATIONS["fcc screw"]
    with pytest.raises(InvalidParameter, match="one of"):
        dislocation_cylinder("hcp screw", *CU[:2], 10.0)
    with pytest.raises(InvalidParameter, match="cubic"):
        Dislocation("hcp", spec.orientation, spec.burgers)
    with pytest.raises(InvalidParameter, match="right-handed"):
        Dislocation("fcc", ((1, 1, -2), (1, 1, 1), (-1, 1, 0)), spec.burgers)
    with pytest.raises(InvalidParameter, match="orthogonal"):
        Dislocation("fcc", ((1, 1, 0), (1, 1, 1), (1, -1, 0)), spec.burgers)
    with pytest.raises(InvalidParameter, match="three directions"):
        Dislocation("fcc", ((1, 1, 0), (1, 1, 1)), spec.burgers)
    with pytest.raises(InvalidParameter, match="glide plane"):
        Dislocation("fcc", spec.orientation, (0.5, 0.5, 0.0))
    with pytest.raises(InvalidParameter, match="nonzero"):
        Dislocation("fcc", spec.orientation, (0.0, 0.0, 0.0))
    for options, message in [
        ({"radius": 0.0}, "radius"),
        ({"radius": 10.0, "fixed": -1.0}, "fixed"),
        ({"radius": 10.0, "length": 0}, "length"),
        ({"radius": 10.0, "dissociation": -1.0}, "dissociation"),
    ]:
        with pytest.raises(InvalidParameter, match=message):
            dislocation_cylinder("fcc screw", *CU[:2], **options)
    with pytest.raises(InvalidParameter, match="a must be positive"):
        dislocation_cylinder("fcc screw", -1.0, CU[1], 10.0)
    with pytest.raises(InvalidParameter, match="only fcc and diamond"):
        dislocation_cylinder("bcc edge", *FE[:2], 10.0, dissociation=5.0)
    with pytest.raises(InvalidParameter, match="does not split"):
        dislocation_cylinder("diamond 30 partial", *SI[:2], 10.0, dissociation=5.0)
    with pytest.raises(InvalidParameter, match="Shockley"):
        weird = Dislocation("fcc", spec.orientation, (1.0, -1.0, 0.0))
        dislocation_cylinder(weird, *CU[:2], 10.0, dissociation=5.0)
    with pytest.raises(InvalidParameter, match="111"):
        cube = Dislocation("fcc", ((1, 0, 0), (0, 1, 0), (0, 0, 1)), (0.5, 0, 0.5))
        dislocation_cylinder(cube, *CU[:2], 10.0, dissociation=5.0)
    with pytest.raises(InvalidParameter, match="perfect dislocation"):
        dislocation_quadrupole("diamond 90 partial", *SI[:2], (4, 4, 1))
    with pytest.raises(InvalidParameter, match="repeat"):
        dislocation_quadrupole("fcc screw", *CU[:2], (1, 4, 1))
