"""Potentials on the GPU against the same potentials on the processor, atom
by atom: in double precision to rounding, in mixed precision to single
precision; the neighbor list of the GPU against that of the processor, and
when it is built again. Skipped without the GPU part of MDAPY or a device."""

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.calculators import (
    ADP,
    EAM,
    MEAM,
    NEP,
    LennardJones,
    Morse,
    StillingerWeber,
    Tersoff,
)
from mdapy.errors import InvalidParameter
from mdapy.neighbor import NeighborList

from calculators.eam_files import write_funcfl, write_setfl
from calculators.meam_files import CASES as MEAM_CASES
from calculators.meam_files import write_library, write_parameters
from calculators.nep_files import write_nep
from calculators.potential_files import write_adp, write_sw, write_tersoff
from calculators.test_meam import crystal as meam_crystal

cuda = pytest.importorskip("mdapy._cuda")
try:
    cuda.device_name()
except RuntimeError:
    pytest.skip("no CUDA device", allow_module_level=True)

ALL = ("energy", "atomic_energies", "forces", "virial", "atomic_virials")
# of the largest value; the total virial of mixed precision sums the single
# precision terms of every pair, which mostly cancel near zero pressure
TOLERANCE = {"double": 1e-12, "mixed": 2e-5}
VIRIAL = {"double": 1e-12, "mixed": 1e-4}


def shaken(a, elements, repeat, seed=0, tilt=0.0, pbc=True, jitter=0.1):
    """A shaken fcc crystal of elements, its box tilted by a fraction."""
    repeat = (repeat,) * 3 if isinstance(repeat, int) else repeat
    crystal = build_crystal("fcc", a, elements=elements[0], repeat=repeat)
    matrix = crystal.box.matrix.copy()
    matrix[1, 0] += tilt * matrix[0, 0]
    matrix[2, 1] += 0.5 * tilt * matrix[1, 1]
    system = crystal.with_box(
        mdapy.Box(matrix, origin=crystal.box.origin, pbc=pbc), scale_positions=True
    )
    rng = np.random.default_rng(seed)
    return system.with_columns(
        positions=system["positions"] + rng.normal(0, jitter, (len(system), 3)),
        element=list(rng.choice(list(elements), len(system))),
    )


def compare(gpu, cpu, system, precision):
    a, b = gpu.compute(system, ALL), cpu.compute(system, ALL)
    tolerance = TOLERANCE[precision]
    for name in ("forces", "atomic_energies", "virial", "atomic_virials"):
        x, y = getattr(a, name), getattr(b, name)
        assert x.shape == y.shape, name
        scale = VIRIAL[precision] if name == "virial" else tolerance
        np.testing.assert_allclose(
            x, y, rtol=0, atol=scale * np.abs(y).max(), err_msg=name
        )
    assert a.energy == pytest.approx(b.energy, rel=tolerance, abs=tolerance)
    assert a.volume == b.volume


@pytest.mark.parametrize("precision", ["double", "mixed"])
@pytest.mark.parametrize("tilt", [0.0, 0.3])
def test_eam(tmp_path, precision, tilt):
    setfl = write_setfl(tmp_path / "CuNi.eam.alloy")
    fs = write_setfl(tmp_path / "CuNi.eam.fs", fs=True)
    funcfl = [
        write_funcfl(tmp_path / "Cu.eam", "Cu"),
        write_funcfl(tmp_path / "Ni.eam", "Ni"),
    ]
    system = shaken(3.3, ("Cu", "Ni"), 5, tilt=tilt)
    for path in (setfl, fs, funcfl):
        compare(
            EAM(path, device="cuda", precision=precision), EAM(path), system, precision
        )
    zhou = EAM.zhou(["Cu", "Ni"])
    gpu = EAM.zhou(["Cu", "Ni"], device="cuda", precision=precision)
    crystal = shaken(3.6, ("Cu", "Ni"), 5, tilt=tilt)
    compare(gpu, zhou, crystal, precision)
    # the average atom keeps the device
    average = gpu.average({"Cu": 0.3, "Ni": 0.7})
    assert (average.device, average.precision) == ("cuda", precision)
    compare(average, zhou.average({"Cu": 0.3, "Ni": 0.7}), crystal, precision)


@pytest.mark.parametrize("precision", ["double", "mixed"])
@pytest.mark.parametrize("tilt", [0.0, 0.3])
def test_adp(tmp_path, precision, tilt):
    path = write_adp(tmp_path / "CuNi.adp")
    system = shaken(3.6, ("Cu", "Ni"), 5, tilt=tilt)
    compare(ADP(path, device="cuda", precision=precision), ADP(path), system, precision)


def diamond(elements, a=5.43, repeat=3, tilt=0.0, seed=0, jitter=0.1):
    """A shaken diamond crystal of random elements, its box tilted."""
    system = build_crystal("diamond", a, elements="Si", repeat=(repeat,) * 3)
    matrix = system.box.matrix.copy()
    matrix[1, 0] += tilt * matrix[0, 0]
    system = system.with_box(mdapy.Box(matrix, pbc=True), scale_positions=True)
    rng = np.random.default_rng(seed)
    return system.with_columns(
        positions=system["positions"] + rng.normal(0, jitter, (len(system), 3)),
        element=list(rng.choice(list(elements), len(system))),
    )


@pytest.mark.parametrize("precision", ["double", "mixed"])
@pytest.mark.parametrize("tilt", [0.0, 0.3])
def test_stillinger_weber(tmp_path, precision, tilt):
    """Three elements whose entries disagree, so that the means of the
    processor are taken; the atoms' own energies and virials shared out."""
    path = write_sw(tmp_path / "SiGeC.sw", ("Si", "Ge", "C"))
    system = diamond(("Si", "Ge", "C"), repeat=4, tilt=tilt)
    compare(
        StillingerWeber(path, device="cuda", precision=precision),
        StillingerWeber(path),
        system,
        precision,
    )


# Stillinger and Weber's silicon, as in the potentials of LAMMPS
SILICON = "Si Si Si 2.1683 2.0951 1.80 21.0 1.20 -0.333333333333 7.049556277 0.6022245584 4.0 0.0 0.0\n"


@pytest.mark.parametrize("precision", ["double", "mixed"])
def test_stillinger_weber_at_its_cutoff(tmp_path, precision):
    """Pairs within the rounding of single precision of the cutoff, where
    1 / (r - a sigma) is infinite for an r^2 just below the square of the
    cutoff whose square root rounds to the cutoff itself: their terms are
    zero, as on the processor."""
    path = tmp_path / "Si.sw"
    path.write_text(SILICON)
    cutoff = 1.80 * 2.0951
    rng = np.random.default_rng(1)
    grid = np.stack(np.meshgrid(*[np.arange(10)] * 3, indexing="ij"), -1)
    centres = 8.0 * grid.reshape(-1, 3) + 4.0
    directions = rng.normal(size=centres.shape)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    half = 0.5 * (cutoff + rng.uniform(-1e-5, 1e-5, len(centres)))[:, None]
    positions = np.concatenate(
        [centres - half * directions, centres + half * directions]
    )
    system = mdapy.System(positions, mdapy.Box(80.0, pbc=False)).with_columns(
        element=["Si"] * len(positions)
    )
    gpu = StillingerWeber(path, device="cuda", precision=precision)
    a = gpu.compute(system, ("energy", "forces", "virial"))
    b = StillingerWeber(path).compute(system, ("energy", "forces", "virial"))
    assert np.isfinite(a.forces).all()
    assert abs(a.energy - b.energy) < 1e-9
    np.testing.assert_allclose(a.forces, b.forces, rtol=0, atol=1e-6)


@pytest.mark.parametrize("precision", ["double", "mixed"])
@pytest.mark.parametrize("tilt", [0.0, 0.3])
def test_tersoff(tmp_path, precision, tilt):
    """Three elements whose repulsions disagree, so that the means of the
    processor are taken; the atoms' own energies and virials shared out."""
    path = write_tersoff(tmp_path / "SiCGe.tersoff", ("Si", "C", "Ge"))
    system = diamond(("Si", "C", "Ge"), repeat=4, tilt=tilt)
    compare(
        Tersoff(path, device="cuda", precision=precision),
        Tersoff(path),
        system,
        precision,
    )


@pytest.mark.parametrize("precision", ["double", "mixed"])
@pytest.mark.parametrize("case", sorted(MEAM_CASES))
def test_meam(tmp_path, precision, case):
    """Every case of the CPU's tests: the reference structures, the alloys of
    each ialloy, screening within and beyond rc."""
    names, lines, structure, a = MEAM_CASES[case]
    library = write_library(tmp_path / "library.meam")
    parameters = None if lines is None else write_parameters(tmp_path / "x.meam", lines)
    cpu = MEAM(library, names, parameters)
    gpu = MEAM(library, names, parameters, device="cuda", precision=precision)
    repeat = int(np.ceil((2 * (cpu.cutoff + 1.0) + 1.0) / a)) + 1
    system = meam_crystal(
        structure, a, cpu.elements, repeat=repeat, jitter=0.15, seed=3
    )
    compare(gpu, cpu, system, precision)


@pytest.mark.parametrize("precision", ["double", "mixed"])
@pytest.mark.parametrize(
    "options",
    [
        {"zbl": "universal", "cutoffs": ((5.0, 4.0), (4.4, 3.2))},
        {"zbl": "typewise"},
        {"zbl": "flexible"},
        {"l_max": (4, 0, 0)},
        {"l_max": (8, 2, 1), "n_max": (6, 5), "basis": (8, 7)},
        {"l_max": (1, 0, 0), "n_max": (0, 0), "basis": (0, 0)},
        {"n_max": (19, 19), "basis": (19, 19), "neurons": 30},
    ],
)
def test_nep(tmp_path, precision, options):
    """Every ZBL, cutoffs of each element, the harmonics up to l = 8, the
    fewest and the most functions the GPU takes."""
    path = write_nep(tmp_path / "nep.txt", **options)
    system = shaken(3.0, ("Cu", "Ni"), 5, tilt=0.2, jitter=0.2)
    compare(NEP(path, device="cuda", precision=precision), NEP(path), system, precision)


def test_nep_with_too_many_functions(tmp_path):
    path = write_nep(tmp_path / "nep.txt", n_max=(20, 3))
    NEP(path)
    with pytest.raises(InvalidParameter, match="at most 20 functions"):
        NEP(path, device="cuda")


@pytest.mark.parametrize("precision", ["double", "mixed"])
def test_pair_potentials(precision):
    system = shaken(5.3, ("Ar", "Kr"), 5, tilt=0.2, jitter=0.2)
    parameters = {"Ar": (0.0104, 3.40), "Kr": (0.0140, 3.65, 9.0)}
    for options in ({}, {"shift": True, "mixing": "arithmetic"}):
        compare(
            LennardJones(
                parameters, 8.5, device="cuda", precision=precision, **options
            ),
            LennardJones(parameters, 8.5, **options),
            system,
            precision,
        )
    morse = {
        "Ar": (0.01, 1.4, 3.8),
        "Kr": (0.012, 1.3, 4.0, 7.0),
        ("Ar", "Kr"): (0.011, 1.35, 3.9),
    }
    compare(
        Morse(morse, 8.0, device="cuda", precision=precision, shift=True),
        Morse(morse, 8.0, shift=True),
        system,
        precision,
    )


def test_atoms_on_the_faces_of_the_box():
    """Atoms on the faces and a hair outside, which the wrap may leave at 1
    by rounding: each must be in the cell where it is."""
    zhou = EAM.zhou(["Cu", "Ni"])
    gpu = EAM.zhou(["Cu", "Ni"], device="cuda", precision="double")
    system = shaken(3.6, ("Cu", "Ni"), 6, tilt=0.2, jitter=0.0)
    fractional = system["positions"] @ np.linalg.inv(system.box.matrix)
    # the same places, through the faces: 0 as a hair below it, or as 1
    on = np.isclose(fractional, 0.0, atol=1e-12)
    assert on.any(axis=0).all()
    fractional[on[:, 0], 0] = -1e-17
    fractional[on[:, 1], 1] = 1.0 - 1e-17
    fractional[on[:, 2], 2] = 1.0
    system = system.with_columns(positions=fractional @ system.box.matrix)
    compare(gpu, zhou, system, "double")


def test_atoms_outside_directions_that_are_not_periodic():
    """A slab with vacuum along z and a cluster, some atoms outside the box
    along the directions that are not periodic."""
    zhou = EAM.zhou(["Cu"])
    gpu = EAM.zhou(["Cu"], device="cuda", precision="double")
    slab = shaken(3.6, ("Cu",), (5, 5, 3), pbc=(True, True, False))
    moved = slab["positions"].copy()
    moved[:20, 2] -= 3.0  # below the box
    slab = slab.with_columns(positions=moved)
    compare(gpu, zhou, slab, "double")
    cluster = shaken(3.6, ("Cu",), 3, pbc=False)
    cluster = cluster.with_box(mdapy.Box(5.0, origin=cluster.box.origin, pbc=False))
    compare(gpu, zhou, cluster, "double")


def test_the_neighbor_list_is_that_of_the_processor():
    system = shaken(3.6, ("Cu",), 6, tilt=0.3, jitter=0.2)
    atoms = cuda.Atoms(len(system))
    atoms.set_species(np.zeros(len(system), dtype=np.int32))
    atoms.set_positions(np.ascontiguousarray(system["positions"]))
    box = system.box
    args = (
        np.ascontiguousarray(box.matrix),
        np.ascontiguousarray(box.origin, dtype=float),
        (True,) * 3,
    )
    neighbors = cuda.Neighbors(5.0, 0.0)
    assert neighbors.update(atoms, *args)
    counts, indices, images = neighbors.get(atoms)
    found = {(i, int(j)) for i in range(len(system)) for j in indices[i, : counts[i]]}
    cpu = NeighborList.build(system, 5.0)
    expected = {
        (i, int(j))
        for i in range(len(system))
        for j in cpu.j[cpu.offsets[i] : cpu.offsets[i + 1]]
    }
    # every pair, and a few more found in single precision within 1e-3 Å
    assert expected <= found
    wider = NeighborList.build(system, 5.002)
    assert found <= {
        (i, int(j))
        for i in range(len(system))
        for j in wider.j[wider.offsets[i] : wider.offsets[i + 1]]
    }
    assert (indices[counts[:, None] <= np.arange(indices.shape[1])] == -1).all()
    # each pair through its image: x_j - x_i + s h, s of -1, 0, 1 by the code,
    # the atoms wrapped into the box
    x = atoms.positions()
    assert (x @ np.linalg.inv(box.matrix) >= -1e-12).all()
    i, k = np.nonzero(np.arange(indices.shape[1]) < counts[:, None])
    code = images[i, k]
    shift = np.stack([code % 3 - 1, code // 3 % 3 - 1, code // 9 - 1], axis=1)
    r = x[indices[i, k]] - x[i] + shift @ box.matrix
    assert (np.linalg.norm(r, axis=1) < 5.002).all()
    unwrapped = x + atoms.images() @ box.matrix
    np.testing.assert_allclose(unwrapped, system["positions"], atol=1e-12)


def test_the_list_is_built_again_when_the_atoms_moved_enough():
    system = shaken(3.6, ("Cu",), 6, jitter=0.0)
    n = len(system)
    atoms = cuda.Atoms(n)
    atoms.set_species(np.zeros(n, dtype=np.int32))
    x = np.ascontiguousarray(system["positions"])
    atoms.set_positions(x)
    h = np.ascontiguousarray(system.box.matrix)
    lo = np.ascontiguousarray(system.box.origin, dtype=float)
    neighbors = cuda.Neighbors(5.0, 1.0)
    assert neighbors.update(atoms, h, lo, (True,) * 3)
    assert not neighbors.update(atoms, h, lo, (True,) * 3)  # nothing moved
    shifted = x.copy()
    shifted[0, 0] += 0.4  # less than half the skin
    atoms.set_positions(shifted)
    assert not neighbors.update(atoms, h, lo, (True,) * 3)
    shifted[1, 0] -= 0.6
    atoms.set_positions(shifted)
    assert neighbors.update(atoms, h, lo, (True,) * 3)
    # moving with the box is not moving: the atoms of a strain of 1e-3
    stretched = shifted @ np.linalg.inv(h) @ (h * 1.001)
    atoms.set_positions(np.ascontiguousarray(stretched))
    assert not neighbors.update(atoms, h * 1.001, lo, (True,) * 3)
    stretched = shifted @ np.linalg.inv(h) @ (h * 1.2)
    atoms.set_positions(np.ascontiguousarray(stretched))
    assert neighbors.update(atoms, h * 1.2, lo, (True,) * 3)
    assert neighbors.builds == 3
    # positions gone wrong are told
    broken = stretched.copy()
    broken[3, 0] = np.nan
    atoms.set_positions(np.ascontiguousarray(broken))
    with pytest.raises(cuda.NotFinite, match="not finite"):
        neighbors.update(atoms, h * 1.2, lo, (True,) * 3)


def test_errors():
    gpu = EAM.zhou(["Cu"], device="cuda")
    thin = build_crystal("fcc", 3.6, elements="Cu", repeat=(3, 6, 6))
    with pytest.raises(InvalidParameter, match="thicker than twice the cutoff"):
        gpu.compute(thin)
    crystal = build_crystal("fcc", 3.6, elements="Cu", repeat=(6, 6, 6))
    result = gpu.compute(crystal, ("forces",))
    assert (
        result.energy is None
        and result.virial is None
        and result.forces.shape == (864, 3)
    )
    assert "device='cuda', precision='mixed'" in repr(gpu)
    assert "precision='double'" in repr(
        LennardJones((0.01, 3.4), 8.0, device="cuda", precision="double")
    )
