# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""DFT-D3 against s-dftd3 (the dftd3 package of PyPI; skipped without it) on
molecules, crystals, slabs and cells smaller than the cutoffs, of every
element, both dampings, with and without the three-body term; against LAMMPS
pair_style dispersion/d3 atom by atom and in a run step by step; against
finite differences and its invariances; overlaid; on the GPU stood in."""

import math

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.calculators import D3, EAM, Overlay, _d3, _d3_tables, _gpu
from mdapy.calculators.testing import check_calculator
from mdapy.core.threads import CancelToken
from mdapy.elements import SYMBOLS
from mdapy.errors import Cancelled, InvalidParameter, MissingProperty
from mdapy.transform import replicate

from .gpu_stand_in import stand_in

PROPERTIES = ("energy", "atomic_energies", "forces", "virial", "atomic_virials")


def _codata2018():
    """The bohr in Å and the hartree in eV as mctc-lib, and so s-dftd3,
    computes them from CODATA 2018."""
    h, me, c, alpha, e = 6.62607015e-34, 9.1093837015e-31, 299792458.0, 7.2973525693e-3, 1.602176634e-19  # fmt: skip
    hbar = h / (2.0 * math.pi)
    bohr = hbar / (me * c * alpha)
    return bohr * 1e10, me * c**2 * alpha**2 / e


@pytest.fixture
def sdftd3_constants(monkeypatch):
    """The constants of s-dftd3 in MDAPY's D3: with them the two agree to
    rounding; with MDAPY's (CODATA 2022) as these differ, ~1e-9."""
    bohr, hartree = _codata2018()
    monkeypatch.setattr(_d3, "_BOHR", bohr)
    monkeypatch.setattr(_d3, "_HARTREE", hartree)
    return bohr, hartree


def reference(system, damping, functional, three_body, cutoffs, constants, **given):
    """What s-dftd3 computes for the system: the energy, the forces, the
    virial and the energy of each atom, in eV and Å."""
    interface = pytest.importorskip("dftd3.interface")
    bohr, hartree = constants
    numbers = system.table["element"].to_physical().to_numpy()
    to_bohr = 1.0 / bohr
    model = interface.DispersionModel(
        numbers,
        system["positions"] * to_bohr,
        system.box.matrix * to_bohr,
        np.asarray(system.box.pbc),
    )
    cutoff, cn_cutoff, three_body_cutoff = cutoffs
    model.set_realspace_cutoff(
        cutoff * to_bohr, three_body_cutoff * to_bohr, cn_cutoff * to_bohr
    )
    kind = (
        interface.RationalDampingParam
        if damping == "bj"
        else interface.ZeroDampingParam
    )
    if functional is None:
        param = kind(s9=1.0 if three_body else 0.0, **given)
    else:
        param = kind(method=functional, atm=three_body)
    out = model.get_dispersion(param, grad=True)
    pairs = model.get_pairwise_dispersion(param)
    atomic = pairs["additive pairwise energy"] + pairs["non-additive pairwise energy"]
    return {
        "energy": float(out["energy"]) * hartree,
        "forces": -out["gradient"] * hartree / bohr,
        "virial": -out["virial"] * hartree,
        "atomic_energies": atomic.sum(axis=1) * hartree,
    }


def assert_close(result, expected, rtol):
    """Energies, forces and virials within rtol of the largest of each."""
    assert result.energy == pytest.approx(expected["energy"], rel=rtol)
    for name in ("forces", "virial", "atomic_energies"):
        scale = np.abs(expected[name]).max()
        np.testing.assert_allclose(
            getattr(result, name), expected[name], rtol=0, atol=rtol * scale,
            err_msg=name,
        )  # fmt: skip


def apart(positions, closest):
    """Whether no two points are closer than closest (no images)."""
    d = np.linalg.norm(positions[:, None] - positions[None], axis=-1)
    return d[np.triu_indices(len(positions), 1)].min() > closest


def cluster(elements, seed, spread=2.5, closest=1.6):
    """The elements at random places, apart, in a box with no periodic
    directions."""
    rng = np.random.default_rng(seed)
    while True:
        x = rng.normal(size=(len(elements), 3)) * spread
        if apart(x, closest):
            return mdapy.System(x, mdapy.Box(40.0, pbc=False), elements=list(elements))


def random_cell(elements, seed, pbc=(True, True, True), closest=1.6):
    """The elements at random places of a tilted cell of about 7 Å, smaller
    than the cutoffs, so that the atoms meet many images of themselves."""
    rng = np.random.default_rng(seed)
    h = np.diag([7.0, 6.5, 8.0]) + rng.normal(size=(3, 3)) * 0.8
    while True:
        x = rng.random((len(elements), 3)) @ h
        system = mdapy.System(x, mdapy.Box(h, pbc=pbc), elements=list(elements))
        if len(mdapy.NeighborList.build(system, closest)) == 0:
            return system


def crystal():
    """Diamond silicon, slightly disordered."""
    system = build_crystal("diamond", 5.43, elements="Si", repeat=(2, 2, 2))
    rng = np.random.default_rng(5)
    return system.with_columns(
        positions=system["positions"] + rng.normal(0, 0.05, (len(system), 3))
    )


MIXTURE = ["H", "C", "N", "O", "S", "Cl", "Si", "Fe", "Mo", "Ag", "Cs", "Ne", "U"]


def systems():
    rng = np.random.default_rng(11)
    return {
        "molecule": cluster(rng.choice(MIXTURE[:8], 14), seed=1),
        "crystal": crystal(),
        "small cell": random_cell(rng.choice(MIXTURE, 8), seed=2),
        "slab": random_cell(rng.choice(MIXTURE, 6), seed=3, pbc=(True, True, False)),
    }


CASES = [
    ("bj", "pbe", False),
    ("bj", "pbe", True),
    ("zero", "pbe", False),
    ("zero", "pbe", True),
    ("bj", "r2scan", False),
    ("zero", "b3lyp", True),
]


@pytest.mark.parametrize("name", ["molecule", "crystal", "small cell", "slab"])
@pytest.mark.parametrize("damping, functional, three_body", CASES)
def test_against_s_dftd3(sdftd3_constants, name, damping, functional, three_body):
    system = systems()[name]
    # the three-body term at a shorter cutoff, which s-dftd3 takes long for
    cutoffs = (_d3.CUTOFF, _d3.CN_CUTOFF, 9.0)
    d3 = D3(functional, damping, three_body=three_body, three_body_cutoff=9.0)
    result = d3.compute(system, PROPERTIES)
    expected = reference(
        system, damping, functional, three_body, cutoffs, sdftd3_constants
    )
    assert_close(result, expected, rtol=1e-12)
    assert result.atomic_virials.sum(axis=0) == pytest.approx(result.virial, abs=1e-12)


@pytest.mark.parametrize("damping", ["bj", "zero"])
def test_with_the_constants_of_mdapy(damping):
    """With CODATA 2022 the tables in atomic units and those in Å move apart by
    the difference of the bohrs, 7e-10, which zero damping raises to its 14th
    power."""
    system = systems()["small cell"]
    cutoffs = (_d3.CUTOFF, _d3.CN_CUTOFF, 9.0)
    result = D3("pbe", damping, three_body=True, three_body_cutoff=9.0).compute(
        system, PROPERTIES
    )
    expected = reference(
        system, damping, "pbe", True, cutoffs, (_d3._BOHR, _d3._HARTREE)
    )
    assert_close(result, expected, rtol=1e-7)


# s-dftd3 1.7.0 gives Cm six references, its table seven, and Md six, its
# table five, the sixth at CN -1 with no C6; MDAPY, as tad-dftd3, takes the
# references of the tables: those two are left out.
EVERY = [z for z in range(1, _d3_tables.MAX_Z + 1) if SYMBOLS[z] not in ("Cm", "Md")]


@pytest.mark.parametrize("first", range(0, len(EVERY), 8))
def test_every_element(sdftd3_constants, first):
    """Each element in a cluster of eight with as many hydrogen atoms, at a
    range of coordination numbers, with both dampings and the three-body term."""
    numbers = EVERY[first : first + 8]
    elements = [SYMBOLS[z] for z in numbers] + ["H"] * len(numbers)
    system = cluster(elements, seed=first)
    cutoffs = (25.0, 20.0, 20.0)
    for damping in ("bj", "zero"):
        d3 = D3("pbe", damping, three_body=True, cutoff=25.0, cn_cutoff=20.0,
                three_body_cutoff=20.0)  # fmt: skip
        result = d3.compute(system, PROPERTIES)
        expected = reference(system, damping, "pbe", True, cutoffs, sdftd3_constants)
        assert_close(result, expected, rtol=1e-12)


def test_the_functionals(sdftd3_constants):
    """The numbers of every functional, against those s-dftd3 loads by name."""
    system = systems()["molecule"]
    cutoffs = (_d3.CUTOFF, _d3.CN_CUTOFF, _d3.THREE_BODY_CUTOFF)
    for functional, dampings in _d3.FUNCTIONALS.items():
        for damping in dampings:
            result = D3(functional, damping).compute(system, PROPERTIES)
            expected = reference(
                system, damping, functional, False, cutoffs, sdftd3_constants
            )
            assert_close(result, expected, rtol=1e-12)


def test_numbers_given(sdftd3_constants):
    system = systems()["small cell"]
    cutoffs = (_d3.CUTOFF, _d3.CN_CUTOFF, _d3.THREE_BODY_CUTOFF)
    numbers = {"s6": 0.9, "s8": 1.1, "a1": 0.3, "a2": 4.0}
    result = D3(None, **numbers).compute(system, PROPERTIES)
    expected = reference(
        system, "bj", None, False, cutoffs, sdftd3_constants, **numbers
    )
    assert_close(result, expected, rtol=1e-12)
    numbers = {"s6": 1.0, "s8": 0.8, "rs6": 1.1, "rs8": 1.2}
    result = D3(None, "zero", **numbers).compute(system, PROPERTIES)
    expected = reference(
        system, "zero", None, False, cutoffs, sdftd3_constants, **numbers
    )
    assert_close(result, expected, rtol=1e-12)
    # a number given replaces that of the functional
    d3 = D3("pbe", s8=0.0)
    assert d3.parameters == {
        "s6": 1.0,
        "s8": 0.0,
        "a1": 0.4289,
        "a2": 4.4407,
        "s9": 0.0,
    }
    assert D3("pbe", "zero", three_body=True).parameters == {
        "s6": 1.0, "s8": 0.722, "rs6": 1.217, "rs8": 1.0, "s9": 1.0
    }  # fmt: skip


@pytest.mark.parametrize("damping", ["bj", "zero"])
@pytest.mark.parametrize("three_body", [False, True])
def test_check_calculator(damping, three_body):
    """Finite differences of the forces and the virial, sums of the atoms,
    translation, rotation, reordering and a doubled cell."""
    system = random_cell(["C", "H", "O", "Mo", "S", "Ag", "C", "H"], seed=7)
    d3 = D3("pbe", damping, three_body=three_body, cutoff=14.0, cn_cutoff=10.0,
            three_body_cutoff=8.0)  # fmt: skip
    check_calculator(d3, system, rtol=1e-6, atol=1e-8)
    check_calculator(d3, cluster(["C", "H", "N", "O", "Fe", "H"], seed=3), atol=1e-8)


def test_a_cluster_replicated():
    """A cluster far from its images, in a periodic box, is the cluster alone;
    the energy of a crystal per atom is that of its doubled cell."""
    alone = cluster(["C", "H", "O", "N", "S", "H"], seed=4)
    d3 = D3("pbe", three_body=True, cutoff=15.0, cn_cutoff=12.0, three_body_cutoff=8.0)
    free = d3.compute(alone, PROPERTIES)
    boxed = alone.with_box(mdapy.Box(np.diag([60.0, 60.0, 60.0]), origin=[-30.0] * 3))
    periodic = d3.compute(boxed, PROPERTIES)
    assert periodic.energy == pytest.approx(free.energy, rel=1e-14)
    np.testing.assert_allclose(periodic.forces, free.forces, rtol=1e-12, atol=1e-15)
    small = crystal()
    one = D3("pbe", three_body=True, three_body_cutoff=8.0).compute(small, PROPERTIES)
    big = D3("pbe", three_body=True, three_body_cutoff=8.0).compute(
        replicate(small, (2, 1, 1)), PROPERTIES
    )
    assert big.energy == pytest.approx(2 * one.energy, rel=1e-12)
    np.testing.assert_allclose(big.virial, 2 * one.virial, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(big.forces[: len(small)], one.forces, atol=1e-12)


def test_cutoffs():
    """Pairs beyond each cutoff are passed over: a list wider than the cutoff
    gives the same, and a pair just beyond the cutoff adds nothing."""
    system = random_cell(["C", "H", "O", "Mo"], seed=9)
    d3 = D3("pbe", three_body=True, cutoff=12.0, cn_cutoff=9.0, three_body_cutoff=7.0)
    wide = mdapy.NeighborList.build(system, 16.0)
    a = d3.compute(system, PROPERTIES)
    b = d3.compute(system, PROPERTIES, neighbors=wide)
    for name in PROPERTIES:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    with pytest.raises(InvalidParameter, match="below the required"):
        d3.compute(system, neighbors=mdapy.NeighborList.build(system, 10.0))
    pair = mdapy.System([[0, 0, 0], [0, 0, 12.0 + 1e-9]], mdapy.Box(40.0, pbc=False),
                        elements=["Ar", "Ar"])  # fmt: skip
    assert D3("pbe", cutoff=12.0, cn_cutoff=9.0).compute(pair).energy == 0.0
    assert d3.cutoff == 12.0 and d3.cn_cutoff == 9.0 and d3.three_body_cutoff == 7.0
    # the defaults, those of s-dftd3
    default = D3()
    assert default.cutoff == pytest.approx(31.7506, abs=1e-4)
    assert (
        default.cn_cutoff
        == default.three_body_cutoff
        == pytest.approx(21.1671, abs=1e-4)
    )


def test_what_it_leaves_out():
    """Without the virial when the dynamics leaves it out, and the properties
    asked."""
    from mdapy.calculators._base import WITHOUT_VIRIAL

    system = random_cell(["C", "H"], seed=1)
    d3 = D3("pbe", cutoff=10.0, cn_cutoff=8.0)
    out = d3.compute(system, ("energy", "forces", WITHOUT_VIRIAL))
    assert out.virial is None and out.atomic_virials is None
    full = d3.compute(system, PROPERTIES)
    np.testing.assert_array_equal(out.forces, full.forces)
    assert d3.implemented == frozenset(PROPERTIES)
    assert full.stress.shape == (3, 3)


def test_overlay():
    """Added to EAM, D3 adds its own energies, forces and virials."""
    system = build_crystal("fcc", 3.615, elements="Cu", repeat=(3, 3, 3))
    rng = np.random.default_rng(2)
    system = system.with_columns(
        positions=system["positions"] + rng.normal(0, 0.05, (len(system), 3))
    )
    eam, d3 = EAM.zhou(["Cu"]), D3("pbe", cutoff=12.0, cn_cutoff=8.0)
    total = Overlay([eam, d3]).compute(system, PROPERTIES)
    a, b = eam.compute(system, PROPERTIES), d3.compute(system, PROPERTIES)
    assert b.energy < -0.1 * len(system) / 10  # D3 binds
    assert total.energy == pytest.approx(a.energy + b.energy, rel=1e-13)
    for name in PROPERTIES[1:]:
        np.testing.assert_allclose(
            getattr(total, name), getattr(a, name) + getattr(b, name), rtol=1e-12,
            atol=1e-12, err_msg=name,
        )  # fmt: skip
    assert Overlay([eam, d3]).elements is None


def test_progress_and_cancel():
    system = crystal()
    d3 = D3("pbe")
    seen = []
    d3.compute(system, progress=lambda done, total: seen.append((done, total)) or True)
    assert seen and seen[-1][0] == seen[-1][1]
    token = CancelToken()
    token.cancel()
    with pytest.raises(Cancelled):
        d3.compute(system, cancel=token)


def test_tables():
    """The tables as MDAPY keeps them: the references of each element, the
    C6 of the pairs of references, the radii, and the first and last C6 of
    Grimme et al. (2010), of H–H and Lr–Lr."""
    counts = [len(r) for r in _d3_tables.REFERENCE_CN]
    assert len(counts) == _d3_tables.MAX_Z == 103 and sum(counts) == 338
    assert max(counts) == 7 and min(counts) == 1
    i, j = np.tril_indices(103)
    c6 = _d3_tables.reference_c6()
    assert c6.size == int((np.array(counts)[i] * np.array(counts)[j]).sum())
    assert c6[0] == 3.0267 and c6.min() > 1.0
    radii = _d3_tables.vdw_radii()
    assert radii.shape == (103, 103) and (radii == radii.T).all()
    assert radii[0, 0] == 2.1823 and radii.min() > 1.0
    assert len(_d3_tables.R4_OVER_R2) == len(_d3_tables.COVALENT_RADII) == 103
    assert (
        _d3_tables.COVALENT_RADII[0] == 0.32 and _d3_tables.COVALENT_RADII[-1] == 1.45
    )
    with pytest.raises(RuntimeError, match="damaged"):
        _d3_tables._unpacked(_d3_tables._R0, "<u2", 7)


def test_elements():
    """Any element from H to Lr, or those given in their order."""
    system = cluster(["Cu", "C", "H", "Cu"], seed=2)
    every = D3("pbe").compute(system, PROPERTIES)
    some = D3("pbe", elements=["H", "C", "Cu"]).compute(system, PROPERTIES)
    np.testing.assert_allclose(some.forces, every.forces, rtol=1e-13, atol=1e-15)
    assert some.energy == pytest.approx(every.energy, rel=1e-14)
    assert D3("pbe", elements="Cu").elements == ("Cu",)
    with pytest.raises(InvalidParameter, match=r"no parameters for \['C', 'H'\]"):
        D3("pbe", elements=["Cu"]).compute(system)
    beyond = cluster(["Rf", "H"], seed=1)
    with pytest.raises(InvalidParameter, match=r"no references for \['Rf'\]"):
        D3("pbe").compute(beyond)
    with pytest.raises(MissingProperty, match="element column"):
        D3("pbe").compute(mdapy.System([[0, 0, 0]], mdapy.Box(10.0)))
    with pytest.raises(MissingProperty, match="element column"):
        D3("pbe", elements=["H"]).compute(mdapy.System([[0, 0, 0]], mdapy.Box(10.0)))


@pytest.mark.parametrize(
    "make, message",
    [
        (lambda: D3("pw91"), "no parameters for 'pw91'"),
        (lambda: D3(5), "must be a name or None"),
        (lambda: D3("r2scan", "zero"), "no parameters of 'zero' damping"),
        (lambda: D3("pbe", "bjm"), "damping must be one of"),
        (lambda: D3(None), "give s8, a1, a2"),
        (lambda: D3(None, "zero", s8=1.0), "give rs6"),
        (lambda: D3("pbe", rs6=1.0), "parameter of the other damping"),
        (lambda: D3("pbe", s8="x"), "s8 must be a number"),
        (lambda: D3("pbe", s8=math.inf), "s8 must be finite"),
        (lambda: D3("pbe", a1=-0.1), "must not be negative"),
        (lambda: D3(None, s8=1.0, a1=0.0, a2=0.0), "cannot both be zero"),
        (lambda: D3("pbe", "zero", rs6=0.0), "must be positive"),
        (lambda: D3("pbe", cutoff=0.0), "cutoff must be positive"),
        (lambda: D3("pbe", cutoff=10.0), "cn_cutoff 21.1"),
        (
            lambda: D3("pbe", cutoff=10.0, cn_cutoff=8.0, three_body=True),
            "three_body_cutoff 21.1",
        ),  # fmt: skip
        (lambda: D3("pbe", three_body=1), "three_body must be True or False"),
        (lambda: D3("pbe", elements=[]), "each element once"),
        (lambda: D3("pbe", elements=["H", "H"]), "each element once"),
        (lambda: D3("pbe", elements=["Og"]), "covers H to Lr"),
        (lambda: D3("pbe", device="tpu"), "device must be one of"),
        (lambda: D3("pbe", precision="mixed"), "double precision only"),
    ],
)
def test_errors(make, message):
    with pytest.raises(InvalidParameter, match=message):
        make()


def test_names_and_repr():
    d3 = D3("PBEsol", "zero", three_body=True)
    assert d3.functional == "pbesol" and d3.damping == "zero" and d3.three_body
    assert D3("r2SCAN").functional == "r2scan"
    assert D3("B3-LYP").functional == "b3lyp"
    assert repr(D3("pbe", cutoff=12.0, cn_cutoff=8.0)) == (
        "D3('pbe', damping='bj', three_body=False, cutoff=12.0, cn_cutoff=8.0)"
    )
    assert D3(None, s8=1.0, a1=0.4, a2=4.0).functional is None
    assert D3("pbe").device == "cpu" and D3("pbe").precision == "double"


def test_on_the_gpu_stood_in(monkeypatch):
    monkeypatch.setattr(_gpu, "cuda", stand_in)
    system = random_cell(["C", "H", "O", "S"], seed=4)
    cpu = D3("pbe", three_body=True, cutoff=12.0, cn_cutoff=9.0, three_body_cutoff=7.0)
    gpu = D3("pbe", three_body=True, cutoff=12.0, cn_cutoff=9.0, three_body_cutoff=7.0,
             device="cuda", precision="double")  # fmt: skip
    assert gpu._on_device is not None and cpu._on_device is None
    assert "device='cuda', precision='double'" in repr(gpu)
    a, b = cpu.compute(system, PROPERTIES), gpu.compute(system, PROPERTIES)
    assert b.energy == pytest.approx(a.energy, rel=1e-13)
    np.testing.assert_allclose(b.forces, a.forces, rtol=1e-12, atol=1e-14)
    # overlaid on the GPU with the elements of the other
    eam = EAM.zhou(["Cu"], device="cuda", precision="double")
    copper = build_crystal("fcc", 3.615, elements="Cu", repeat=(5, 5, 5))
    overlay = Overlay([eam, D3("pbe", cutoff=12.0, cn_cutoff=8.0, elements=["Cu"],
                               device="cuda", precision="double")])  # fmt: skip
    on_cpu = Overlay([EAM.zhou(["Cu"]), D3("pbe", cutoff=12.0, cn_cutoff=8.0)])
    assert overlay.compute(copper).energy == pytest.approx(
        on_cpu.compute(copper).energy, rel=1e-12
    )


def test_not_split_among_processes():
    from mdapy.simulation._domain import _reach

    for calculator in (D3("pbe"), Overlay([EAM.zhou(["Cu"]), D3("pbe")])):
        with pytest.raises(InvalidParameter, match="does not split D3"):
            _reach(calculator)


def test_native_checks():
    """The tables the compiled potential takes, checked there."""
    from mdapy import _core

    tables = _d3._tables([1, 6])
    arguments = dict(
        damping="bj", **tables, s6=1.0, s8=0.8, s9=0.0, a1=0.4, a2=2.3, rs6=0.0,
        rs8=0.0, alpha=14.0, hartree=_d3._HARTREE, cutoff=10.0, cn_cutoff=8.0,
        three_body_cutoff=8.0,
    )  # fmt: skip
    assert _core.calculators.make_d3(**arguments).reach == 10.0
    for name, value, message in [
        ("damping", "bjm", "unknown D3 damping"),
        ("alpha", 13.0, "must be an even number"),
        ("cutoff", -1.0, "must be positive"),
        ("c6", tables["c6"][:-1], "does not fit its references"),
        ("rcov", tables["rcov"][:1], "inconsistent D3 tables"),
        ("references", np.array([0, 2], dtype=np.int32), "1 to 7 references"),
    ]:
        with pytest.raises(ValueError, match=message):
            _core.calculators.make_d3(**{**arguments, name: value})
