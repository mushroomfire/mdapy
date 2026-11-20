# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The energies and virials of each atom that give the heat flux, which the
heat flux asks for with HEAT_FLUX_TERMS. In a free cluster the potential part
of the heat flux is the change of the moment of the energies: for any
velocities v, sum_i W_i v_i = d/dt sum_i r_i U_i + sum_i r_i (F_i . v_i), the
last term what the kinetic energies give back in real dynamics. EAM gives
otherwise its pair forces split in halves, as LAMMPS's centroid stress, and
Stillinger-Weber and Tersoff LAMMPS's shares of their terms, which do not."""

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.calculators import EAM, NEP, LennardJones, StillingerWeber, Tersoff
from mdapy.calculators._base import HEAT_FLUX_TERMS

from .eam_files import write_setfl
from .nep_files import write_nep
from .potential_files import write_sw, write_tersoff

HEAT = ("atomic_energies", "atomic_virials", HEAT_FLUX_TERMS)


def jittered(structure, a, elements, repeat=3, seed=0):
    """A crystal of random elements, its atoms moved at random."""
    rng = np.random.default_rng(seed)
    crystal = build_crystal(structure, a, elements=elements[0], repeat=repeat)
    chosen = np.asarray(elements)[rng.integers(0, len(elements), len(crystal))]
    moved = crystal["positions"] + rng.normal(scale=0.12, size=(len(crystal), 3))
    return mdapy.System(moved, crystal.box).with_columns(element=chosen)


def cluster(system, radius):
    """The atoms within radius of the middle, in a large open box."""
    x = system["positions"]
    middle = x.mean(axis=0)
    keep = np.linalg.norm(x - middle, axis=1) < radius
    box = mdapy.Box(80.0, pbc=(False, False, False))
    return mdapy.System(x[keep] - middle + 40.0, box).with_columns(
        element=system["element"][keep]
    )


def flux_and_moment(potential, system, heat, seed=1, step=1e-5):
    """sum_i W_i v_i, and the change of the moment of the energies by central
    differences along v, plus sum_i r_i (F_i . v_i), for random v."""
    v = np.random.default_rng(seed).normal(size=(len(system), 3))
    asked = HEAT if heat else HEAT[:2]
    result = potential.compute(system, asked + ("forces",))
    flux = np.einsum("iab,ib->a", result.atomic_virials, v)
    r = system["positions"]

    def energies(sign):
        moved = system.with_columns(positions=r + sign * step * v)
        return potential.compute(moved, asked).atomic_energies

    change = (energies(1) - energies(-1)) / (2 * step)
    moment = r.T @ change + r.T @ (result.forces * v).sum(axis=1)
    return flux, moment


def potentials(tmp_path):
    return {
        "Stillinger-Weber": (
            StillingerWeber(write_sw(tmp_path / "a.sw", ("Si", "Ge"))),
            jittered("diamond", 5.3, ("Si", "Ge")),
        ),
        "Tersoff": (
            Tersoff(write_tersoff(tmp_path / "a.tersoff", ("Si", "C", "Ge"))),
            jittered("diamond", 5.0, ("Si", "C", "Ge")),
        ),
        "Lennard-Jones": (
            LennardJones({"Cu": (0.4, 2.3), "Ni": (0.5, 2.2)}, 6.0),
            jittered("fcc", 3.6, ("Cu", "Ni")),
        ),
        "EAM": (EAM.zhou(["Cu", "Ni"]), jittered("fcc", 3.6, ("Cu", "Ni"))),
        # a density of each pair of elements
        "EAM/fs": (
            EAM(write_setfl(tmp_path / "CuNi.eam.fs", fs=True)),
            jittered("fcc", 3.3, ("Cu", "Ni")),
        ),
        "NEP": (
            NEP(write_nep(tmp_path / "nep.txt")),
            jittered("fcc", 3.6, ("Cu", "Ni")),
        ),
    }


@pytest.mark.parametrize(
    "name", ["Stillinger-Weber", "Tersoff", "Lennard-Jones", "NEP", "EAM", "EAM/fs"]
)
def test_the_flux_is_the_change_of_the_moment_of_the_energies(tmp_path, name):
    potential, crystal = potentials(tmp_path)[name]
    flux, moment = flux_and_moment(potential, cluster(crystal, 7.0), heat=True)
    assert np.abs(flux).max() > 1.0
    np.testing.assert_allclose(flux, moment, rtol=0, atol=2e-6 * np.abs(flux).max())


@pytest.mark.parametrize("name", ["Stillinger-Weber", "Tersoff", "EAM"])
def test_the_shares_of_lammps_do_not_give_the_flux(tmp_path, name):
    potential, crystal = potentials(tmp_path)[name]
    flux, moment = flux_and_moment(potential, cluster(crystal, 7.0), heat=False)
    assert np.abs(flux - moment).max() > 0.02 * np.abs(moment).max()


@pytest.mark.parametrize("name", ["Stillinger-Weber", "Tersoff", "EAM"])
@pytest.mark.parametrize("repeat", [3, 1])  # 1: images of an atom among its pairs
def test_the_terms_of_the_flux_sum_to_the_totals(tmp_path, name, repeat):
    potential, _ = potentials(tmp_path)[name]
    structure, a, elements = {
        "Stillinger-Weber": ("diamond", 5.3, ("Si", "Ge")),
        "Tersoff": ("diamond", 5.0, ("Si", "C", "Ge")),
        "EAM": ("fcc", 3.6, ("Cu", "Ni")),
    }[name]
    system = jittered(structure, a, elements, repeat=repeat, seed=repeat)
    asked = ("energy", "forces", "virial", "atomic_energies", "atomic_virials")
    shares = potential.compute(system, asked)
    terms = potential.compute(system, asked + (HEAT_FLUX_TERMS,))
    assert np.array_equal(terms.forces, shares.forces)
    assert terms.energy == pytest.approx(shares.energy, rel=1e-13)
    assert terms.atomic_energies.sum() == pytest.approx(shares.energy, rel=1e-13)
    np.testing.assert_allclose(
        terms.atomic_virials.sum(axis=0), shares.virial, rtol=0, atol=1e-10
    )
    # each atom's differ: the terms are not shared out; EAM's energies are
    # those of its sites either way, its virials differ
    if name == "EAM":
        np.testing.assert_array_equal(terms.atomic_energies, shares.atomic_energies)
        assert np.abs(terms.atomic_virials - shares.atomic_virials).max() > 1e-3
    else:
        assert np.abs(terms.atomic_energies - shares.atomic_energies).max() > 1e-3


def test_the_energy_of_each_atom_holds_its_terms_whole(tmp_path):
    """Stillinger-Weber, one element: half of each pair with each atom, each
    three-body term whole with its centre, against NumPy."""
    path = write_sw(tmp_path / "Si.sw", ("Si",))
    eps, sigma, a, lam, gamma, cos0, big_a, big_b, p, q, _ = (
        float(v) for v in path.read_text().split("\n")[2].split()[3:]
    )
    system = jittered("diamond", 5.3, ("Si",), seed=4)
    got = StillingerWeber(path).compute(system, HEAT).atomic_energies
    nl = mdapy.NeighborList.build(system, a * sigma)
    expected = np.zeros(len(system))
    for i in range(len(system)):
        rows = slice(nl.offsets[i], nl.offsets[i + 1])
        d = nl.vectors[rows]
        r = np.linalg.norm(d, axis=1)
        s = sigma / r
        pair = big_a * eps * (big_b * s**p - s**q) * np.exp(sigma / (r - a * sigma))
        expected[i] += 0.5 * pair.sum()
        decay = np.exp(gamma * sigma / (r - a * sigma))
        for j in range(len(r)):
            for k in range(j + 1, len(r)):
                cos = d[j] @ d[k] / (r[j] * r[k])
                expected[i] += lam * eps * (cos - cos0) ** 2 * decay[j] * decay[k]
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_on_the_gpu_stand_in(tmp_path, monkeypatch):
    """The device's compute_heat, its module stood in for on the processor."""
    from mdapy.calculators import _gpu

    from calculators.gpu_stand_in import stand_in

    monkeypatch.setattr(_gpu, "cuda", stand_in)
    system = jittered("diamond", 5.3, ("Si", "Ge"))
    path = write_sw(tmp_path / "a.sw", ("Si", "Ge"))
    alloy = jittered("fcc", 3.6, ("Cu", "Ni"), repeat=4)
    for make, atoms in (
        (lambda **o: StillingerWeber(path, **o), system),
        (lambda **o: EAM.zhou(["Cu", "Ni"], **o), alloy),
    ):
        on_gpu = make(device="cuda").compute(atoms, HEAT)
        here = make().compute(atoms, HEAT)
        np.testing.assert_allclose(
            on_gpu.atomic_energies, here.atomic_energies, atol=1e-12
        )
        np.testing.assert_allclose(
            on_gpu.atomic_virials, here.atomic_virials, atol=1e-12
        )
