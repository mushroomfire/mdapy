# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""DFT-D3 against LAMMPS pair_style dispersion/d3 (package EXTRA-PAIR), atom
by atom, and an overlay on EAM in a run step by step. Skipped where LAMMPS or
the style is missing.

LAMMPS converts with a bohr of 0.52917725 Å and a hartree of 27.21140795 eV;
a system scaled by MDAPY's bohr over LAMMPS's, its energies by LAMMPS's
hartree over MDAPY's, maps one set of constants on the other exactly. What is
left is LAMMPS's tables: its covalent radii, r2r4 and R0AB were converted to
bohr from Å with another bohr than the one it converts them back with, some
4e-8 apart, which zero damping raises to its 14th power; LAMMPS has no
three-body term and its r²SCAN has four decimals."""

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.calculators import D3, EAM, LAMMPS, Overlay, _d3
from mdapy.simulation import MolecularDynamics

from .lammps_reference import has_style, reference

pytest.importorskip("lammps")
if not has_style("dispersion/d3"):  # pragma: no cover - CI's LAMMPS has it
    pytest.skip("LAMMPS has no pair_style dispersion/d3", allow_module_level=True)

BOHR, HARTREE = 0.52917725, 27.21140795
SCALE = _d3._BOHR / BOHR
ENERGY = HARTREE / _d3._HARTREE
PROPERTIES = ("energy", "atomic_energies", "forces", "virial", "atomic_virials")


def random_cell(elements, seed, pbc=(True, True, True)):
    """The elements at random places of a tilted cell as LAMMPS takes them,
    a along x and b in the xy plane, smaller than the cutoffs."""
    rng = np.random.default_rng(seed)
    h = np.array(
        [[9.0, 0.0, 0.0], [rng.normal(0, 0.7), 8.5, 0.0], rng.normal(0, 0.7, 3)]
    )
    h[2, 2] = 10.0
    if not pbc[2]:  # a slab: LAMMPS takes free directions along an axis
        h[2, :2] = 0.0
    while True:
        x = rng.random((len(elements), 3)) @ h
        system = mdapy.System(x, mdapy.Box(h, pbc=pbc), elements=list(elements))
        if len(mdapy.NeighborList.build(system, 1.3)) == 0:
            return system


def scaled(system, k):
    box = system.box
    return mdapy.System(
        system["positions"] * k,
        mdapy.Box(box.matrix * k, origin=box.origin * k, pbc=box.pbc),
        elements=list(system.table["element"].cast(str)),
    )


def mapped(d3, system):
    """What D3 computes for the system with LAMMPS's constants."""
    r = d3.compute(scaled(system, SCALE), PROPERTIES)
    return {
        "energy": ENERGY * r.energy,
        "atomic_energies": ENERGY * r.atomic_energies,
        "forces": ENERGY * SCALE * r.forces,
        "virial": ENERGY * r.virial,
        "atomic_virials": ENERGY * r.atomic_virials,
    }


def assert_near(result, expected, rtol):
    """The energy within rtol, the others within 1e-5 of the largest value
    of each: the coordination numbers carry the difference of the tables to
    the forces most."""
    assert result["energy"] == pytest.approx(expected["energy"], rel=rtol)
    for name in PROPERTIES[1:]:
        scale = np.abs(expected[name]).max()
        np.testing.assert_allclose(result[name], expected[name], rtol=0,
                                   atol=1e-5 * scale, err_msg=name)  # fmt: skip


MIXTURE = ["C", "H", "O", "Mo", "S", "Ag"]

# (damping of MDAPY, of LAMMPS, functional of LAMMPS, the tolerance)
CASES = [
    ("bj", "bj", "pbe", 1e-7),
    ("bj", "bj", "b3-lyp", 1e-7),
    ("bj", "bj", "pbesol", 1e-7),
    ("zero", "original", "pbe", 1e-6),
    ("zero", "original", "b3-lyp", 1e-6),
]


@pytest.mark.parametrize("damping, style, functional, rtol", CASES)
@pytest.mark.parametrize("pbc", [(True, True, True), (True, True, False)])
def test_against_lammps(tmp_path, damping, style, functional, rtol, pbc):
    system = random_cell(np.random.default_rng(1).choice(MIXTURE, 24), seed=2, pbc=pbc)
    names = sorted(set(system.table["element"].cast(str)))
    cutoff, cn_cutoff = 15.0, 10.0  # in LAMMPS's Å
    pair = (
        f"pair_style dispersion/d3 {style} {functional} {cutoff} {cn_cutoff}\n"
        f"pair_coeff * * {' '.join(names)}"
    )
    expected = reference(system, names, pair, tmp_path)
    d3 = D3(functional.replace("-", ""), damping, cutoff=cutoff * SCALE,
            cn_cutoff=cn_cutoff * SCALE)  # fmt: skip
    assert_near(mapped(d3, system), expected, rtol)


def test_r2scan_of_lammps(tmp_path):
    """LAMMPS's r²SCAN, with its numbers of four decimals given."""
    system = random_cell(["C", "H", "O", "S"] * 4, seed=3)
    pair = "pair_style dispersion/d3 bj r2scan 12.0 8.0\npair_coeff * * C H O S"
    expected = reference(system, ["C", "H", "O", "S"], pair, tmp_path)
    d3 = D3(None, s8=0.7898, a1=0.4948, a2=5.7308,
            cutoff=12.0 * SCALE, cn_cutoff=8.0 * SCALE)  # fmt: skip
    assert_near(mapped(d3, system), expected, 1e-7)
    # the published numbers of eight decimals differ from those by 1e-4
    published = mapped(D3("r2scan", cutoff=12.0 * SCALE, cn_cutoff=8.0 * SCALE), system)
    assert abs(published["energy"] / expected["energy"] - 1) > 1e-5


def test_a_run_overlaid_on_eam(tmp_path):
    """EAM with D3 added, in MDAPY's dynamics, with LAMMPS computing the
    forces of pair_style hybrid/overlay and with MDAPY's: the same run, step
    by step, to the difference of their constants."""
    path = tmp_path / "Cu.eam.alloy"
    EAM.zhou(["Cu"]).write(path)
    system = build_crystal("fcc", 3.615, elements="Cu", repeat=(4, 4, 4))
    rng = np.random.default_rng(0)
    system = system.with_columns(
        positions=system["positions"] + rng.normal(0, 0.05, (len(system), 3))
    )
    lammps = LAMMPS(
        f"pair_style hybrid/overlay eam/alloy dispersion/d3 bj pbe 12.0 8.0\n"
        f"pair_coeff * * eam/alloy {path} Cu\npair_coeff * * dispersion/d3 Cu",
        ["Cu"],
    )
    native = Overlay([EAM(path), D3("pbe", cutoff=12.0, cn_cutoff=8.0)])
    with MolecularDynamics(system, native, timestep=0.002) as md:
        md.velocities(600, seed=3)
        system = md.system.drop("image")
    runs = []
    for calculator in (lammps, native):
        with MolecularDynamics(system, calculator, timestep=0.002) as md:
            md.run(200, thermo=10)
            runs.append((md.system, md.thermo))
    (a, thermo_a), (b, thermo_b) = runs
    d = a.box.to_fractional(a["positions"]) - b.box.to_fractional(b["positions"])
    assert np.abs((d - np.round(d)) @ a.box.matrix).max() < 1e-7
    assert np.abs(a["velocities"] - b["velocities"]).max() < 2e-5
    # the energy of D3 apart by the ratio of the constants, 1e-6
    np.testing.assert_allclose(thermo_a["potential_energy"], thermo_b["potential_energy"],
                               rtol=1e-6)  # fmt: skip
    np.testing.assert_allclose(thermo_a["pressure"], thermo_b["pressure"], rtol=1e-5)
    # D3 holds the crystal: its share of the energy
    d3 = D3("pbe", cutoff=12.0, cn_cutoff=8.0).compute(b).energy
    assert d3 < -0.2 * len(b)
