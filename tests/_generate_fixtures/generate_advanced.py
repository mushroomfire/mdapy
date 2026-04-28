# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""
Reference fixtures for the heavier tests that compare against
sklearn / ASE / pymatgen / pynep / LAMMPS-NEP.

Each fixture is a single .npz file in tests/fixtures/advanced/ with the
inputs and the reference outputs. The test suite never imports any of
these libraries — it just loads the fixtures.

Algorithms covered:

    pca                — sklearn PCA reference on a fixed random matrix
    build_crystal      — ASE bulk + Miller-oriented FCC/BCC/diamond cells
    elastic_constant   — pymatgen elastic tensor on FCC Al + NEP
    minimize           — ASE FIRE2 final stress/force/energy on AlCrNi+NEP
                         across 8 minimization modes
    nep                — pynep reference (energies/forces/stress/descriptor/
                         latent) on AlCrNi + the energy trajectory under
                         ASE FIRE relaxation
    nep_lmp            — LAMMPS-NEP reference outputs on hea1 and hea2

Run manually whenever the underlying algorithms or input files change:

    python tests/_generate_fixtures/generate_advanced.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from configs import INPUT_DIR        # noqa: E402

import mdapy as mp                    # noqa: E402

OUT_DIR = HERE.parent / "fixtures" / "advanced"


# ===========================================================================
# PCA — sklearn reference on a fixed random matrix
# ===========================================================================

def gen_pca():
    from sklearn.decomposition import PCA as PCA_S
    np.random.seed(1)
    des = np.random.random((100, 20))
    pca = PCA_S(n_components=3)
    res = pca.fit_transform(des)
    np.savez_compressed(
        OUT_DIR / "pca.npz",
        seed=np.int32(1),
        n_samples=np.int32(100),
        n_features=np.int32(20),
        n_components=np.int32(3),
        transformed=np.asarray(res, dtype=np.float64),
        explained_variance=np.asarray(pca.explained_variance_, dtype=np.float64),
    )


# ===========================================================================
# build_crystal — ASE reference for bulk and Miller-oriented cells
# ===========================================================================

def gen_build_crystal():
    from ase.build import bulk
    from ase.lattice.cubic import FaceCenteredCubic, BodyCenteredCubic, Diamond

    out = {}

    # Bulk (cubic=True): exact reference for FCC/BCC/diamond conventional cells
    for tag, sym, struct, a in [
        ("Cu_fcc", "Cu", "fcc", 3.615),
        ("Fe_bcc", "Fe", "bcc", 2.83),
        ("C_diamond", "C", "diamond", 3.6),
    ]:
        atoms = bulk(sym, struct, a=a, cubic=True)
        out[f"bulk__{tag}__box"] = np.asarray(atoms.get_cell(), dtype=np.float64)
        out[f"bulk__{tag}__pos"] = np.asarray(atoms.get_positions(), dtype=np.float64)

    # Miller orientation: only check box matrix and atom count (atom ordering
    # may differ between ASE's lattice generator and mdapy).
    for tag, gen_cls, sym, a, miller in [
        ("Cu_fcc",  FaceCenteredCubic, "Cu", 3.615, [[1,-1,0], [1,1,-2], [1,1,1]]),
        ("Fe_bcc",  BodyCenteredCubic, "Fe", 2.83,  [[1,2,1], [-1,0,1], [1,-1,1]]),
        ("C_diamond", Diamond,         "C",  3.6,   [[1,2,1], [-1,0,1], [1,-1,1]]),
    ]:
        if gen_cls is FaceCenteredCubic:
            atoms = gen_cls(miller=miller, symbol=sym, latticeconstant=a)
        else:
            atoms = gen_cls(directions=miller, symbol=sym, latticeconstant=a)
        out[f"miller__{tag}__box"] = np.asarray(atoms.get_cell(), dtype=np.float64)
        out[f"miller__{tag}__N"] = np.int32(len(atoms))

    np.savez_compressed(OUT_DIR / "build_crystal.npz", **out)


# ===========================================================================
# elastic_constant — pymatgen elastic tensor on relaxed FCC Al + NEP
# ===========================================================================

def gen_elastic_constant():
    from mdapy.elastic import _get_stress
    from mdapy import build_crystal, FIRE, System
    from mdapy.nep import NEP
    from pymatgen.core.elasticity.strain import DeformedStructureSet
    from pymatgen.core.elasticity.elastic import Strain
    from pymatgen.core.elasticity.elastic import ElasticTensor
    from pymatgen.core.structure import Structure

    system = build_crystal("Al", "fcc", 4.05)
    calc = NEP(str(INPUT_DIR / "UNEP-v1.txt"))
    system.calc = calc

    fy = FIRE(system, optimize_cell=True)
    assert fy.run(fmax=1e-4, steps=10000, show_process=False)
    equi_stress = _get_stress(system)

    relax = Structure.from_ase_atoms(system.to_ase())
    norm_strains = (-0.01, -0.005, 0.005, 0.01)
    shear_strains = (-0.06, -0.03, 0.03, 0.06)
    dfm_ss = DeformedStructureSet(
        relax, symmetry=False, norm_strains=norm_strains, shear_strains=shear_strains)

    strain_list = []
    stress_list = []
    for i, j in enumerate(dfm_ss):
        atoms = j.to_ase_atoms()
        dfm_system = System(ase_atom=atoms)
        dfm_system.calc = calc
        fy = FIRE(dfm_system)
        assert fy.run(fmax=1e-4, steps=10000, show_process=False)
        stress_list.append(_get_stress(dfm_system))
        strain_list.append(Strain.from_deformation(dfm_ss.deformations[i]))

    et = ElasticTensor.from_independent_strains(
        strain_list, stress_list, eq_stress=equi_stress)

    np.savez_compressed(
        OUT_DIR / "elastic_constant.npz",
        potential_filename="input_files/UNEP-v1.txt",
        symbol="Al",
        structure="fcc",
        a=np.float64(4.05),
        voigt=np.asarray(et.voigt, dtype=np.float64),
    )


# ===========================================================================
# minimize — ASE FIRE2 final stress/force/energy across 8 modes
# ===========================================================================

_MIN_MODES = [
    # (use_abc, optimize_cell, mask, hydrostatic_strain, constant_volume, scalar_pressure)
    (False, False, None,                 False, False, 0),
    (True,  False, None,                 False, False, 0),
    (False, True,  None,                 False, False, 0),
    (True,  True,  None,                 False, False, 0),
    (False, True,  [1, 0, 0, 0, 0, 0],   False, False, 0),
    (False, True,  None,                 True,  False, 0),
    (False, True,  None,                 False, True,  0),
    (False, True,  None,                 False, False, 1),
]


def _min_key(mode_idx):
    return f"mode_{mode_idx}"


def gen_minimize():
    from ase.io import read
    from ase.optimize import FIRE2
    from ase.filters import UnitCellFilter
    from mdapy.nep4ase import NEP4ASE

    out = {"potential_filename": "input_files/UNEP-v1.txt",
           "input_filename": "input_files/AlCrNi.xyz",
           "steps": np.int32(5)}

    for idx, (use_abc, optimize_cell, mask, hydrostatic_strain, constant_volume,
              scalar_pressure) in enumerate(_MIN_MODES):
        atoms = read(str(INPUT_DIR / "AlCrNi.xyz"), format="extxyz")
        atoms.calc = NEP4ASE(str(INPUT_DIR / "UNEP-v1.txt"))
        if optimize_cell:
            aa = UnitCellFilter(
                atoms,
                mask=mask,
                hydrostatic_strain=hydrostatic_strain,
                constant_volume=constant_volume,
                scalar_pressure=scalar_pressure,
            )
        else:
            aa = atoms
        fy = FIRE2(aa, use_abc=use_abc, logfile=None)
        fy.run(steps=5)

        out[f"{_min_key(idx)}__stress"] = np.asarray(atoms.get_stress(), dtype=np.float64)
        out[f"{_min_key(idx)}__forces"] = np.asarray(atoms.get_forces(), dtype=np.float64)
        out[f"{_min_key(idx)}__energies"] = np.asarray(atoms.get_potential_energies(),
                                                        dtype=np.float64)

    np.savez_compressed(OUT_DIR / "minimize.npz", **out)


# ===========================================================================
# NEP — pynep reference + ASE FIRE energy trajectory
# ===========================================================================

def gen_nep():
    from ase.io import read
    from ase.optimize import FIRE
    from pynep.calculate import NEP as pyNEP

    atom = read(str(INPUT_DIR / "AlCrNi.xyz"), format="extxyz")
    pynep = pyNEP(str(INPUT_DIR / "UNEP-v1.txt"))
    atom.calc = pynep

    e_ref = atom.get_potential_energies()
    f_ref = atom.get_forces()
    s_ref = atom.get_stress()
    des_ref = pynep.get_property("descriptor", atom)
    lat_ref = pynep.get_property("latent", atom)

    # Energy trajectory under ASE FIRE relaxation (10 steps cap)
    atom_min = read(str(INPUT_DIR / "AlCrNi.xyz"), format="extxyz")
    atom_min.calc = pyNEP(str(INPUT_DIR / "UNEP-v1.txt"))
    energy_log = []
    dyn = FIRE(atom_min, logfile=None)
    dyn.attach(lambda: energy_log.append(atom_min.get_potential_energy()), interval=1)
    dyn.run(fmax=1e-4, steps=10)

    np.savez_compressed(
        OUT_DIR / "nep.npz",
        potential_filename="input_files/UNEP-v1.txt",
        input_filename="input_files/AlCrNi.xyz",
        energies=np.asarray(e_ref, dtype=np.float64),
        forces=np.asarray(f_ref, dtype=np.float64),
        stress=np.asarray(s_ref, dtype=np.float64),
        descriptor=np.asarray(des_ref, dtype=np.float64),
        latent=np.asarray(lat_ref, dtype=np.float64),
        fire_energy_trajectory=np.asarray(energy_log, dtype=np.float64),
    )


# ===========================================================================
# EAM via LAMMPS — energies/forces/stress/virials on multiple HEA configs
# ===========================================================================

def gen_eam():
    """LAMMPS-EAM reference for two scenarios (different alloy + potential)."""
    from mdapy.lammps_potential import LammpsPotential
    from mdapy import build_hea

    out = {}

    # --- Case 1: 5-element CoNiFeAlCu, 3x3x3 cell, no displacement
    elements_1 = ["Co", "Ni", "Fe", "Al", "Cu"]
    eam_lmp_1 = LammpsPotential(
        pair_parameter=f"""
    pair_style eam/alloy
    pair_coeff * * {INPUT_DIR / "CoNiFeAlCu.eam.alloy"} Co Ni Fe Al Cu
    """,
        element_list=elements_1,
    )
    model = build_hea(
        elements_1, [0.25, 0.25, 0.25, 0.075, 0.175], "fcc",
        3.6, nx=3, ny=3, nz=3, random_seed=1,
    )
    model.calc = eam_lmp_1
    out["case1__energies"] = np.asarray(model.get_energies(), dtype=np.float64)
    out["case1__forces"]   = np.asarray(model.get_force(), dtype=np.float64)
    out["case1__virials"]  = np.asarray(model.get_virials(), dtype=np.float64)
    out["case1__stress"]   = np.asarray(model.get_stress(), dtype=np.float64)

    # --- Case 2 / 3: 3-element CoNiCr, 4x4x4 cell, with random displacement
    #     Two different EAM potential files.
    import polars as pl
    elements_2 = ["Co", "Ni", "Cr"]
    for case_idx, fname in enumerate([
        "NiCoCr.lammps.eam",
        "FeNiCrCoTi-heamix.setfl",
    ], start=2):
        eam_lmp = LammpsPotential(
            pair_parameter=f"""
        pair_style eam/alloy
        pair_coeff * * {INPUT_DIR / fname} Co Ni Cr
        """,
            element_list=elements_2,
        )
        model = build_hea(
            elements_2, [0.2, 0.3, 0.5], "fcc",
            3.6, nx=4, ny=4, nz=4, random_seed=1,
        )
        np.random.seed(1)
        noise = (np.random.random((model.N, 3)) - 0.5) * 1.4
        model.update_data(
            model.data.with_columns(
                pl.col("x") + noise[:, 0],
                pl.col("y") + noise[:, 1],
                pl.col("z") + noise[:, 2],
            )
        )
        model.calc = eam_lmp
        out[f"case{case_idx}__potential"] = fname
        out[f"case{case_idx}__energies"] = np.asarray(model.get_energies(), dtype=np.float64)
        out[f"case{case_idx}__forces"]   = np.asarray(model.get_force(), dtype=np.float64)
        out[f"case{case_idx}__virials"]  = np.asarray(model.get_virials(), dtype=np.float64)
        out[f"case{case_idx}__stress"]   = np.asarray(model.get_stress(), dtype=np.float64)

    np.savez_compressed(OUT_DIR / "eam.npz", **out)


# ===========================================================================
# NEP via LAMMPS — energies/forces/stress/virials on hea1 + hea2
# ===========================================================================

def gen_nep_lmp():
    """Requires the `nep` LAMMPS pair style. Run with a working LAMMPS
    install in the path."""
    from mdapy.lammps_potential import LammpsPotential
    from mdapy import build_hea, System

    hea1 = build_hea(["Al", "Cr", "Ni"], [1/3, 1/3, 1/3], "fcc",
                     a=3.6, nx=3, ny=3, nz=3, random_seed=1)
    hea2 = System(str(INPUT_DIR / "AlCrNi.xyz"))

    nep_lmp = LammpsPotential(
        f"""pair_style nep
        pair_coeff * * {INPUT_DIR / "UNEP-v1.txt"} Al Cr Ni
        """,
        ["Al", "Cr", "Ni"],
        centroid_stress=True,
    )

    out = {"potential_filename": "input_files/UNEP-v1.txt"}
    for i, system in enumerate([hea1, hea2]):
        system.calc = nep_lmp
        out[f"sys{i}__energies"] = np.asarray(system.get_energies(), dtype=np.float64)
        out[f"sys{i}__forces"] = np.asarray(system.get_force(), dtype=np.float64)
        out[f"sys{i}__stress"] = np.asarray(system.get_stress(), dtype=np.float64)
        out[f"sys{i}__virials"] = np.asarray(system.get_virials(), dtype=np.float64)

    # Also store the inputs needed to reconstruct hea1 (build_hea is in mdapy).
    out["hea1_kwargs"] = np.array(["Al,Cr,Ni", "fcc", "3.6", "3", "3", "3", "1"], dtype="<U16")
    out["hea2_filename"] = "input_files/AlCrNi.xyz"

    np.savez_compressed(OUT_DIR / "nep_lmp.npz", **out)


# ---------------------------------------------------------------------------

GENERATORS = [
    ("pca",                 gen_pca),
    ("build_crystal",       gen_build_crystal),
    ("nep",                 gen_nep),
    ("minimize",            gen_minimize),
    ("elastic_constant",    gen_elastic_constant),
    ("eam",                 gen_eam),
    ("nep_lmp",             gen_nep_lmp),
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in OUT_DIR.glob("*.npz"):
        f.unlink()
    print(f"Writing fixtures to {OUT_DIR.relative_to(HERE.parent.parent)}/\n")
    for name, fn in GENERATORS:
        try:
            fn()
            size_kb = (OUT_DIR / f"{name}.npz").stat().st_size / 1024
            print(f"  {name:<20s}  {size_kb:7.1f} KB")
        except Exception as e:
            print(f"  ! {name}: {type(e).__name__}: {e}")
            # Don't `raise` — these heavy fixtures may legitimately be
            # uninstallable in some environments. The test for that
            # algorithm will skip if the fixture is missing.

    total_kb = sum(p.stat().st_size for p in OUT_DIR.glob("*.npz")) / 1024
    n = len(list(OUT_DIR.glob("*.npz")))
    print(f"\n  {'TOTAL':<20s}  {total_kb:7.1f} KB ({n} files)")


if __name__ == "__main__":
    main()
