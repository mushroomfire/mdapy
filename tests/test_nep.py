# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""NEP potential — fixture-driven, no pynep / ASE / LAMMPS at runtime.

Reference outputs were generated once with pynep (CPU NEP reference
implementation) and LAMMPS' `nep` pair style; see
`tests/_generate_fixtures/generate_advanced.py`.
"""

import numpy as np

from mdapy.system import System
from mdapy import build_hea
from mdapy.nep import NEP
from _fixture_helper import load_advanced, input_path


def test_nep_reference():
    """mdapy.NEP must match pynep on AlCrNi for energies, forces, stress,
    descriptor, and latent space."""
    data = load_advanced("nep")
    system = System(input_path("AlCrNi.xyz"))
    mynep = NEP(input_path("UNEP-v1.txt"))
    system.calc = mynep

    assert np.allclose(system.get_energies(), data["energies"]), "energies differ"
    assert np.allclose(system.get_force(),    data["forces"]),   "forces differ"
    assert np.allclose(system.get_stress(),   data["stress"]),   "stress differs"
    assert np.allclose(mynep.get_descriptor(system.data, system.box),
                       data["descriptor"]), "descriptor differs"
    assert np.allclose(mynep.get_latentspace(system.data, system.box),
                       data["latent"]), "latent differs"


def test_nep_lmp():
    """mdapy.NEP must match the LAMMPS-NEP plugin on hea1 (built via
    build_hea) and on AlCrNi."""
    data = load_advanced("nep_lmp")
    hea1 = build_hea(["Al", "Cr", "Ni"], [1/3, 1/3, 1/3], "fcc",
                     a=3.6, nx=3, ny=3, nz=3, random_seed=1)
    hea2 = System(input_path("AlCrNi.xyz"))
    nep_mda = NEP(input_path("UNEP-v1.txt"))

    for i, system in enumerate([hea1, hea2]):
        system.calc = nep_mda
        assert np.allclose(system.get_energies(), data[f"sys{i}__energies"]), (
            f"sys{i}: energies differ"
        )
        assert np.allclose(system.get_force(), data[f"sys{i}__forces"]), (
            f"sys{i}: forces differ"
        )
        assert np.allclose(system.get_stress(), data[f"sys{i}__stress"]), (
            f"sys{i}: stress differs"
        )
        assert np.allclose(system.get_virials(), data[f"sys{i}__virials"]), (
            f"sys{i}: virials differ"
        )
