# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""NEP against results of GPUMD and NEP_CPU kept in the tests of torchnep.

The data are read from a torchnep checkout, by default next to this
repository or else at $MDAPY_TORCHNEP_DATA (its tests/data directory);
the tests are skipped without it.
"""

import os
from pathlib import Path

import numpy as np
import pytest

import mdapy
from mdapy.calculators import NEP
from mdapy.calculators._nep import _read_nep

from .nep_reference import descriptors_and_energies

DATA = Path(
    os.environ.get(
        "MDAPY_TORCHNEP_DATA",
        Path(__file__).resolve().parents[3] / "torchnep" / "tests" / "data",
    )
)
pytestmark = pytest.mark.skipif(not DATA.is_dir(), reason="no torchnep test data")

GPUMD = ["CrCoNi", "CrCoNi_unizbl", "CrCoNi_flexzbl", "CrCoNi_multicut"]
NEP_CPU = sorted(p.name.split(".")[0] for p in (DATA / "zbl").glob("*.nepcpu.npz"))


def frames(path):
    return list(mdapy.read_trajectory(path))


@pytest.mark.parametrize("name", GPUMD)
def test_against_gpumd(name):
    """GPUMD computes in single precision and prints six digits."""
    potential = NEP(DATA / f"nep_{name}.txt")
    reference = np.load(DATA / f"{name}.gpumd.npz")
    systems = frames(DATA / "CrCoNi.xyz")
    results = [potential.compute(s, ("energy", "forces", "virial")) for s in systems]
    energies = [r.energy / len(s) for r, s in zip(results, systems)]
    np.testing.assert_allclose(energies, reference["E_per_atom"], rtol=1e-5, atol=2e-5)
    forces = np.concatenate([r.forces for r in results])
    np.testing.assert_allclose(forces, reference["F"], rtol=1e-4, atol=1e-3)
    # GPUMD orders the virial xx, yy, zz, xy, yz, zx
    virials = [
        r.virial.ravel()[[0, 4, 8, 1, 5, 6]] / len(s) for r, s in zip(results, systems)
    ]
    np.testing.assert_allclose(virials, reference["V_per_atom"], rtol=1e-4, atol=1e-4)
    descriptors = np.concatenate([potential.descriptors(s) for s in systems])
    np.testing.assert_allclose(
        descriptors, reference["D_per_atom"], rtol=1e-4, atol=2e-5
    )


@pytest.mark.parametrize("name", NEP_CPU)
def test_against_nep_cpu(name):
    """NEP_CPU computes in double precision, as mdapy does."""
    path = DATA / "zbl" / f"nep_{name}.txt"
    potential = NEP(path)
    reference = np.load(DATA / "zbl" / f"{name}.nepcpu.npz")
    systems = frames(DATA / "zbl" / f"{name.split('_')[0]}.xyz")
    results = [
        potential.compute(s, ("atomic_energies", "forces", "atomic_virials"))
        for s in systems
    ]

    def close(values, expected):
        scale = np.abs(expected).max()
        np.testing.assert_allclose(values, expected, rtol=1e-10, atol=1e-13 * scale)

    close(np.concatenate([r.atomic_energies for r in results]), reference["e_atom"])
    close(np.concatenate([r.forces for r in results]), reference["forces"])
    close(np.array([r.virial for r in results]), reference["virial"])
    close(
        np.concatenate([r.atomic_virials.reshape(-1, 9) for r in results]),
        reference["virial_atom"],
    )


@pytest.mark.parametrize("name", NEP_CPU[:4])
def test_numpy_reference_against_nep_cpu(name):
    """The reference of the other tests agrees with NEP_CPU too."""
    model = _read_nep(DATA / "zbl" / f"nep_{name}.txt")
    reference = np.load(DATA / "zbl" / f"{name}.nepcpu.npz")
    systems = frames(DATA / "zbl" / f"{name.split('_')[0]}.xyz")
    energies = np.concatenate([descriptors_and_energies(model, s)[1] for s in systems])
    scale = np.abs(reference["e_atom"]).max()
    np.testing.assert_allclose(
        energies, reference["e_atom"], rtol=1e-10, atol=1e-13 * scale
    )
