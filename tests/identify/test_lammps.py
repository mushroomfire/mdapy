# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Structure identification and order parameters against LAMMPS computes.
Skipped where LAMMPS is not installed."""

import numpy as np
import pytest

import mdapy
from mdapy.analysis import EntropyFingerprint, SteinhardtParameters
from mdapy.identify import (
    AcklandJonesAnalysis,
    CentroSymmetryParameter,
    CommonNeighborAnalysis,
)

from lattices import crystal

lammps = pytest.importorskip("lammps")

# LAMMPS cna/atom: 1 fcc, 2 hcp, 3 bcc, 4 icosahedral, 5 other
FROM_LAMMPS = np.array([0, 1, 2, 3, 4, 0])
# LAMMPS ackland/atom: 0 unknown, 1 bcc, 2 fcc, 3 hcp, 4 icosahedral
FROM_ACKLAND = np.array([0, 3, 1, 2, 4])


def per_atom(tmp_path, system, commands, names):
    """Per-atom values of LAMMPS computes, by atom ID."""
    data = tmp_path / "in.data"
    mdapy.write(data, system)
    lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    try:
        lmp.commands_string(
            f"""
units metal
atom_style atomic
read_data {data}
mass * 1.0
pair_style zero 6.0
pair_coeff * *
{commands}
run 0
"""
        )
        n = lmp.extract_global("nlocal")
        order = np.argsort(np.array(lmp.numpy.extract_atom("id")[:n], copy=True))
        values = {}
        for name in names:
            style = 2 if name.startswith("q") else 1  # per-atom array or vector
            column = lmp.numpy.extract_compute(name, 1, style)
            values[name] = np.array(column[:n], copy=True)[order]
    finally:
        lmp.close()
    return values


@pytest.mark.parametrize(
    "kind, a, cutoff, jitter",
    [("fcc", 3.6, 3.07, 0.12), ("hcp", 2.5, 3.0, 0.15), ("bcc", 2.87, 3.45, 0.15)],
)
def test_cna_and_centro_symmetry(tmp_path, kind, a, cutoff, jitter):
    system = crystal(kind, a, (4, 4, 4), jitter=jitter, seed=5)
    k = 8 if kind == "bcc" else 12
    values = per_atom(
        tmp_path,
        system,
        f"compute cna all cna/atom {cutoff}\ncompute csp all centro/atom {k}",
        ["cna", "csp"],
    )
    structure = CommonNeighborAnalysis(cutoff=cutoff).compute(system).structure
    expected = FROM_LAMMPS[values["cna"].astype(int)]
    np.testing.assert_array_equal(structure, expected)
    assert 0 < np.count_nonzero(structure) < len(system)  # a mixture
    csp = CentroSymmetryParameter(k).compute(system).csp
    np.testing.assert_allclose(csp, values["csp"], rtol=1e-10, atol=1e-12)


def test_steinhardt(tmp_path):
    system = crystal("fcc", 3.6, (4, 4, 4), jitter=0.15, seed=6)
    values = per_atom(
        tmp_path,
        system,
        "compute q all orientorder/atom degrees 4 2 4 6 8 nnn 12 cutoff 5.0\n"
        "compute qc all orientorder/atom degrees 3 2 4 6 nnn NULL cutoff 3.0",
        ["q", "qc"],
    )
    result = SteinhardtParameters((2, 4, 6, 8)).compute(system)
    np.testing.assert_allclose(result.q, values["q"], rtol=1e-10, atol=1e-12)
    # all neighbors closer than a cutoff
    within = SteinhardtParameters((2, 4, 6), cutoff=3.0).compute(system)
    np.testing.assert_allclose(within.q, values["qc"], rtol=1e-10, atol=1e-12)


def test_entropy_fingerprint(tmp_path):
    """LAMMPS smears each neighbor over seven bins only; mdapy over all of
    them, so that equal atoms get equal values. They agree to 1 %."""
    system = crystal("fcc", 3.6, (4, 4, 4), jitter=0.1, seed=7)
    values = per_atom(
        tmp_path,
        system,
        "compute e all entropy/atom 0.2 5.0\ncompute a all entropy/atom 0.2 5.0 avg yes 3.0",
        ["e", "a"],
    )
    plain = EntropyFingerprint(5.0, 0.2).compute(system).entropy
    averaged = EntropyFingerprint(5.0, 0.2, average_cutoff=3.0).compute(system).entropy
    np.testing.assert_allclose(plain, values["e"], rtol=1e-2)
    np.testing.assert_allclose(averaged, values["a"], rtol=1e-2)


@pytest.mark.parametrize(
    "kind, a, jitter",
    [("fcc", 3.6, 0.15), ("hcp", 2.5, 0.12), ("bcc", 2.87, 0.12)],
)
def test_ackland_jones_as_the_original_implementation(tmp_path, kind, a, jitter):
    """LAMMPS with legacy yes; its neighbors are all those within the pair
    cutoff, here never more than the 14 nearest that mdapy takes."""
    system = crystal(kind, a, (4, 4, 4), jitter=jitter, seed=7)
    values = per_atom(
        tmp_path, system, "compute aja all ackland/atom legacy yes", ["aja"]
    )
    structure = AcklandJonesAnalysis().compute(system).structure
    np.testing.assert_array_equal(structure, FROM_ACKLAND[values["aja"].astype(int)])
    assert len(np.unique(structure)) > 1  # not all atoms keep their structure
