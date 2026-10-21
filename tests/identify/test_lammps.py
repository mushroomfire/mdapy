# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Structure identification and order parameters against LAMMPS computes.
Skipped where LAMMPS is not installed."""

import numpy as np
import pytest
from lattices import crystal

import mdapy
from mdapy.analysis import SteinhardtParameters
from mdapy.identify import CentroSymmetryParameter, CommonNeighborAnalysis

lammps = pytest.importorskip("lammps")

# LAMMPS cna/atom: 1 fcc, 2 hcp, 3 bcc, 4 icosahedral, 5 other
FROM_LAMMPS = np.array([0, 1, 2, 3, 4, 0])


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
        "compute q all orientorder/atom degrees 4 2 4 6 8 nnn 12 cutoff 5.0",
        ["q"],
    )
    result = SteinhardtParameters((2, 4, 6, 8)).compute(system)
    np.testing.assert_allclose(result.q, values["q"], rtol=1e-10, atol=1e-12)
