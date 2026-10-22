# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Radial distribution, coordination numbers and clusters against LAMMPS
computes. Skipped where LAMMPS is not installed."""

import numpy as np
import pytest
from lattices import crystal

import mdapy
from mdapy.analysis import (
    ClusterAnalysis,
    CoordinationNumber,
    RadialDistributionFunction,
)

lammps = pytest.importorskip("lammps")


def computed(tmp_path, system, commands, per_atom=(), arrays=()):
    """Per-atom vectors (by atom ID) and global arrays of LAMMPS computes."""
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
        for name in per_atom:
            column = lmp.numpy.extract_compute(name, 1, 1)
            values[name] = np.array(column[:n], copy=True)[order]
        for name in arrays:
            values[name] = np.array(lmp.numpy.extract_compute(name, 0, 2), copy=True)
    finally:
        lmp.close()
    return values


def test_rdf_coordination_and_clusters(tmp_path):
    system = crystal("fcc", 3.6, (5, 5, 5), jitter=0.25, seed=12)
    values = computed(
        tmp_path,
        system,
        "compute rdf all rdf 120\n"
        "compute coord all coord/atom cutoff 3.0\n"
        "compute cluster all cluster/atom 2.3",
        per_atom=("coord", "cluster"),
        arrays=("rdf",),
    )
    rdf = RadialDistributionFunction(6.0, 120).compute(system)
    np.testing.assert_allclose(rdf.r, values["rdf"][:, 0], rtol=1e-12)
    # LAMMPS counts the N - 1 other atoms, mdapy the density N / V
    n = len(system)
    np.testing.assert_allclose(
        rdf.g * n / (n - 1), values["rdf"][:, 1], rtol=1e-10, atol=1e-12
    )
    coordination = CoordinationNumber(3.0).compute(system).coordination
    np.testing.assert_array_equal(coordination, values["coord"])
    # LAMMPS numbers each cluster by the smallest atom ID in it
    cluster = ClusterAnalysis(2.3).compute(system).cluster
    pairs = np.unique(np.stack([cluster, values["cluster"]], axis=1), axis=0)
    assert len(pairs) == len(np.unique(cluster)) == len(np.unique(values["cluster"]))
    assert 10 < len(pairs) < len(system) / 2
