# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Voronoi cells against LAMMPS compute voronoi/atom (voro++, the VORONOI
package). Skipped where LAMMPS is not installed."""

import numpy as np
import pytest

import mdapy
from mdapy.analysis import VoronoiAnalysis

from lattices import crystal

lammps = pytest.importorskip("lammps")


@pytest.mark.parametrize("pbc", [True, (True, False, True)])
def test_volumes_and_faces_as_voro(tmp_path, pbc):
    system = crystal("fcc", 3.6, (4, 4, 4), jitter=0.25, seed=13)
    pbc = np.broadcast_to(pbc, 3)
    # vacuum along non-periodic directions, so that every atom is inside
    matrix = system.box.matrix * np.where(pbc, 1.0, 1.5)[:, None]
    shift = np.where(pbc, 0.0, 0.25) @ matrix
    system = mdapy.System(
        system.positions + shift, mdapy.Box(matrix, pbc=pbc), elements="Cu"
    )
    data = tmp_path / "in.data"
    mdapy.write(data, system)
    lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    try:
        boundary = " ".join("p" if p else "f" for p in system.box.pbc)
        lmp.commands_string(
            f"""
units metal
atom_style atomic
boundary {boundary}
read_data {data}
mass * 1.0
pair_style zero 6.0
pair_coeff * *
compute voronoi all voronoi/atom
run 0
"""
        )
        n = lmp.extract_global("nlocal")
        order = np.argsort(np.array(lmp.numpy.extract_atom("id")[:n], copy=True))
        cells = np.array(lmp.numpy.extract_compute("voronoi", 1, 2)[:n], copy=True)[
            order
        ]
    finally:
        lmp.close()
    result = VoronoiAnalysis().compute(system)
    np.testing.assert_allclose(result.volume, cells[:, 0], rtol=1e-10)
    # LAMMPS counts the walls of the box among the faces; coordination does not
    faces = np.diff(result.offsets)
    np.testing.assert_array_equal(faces, cells[:, 1])
    walls = np.add.reduceat(result.j < 0, result.offsets[:-1])
    np.testing.assert_array_equal(result.coordination, faces - walls)
    assert walls.any() == (not np.all(pbc))
