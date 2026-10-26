# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Dump files exchanged with LAMMPS. Skipped where LAMMPS is not installed."""

import numpy as np
import pytest

import mdapy

lammps = pytest.importorskip("lammps")

SETUP = """
units {units}
atom_style charge
boundary p p f
region b prism 0 8 0 9 0 10 1.0 0.5 -0.5
create_box 2 b
create_atoms 1 random 20 12345 b
create_atoms 2 random 20 54321 b
mass 1 63.546
mass 2 91.224
set type 1 charge 0.5
set type 2 charge -0.5
velocity all create 300 4928459
pair_style lj/cut 3.0
pair_coeff * * 0.01 2.0
"""


def atoms(lmp, name):
    """Per-atom array of the running LAMMPS (owned atoms), in atom ID order."""
    n = lmp.extract_global("nlocal")
    ids = np.array(lmp.numpy.extract_atom("id")[:n], copy=True)
    values = np.array(lmp.numpy.extract_atom(name)[:n], copy=True)
    return values[np.argsort(ids)]


@pytest.mark.parametrize("units", ["metal", "real"])
def test_mdapy_reads_lammps_dumps(tmp_path, units):
    path = tmp_path / "out.dump"
    lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    try:
        lmp.commands_string(
            SETUP.format(units=units)
            + f"dump d all custom 1 {path} id type element x y z xs ys zs vx vy vz "
            "fx fy fz q mass\n"
            "dump_modify d units yes time yes element Cu Zr format float %.17g sort id\n"
            "run 0"
        )
        expected = {name: atoms(lmp, name) for name in ("x", "v", "f", "q")}
    finally:
        lmp.close()
    system = mdapy.read(path)
    table = system.table
    np.testing.assert_array_equal(system["positions"], expected["x"])
    np.testing.assert_allclose(
        system.box.to_cartesian(
            np.column_stack([table[c] for c in ("xs", "ys", "zs")])
        ),
        expected["x"],
        atol=1e-13,
    )
    # velocities: Å/ps; in real units LAMMPS stores Å/fs
    scale = 1000.0 if units == "real" else 1.0
    np.testing.assert_allclose(
        table["velocities"].to_numpy(), expected["v"] * scale, rtol=1e-15
    )
    factor = mdapy.units.KCAL_PER_MOL_TO_EV if units == "real" else 1.0
    np.testing.assert_allclose(
        table["forces"].to_numpy(), expected["f"] * factor, rtol=1e-15
    )
    assert table["charge"].to_list() == expected["q"].tolist()
    assert table["element"].to_list() == ["Cu"] * 20 + ["Zr"] * 20
    np.testing.assert_allclose(
        system.box.matrix, [[8, 0, 0], [1.0, 9, 0], [0.5, -0.5, 10]], atol=1e-14
    )
    assert system.box.pbc.tolist() == [True, True, False]


def test_lammps_reads_mdapy_dumps(tmp_path):
    rng = np.random.default_rng(4)
    box = mdapy.Box.from_lammps([0.5, -1, 2], [8.5, 8, 12], (1.5, -0.5, 0.75))
    system = mdapy.System(
        box.to_cartesian(rng.random((30, 3))), box, elements=["Cu", "Zr"] * 15
    ).with_columns(velocities=rng.standard_normal((30, 3)))
    path = tmp_path / "in.dump"
    mdapy.write(path, system)
    lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    try:
        lmp.commands_string(
            # read_dump needs a triclinic box for a triclinic dump
            "units metal\natom_style atomic\nregion b prism 0 1 0 1 0 1 0 0 0\n"
            "create_box 2 b\nmass * 1.0\n"
            f"read_dump {path} 0 x y z vx vy vz box yes add keep"
        )
        positions, velocities = atoms(lmp, "x"), atoms(lmp, "v")
        boxlo, boxhi, xy, yz, xz, _, _ = lmp.extract_box()
    finally:
        lmp.close()
    np.testing.assert_allclose(boxlo, box.origin, atol=1e-14)
    np.testing.assert_allclose(np.array(boxhi) - boxlo, np.diag(box.matrix), rtol=1e-14)
    np.testing.assert_allclose([xy, xz, yz], [1.5, -0.5, 0.75], rtol=1e-14)
    # LAMMPS wraps atoms into the box; compare in fractional coordinates
    expected = box.to_fractional(system["positions"]) % 1.0
    found = box.to_fractional(positions) % 1.0
    np.testing.assert_allclose(
        np.minimum(abs(found - expected), 1 - abs(found - expected)), 0, atol=1e-13
    )
    np.testing.assert_array_equal(velocities, system.table["velocities"].to_numpy())
