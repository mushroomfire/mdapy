# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Data files exchanged with LAMMPS. Skipped where LAMMPS is not installed.

LAMMPS reads each file mdapy writes and writes it back with write_data;
reading that file must give the same system.
"""

import numpy as np
import pytest

import mdapy

lammps = pytest.importorskip("lammps")


def through_lammps(tmp_path, system, setup, write_options="", after="", **options):
    """setup runs before read_data, after between read_data and write_data."""
    path = tmp_path / "in.data"
    out = tmp_path / "out.data"
    mdapy.write(path, system, **options)
    lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    try:
        lmp.commands_string(
            f"{setup}\nread_data {path}\n{after}\nwrite_data {out} {write_options}"
        )
    finally:
        lmp.close()
    return mdapy.read(out)


def unwrapped_fractions(system):
    positions = (
        system["positions"] + system.table["image"].to_numpy() @ system.box.matrix
    )
    return system.box.to_fractional(positions)


def water():
    rng = np.random.default_rng(0)
    oxygens = rng.random((4, 3)) * 12 + 1
    positions = (
        np.concatenate([oxygens, oxygens + [0.96, 0, 0], oxygens + [-0.24, 0.93, 0]])
        .reshape(3, 4, 3)
        .transpose(1, 0, 2)
        .reshape(12, 3)
    )
    bonds = [[3 * k, 3 * k + i] for k in range(4) for i in (1, 2)]
    angles = [[3 * k + 1, 3 * k, 3 * k + 2] for k in range(4)]
    system = mdapy.System(
        positions,
        mdapy.Box(15.0),
        elements=["O", "H", "H"] * 4,
        topology=mdapy.Topology(bonds=bonds, angles=angles),
    )
    return system.with_columns(
        charge=[-0.8476, 0.4238, 0.4238] * 4,
        molecule=np.repeat(np.arange(1, 5), 3),
        velocities=rng.standard_normal((12, 3)),
        type_label=["OW", "HW", "HW"] * 4,
    )


@pytest.mark.parametrize("units", ["metal", "real"])
def test_full_style(tmp_path, units):
    system = water()
    setup = f"units {units}\natom_style full\nbond_style zero\nangle_style zero\n"
    back = through_lammps(
        tmp_path, system, setup, after="bond_coeff *\nangle_coeff *", units=units
    )
    for name in ("id", "molecule", "type", "charge", "type_label", "element", "mass"):
        assert (
            back.table[name].to_list()
            == mdapy.read(tmp_path / "in.data").table[name].to_list()
        ), name
    # Exact in metal units. In real units LAMMPS stores Å/fs as a double, which
    # rounds once: at most 1 unit in the last place.
    np.testing.assert_array_max_ulp(
        back.table["velocities"].to_numpy(),
        system.table["velocities"].to_numpy(),
        maxulp=0 if units == "metal" else 1,
    )
    np.testing.assert_allclose(
        unwrapped_fractions(back),
        system.box.to_fractional(system["positions"]),
        atol=1e-15,
    )
    assert (
        back.topology.bonds["atoms"].to_list()
        == system.topology.bonds["atoms"].to_list()
    )
    assert (
        back.topology.angles["atoms"].to_list()
        == system.topology.angles["atoms"].to_list()
    )
    assert back.info["lammps"]["units"] == units


@pytest.mark.parametrize("output", ["", "triclinic/general"])
def test_general_triclinic_box(tmp_path, output):
    rng = np.random.default_rng(1)
    matrix = np.array([[4.0, 0.3, 0.2], [0.5, 4.2, -0.1], [0.3, 0.4, 3.9]])
    box = mdapy.Box(matrix, origin=[1.0, -2.0, 0.5])
    fractions = rng.random((20, 3))
    system = mdapy.System(
        fractions @ matrix + box.origin, box, elements=rng.choice(["Cu", "Zr"], 20)
    ).with_columns(velocities=rng.standard_normal((20, 3)))
    back = through_lammps(tmp_path, system, "units metal\natom_style atomic", output)
    np.testing.assert_allclose(unwrapped_fractions(back), fractions, atol=1e-14)
    speeds = np.linalg.norm(back.table["velocities"].to_numpy(), axis=1)
    expected = np.linalg.norm(system.table["velocities"].to_numpy(), axis=1)
    np.testing.assert_allclose(speeds, expected, rtol=1e-14)
    assert back.table["element"].to_list() == system.table["element"].to_list()
    if output:
        np.testing.assert_allclose(back.box.matrix, matrix, atol=1e-14)
        np.testing.assert_allclose(back.box.origin, box.origin, atol=1e-14)
    else:
        np.testing.assert_allclose(back.box.volume, box.volume, rtol=1e-14)


def test_restricted_triclinic_box_and_charges(tmp_path):
    rng = np.random.default_rng(2)
    box = mdapy.Box([[5.0, 0, 0], [1.0, 5.0, 0], [0.5, -0.5, 5.0]], origin=[-1, 0, 2])
    system = mdapy.System(
        rng.random((10, 3)) @ box.matrix + box.origin,
        box,
        elements=["Na", "Cl"] * 5,
    ).with_columns(charge=[1.0, -1.0] * 5)
    back = through_lammps(tmp_path, system, "units metal\natom_style charge")
    assert back.box.allclose(box)
    assert back.table["charge"].to_list() == [1.0, -1.0] * 5
    assert back.table["element"].to_list() == ["Na", "Cl"] * 5
    np.testing.assert_allclose(
        unwrapped_fractions(back), box.to_fractional(system["positions"]), atol=1e-15
    )
