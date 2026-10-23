# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""replicate against LAMMPS replicate. Skipped where LAMMPS is not installed."""

import numpy as np
import pytest

import mdapy
from mdapy.transform import replicate

lammps = pytest.importorskip("lammps")


def water(seed):
    """Water molecules in a triclinic box, some across its faces."""
    rng = np.random.default_rng(seed)
    box = mdapy.Box(
        [[6.0, 0, 0], [1.5, 5.5, 0], [-0.8, 1.1, 6.2]], origin=[1.0, -2.0, 0.5]
    )
    centers = box.to_cartesian(rng.random((8, 3)))
    shape = np.array([[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    positions, image = box.wrap((centers[:, None, :] + shape).reshape(-1, 3))
    system = mdapy.System(
        positions,
        box,
        elements=["O", "H", "H"] * 8,
        topology=mdapy.Topology(
            bonds=[[3 * m, 3 * m + k] for m in range(8) for k in (1, 2)],
            angles=[[3 * m + 1, 3 * m, 3 * m + 2] for m in range(8)],
        ),
    )
    return system.with_columns(
        id=rng.permutation(24) + 1,
        molecule=np.repeat(rng.permutation(8) + 1, 3),
        charge=[-0.8, 0.4, 0.4] * 8,
        image=image,
    )


def by_id(system):
    order = np.argsort(system.table["id"].to_numpy())
    return system.select(order)


def bonded_ids(system):
    ids = system.table["id"].to_numpy()
    return {tuple(sorted(ids[b])) for b in system.topology.bonds["atoms"].to_numpy()}


@pytest.mark.parametrize("repeat", [(2, 3, 1), (1, 2, 2)])
def test_the_copies_of_lammps(tmp_path, repeat):
    path, out = tmp_path / "in.data", tmp_path / "out.data"
    mdapy.write(path, water(seed=sum(repeat)))
    system = mdapy.read(path)
    lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    try:
        lmp.commands_string(
            "units metal\natom_style full\nbond_style zero\nangle_style zero\n"
            f"read_data {path}\nbond_coeff *\nangle_coeff *\n"
            f"replicate {' '.join(map(str, repeat))}\nwrite_data {out}"
        )
    finally:
        lmp.close()
    expected = by_id(mdapy.read(out))
    copies = by_id(replicate(system, repeat))
    np.testing.assert_allclose(copies.box.matrix, expected.box.matrix, atol=1e-12)
    for name in ("id", "molecule", "image"):
        np.testing.assert_array_equal(
            copies.table[name].to_numpy(), expected.table[name].to_numpy(), err_msg=name
        )
    np.testing.assert_allclose(copies.positions, expected.positions, atol=1e-10)
    assert bonded_ids(copies) == bonded_ids(expected)
    assert len(copies.topology.angles) == len(expected.topology.angles)
