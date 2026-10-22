# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Mean squared displacement against LAMMPS compute msd. Skipped where LAMMPS
is not installed."""

import numpy as np
import pytest

import mdapy
from mdapy.analysis import MeanSquaredDisplacement

lammps = pytest.importorskip("lammps")


@pytest.mark.parametrize("scale", [0.4, 4.0])
def test_msd_from_the_first_frame_as_lammps(tmp_path, scale):
    """Both unwrap by the image flags of the dump, so steps longer than half
    the box count; with short steps the nearest image gives the same."""
    rng = np.random.default_rng(14)
    box = mdapy.Box([[12.0, 0, 0], [2.0, 11.0, 0], [-1.0, 1.5, 13.0]])
    start = box.to_cartesian(rng.random((60, 3)))
    walk = start + np.cumsum(rng.normal(scale=scale, size=(25, 60, 3)), axis=0)
    frames = []
    for t, positions in enumerate(walk):
        wrapped, image = box.wrap(positions)
        frame = mdapy.System(wrapped, box, elements="Ar").with_columns(
            id=np.arange(1, 61), image=image
        )
        frames.append(frame.with_info(timestep=t))
    data, dump, out = tmp_path / "in.data", tmp_path / "md.dump", tmp_path / "msd.txt"
    mdapy.write(data, frames[0])
    mdapy.write(dump, frames)
    lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    try:
        lmp.commands_string(
            f"""
units metal
atom_style atomic
read_data {data}
mass * 1.0
pair_style zero 3.0
pair_coeff * *
compute msd all msd
fix out all ave/time 1 1 1 c_msd[4] file {out} format " %.17g"
rerun {dump} dump x y z ix iy iz
"""
        )
    finally:
        lmp.close()
    expected = np.loadtxt(out)[:, 1]
    result = MeanSquaredDisplacement(window=False).compute(frames)
    np.testing.assert_allclose(result.msd, expected, rtol=1e-10, atol=1e-12)
    assert expected[-1] > 10.0  # the atoms moved by several of their steps
    nearest = MeanSquaredDisplacement(window=False).compute(
        [frame.with_table(frame.table.drop("image")) for frame in frames]
    )
    if scale < 1.0:
        np.testing.assert_allclose(nearest.msd, expected, rtol=1e-10, atol=1e-12)
    else:  # steps longer than half the box are taken to other images
        assert abs(nearest.msd[-1] - expected[-1]) > 0.1 * expected[-1]
