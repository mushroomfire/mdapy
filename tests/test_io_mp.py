# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""
mdapy's `.mp` (parquet) IO tests.

The format is just a Polars parquet file with extra metadata
(box / origin / boundary / energy / stress / virial / timestep) attached
in the parquet key/value metadata block.
"""

from pathlib import Path

import numpy as np
import polars as pl

import mdapy as mp
from mdapy.box import Box


def _make_full_system():
    """A System with positions, types, elements, velocities, and a small
    custom column — exercises every roundtrip surface of the .mp format."""
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    s = mp.System(pos=pos, box=Box(np.diag([4.0, 5.0, 6.0]), boundary=[1, 1, 0]))
    s.box.set_origin([-1.0, -2.0, -3.0])
    s.update_data(s.data.with_columns(
        type=pl.Series([1, 2, 1], dtype=pl.Int32),
        element=pl.Series(["Al", "Cu", "Al"]),
        vx=pl.Series([0.1, 0.2, 0.3]),
        vy=pl.Series([0.4, 0.5, 0.6]),
        vz=pl.Series([0.7, 0.8, 0.9]),
        energy=pl.Series([-3.5, -3.6, -3.7]),  # custom per-atom column
    ))
    # Recreate so the box update actually sticks (System.box is set lazily).
    return s


def test_mp_roundtrip(tmp_path):
    s = _make_full_system()
    out = tmp_path / "rt.mp"
    s.write_mp(str(out))
    s2 = mp.System(str(out))

    # Box matrix + boundary + origin must all round-trip.
    np.testing.assert_allclose(s.box.box, s2.box.box, atol=1e-12)
    np.testing.assert_allclose(s.box.origin, s2.box.origin, atol=1e-12)
    assert s.box.boundary.tolist() == s2.box.boundary.tolist()

    # Atom data — every column.
    for col in ("type", "element"):
        assert s.data[col].to_list() == s2.data[col].to_list()
    for col in ("x", "y", "z", "vx", "vy", "vz", "energy"):
        np.testing.assert_allclose(
            s.data[col].to_numpy(), s2.data[col].to_numpy(), atol=1e-12
        )


def test_mp_global_info_roundtrip(tmp_path):
    """The known global_info keys (energy/stress/virial/timestep) must be
    stored in the parquet metadata and recovered on read. Other keys are
    silently dropped (per the current write_mp filter)."""
    s = _make_full_system()
    s.global_info["timestep"] = 42
    s.global_info["energy"] = -123.456
    s.global_info["other"] = "should be dropped"   # not in the allowlist

    out = tmp_path / "rt_meta.mp"
    s.write_mp(str(out))
    s2 = mp.System(str(out))

    # Stored keys come back as strings (parquet metadata is a str→str map).
    assert s2.global_info.get("timestep") == "42"
    assert float(s2.global_info.get("energy")) == -123.456
    assert "other" not in s2.global_info
