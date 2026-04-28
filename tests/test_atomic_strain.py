# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""Atomic strain — fixture-driven, no OVITO at runtime."""

import numpy as np

from mdapy import System, AtomicStrain
from _fixture_helper import load_misc, input_path


def _run(affine: bool):
    data = load_misc("atomic_strain")
    ref = System(input_path("strain.0.xyz"))
    cur = System(input_path("strain.1.xyz"))
    AS = AtomicStrain(float(data["cutoff"]), ref, max_neigh=30, affine=affine)
    AS.compute(cur)
    return cur


def test_atom_strain_plain():
    data = load_misc("atomic_strain")
    cur = _run(affine=False)
    assert np.allclose(cur.data["shear_strain"].to_numpy(allow_copy=False),
                       data["shear_strain"], atol=1e-6), "shear strain (plain) differs"
    assert np.allclose(cur.data["volumetric_strain"].to_numpy(allow_copy=False),
                       data["volumetric_strain"], atol=1e-6), "volumetric strain (plain) differs"


def test_atom_strain_affine():
    data = load_misc("atomic_strain")
    cur = _run(affine=True)
    assert np.allclose(cur.data["shear_strain"].to_numpy(allow_copy=False),
                       data["shear_strain_affine"], atol=1e-6), "shear strain (affine) differs"
    assert np.allclose(cur.data["volumetric_strain"].to_numpy(allow_copy=False),
                       data["volumetric_strain_affine"], atol=1e-6), "volumetric strain (affine) differs"
