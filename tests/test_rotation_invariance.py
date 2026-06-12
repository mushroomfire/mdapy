# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""
General-box (rotation-invariance) checks.

Every analysis must give identical per-atom results for a structure in a
tilted axis-anchored cell and for the same structure rigidly rotated so
that *no* cell vector is axis-aligned. This is the box class produced by
e.g. GPUMD, where the lattice is arbitrarily oriented in the lab frame;
LAMMPS-style lower-triangular fixtures do not exercise it.

Regression context: the kd-tree behind `build_nearest_neighbor` used an
axis-aligned AABB for its node bounds while pruning with triclinic cell
face normals, which silently dropped true neighbors for rotated cells
(85% wrong atoms on a real GPUMD diamond snapshot) and broke every
downstream analysis (identify_diamond_structure classified all atoms as
Other). These tests would have caught that.
"""

import numpy as np
import pytest

import mdapy as mp
from mdapy.box import Box
from mdapy.atomic_strain import AtomicStrain
from mdapy.wigner_seitz_defect import WignerSeitzAnalysis


def _rotation_matrix(rng):
    """A uniformly random proper rotation (QR of a Gaussian matrix)."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _fcc(n=4, a=4.05):
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    cells = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"),
                     axis=-1).reshape(-1, 3)
    frac = (cells[:, None, :] + basis[None, :, :]).reshape(-1, 3) / n
    box = np.array([[n * a, 0, 0],
                    [0.2 * n * a, n * a, 0],
                    [0.1 * n * a, 0.15 * n * a, n * a]])
    return frac @ box, box


def _diamond(n=3, a=3.567):
    basis = np.array([
        [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
        [0.25, 0.25, 0.25], [0.75, 0.75, 0.25],
        [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]])
    cells = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"),
                     axis=-1).reshape(-1, 3)
    frac = (cells[:, None, :] + basis[None, :, :]).reshape(-1, 3) / n
    box = np.array([[n * a, 0, 0],
                    [0.25 * n * a, n * a, 0],
                    [0.1 * n * a, 0.2 * n * a, n * a]])
    return frac @ box, box


RNG = np.random.default_rng(42)
ROT = _rotation_matrix(RNG)

_POS_FCC, _BOX_FCC = _fcc()
_POS_FCC = _POS_FCC + RNG.normal(0, 0.08, _POS_FCC.shape)
_POS_DIA, _BOX_DIA = _diamond()
_POS_DIA = _POS_DIA + RNG.normal(0, 0.05, _POS_DIA.shape)

STRUCTURES = {"fcc": (_POS_FCC, _BOX_FCC), "diamond": (_POS_DIA, _BOX_DIA)}


def _pair(structure):
    pos, box = STRUCTURES[structure]
    s1 = mp.System(pos=pos, box=Box(box, boundary=[1, 1, 1]))
    s2 = mp.System(pos=pos @ ROT, box=Box(box @ ROT, boundary=[1, 1, 1]))
    return s1, s2


def _assert_column_equal(s1, s2, col, atol=1e-5):
    v1 = s1.data[col].to_numpy()
    v2 = s2.data[col].to_numpy()
    if v1.dtype.kind in "fc":
        assert np.allclose(v1, v2, atol=atol, equal_nan=True), (
            f"{col}: max |diff| = {np.nanmax(np.abs(v1 - v2))}"
        )
    else:
        assert np.array_equal(v1, v2), (
            f"{col}: {np.count_nonzero(v1 != v2)} atoms differ"
        )


@pytest.mark.parametrize("structure", ["fcc", "diamond"])
def test_cutoff_neighbor(structure):
    s1, s2 = _pair(structure)
    s1.build_neighbor(rc=4.0, max_neigh=80)
    s2.build_neighbor(rc=4.0, max_neigh=80)
    assert np.array_equal(s1.neighbor_number, s2.neighbor_number)
    assert np.allclose(np.sort(s1.distance_list, axis=1),
                       np.sort(s2.distance_list, axis=1), atol=1e-8)


@pytest.mark.parametrize("structure", ["fcc", "diamond"])
def test_nearest_neighbor(structure):
    s1, s2 = _pair(structure)
    s1.build_nearest_neighbor(12)
    s2.build_nearest_neighbor(12)
    assert np.array_equal(s1.verlet_list, s2.verlet_list)
    assert np.allclose(s1.distance_list, s2.distance_list, atol=1e-8)


PER_ATOM_CASES = [
    ("cal_common_neighbor_analysis", "cna", {"rc": 3.4}),
    ("cal_centro_symmetry_parameter", "csp", {"N": 12}),
    ("cal_ackland_jones_analysis", "aja", {}),
    ("cal_polyhedral_template_matching", "ptm", {"return_rmsd": False}),
    ("cal_steinhardt_bond_orientation", "ql6", {"llist": [6], "nnn": 12}),
    ("cal_voronoi_volume", "volume", {}),
    ("cal_cluster_analysis", "cluster_id", {"rc": 2.0}),
    ("cal_structure_entropy", "entropy", {"rc": 4.0, "sigma": 0.2}),
]


@pytest.mark.parametrize("structure", ["fcc", "diamond"])
@pytest.mark.parametrize("method,col,kwargs", PER_ATOM_CASES,
                         ids=[c[0] for c in PER_ATOM_CASES])
def test_per_atom_analysis(structure, method, col, kwargs):
    s1, s2 = _pair(structure)
    getattr(s1, method)(**kwargs)
    getattr(s2, method)(**kwargs)
    _assert_column_equal(s1, s2, col)


def test_identify_diamond_structure():
    s1, s2 = _pair("diamond")
    s1.cal_identify_diamond_structure()
    s2.cal_identify_diamond_structure()
    _assert_column_equal(s1, s2, "ids")


def test_rdf():
    s1, s2 = _pair("fcc")
    g1 = s1.cal_radial_distribution_function(rc=5.0, nbin=100)
    g2 = s2.cal_radial_distribution_function(rc=5.0, nbin=100)
    assert np.allclose(g1.g_total, g2.g_total, atol=1e-8)


def test_structure_factor():
    s1, s2 = _pair("fcc")
    f1 = s1.cal_structure_factor(k_min=1.0, k_max=8.0, nbins=60)
    f2 = s2.cal_structure_factor(k_min=1.0, k_max=8.0, nbins=60)
    assert np.allclose(f1.Sk, f2.Sk, atol=1e-6)


def test_common_neighbor_parameter():
    s1, s2 = _pair("fcc")
    s1.cal_common_neighbor_parameter(rc=3.4)
    s2.cal_common_neighbor_parameter(rc=3.4)
    _assert_column_equal(s1, s2, "cnp")


def test_bond_analysis():
    s1, s2 = _pair("fcc")
    b1 = s1.cal_bond_analysis(rc=3.4, nbin=60)
    b2 = s2.cal_bond_analysis(rc=3.4, nbin=60)
    assert np.allclose(b1.bond_length_distribution,
                       b2.bond_length_distribution, atol=1e-8)
    assert np.allclose(b1.bond_angle_distribution,
                       b2.bond_angle_distribution, atol=1e-8)


def test_atomic_strain():
    pos_ref, box = _fcc()
    disp = np.random.default_rng(3).normal(0, 0.08, pos_ref.shape)
    pos_cur = pos_ref + disp
    cur1 = mp.System(pos=pos_cur, box=Box(box, boundary=[1, 1, 1]))
    cur2 = mp.System(pos=pos_cur @ ROT, box=Box(box @ ROT, boundary=[1, 1, 1]))
    AtomicStrain(4.0, mp.System(pos=pos_ref, box=Box(box, boundary=[1, 1, 1]))).compute(cur1)
    AtomicStrain(4.0, mp.System(pos=pos_ref @ ROT, box=Box(box @ ROT, boundary=[1, 1, 1]))).compute(cur2)
    for col in ("shear_strain", "volumetric_strain"):
        _assert_column_equal(cur1, cur2, col)


def test_wigner_seitz():
    pos_ref, box = _fcc()
    pos_cur = pos_ref + np.random.default_rng(4).normal(0, 0.08, pos_ref.shape)
    keep = np.ones(len(pos_cur), bool)
    keep[[10, 50, 100]] = False
    pos_cur = pos_cur[keep]

    ws1 = WignerSeitzAnalysis(mp.System(pos=pos_ref, box=Box(box, boundary=[1, 1, 1])))
    out1 = ws1.compute(mp.System(pos=pos_cur, box=Box(box, boundary=[1, 1, 1])))
    ws2 = WignerSeitzAnalysis(mp.System(pos=pos_ref @ ROT, box=Box(box @ ROT, boundary=[1, 1, 1])))
    out2 = ws2.compute(mp.System(pos=pos_cur @ ROT, box=Box(box @ ROT, boundary=[1, 1, 1])))
    for key in out1:
        assert np.array_equal(np.asarray(out1[key]), np.asarray(out2[key])), key
