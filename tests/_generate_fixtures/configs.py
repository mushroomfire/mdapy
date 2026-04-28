# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""
Reference configurations used by the fixture generators.

The set is intentionally broad:

  * Perfect crystals (FCC / BCC / HCP / diamond) — the algorithm should
    return its analytic value (CSP = 0, CNA type = 1/2/3, etc.).
  * Thermally rattled crystals — small Gaussian noise; ensures the test
    is sensitive to numerical precision and integration-vs-OVITO drift.
  * Crystals with point defects (vacancy / interstitial) — exercises
    atoms with non-perfect coordination shells.
  * Strained / sheared crystals — exercises non-cubic and triclinic
    bounding boxes.
  * Stacking-fault inserted via Miller-index orientation — exercises
    intermediate ordered-but-defective neighborhoods.
  * Slab geometries (one or two open boundaries) — exercises atoms
    with truncated coordination shells.
  * Multi-element alloy — small set of mixed types.

Each entry is a callable returning an `mdapy.System`. Keep them small
(≈100–500 atoms) so the saved fixtures stay tiny on disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import mdapy as mp
from mdapy.box import Box


def _rattle(system: mp.System, sigma: float, seed: int) -> mp.System:
    rng = np.random.default_rng(seed)
    pos = system.data.select(["x", "y", "z"]).to_numpy().copy()
    pos += rng.normal(0.0, sigma, pos.shape)
    return mp.System(pos=pos, box=Box(system.box.box.copy(),
                                      boundary=list(system.box.boundary)))


def _drop_atom(system: mp.System, idx: int) -> mp.System:
    pos = system.data.select(["x", "y", "z"]).to_numpy()
    keep = np.ones(pos.shape[0], dtype=bool); keep[idx] = False
    return mp.System(pos=pos[keep], box=Box(system.box.box.copy(),
                                            boundary=list(system.box.boundary)))


def _insert_atom(system: mp.System, p) -> mp.System:
    pos = system.data.select(["x", "y", "z"]).to_numpy()
    pos = np.vstack([pos, np.asarray(p, dtype=float).reshape(1, 3)])
    return mp.System(pos=pos, box=Box(system.box.box.copy(),
                                      boundary=list(system.box.boundary)))


def _strain(system: mp.System, e_xx: float, shear_xy: float = 0.0) -> mp.System:
    pos = system.data.select(["x", "y", "z"]).to_numpy().copy()
    box = system.box.box.copy()
    F = np.eye(3)
    F[0, 0] += e_xx
    F[0, 1] += shear_xy
    pos = pos @ F.T
    box = box @ F.T
    return mp.System(pos=pos, box=Box(box, boundary=list(system.box.boundary)))


def _open_boundary(system: mp.System, axis: int) -> mp.System:
    bnd = list(system.box.boundary); bnd[axis] = 0
    pos = system.data.select(["x", "y", "z"]).to_numpy()
    return mp.System(pos=pos, box=Box(system.box.box.copy(), boundary=bnd))


# --- the registry of configs we test against -------------------------------

def _perfect_fcc(): return mp.build_crystal("Al", "fcc", 4.05, nx=4, ny=4, nz=4)
def _perfect_bcc(): return mp.build_crystal("Fe", "bcc", 2.86, nx=4, ny=4, nz=4)
def _perfect_hcp(): return mp.build_crystal("Mg", "hcp", 3.21, nx=4, ny=4, nz=3)
def _perfect_diamond(): return mp.build_crystal("Si", "diamond", 5.43, nx=3, ny=3, nz=3)


def _rattled_fcc():       return _rattle(_perfect_fcc(), sigma=0.05, seed=0)
def _hot_fcc():           return _rattle(_perfect_fcc(), sigma=0.20, seed=1)
def _rattled_bcc():       return _rattle(_perfect_bcc(), sigma=0.05, seed=2)
def _vacancy_fcc():       return _drop_atom(_perfect_fcc(), idx=10)
def _interstitial_fcc():
    s = _perfect_fcc()
    cell_origin = np.array([s.box.box[0, 0], s.box.box[1, 1], s.box.box[2, 2]]) * 0.25
    return _insert_atom(s, cell_origin + np.array([0.5, 0.5, 0.5]))


def _strained_fcc():      return _strain(_perfect_fcc(), e_xx=0.04)
def _sheared_fcc():       return _strain(_perfect_fcc(), e_xx=0.0, shear_xy=0.10)


def _slab_fcc():          return _open_boundary(_perfect_fcc(), axis=2)
def _wire_fcc():
    s = _open_boundary(_perfect_fcc(), axis=1)
    return _open_boundary(s, axis=2)


def _highly_tilted_fcc():
    """Strongly sheared cell — exercises the triclinic path with a
    near-degenerate box. Atoms are still FCC; the simulation cell is just
    skewed so triclinic-aware code is forced to handle non-orthogonal
    minimum-image conventions."""
    s = _perfect_fcc()
    return _strain(s, e_xx=0.0, shear_xy=0.30)


def _random_atoms():
    """Uniformly random positions in a cubic PBC box. There is no crystal
    structure at all, so structure-classification algorithms must be
    robust against fully disordered input."""
    rng = np.random.default_rng(7)
    L = 12.0
    N = 300
    pos = rng.random((N, 3)) * L
    return mp.System(pos=pos, box=Box(np.eye(3) * L, boundary=[1, 1, 1]))


# Path to the canonical input-file directory; misc fixture generators use it
# to load the sample data files that ship with the repo.
INPUT_DIR = Path(__file__).parent.parent / "input_files"


CONFIGS = {
    # name -> System factory. Names are also used as fixture filenames.
    "perfect_fcc":       _perfect_fcc,
    "perfect_bcc":       _perfect_bcc,
    "perfect_hcp":       _perfect_hcp,
    "perfect_diamond":   _perfect_diamond,
    "rattled_fcc":       _rattled_fcc,
    "hot_fcc":           _hot_fcc,
    "rattled_bcc":       _rattled_bcc,
    "vacancy_fcc":       _vacancy_fcc,
    "interstitial_fcc":  _interstitial_fcc,
    "strained_fcc":      _strained_fcc,
    "sheared_fcc":       _sheared_fcc,
    "highly_tilted_fcc": _highly_tilted_fcc,
    "slab_fcc":          _slab_fcc,
    "wire_fcc":          _wire_fcc,
    "random_atoms":      _random_atoms,
}


# Per-config nominal lattice constants. Used by algorithms that need a
# physical cutoff (CNA, CNP). For configs that aren't FCC-like, we fall
# back to a value matching the structure's first/second NN gap.
LATTICE_A = {
    "perfect_fcc":       4.05,
    "rattled_fcc":       4.05,
    "hot_fcc":           4.05,
    "vacancy_fcc":       4.05,
    "interstitial_fcc":  4.05,
    "strained_fcc":      4.05,
    "sheared_fcc":       4.05,
    "highly_tilted_fcc": 4.05,
    "slab_fcc":          4.05,
    "wire_fcc":          4.05,
    "perfect_bcc":       2.86,
    "rattled_bcc":       2.86,
    "perfect_hcp":       3.21,
    "perfect_diamond":   5.43,
    # random_atoms has no lattice — use a cutoff that yields a few neighbors
    "random_atoms":      None,
}


# Per-test overrides. CSP, for example, is well-defined only when
# `num_neighbors` matches a structure's natural coordination number;
# k=12 on BCC or diamond falls into a degenerate shell where tie-
# breaking differs between implementations. Map each algorithm to its
# preferred per-config parameters here.

CSP_NUM_NEIGHBORS = {
    # Structure-specific natural coordination numbers
    "perfect_bcc":      8,
    "rattled_bcc":      8,
    "perfect_diamond":  4,
    # Surfaces/wires lose neighbors on the open faces — k=12 forces the
    # algorithm to reach into a degenerate shell where the 12-th neighbor
    # is ambiguous. We omit them from the strict per-atom CSP test.
}

CSP_SKIP = {"slab_fcc", "wire_fcc"}
