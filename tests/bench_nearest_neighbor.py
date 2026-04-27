# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""
Benchmark mdapy's k-nearest-neighbor finder against ovito's
NearestNeighborFinder on multiple cell geometries (orthogonal, triclinic,
mixed PBC) and several system sizes up to ~4M atoms.

Each timing isolates two stages:
  * build  — constructing the spatial index
  * query  — running the k-NN query for every atom

Run with:
    python tests/bench_nearest_neighbor.py [--k 12] [--reps 3] [--max 4000000]
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np

import mdapy as mp
from mdapy import _fast_knn
from mdapy.box import Box


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _time_best(fn: Callable[[], None], reps: int) -> float:
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _fcc_supercell(n: int) -> mp.System:
    return mp.build_crystal("Al", "FCC", 4.05, nx=n, ny=n, nz=n)


def _shear(system: mp.System, shear: float = 0.15) -> mp.System:
    """Return a copy of `system` with a triclinic shear applied."""
    pos = system.data.select(["x", "y", "z"]).to_numpy().copy()
    box_mat = system.box.box.copy()
    # shear b along x
    box_mat[1, 0] += shear * box_mat[1, 1]
    pos[:, 0] += shear * pos[:, 1]
    return mp.System(pos=pos, box=Box(box_mat, boundary=[1, 1, 1]))


def _set_boundary(system: mp.System, boundary) -> mp.System:
    pos = system.data.select(["x", "y", "z"]).to_numpy().copy()
    return mp.System(pos=pos, box=Box(system.box.box.copy(), boundary=boundary))


# ---------------------------------------------------------------------------
#  Per-stage timing
# ---------------------------------------------------------------------------

@dataclass
class Timing:
    build: float
    query: float

    @property
    def total(self) -> float:
        return self.build + self.query


def _time_mdapy(system: mp.System, k: int, reps: int) -> Timing:
    """Time mdapy's KNN build and query phases separately via the Tree wrapper."""
    pos = system.data.select(["x", "y", "z"]).to_numpy()
    x = np.ascontiguousarray(pos[:, 0])
    y = np.ascontiguousarray(pos[:, 1])
    z = np.ascontiguousarray(pos[:, 2])
    box_mat = system.box.box
    origin = system.box.origin
    boundary = system.box.boundary
    N = system.N

    # warm-up
    tree = _fast_knn.Tree()
    tree.build_with_coords(x, y, z, box_mat, origin, boundary)
    idx = np.zeros((N, k), np.int32)
    dst = np.zeros((N, k), np.float64)
    tree.query_knn_batch(x, y, z, k, True, idx, dst)

    # build phase
    def _build():
        t = _fast_knn.Tree()
        t.build_with_coords(x, y, z, box_mat, origin, boundary)
        # keep a reference so the destructor doesn't run inside the timer
        global _last_tree
        _last_tree = t
    t_build = _time_best(_build, reps)

    # query phase (against the warmed-up tree)
    def _query():
        tree.query_knn_batch(x, y, z, k, True, idx, dst)
    t_query = _time_best(_query, reps)

    return Timing(t_build, t_query)


def _time_ovito(system: mp.System, k: int, reps: int) -> Timing:
    from ovito.data import NearestNeighborFinder
    data = system.to_ovito()

    # warm-up
    f = NearestNeighborFinder(k, data)
    f.find_all()

    def _build():
        global _last_finder
        _last_finder = NearestNeighborFinder(k, data)
    t_build = _time_best(_build, reps)

    def _query():
        f.find_all()
    t_query = _time_best(_query, reps)

    return Timing(t_build, t_query)


# ---------------------------------------------------------------------------
#  Reporting
# ---------------------------------------------------------------------------

def _print_header(label: str):
    print(f"\n=== {label} ===")
    print(f"{'N atoms':>10} | "
          f"{'mdapy build':>11} {'mdapy query':>11} {'mdapy total':>11} | "
          f"{'ovito build':>11} {'ovito query':>11} {'ovito total':>11} | "
          f"{'speedup':>8}")
    print("-" * 116)


def _print_row(N, mt: Timing, ot: Timing | None):
    if ot is None:
        print(f"{N:>10d} | "
              f"{mt.build:>11.4f} {mt.query:>11.4f} {mt.total:>11.4f} | "
              f"{'-':>11} {'-':>11} {'-':>11} | {'-':>8}")
    else:
        speedup = ot.total / mt.total if mt.total > 0 else float("nan")
        print(f"{N:>10d} | "
              f"{mt.build:>11.4f} {mt.query:>11.4f} {mt.total:>11.4f} | "
              f"{ot.build:>11.4f} {ot.query:>11.4f} {ot.total:>11.4f} | "
              f"{speedup:>7.2f}x")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def _build_systems(max_atoms: int) -> List[Tuple[str, mp.System]]:
    """Build a sequence of FCC supercells of growing size, capped at max_atoms."""
    systems = []
    # 4*n^3 atoms; pick n so the max system is just under max_atoms
    for n in (20, 30, 40, 60, 80, 100):
        N = 4 * n ** 3
        if N > max_atoms:
            break
        systems.append((f"FCC ortho n={n}", _fcc_supercell(n)))
    return systems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--max", type=int, default=4_000_000,
                        help="maximum number of atoms to benchmark")
    args = parser.parse_args()

    have_ovito = True
    try:
        from ovito.data import NearestNeighborFinder  # noqa: F401
    except ImportError:
        have_ovito = False
        print("WARNING: ovito not installed; only mdapy timings will be shown")

    print(f"\nBenchmark: k={args.k}, reps={args.reps} (best of), max={args.max} atoms")

    base_systems = _build_systems(args.max)

    # ---- Orthogonal, full PBC (largest set) ----
    _print_header("Orthogonal, PBC=[1,1,1]")
    for label, sys_obj in base_systems:
        mt = _time_mdapy(sys_obj, args.k, args.reps)
        ot = _time_ovito(sys_obj, args.k, args.reps) if have_ovito else None
        _print_row(sys_obj.N, mt, ot)

    # ---- Triclinic ----
    _print_header("Triclinic (sheared 0.15), PBC=[1,1,1]")
    for label, sys_obj in base_systems:
        if sys_obj.N > min(args.max, 2_000_000):
            continue  # skip biggest in triclinic to save time
        sheared = _shear(sys_obj, 0.15)
        mt = _time_mdapy(sheared, args.k, args.reps)
        ot = _time_ovito(sheared, args.k, args.reps) if have_ovito else None
        _print_row(sheared.N, mt, ot)

    # ---- Mixed PBC: open in z (slab) ----
    _print_header("Orthogonal slab, PBC=[1,1,0]")
    for label, sys_obj in base_systems:
        if sys_obj.N > min(args.max, 2_000_000):
            continue
        slab = _set_boundary(sys_obj, [1, 1, 0])
        mt = _time_mdapy(slab, args.k, args.reps)
        ot = _time_ovito(slab, args.k, args.reps) if have_ovito else None
        _print_row(slab.N, mt, ot)


if __name__ == "__main__":
    main()
