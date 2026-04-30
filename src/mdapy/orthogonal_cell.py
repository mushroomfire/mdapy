# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""
Convert a triclinic, fully-periodic :class:`mdapy.System` to an
equivalent orthogonal supercell.

This is the mdapy port of atomsk's ``-orthogonal-cell`` option. The
algorithm follows ``opt_orthocell.f90``: along each Cartesian axis we
search integer combinations of the original lattice vectors that
produce a vector aligned with that axis, then keep the shortest such
combination. Replicating the original cell over those integer ranges
and filtering atoms inside the new box yields the orthogonal supercell.

A typical use case is converting an HCP / wurtzite / graphite primitive
cell (120° hex) into its conventional orthogonal supercell — useful
when downstream tooling does not handle triclinic boxes.

Usage
-----
>>> import mdapy as mp
>>> hcp = mp.build_crystal("Mg", "hcp", a=3.21, c=5.21)
>>> ortho = mp.orthogonal_cell(hcp)              # 4-atom orthogonal cell
>>> ortho_min = mp.orthogonal_cell(hcp, find_minimal=True)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import polars as pl

from mdapy.box import Box
from mdapy.system import System


__all__ = ["orthogonal_cell"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_axis_aligned(box: np.ndarray, i: int, tol: float) -> bool:
    """Is row ``i`` of ``box`` already (positive) along Cartesian axis ``i``?"""
    v = box[i]
    return (
        abs(np.linalg.norm(v) - abs(v[i])) < tol
        and v[i] > tol
    )


def _find_axis_combination(
    box: np.ndarray, axis: int, max_search: int, tol: float,
) -> Tuple[Optional[Tuple[int, int, int]], float]:
    """Search integer ``(m, n, o)`` such that ``m H1 + n H2 + o H3`` is
    parallel to Cartesian axis ``axis`` and as short as possible.

    Returns ``((m, n, o), length)`` or ``(None, inf)`` if nothing fits."""
    j = (axis + 1) % 3
    k = (axis + 2) % 3
    best: Optional[Tuple[int, int, int]] = None
    best_len = np.inf
    for m in range(-max_search, max_search + 1):
        for n in range(-max_search, max_search + 1):
            for o in range(-max_search, max_search + 1):
                if m == 0 and n == 0 and o == 0:
                    continue
                v = m * box[0] + n * box[1] + o * box[2]
                # Must lie along axis: orthogonal-axis components ≈ 0
                # and primary component must be positive (we want
                # right-handed output).
                if abs(v[j]) > tol or abs(v[k]) > tol:
                    continue
                if v[axis] <= tol:
                    continue
                if v[axis] < best_len:
                    best_len = v[axis]
                    best = (m, n, o)
    return best, float(best_len)


def _find_minimal_orthogonal_cell(
    box: np.ndarray, pos: np.ndarray, elements: Optional[np.ndarray],
    extras: Optional[pl.DataFrame], max_search: int, tol: float,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[pl.DataFrame]]:
    """Brute-force search a smaller orthogonal sub-cell that, when
    replicated ``(nx, ny, nz)`` times, reproduces the input atoms
    (positions + elements). Species-aware: a sub-cell is rejected if
    two replicated images map to the same site with different elements.
    """
    n_atoms = pos.shape[0]
    if n_atoms == 0:
        return box, pos, elements, extras

    a, b, c = box[0, 0], box[1, 1], box[2, 2]
    # Wrap into [0, 1) fractional coords for the search.
    frac = pos / np.array([a, b, c])
    frac -= np.floor(frac + tol)

    best_box = box.copy()
    best_pos = pos.copy()
    best_ele = None if elements is None else elements.copy()
    best_extras = extras.clone() if extras is not None else None
    best_n = n_atoms

    for nx in range(1, max_search + 1):
        for ny in range(1, max_search + 1):
            for nz in range(1, max_search + 1):
                if nx == 1 and ny == 1 and nz == 1:
                    continue
                n_div = nx * ny * nz
                if n_atoms % n_div:
                    continue
                expected = n_atoms // n_div
                if expected >= best_n:
                    continue

                inv = np.array([nx, ny, nz])
                low = np.array([1.0 / nx - tol, 1.0 / ny - tol, 1.0 / nz - tol])
                in_first = np.all(
                    (frac >= -tol) & (frac < low), axis=1
                )
                if int(in_first.sum()) != expected:
                    continue

                small_frac = (frac[in_first] * inv) % 1.0
                small_ele = None if elements is None else elements[in_first]

                # Verify replication recovers the original (positions
                # AND elements).
                ok = True
                rep = []
                rep_ele_idx = []
                for ix in range(nx):
                    for iy in range(ny):
                        for iz in range(nz):
                            shifted = (small_frac + np.array([ix, iy, iz])) / inv
                            shifted -= np.floor(shifted + tol)
                            rep.append(shifted)
                            if elements is not None:
                                rep_ele_idx.append(small_ele)
                rep = np.vstack(rep)
                rep_ele = (
                    None
                    if elements is None
                    else np.concatenate(rep_ele_idx)
                )

                # Match each replicated point to an original point.
                matched = np.zeros(n_atoms, bool)
                for r_idx, r in enumerate(rep):
                    diff = frac - r
                    diff -= np.round(diff)
                    candidate = np.where(
                        (np.linalg.norm(diff, axis=1) < tol) & (~matched)
                    )[0]
                    if candidate.size == 0:
                        ok = False
                        break
                    idx = candidate[0]
                    if rep_ele is not None and rep_ele[r_idx] != elements[idx]:
                        ok = False
                        break
                    matched[idx] = True
                if not ok or not matched.all():
                    continue

                if expected < best_n:
                    best_n = expected
                    best_box = np.diag([a / nx, b / ny, c / nz]).astype(float)
                    best_pos = small_frac * np.array(
                        [a / nx, b / ny, c / nz]
                    )
                    best_ele = None if elements is None else small_ele
                    if extras is not None:
                        best_extras = extras[np.flatnonzero(in_first).tolist()]
                    else:
                        best_extras = None

    return best_box, best_pos, best_ele, best_extras


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def orthogonal_cell(
    system: System,
    find_minimal: bool = False,
    max_search: int = 20,
    tol: float = 1e-6,
) -> System:
    """Convert a triclinic :class:`System` to an equivalent orthogonal
    supercell.

    Parameters
    ----------
    system : System
        Input crystal. Must have all three boundaries periodic; an open
        boundary makes the orthogonal-supercell construction
        ill-defined and is rejected.
    find_minimal : bool, default=False
        After building the orthogonal cell, also search for the
        smallest orthogonal sub-cell that reproduces the same crystal
        when replicated. Species-aware (won't merge sites that carry
        different elements).
    max_search : int, default=20
        Range of integer combinations searched along each lattice
        vector. Increase for very oblique cells where the canonical
        orthogonal supercell needs large coefficients; the cost grows
        as ``(2 max_search + 1)^3``.
    tol : float, default=1e-6
        Numerical tolerance for axis alignment, fractional-coordinate
        wrapping, and duplicate-atom detection.

    Returns
    -------
    System
        New System whose ``box`` is diagonal and whose atoms reproduce
        the input crystal exactly. The output is fully periodic.

    Raises
    ------
    ValueError
        If the input has any open boundary, if the input box is
        singular, or if no orthogonal supercell can be found within
        ``max_search``.

    Notes
    -----
    Algorithm follows atomsk's ``opt_orthocell.f90``:

    1. For each Cartesian axis, find the shortest non-zero integer
       combination of the input lattice vectors that produces a
       cartesian vector aligned with that axis.
    2. Replicate the input cell by enough copies to cover the
       resulting orthogonal box.
    3. Wrap atoms into the orthogonal box and deduplicate with
       ``tol`` precision; species-aware (an exact-position duplicate
       carrying a different element raises).
    4. (Optional) reduce to the smallest periodic sub-cell.

    Examples
    --------
    HCP primitive (120° hex) → 4-atom orthogonal supercell:

    >>> hcp = mp.build_crystal("Mg", "hcp", a=3.21, c=5.21)
    >>> ortho = mp.orthogonal_cell(hcp)
    >>> assert ortho.N == 4

    Wurtzite GaN → orthogonal supercell with both species preserved:

    >>> wz = mp.build_crystal(("Ga", "N"), "wurtzite", a=3.19, c=5.18)
    >>> ortho = mp.orthogonal_cell(wz)
    >>> sorted(set(ortho.data["element"].to_list()))
    ['Ga', 'N']
    """
    if not all(b == 1 for b in system.box.boundary):
        raise ValueError(
            "orthogonal_cell requires a fully periodic input "
            "(box.boundary must be [1, 1, 1])."
        )

    box = np.asarray(system.box.box, dtype=float)
    origin = np.asarray(system.box.origin, dtype=float)
    if abs(np.linalg.det(box)) < tol:
        raise ValueError("Input box is singular (zero volume).")

    # ----------------------------------------------------------------
    # Step 1 — find shortest axis-aligned integer combinations.
    # ----------------------------------------------------------------
    mno = np.zeros((3, 3), dtype=np.int64)  # rows: new vec in old basis
    for i in range(3):
        if _is_axis_aligned(box, i, tol):
            mno[i, i] = 1
            continue
        # Atomsk-style escalating search: try a small bound first, then
        # widen if needed. Small max_search keeps simple cells fast.
        found = None
        for bound in (max_search, max_search * 2, max_search * 5):
            best, _ = _find_axis_combination(box, i, bound, tol)
            if best is not None:
                found = best
                break
        if found is None:
            raise ValueError(
                f"Could not find an integer combination of the input "
                f"lattice vectors aligned with Cartesian axis "
                f"{['x', 'y', 'z'][i]} within max_search={max_search * 5}. "
                "Either the input cell is too oblique or numerical "
                "noise prevents alignment; try increasing `max_search` "
                "or `tol`."
            )
        mno[i] = found

    # New orthogonal box: row i is mno[i] @ box (which is along axis i).
    new_box_rows = mno @ box
    # Defensive: zero out the off-axis components introduced by
    # floating-point in `mno @ box`.
    new_lengths = np.array([new_box_rows[i, i] for i in range(3)])
    if np.any(new_lengths <= 0):
        raise ValueError(
            "Computed new lattice vectors are not positive — "
            "input box may not be right-handed."
        )
    new_box = np.diag(new_lengths)

    # ----------------------------------------------------------------
    # Step 2 — replicate the input cell over a range that fully covers
    # the new orthogonal box, then filter / dedupe.
    # ----------------------------------------------------------------
    # The orthogonal cell volume is |det(mno)| times the input volume,
    # so we need at least that many replicas. To be safe (atoms may
    # spill across mno's bounding box), use a margin of |mno| max + 1.
    margin = int(np.max(np.abs(mno))) + 1
    rng = range(-margin, margin + 1)

    pos = system.data.select("x", "y", "z").to_numpy() - origin
    n_atoms = pos.shape[0]
    has_element = "element" in system.data.columns
    elements = (
        np.asarray(system.data["element"].to_list(), dtype=object)
        if has_element else None
    )
    # Carry per-atom non-position columns through the transform.
    extras_cols = [
        c for c in system.data.columns if c not in ("x", "y", "z", "element")
    ]
    extras = system.data.select(extras_cols) if extras_cols else None

    n_replicas = (2 * margin + 1) ** 3
    rep_pos = np.empty((n_replicas * n_atoms, 3), dtype=float)
    rep_ele = (
        np.empty(n_replicas * n_atoms, dtype=object) if has_element else None
    )
    rep_src = np.empty(n_replicas * n_atoms, dtype=np.int64)
    cursor = 0
    for ix in rng:
        for iy in rng:
            for iz in rng:
                shift = ix * box[0] + iy * box[1] + iz * box[2]
                rep_pos[cursor:cursor + n_atoms] = pos + shift
                if rep_ele is not None:
                    rep_ele[cursor:cursor + n_atoms] = elements
                rep_src[cursor:cursor + n_atoms] = np.arange(n_atoms)
                cursor += n_atoms

    # Keep atoms inside [0, L) along each axis (right-edge exclusive).
    inside = np.all(
        (rep_pos > -tol)
        & (rep_pos < new_lengths - tol),
        axis=1,
    )
    sel_pos = rep_pos[inside]
    sel_ele = rep_ele[inside] if rep_ele is not None else None
    sel_src = rep_src[inside]

    # Sanity: expected atom count is |det(mno)| * n_atoms.
    expected_n = int(round(abs(np.linalg.det(mno.astype(float)))) * n_atoms)
    if sel_pos.shape[0] != expected_n:
        raise ValueError(
            f"orthogonal_cell: produced {sel_pos.shape[0]} atoms but "
            f"expected {expected_n} = |det(mno)| * N. The cell may have "
            "atoms exactly on the boundary; try perturbing positions or "
            "tightening `tol`."
        )

    # Wrap into [0, L). Atoms on the right edge fold back to 0; atoms
    # at sub-tol negative offsets (from floating-point in the basis →
    # cartesian conversion) snap to 0 instead of getting wrapped to L.
    sel_pos = sel_pos - np.floor(sel_pos / new_lengths + tol) * new_lengths
    sel_pos = np.where(np.abs(sel_pos) < tol, 0.0, sel_pos)

    # ----------------------------------------------------------------
    # Step 3 — optional minimal-cell reduction.
    # ----------------------------------------------------------------
    sel_extras = extras[sel_src.tolist()] if extras is not None else None
    if find_minimal:
        new_box_arr, sel_pos, sel_ele, sel_extras = _find_minimal_orthogonal_cell(
            new_box, sel_pos, sel_ele, sel_extras, max_search, tol,
        )
        new_box = new_box_arr

    # ----------------------------------------------------------------
    # Step 4 — assemble the output System.
    # ----------------------------------------------------------------
    df = pl.from_numpy(sel_pos, schema=["x", "y", "z"])
    if sel_ele is not None:
        df = df.with_columns(pl.lit(sel_ele).alias("element"))
    if sel_extras is not None and sel_extras.shape[1] > 0:
        df = pl.concat([df, sel_extras], how="horizontal")

    return System(data=df, box=Box(new_box, boundary=[1, 1, 1]))
