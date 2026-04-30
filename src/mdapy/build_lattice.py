# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

from mdapy import _repeat_cell
from mdapy.system import System
from mdapy.box import Box
import polars as pl
import numpy as np
from typing import Optional, Tuple


def _get_basispos_and_box_cubic(
    structure: str, a: float, c_over_a: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the orthogonal-supercell representation used by the legacy
    ``hcp`` and ``graphene`` paths and by the cubic FCC/BCC/diamond cases.
    Returns ``(box, basis_pos)`` only — species ordering is the implicit
    "first species at every site". For multi-species or non-orthogonal
    cells, use :func:`_get_structure_definition`.
    """
    if c_over_a is None:
        c_over_a = np.sqrt(8 / 3)
    s = structure.lower()
    if s in ("fcc", "bcc", "diamond"):
        box = np.array([[a, 0, 0], [0, a, 0], [0, 0, a]], dtype=float)
        if s == "fcc":
            basis_pos = np.array(
                [
                    (0.0, 0.0, 0.0),
                    (0.0, 0.5, 0.5),
                    (0.5, 0.0, 0.5),
                    (0.5, 0.5, 0.0),
                ]
            )

        elif s == "bcc":
            basis_pos = np.array(
                [
                    (0.0, 0.0, 0.0),
                    (0.5, 0.5, 0.5),
                ]
            )
        elif s == "diamond":
            basis_pos = np.array(
                [
                    (0.0, 0.0, 0.0),
                    (0.25, 0.25, 0.25),
                    (0.0, 0.5, 0.5),
                    (0.25, 0.75, 0.75),
                    (0.5, 0.0, 0.5),
                    (0.75, 0.25, 0.75),
                    (0.5, 0.5, 0.0),
                    (0.75, 0.75, 0.25),
                ]
            )
    elif s == "hcp":
        box = np.array(
            [
                [a, 0.0, 0.0],
                [0.0, np.sqrt(3) * a, 0.0],
                [0.0, 0.0, a * c_over_a],
            ]
        )
        basis_pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.5, 5 / 6, 0.5],
                [0.0, 1 / 3, 0.5],
            ]
        )
    elif s == "graphene":
        box = np.array(
            [
                [3.0 * a, 0.0, 0.0],
                [0.0, np.sqrt(3) * a, 0.0],
                [0.0, 0.0, 3.4],
            ]
        )
        basis_pos = np.array(
            [[1 / 6, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [2 / 3, 0.5, 0.0]]
        )

    else:
        raise ValueError(
            f"Unsupported structure: {structure}, only support fcc, bcc, hcp, graphene, and diamond"
        )

    return box, basis_pos


# ===========================================================================
# Atomsk-compatible structure definitions
# ===========================================================================
#
# Each entry returns ``(box_3x3, basis_frac, basis_species_idx)``, where:
#   - ``box_3x3``  rows are the lattice vectors in Cartesian coordinates.
#   - ``basis_frac`` is an ``(N, 3)`` array of fractional coordinates inside
#                   that box.
#   - ``basis_species_idx`` is an ``(N,)`` int array; ``0..M-1`` index into
#                   the user-supplied ``name`` tuple. ``M`` must be in
#                   ``allowed_n_species``.
#
# All positions, orderings and box conventions match the corresponding
# CASE block in atomsk's ``mode_create.f90`` so byte-equivalent output is
# possible after sorting on Cartesian position.

_SQRT3 = np.sqrt(3.0)


def _basis_sc(a, c=None):
    box = a * np.eye(3)
    basis = np.array([[0.0, 0.0, 0.0]])
    species = np.array([0], dtype=np.int32)
    return box, basis, species


def _basis_bcc_2sp(a, c=None):
    """B2 / CsCl: identical to BCC but with two distinct species."""
    box = a * np.eye(3)
    basis = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    species = np.array([0, 1], dtype=np.int32)
    return box, basis, species


def _basis_rocksalt(a, c=None):
    """B1 / NaCl: two interpenetrating fcc lattices, species 1 (4 atoms)
    + species 2 (4 atoms) = 8 atoms / cell."""
    box = a * np.eye(3)
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [0.5, 0.5, 0.5],
    ])
    species = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
    return box, basis, species


def _basis_zincblende(a, c=None):
    """B3: diamond with two distinct species. species 1 occupies the 4 fcc
    sites, species 2 the 4 tetrahedral sites."""
    box = a * np.eye(3)
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.25, 0.25, 0.25],
        [0.75, 0.75, 0.25],
        [0.75, 0.25, 0.75],
        [0.25, 0.75, 0.75],
    ])
    species = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
    return box, basis, species


def _basis_fluorite(a, c=None):
    """Fluorite (CaF2): 4 cation FCC sites + 8 anion sites at the corners
    of two interior cubes (species ratio 1:2)."""
    box = a * np.eye(3)
    basis = np.array([
        # cations (species 0) — 4 fcc sites
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        # anions (species 1) — 8 sites at z=1/4 and z=3/4
        [0.25, 0.25, 0.25],
        [0.75, 0.25, 0.25],
        [0.25, 0.75, 0.25],
        [0.75, 0.75, 0.25],
        [0.25, 0.25, 0.75],
        [0.75, 0.25, 0.75],
        [0.25, 0.75, 0.75],
        [0.75, 0.75, 0.75],
    ])
    species = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32)
    return box, basis, species


def _basis_l1_2(a, c=None):
    """L1_2 (Ni3Al / Cu3Au): species 0 occupies the 3 face centers,
    species 1 occupies the corner."""
    box = a * np.eye(3)
    basis = np.array([
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.0, 0.0, 0.0],
    ])
    species = np.array([0, 0, 0, 1], dtype=np.int32)
    return box, basis, species


def _basis_perovskite(a, c=None):
    """Cubic perovskite ABO3. atomsk's ordering is **B, A, O**:
    species 0 at body centre, species 1 at corner, species 2 at face
    centres."""
    box = a * np.eye(3)
    basis = np.array([
        [0.5, 0.5, 0.5],   # B
        [0.0, 0.0, 0.0],   # A
        [0.5, 0.0, 0.0],   # O
        [0.0, 0.5, 0.0],   # O
        [0.0, 0.0, 0.5],   # O
    ])
    species = np.array([0, 1, 2, 2, 2], dtype=np.int32)
    return box, basis, species


def _hexagonal_box(a, c):
    """atomsk-compatible hexagonal primitive cell with 120° angle."""
    return np.array([
        [a, 0.0, 0.0],
        [-0.5 * a, 0.5 * _SQRT3 * a, 0.0],
        [0.0, 0.0, c],
    ])


def _basis_hcp(a, c):
    """atomsk-compatible 2-atom hexagonal primitive cell. The legacy
    orthogonal 4-atom supercell that mdapy used to emit is *not*
    selected via this entry — it is preserved only for the
    ``"legacy_orth"`` dispatch tag and accessible via
    ``_get_basispos_and_box_cubic`` for callers that depend on the old
    layout."""
    box = _hexagonal_box(a, c)
    basis = np.array([
        [0.0, 0.0, 0.0],
        [1.0 / 3.0, 2.0 / 3.0, 0.5],
    ])
    species = np.array([0, 1], dtype=np.int32)
    return box, basis, species


def _basis_wurtzite(a, c):
    """Wurtzite B4 (e.g. GaN): 4-atom hexagonal primitive."""
    box = _hexagonal_box(a, c)
    basis = np.array([
        [1.0 / 3.0, 2.0 / 3.0, 0.0],
        [2.0 / 3.0, 1.0 / 3.0, 0.5],
        [1.0 / 3.0, 2.0 / 3.0, 3.0 / 8.0],
        [2.0 / 3.0, 1.0 / 3.0, 7.0 / 8.0],
    ])
    species = np.array([0, 0, 1, 1], dtype=np.int32)
    return box, basis, species


def _basis_graphite(a, c):
    """Hexagonal graphite (A9): two layers along ``c`` with AB stacking,
    4 atoms per primitive cell."""
    box = _hexagonal_box(a, c)
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5],
        [1.0 / 3.0, 2.0 / 3.0, 0.0],
        [2.0 / 3.0, 1.0 / 3.0, 0.5],
    ])
    # Per atomsk: P(3,4)=P(1,4) (species 0), P(4,4)=P(2,4) (species 1 if
    # two species else 0). With nspecies==2 the A-layer is species 0 and
    # the B-layer is species 1 — useful for hexagonal BN-like structures
    # but with the graphite z-spacing.
    species = np.array([0, 1, 0, 1], dtype=np.int32)
    return box, basis, species


# Each entry: (build_fn, allowed_n_species, c_default_factor_or_None)
# c_default_factor: when c is not given, c = a * factor. If None, ``c``
# is unused (cubic structures).
_STRUCTURES = {
    # CUBIC
    "sc":         (_basis_sc,         (1,),     None),
    "fcc":        ("legacy_orth",     (1,),     None),
    "bcc":        ("legacy_orth",     (1,),     None),
    "diamond":    ("legacy_orth",     (1,),     None),
    "cscl":       (_basis_bcc_2sp,    (2,),     None),
    "b2":         (_basis_bcc_2sp,    (2,),     None),
    "rocksalt":   (_basis_rocksalt,   (2,),     None),
    "b1":         (_basis_rocksalt,   (2,),     None),
    "zincblende": (_basis_zincblende, (2,),     None),
    "b3":         (_basis_zincblende, (2,),     None),
    "fluorite":   (_basis_fluorite,   (2,),     None),
    "l1_2":       (_basis_l1_2,       (2,),     None),
    "l12":        (_basis_l1_2,       (2,),     None),
    "perovskite": (_basis_perovskite, (3,),     None),
    # HEXAGONAL — atomsk-compatible 120° primitive cell.
    "hcp":        (_basis_hcp,        (1, 2),   np.sqrt(8 / 3)),
    "wurtzite":   (_basis_wurtzite,   (1, 2),   np.sqrt(8 / 3)),
    "graphite":   (_basis_graphite,   (1, 2),   None),  # c must be given
    # LEGACY — orthogonal mdapy-specific supercell, kept for backward compat.
    "graphene":   ("legacy_orth",     (1,),     None),
}

# Cubic structures supporting Miller-indexed orientation.
_MILLER_CUBIC = {
    "sc", "fcc", "bcc", "diamond", "cscl", "b2", "rocksalt", "b1",
    "zincblende", "b3", "fluorite", "l1_2", "l12", "perovskite",
}
# Hexagonal structures supporting Miller-Bravais [hkil] (or 3-index [uvw])
# orientation.
_MILLER_HEX = {"hcp", "wurtzite", "graphite"}
_MILLER_SUPPORTED = _MILLER_CUBIC | _MILLER_HEX


def _normalize_structure_name(structure: str) -> str:
    s = structure.lower().strip()
    # Handful of common synonyms.
    aliases = {
        "rs": "rocksalt", "nacl": "rocksalt",
        "zb": "zincblende", "gas": "zincblende",
        "wz": "wurtzite", "b4": "wurtzite",
        "a9": "graphite",
    }
    return aliases.get(s, s)


def _gcd(a: int, b: int) -> int:
    """Calculate greatest common divisor (internal helper)."""
    while b:
        a, b = b, a % b
    return abs(a)


def _gcd_three(a: int, b: int, c: int) -> int:
    """Calculate greatest common divisor of three integers (internal helper)."""
    return _gcd(_gcd(a, b), c)


def _reduce_miller(miller: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Reduce Miller indices to simplest form (internal helper)."""
    h, k, L = miller
    if h == 0 and k == 0 and L == 0:
        raise ValueError("Miller indices cannot be all zeros")

    g = _gcd_three(h, k, L)
    if g == 0:
        return miller
    return (h // g, k // g, L // g)


def _check_orthogonality_miller(
    m1: Tuple[int, int, int], m2: Tuple[int, int, int], m3: Tuple[int, int, int]
) -> bool:
    """Check if three Miller indices are orthogonal in cubic system (internal helper)."""
    # For cubic system, [h1k1l1]⊥[h2k2l2] ⟺ h1*h2 + k1*k2 + l1*l2 = 0
    dot12 = m1[0] * m2[0] + m1[1] * m2[1] + m1[2] * m2[2]
    dot13 = m1[0] * m3[0] + m1[1] * m3[1] + m1[2] * m3[2]
    dot23 = m2[0] * m3[0] + m2[1] * m3[1] + m2[2] * m3[2]
    return dot12 == 0 and dot13 == 0 and dot23 == 0


def _check_right_hand_miller(
    m1: Tuple[int, int, int], m2: Tuple[int, int, int], m3: Tuple[int, int, int]
) -> bool:
    """Check if three Miller indices satisfy right-hand rule (internal helper)."""
    # m1 × m2 should be parallel and in same direction as m3
    cross = np.cross(np.array(m1), np.array(m2))
    dot = np.dot(cross, np.array(m3))
    return dot > 0


def _build_transform_matrix(
    miller1: Tuple[int, int, int],
    miller2: Tuple[int, int, int],
    miller3: Tuple[int, int, int],
) -> np.ndarray:
    """Build transformation matrix from Miller indices (internal helper)."""
    M = np.array(
        [
            [miller1[0], miller2[0], miller3[0]],
            [miller1[1], miller2[1], miller3[1]],
            [miller1[2], miller2[2], miller3[2]],
        ],
        dtype=int,
    )

    return M


def _find_atoms_in_new_cell(
    transform_matrix: np.ndarray,
    basis_positions: np.ndarray,
    basis_species: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find all atoms in the new (Miller-rotated) unit cell.

    Each basis atom carries a species index that follows it through the
    transform; deduplicated atoms inherit the species of the first
    occurrence — which is canonical because the lattice is periodic, so
    every duplicate must share the same species index in any
    well-defined ordered structure."""
    M = transform_matrix.astype(float)
    M_inv = np.linalg.inv(M)

    # Calculate search range
    det = abs(np.linalg.det(M))
    expected_atoms = int(np.round(det * len(basis_positions)))

    # Search range determined by matrix elements
    max_coef = int(np.max(np.abs(M)))
    search_range = max_coef + 1

    new_basis_list = []
    new_species_list = []

    # Iterate through periodic replicas of original lattice
    for i in range(-search_range, search_range + 1):
        for j in range(-search_range, search_range + 1):
            for k in range(-search_range, search_range + 1):
                for basis_idx, basis_atom in enumerate(basis_positions):
                    # Fractional coordinates in original lattice
                    frac_old = basis_atom + np.array([i, j, k], dtype=float)

                    # Transform to new lattice fractional coordinates
                    frac_new = M_inv @ frac_old

                    # Wrap to [0, 1)
                    frac_new = frac_new - np.floor(frac_new + 1e-10)

                    # Check if in [0, 1)
                    if np.all(frac_new >= -1e-8) and np.all(frac_new < 1.0 - 1e-8):
                        # Check for duplicates
                        is_duplicate = False
                        for existing in new_basis_list:
                            diff = frac_new - existing
                            diff = diff - np.round(diff)
                            if np.linalg.norm(diff) < 1e-6:
                                is_duplicate = True
                                break

                        if not is_duplicate:
                            new_basis_list.append(frac_new.copy())
                            new_species_list.append(int(basis_species[basis_idx]))

    new_basis = np.array(new_basis_list)
    new_species = np.array(new_species_list, dtype=np.int32)

    # Verify atom count
    if len(new_basis) != expected_atoms:
        print(f"Warning: Expected {expected_atoms} atoms but found {len(new_basis)}")

    return new_basis, new_species


def _align_box_to_axes(
    box: np.ndarray, basis: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Align orthogonal lattice vectors to coordinate axes (internal helper)."""
    # Extract the three lattice vectors
    v1, v2, v3 = box[0], box[1], box[2]

    # Compute lengths
    len1 = np.linalg.norm(v1)
    len2 = np.linalg.norm(v2)
    len3 = np.linalg.norm(v3)

    # Check orthogonality
    dot12 = np.dot(v1, v2)
    dot13 = np.dot(v1, v3)
    dot23 = np.dot(v2, v3)

    if abs(dot12) > 1e-6 or abs(dot13) > 1e-6 or abs(dot23) > 1e-6:
        raise ValueError(
            f"Lattice vectors are not orthogonal! "
            f"v1·v2={dot12:.6f}, v1·v3={dot13:.6f}, v2·v3={dot23:.6f}"
        )

    # Create aligned box with vectors parallel to axes
    aligned_box = np.array(
        [[len1, 0.0, 0.0], [0.0, len2, 0.0], [0.0, 0.0, len3]], dtype=float
    )

    # Fractional coordinates don't change
    aligned_basis = basis.copy()

    return aligned_box, aligned_basis


def _hkil_to_uvw(miller) -> Tuple[int, int, int]:
    """Convert a 3- or 4-index hexagonal direction to ``(u, v, w)``
    relative to the (a1, a2, c) primitive basis.

    For a 4-index ``[hkil]`` direction the conversion follows atomsk's
    ``HKIL2UVW``: ``u = 2h + k``, ``v = h + 2k``, ``w = l``, then the
    triplet is GCD-reduced so the result is the *minimal* integer
    direction. The Miller-Bravais constraint ``h + k + i == 0`` is
    enforced.

    For a 3-index ``[uvw]`` the values are returned unchanged (also
    GCD-reduced).
    """
    if len(miller) == 4:
        h, k, i, L = miller
        if abs(h + k + i) > 1e-9:
            raise ValueError(
                f"4-index hexagonal direction {tuple(miller)} violates "
                "the Miller-Bravais constraint h + k + i = 0."
            )
        u, v, w = 2 * h + k, h + 2 * k, L
    elif len(miller) == 3:
        u, v, w = miller
    else:
        raise ValueError(
            f"Hexagonal Miller indices must be length 3 ([uvw]) or 4 "
            f"([hkil]); got {tuple(miller)}."
        )
    g = _gcd_three(int(u), int(v), int(w))
    if g > 0:
        u, v, w = u // g, v // g, w // g
    return int(u), int(v), int(w)


def _build_lattice_from_miller_hex(
    structure: str,
    miller1, miller2, miller3,
    a: float,
    c: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hexagonal counterpart of :func:`_build_lattice_from_miller`.

    Each Miller direction is given in 3-index ``[uvw]`` or 4-index
    ``[hkil]`` notation; the latter is converted via atomsk's
    ``HKIL2UVW`` rule. Orthogonality and right-handedness are checked in
    Cartesian coordinates because the hexagonal basis is non-orthogonal
    and the integer-Miller test would be wrong here.
    """
    u1, v1, w1 = _hkil_to_uvw(miller1)
    u2, v2, w2 = _hkil_to_uvw(miller2)
    u3, v3, w3 = _hkil_to_uvw(miller3)

    s = _normalize_structure_name(structure)
    build_fn, _, _ = _STRUCTURES[s]
    if build_fn == "legacy_orth":
        raise ValueError(
            f"Hexagonal Miller orientation is not supported for legacy "
            f"structure '{structure}'."
        )
    box, basis, species = build_fn(a, c)
    # Build M with new vectors as COLUMNS, mirroring _build_transform_matrix.
    M = np.array([
        [u1, u2, u3],
        [v1, v2, v3],
        [w1, w2, w3],
    ], dtype=int)
    # Cartesian new lattice vectors. Row k = sum_j M[j, k] * box[j, :].
    new_lattice = M.T @ box

    # Orthogonality + right-hand check in Cartesian.
    eps = 1e-6
    nl = new_lattice
    if (abs(np.dot(nl[0], nl[1])) > eps or
            abs(np.dot(nl[0], nl[2])) > eps or
            abs(np.dot(nl[1], nl[2])) > eps):
        raise ValueError(
            "Hexagonal Miller directions must be mutually orthogonal in "
            "Cartesian space."
        )
    if np.dot(np.cross(nl[0], nl[1]), nl[2]) <= 0:
        raise ValueError(
            "Hexagonal Miller indices must satisfy the right-hand rule "
            "(miller1 × miller2 must be parallel to and same-sign as miller3)."
        )

    new_basis, new_species = _find_atoms_in_new_cell(M, basis, species)

    aligned_lattice, aligned_basis = _align_box_to_axes(new_lattice, new_basis)
    return aligned_lattice, aligned_basis, new_species


def _build_lattice_from_miller(
    structure: str,
    miller1: Tuple[int, int, int],
    miller2: Tuple[int, int, int],
    miller3: Tuple[int, int, int],
    lattice_constant: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cubic Miller-indexed cell builder. Returns ``(box, basis_frac,
    species_idx)`` for the rotated rectangular cell."""
    miller1 = _reduce_miller(miller1)
    miller2 = _reduce_miller(miller2)
    miller3 = _reduce_miller(miller3)

    if not _check_orthogonality_miller(miller1, miller2, miller3):
        raise ValueError(
            f"Miller indices must be orthogonal. Got {miller1}, {miller2}, {miller3}"
        )
    if not _check_right_hand_miller(miller1, miller2, miller3):
        raise ValueError("Miller indices must satisfy right-hand rule")

    build_fn, _, _ = _STRUCTURES[_normalize_structure_name(structure)]
    if build_fn == "legacy_orth":
        box, basis = _get_basispos_and_box_cubic(structure, lattice_constant)
        species = np.zeros(len(basis), dtype=np.int32)
    else:
        box, basis, species = build_fn(lattice_constant)

    M = _build_transform_matrix(miller1, miller2, miller3)
    new_lattice = box @ M.T
    new_basis, new_species = _find_atoms_in_new_cell(M, basis, species)
    aligned_lattice, aligned_basis = _align_box_to_axes(new_lattice, new_basis)
    return aligned_lattice, aligned_basis, new_species


def build_crystal(
    name,
    structure: str,
    a: float,
    miller1: Optional[Tuple[int, int, int]] = None,
    miller2: Optional[Tuple[int, int, int]] = None,
    miller3: Optional[Tuple[int, int, int]] = None,
    nx: int = 1,
    ny: int = 1,
    nz: int = 1,
    c_over_a: float = np.sqrt(8 / 3),
    c: Optional[float] = None,
) -> System:
    """
    Build a crystal supercell. Output matches atomsk's
    ``--create <structure>`` byte-for-byte (after sorting on Cartesian
    position).

    Parameters
    ----------
    name : str or sequence of str
        Element symbols. Pass a single string to broadcast it to every
        basis atom (single-species crystal); pass a tuple/list whose
        length equals one of the structure's *allowed species counts*
        for an ordered multi-species structure (NaCl, GaN, SrTiO\\ :sub:`3`,
        ...). Each element must be a recognised symbol from
        ``mdapy.data.atomic_numbers``.

    structure : str
        Crystal type. Case-insensitive; common synonyms accepted (e.g.
        ``"NaCl"`` → ``"rocksalt"``, ``"zb"`` → ``"zincblende"``,
        ``"wz"`` → ``"wurtzite"``).

        ============== ============= ============================== =================
        Structure name n_species     Common prototype               Miller-rotatable
        ============== ============= ============================== =================
        ``sc``         1             α-Po                            yes (cubic)
        ``fcc``        1             Cu, Al, Ni                      yes (cubic)
        ``bcc``        1             Fe, W                           yes (cubic)
        ``diamond``    1             C, Si                           yes (cubic)
        ``cscl`` /``b2`` 2           CsCl, NiAl                      yes (cubic)
        ``rocksalt``/``b1`` 2        NaCl, MgO                       yes (cubic)
        ``zincblende``/``b3`` 2      GaAs, ZnS                       yes (cubic)
        ``fluorite``   2             CaF\\ :sub:`2`, UO\\ :sub:`2`    yes (cubic)
        ``l1_2``       2             Ni\\ :sub:`3`\\ Al, Cu\\ :sub:`3`\\ Au yes (cubic)
        ``perovskite`` 3 (B, A, O)   SrTiO\\ :sub:`3`                yes (cubic)
        ``hcp``        1 or 2        Mg, Ti, Zr                      yes (hexagonal)
        ``wurtzite``/``b4`` 1 or 2   GaN, ZnO                        yes (hexagonal)
        ``graphite``   1 or 2        graphite, hex-BN                yes (hexagonal)
        ``graphene``   1             single-layer graphene           no (legacy)
        ============== ============= ============================== =================

        For ``perovskite`` the species order follows atomsk: ``(B, A,
        O)``. e.g. ``name=("Ti", "Sr", "O")`` for SrTiO\\ :sub:`3`.

    a : float
        Primary lattice constant in Å (cubic edge, hexagonal basal
        spacing, or in-plane spacing for graphene).

    miller1, miller2, miller3 : sequence of int, optional
        Crystallographic axes for the three sides of the supercell.
        For cubic structures pass three integers ``[h, k, l]``; for
        hexagonal structures (``hcp`` / ``wurtzite`` / ``graphite``)
        pass either Miller-Bravais ``[h, k, i, l]`` (with
        ``h + k + i == 0``) or 3-index ``[u, v, w]`` directly. The
        three vectors must be mutually orthogonal in Cartesian space
        and obey the right-hand rule. When all three are ``None``,
        the canonical orientation is used.

    nx, ny, nz : int, default=1
        Replication counts along the (possibly Miller-rotated) axes.
        Total atom count = ``len(basis) * nx * ny * nz``.

    c_over_a : float, default=sqrt(8/3)
        Ratio of c to a for hexagonal structures. Used only when ``c``
        is not given explicitly. The default is the ideal hcp packing.

    c : float, optional
        Out-of-plane lattice constant in Å for hexagonal/tetragonal
        structures. Takes precedence over ``c_over_a``. Required for
        ``graphite`` (no sensible interlayer default).

    Returns
    -------
    System
        A :class:`System` with the per-atom DataFrame populated with
        ``x, y, z, element`` and an orthogonal-or-triclinic simulation
        box matching atomsk's convention for the chosen structure.

    Examples
    --------
    Single-species cubic — copper FCC:

    >>> cu = build_crystal("Cu", "fcc", a=3.615, nx=5, ny=5, nz=5)

    Two-species cubic — NaCl:

    >>> nacl = build_crystal(("Na", "Cl"), "rocksalt", a=5.64, nx=4, ny=4, nz=4)

    Three-species perovskite — note B-A-O ordering:

    >>> sto = build_crystal(("Ti", "Sr", "O"), "perovskite", a=3.905)

    Hexagonal — wurtzite GaN with explicit ``c``:

    >>> gan = build_crystal(("Ga", "N"), "wurtzite", a=3.19, c=5.18)

    Cubic Miller-rotated — FCC ``[111]`` slab orientation:

    >>> cu_111 = build_crystal(
    ...     "Cu", "fcc", a=3.615,
    ...     miller1=(1, -1, 0), miller2=(1, 1, -2), miller3=(1, 1, 1),
    ... )

    Hexagonal Miller-Bravais — Mg prismatic ``(11-20)`` plane:

    >>> mg = build_crystal(
    ...     "Mg", "hcp", a=3.21, c=5.21,
    ...     miller1=(1, -1, 0, 0), miller2=(1, 1, -2, 0), miller3=(0, 0, 0, 1),
    ... )

    Notes
    -----
    - Output is **bit-for-bit equivalent to atomsk**: for every supported
      structure, ``tests/_generate_fixtures/generate_build_crystal.py``
      generates an atomsk reference and ``tests/test_build_crystal.py``
      asserts the box, sorted positions and sorted element list match.
    - Hexagonal structures (``hcp``, ``wurtzite``, ``graphite``) emit a
      2-axis 120° primitive cell (``H1=[2-1-10]``, ``H2=[-12-10]``,
      ``H3=[0001]``); the resulting LAMMPS box has a negative ``xy``
      tilt of ``-a/2``.
    - Multi-species structures attach the ``element`` column directly
      from the structure's basis ordering — there is no random-alloy
      placement here. Use :func:`build_hea` if you want random
      multi-element substitution on a single sublattice.
    """
    s = _normalize_structure_name(structure)
    if s not in _STRUCTURES:
        raise ValueError(
            f"Unsupported structure '{structure}'. Supported: "
            f"{sorted(set(_STRUCTURES))}"
        )
    build_fn, allowed_n_species, c_default_factor = _STRUCTURES[s]

    # --- Normalize the `name` argument ---
    # Accept either a single element string (broadcast to every basis atom)
    # or a tuple/list of element strings whose length must equal one of the
    # structure's allowed species counts.
    if isinstance(name, str):
        name_tuple = (name,)
    else:
        name_tuple = tuple(name)
        for ele in name_tuple:
            if not isinstance(ele, str):
                raise TypeError(
                    f"`name` entries must be element symbols (str); got {type(ele).__name__}."
                )
    if len(name_tuple) not in allowed_n_species and len(name_tuple) != 1:
        raise ValueError(
            f"`name` must be a single element symbol or a tuple of length "
            f"{allowed_n_species} for structure '{s}'; got length "
            f"{len(name_tuple)}."
        )

    # --- Resolve `c` for hexagonal/tetragonal structures ---
    # Precedence: explicit `c` > explicit `c_over_a` (computed against `a`)
    # > the structure's default factor.
    if c is None and c_default_factor is not None:
        # If the caller did not override c_over_a from the function
        # default, fall back to the structure's canonical factor.
        if c_over_a == np.sqrt(8 / 3):
            c = a * float(c_default_factor)
        else:
            c = a * float(c_over_a)
    if c is None and s == "graphite":
        # No sensible default for inter-layer spacing; user must set it.
        raise ValueError("`graphite` requires an explicit `c` parameter.")

    # --- Build the (possibly Miller-rotated) unit cell + species index ---
    if miller1 is None and miller2 is None and miller3 is None:
        # Standard orientation
        if build_fn == "legacy_orth":
            old_box, old_pos = _get_basispos_and_box_cubic(structure, a, c_over_a)
            species_idx = np.zeros(len(old_pos), dtype=np.int32)
        else:
            old_box, old_pos, species_idx = build_fn(a, c)
    else:
        if s not in _MILLER_SUPPORTED:
            raise ValueError(
                f"Miller orientation is not yet supported for structure "
                f"'{s}'."
            )
        if s in _MILLER_HEX:
            # Hexagonal: accept 3-index [uvw] or 4-index [hkil].
            for label, m in (("miller1", miller1), ("miller2", miller2),
                             ("miller3", miller3)):
                if len(m) not in (3, 4):
                    raise ValueError(
                        f"Hexagonal Miller indices must be 3-index [uvw] "
                        f"or 4-index [hkil]. Got {label}={tuple(m)}."
                    )
            old_box, old_pos, species_idx = _build_lattice_from_miller_hex(
                s, miller1, miller2, miller3, a, c,
            )
        else:
            if (len(miller1) != 3 or len(miller2) != 3 or len(miller3) != 3):
                raise ValueError(
                    f"Cubic structures require 3-index Miller indices [h,k,l]. "
                    f"Got miller1={miller1}, miller2={miller2}, miller3={miller3}"
                )
            old_box, old_pos, species_idx = _build_lattice_from_miller(
                s, miller1, miller2, miller3, a,
            )
        # Carry species through the minimal-cell reduction.
        old_box, old_pos, species_idx = _find_minimal_cell_with_species(
            old_box, old_pos, species_idx
        )

    # --- Replicate ---
    old_pos = old_pos @ old_box
    n_old = old_pos.shape[0]
    total = n_old * nx * ny * nz * 3
    new_pos = np.zeros(total, dtype=np.float64)
    _repeat_cell.repeat_cell(new_pos, old_box, old_pos, nx, ny, nz)
    new_pos = new_pos.reshape((-1, 3))
    new_box = old_box * np.array([nx, ny, nz]).reshape((3, 1))

    # --- Attach element column ---
    species_full = np.tile(species_idx, nx * ny * nz)
    if len(name_tuple) == 1:
        # Broadcast: all atoms share the same element.
        elements = np.full(species_full.shape[0], name_tuple[0], dtype=object)
    else:
        elements = np.array(name_tuple, dtype=object)[species_full]
    data = pl.from_numpy(new_pos, schema=["x", "y", "z"]).with_columns(
        pl.lit(elements).alias("element")
    )
    return System(data=data, box=Box(new_box))


def _find_minimal_cell_with_species(
    box: np.ndarray, basis: np.ndarray, species: np.ndarray,
    tolerance: float = 1e-6, max_search: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Brute-force smallest periodic sub-cell, species-aware.

    A proposed (nx, ny, nz) sub-cell is valid only if every replicated
    image carries the **same species** as the corresponding atom in the
    original cell — otherwise the reduction would silently merge two
    distinct site classes. Box must be axis-aligned (orthogonal); a
    triclinic box is returned unchanged."""
    off_diag = [box[0, 1], box[0, 2], box[1, 0], box[1, 2], box[2, 0], box[2, 1]]
    if any(abs(x) > tolerance for x in off_diag):
        return box, basis, species
    a_, b_, c_ = box[0, 0], box[1, 1], box[2, 2]
    n_atoms = len(basis)
    if n_atoms == 0:
        return box, basis, species
    basis_w = basis - np.floor(basis + tolerance)

    best_box = box.copy()
    best_basis = basis_w.copy()
    best_species = species.copy()
    min_atoms = n_atoms

    for nx in range(1, max_search + 1):
        for ny in range(1, max_search + 1):
            for nz in range(1, max_search + 1):
                if nx == ny == nz == 1:
                    continue
                n_div = nx * ny * nz
                if n_atoms % n_div != 0:
                    continue
                expected = n_atoms // n_div
                if expected >= min_atoms:
                    continue
                test_box = np.array(
                    [[a_ / nx, 0, 0], [0, b_ / ny, 0], [0, 0, c_ / nz]]
                )

                # Atoms inside the proposed first sub-cell (with their
                # species). If two distinct species share the same site,
                # the sub-cell is invalid → skip.
                small_atoms = []
                small_species = []
                for atom, sp in zip(basis_w, species):
                    if (atom[0] >= -tolerance and atom[0] < 1.0 / nx - tolerance and
                            atom[1] >= -tolerance and atom[1] < 1.0 / ny - tolerance and
                            atom[2] >= -tolerance and atom[2] < 1.0 / nz - tolerance):
                        frac = atom * np.array([nx, ny, nz])
                        frac = frac - np.floor(frac + tolerance)
                        is_dup = False
                        for k, ex in enumerate(small_atoms):
                            d = frac - ex
                            d = d - np.round(d)
                            if np.linalg.norm(d) < tolerance:
                                if small_species[k] != sp:
                                    small_atoms = None  # signal invalid
                                    break
                                is_dup = True
                                break
                        if small_atoms is None:
                            break
                        if not is_dup:
                            small_atoms.append(frac)
                            small_species.append(int(sp))
                    if small_atoms is None:
                        break
                if small_atoms is None:
                    continue
                if len(small_atoms) != expected:
                    continue

                # Replicate and confirm every position+species matches.
                replicated_pos = []
                replicated_sp = []
                for ix in range(nx):
                    for iy in range(ny):
                        for iz in range(nz):
                            for atom, sp in zip(small_atoms, small_species):
                                frac = (atom + np.array([ix, iy, iz])) / np.array(
                                    [nx, ny, nz]
                                )
                                frac = frac - np.floor(frac + tolerance)
                                replicated_pos.append(frac)
                                replicated_sp.append(sp)
                if len(replicated_pos) != n_atoms:
                    continue

                matched = [False] * n_atoms
                ok = True
                for rep, rsp in zip(replicated_pos, replicated_sp):
                    found = False
                    for idx, orig in enumerate(basis_w):
                        if matched[idx]:
                            continue
                        d = rep - orig
                        d = d - np.round(d)
                        if np.linalg.norm(d) < tolerance:
                            if int(species[idx]) != rsp:
                                ok = False
                                break
                            matched[idx] = True
                            found = True
                            break
                    if not ok or not found:
                        ok = False
                        break
                if ok and all(matched) and expected < min_atoms:
                    min_atoms = expected
                    best_box = test_box.copy()
                    best_basis = np.array(small_atoms)
                    best_species = np.array(small_species, dtype=np.int32)

    return best_box, best_basis, best_species


def build_hea(
    element_list: Tuple[str],
    element_ratio: Tuple[float],
    structure: str,
    a: float,
    miller1: Optional[Tuple[int, int, int]] = None,
    miller2: Optional[Tuple[int, int, int]] = None,
    miller3: Optional[Tuple[int, int, int]] = None,
    nx: int = 1,
    ny: int = 1,
    nz: int = 1,
    c_over_a: float = np.sqrt(8 / 3),
    random_seed: Optional[int] = None,
) -> System:
    """
    Build a high-entropy alloy (HEA) crystal structure with an optional orientation
    defined by Miller indices.

    Parameters
    ----------
    element_list : Tuple[str]
        List of element symbols, e.g., ``('Cu', 'Al', 'Mg')``.
    element_ratio : Tuple[float]
        Corresponding atomic ratios. The sum must equal 1.
    structure : str
        Crystal structure type. Supported values are:

        - ``'fcc'`` — Face-centered cubic
        - ``'bcc'`` — Body-centered cubic
        - ``'hcp'`` — Hexagonal close-packed

    a : float
        Lattice constant in Ångströms.
        - For cubic structures (FCC/BCC): edge length of the cubic unit cell.
        - For HCP: basal plane lattice parameter.

    miller1, miller2, miller3 : tuple of int, optional
        Miller indices ``(h, k, l)`` defining the orientation of x-, y-, and z-axes.
        Only applicable to cubic structures.
        If any are ``None``, the standard orthogonal orientation ([1,0,0], [0,1,0], [0,0,1]) is used.

        .. note::
           The three Miller indices must be orthogonal and follow the right-hand rule:
           ``miller1 × miller2 = miller3``.

    nx, ny, nz : int, default=1
        Number of unit cell repetitions along x, y, and z directions.
    c_over_a : float, default=sqrt(8/3)
        The ``c/a`` ratio for the HCP structure.
        The default corresponds to the ideal HCP packing.
        Ignored for other structures.
    random_seed : int, optional
        Random seed for reproducible element assignment.

    Returns
    -------
    System
        A :class:`System` object containing the generated crystal structure
        with atomic positions and simulation box information.

    Examples
    --------
    >>> system = build_hea(
    ...     ("Cr", "Co", "Ni"),
    ...     (0.1, 0.2, 0.7),
    ...     "fcc",
    ...     3.526,
    ...     nx=5,
    ...     ny=5,
    ...     nz=5,
    ...     random_seed=1,
    ... )
    """

    assert structure.lower() in {"fcc", "bcc", "hcp"}, (
        f"Unsupported structure '{structure}'. Choose from 'fcc', 'bcc', or 'hcp'."
    )
    if miller1 is not None and miller2 is not None and miller3 is not None:
        assert structure.lower() != "hcp", "hcp does not support miller orientations."

    # --- Build base crystal ---
    system = build_crystal(
        "X", structure, a, miller1, miller2, miller3, nx, ny, nz, c_over_a
    )

    return build_hea_fromsystem(system, element_list, element_ratio, random_seed)


def build_hea_fromsystem(
    system: System,
    element_list: Tuple[str],
    element_ratio: Tuple[float],
    random_seed: Optional[int] = None,
) -> System:
    """Generate element per atom for a system,

    Parameters
    ----------
    system : System
        System need to be modified.
    element_list : Tuple[str]
        List of element symbols, e.g., ``('Cu', 'Al', 'Mg')``.
    element_ratio : Tuple[float]
        Corresponding atomic ratios. The sum must equal 1.
    random_seed : int, optional
        Random seed for reproducible element assignment.

    Returns
    -------
    System
        A :class:`System` object containing the generated element information.
    """
    # --- Input validation ---
    assert len(element_list) > 1, "At least two elements are required to form an HEA."
    assert len(set(element_list)) == len(element_list), (
        "Each element in element_list must be unique."
    )
    assert len(element_list) == len(element_ratio), (
        "element_list and element_ratio must have the same length."
    )
    assert abs(np.sum(element_ratio) - 1.0) < 1e-6, (
        f"Element ratios must sum to 1 (got {np.sum(element_ratio):.6f})."
    )
    # --- Assign elements by ratio ---
    type_counts = np.floor(system.N * np.array(element_ratio)).astype(int)
    for i in range(len(element_ratio)):
        if type_counts[i] == 0 and element_ratio[i] > 1e-6:
            type_counts[i] += 1
    type_counts[-1] = system.N - type_counts[:-1].sum()

    element_array = np.repeat(element_list, type_counts)
    if random_seed is not None:
        np.random.seed(int(random_seed))
    np.random.shuffle(element_array)

    system.update_data(system.data.with_columns(element=element_array))
    return system


def build_partial_dislocation_fcc(
    name: str,
    a: float,
    nx: int,
    ny: int,
    nz: int,
    element_list: Optional[Tuple[str]] = None,
    element_ratio: Optional[Tuple[float]] = None,
    random_seed: Optional[int] = None,
) -> System:
    """Generate a FCC strcuture with a pair partial dislocations along y axis.
    The crystalline orientation is: x->[1-10], y->[11-2], z->[111]. If `element_list` and
    `element_ratio` are given, it will generate a HEA.
    Similar to Construct a dislocation by superimposing two crystals in atomsk: https://atomsk.univ-lille.fr/tutorial_Al_edge.php

    Parameters
    ----------
    name : str
        Element symbol (e.g., 'Cu', 'Al', 'Mg', 'C').
    a : float
        Lattice constant in Angstroms.
    nx : int, default=1
        Number of repetitions along x-axis direction.
    ny : int, default=1
        Number of repetitions along y-axis direction.
    nz : int, default=1
        Number of repetitions along z-axis direction.
    element_list : Tuple[str], optional
        List of element symbols, e.g., ``('Cu', 'Al', 'Mg')``.
    element_ratio : Tuple[float], optional
        Corresponding atomic ratios. The sum must equal 1.
    random_seed : int, optional
        Random seed for reproducible element assignment.

    Returns
    -------
    System
        A :class:`System` object containing the dislocations.
    """
    upper = build_crystal(
        name,
        "fcc",
        a,
        miller1=[1, -1, 0],
        miller2=[1, 1, -2],
        miller3=[1, 1, 1],
        nx=nx,
        ny=ny,
        nz=nz,
    )
    lower = build_crystal(
        name,
        "fcc",
        a,
        miller1=[1, -1, 0],
        miller2=[1, 1, -2],
        miller3=[1, 1, 1],
        nx=nx + 1,
        ny=ny,
        nz=nz,
    )
    box = upper.box.box.copy()
    box[0, 0] = box[0, 0] * (1 + (0.5 / nx))

    upper.update_box(box, scale_pos=True)
    box = lower.box.box.copy()
    box[0, 0] = box[0, 0] * (1 - (0.5 / (nx + 1)))

    lower.update_box(box, scale_pos=True)
    data = pl.concat(
        [upper.data.with_columns(pl.col("z") + lower.box.box[2, 2]), lower.data]
    )
    box = lower.box.box.copy()
    box[2, 2] += upper.box.box[2, 2]
    system = System(data=data, box=box)
    if element_list is not None and element_ratio is not None:
        system = build_hea_fromsystem(system, element_list, element_ratio, random_seed)
    return system


if __name__ == "__main__":
    fcc = build_hea(
        ["Cr", "Co", "Fe", "Ni"],
        [0, 0, 0, 1],
        "fcc",
        3.53,
        nx=3,
        ny=3,
        nz=3,
        random_seed=1,
    )
    print(fcc)
