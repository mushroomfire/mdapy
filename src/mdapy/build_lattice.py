# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

from mdapy import _repeat_cell
from mdapy.system import System
from mdapy.box import Box
from mdapy.parallel import get_num_threads
import polars as pl
import numpy as np
from typing import Optional, Tuple

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


def _basis_fcc(a, c=None):
    """FCC / A1: 4 atoms / cell, atomsk basis order (P(1)=corner, then
    three face centres). Two-species form places species 1 on two of
    the four sublattices, matching atomsk's two-species FCC."""
    box = a * np.eye(3)
    basis = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
        ]
    )
    species = np.array([0, 0, 1, 1], dtype=np.int32)
    return box, basis, species


def _basis_bcc(a, c=None):
    """BCC / A2: 2 atoms / cell. Two-species variant is the same as the
    CsCl/B2 ordering."""
    box = a * np.eye(3)
    basis = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    species = np.array([0, 1], dtype=np.int32)
    return box, basis, species


def _basis_diamond(a, c=None):
    """Cubic diamond (A4) in atomsk basis order: 4 fcc sites first,
    then 4 tetrahedral sites. Two-species variant is the zincblende
    (B3) structure with the same Cartesian positions but distinct
    species indices on the two interpenetrating fcc sublattices."""
    box = a * np.eye(3)
    basis = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75],
            [0.25, 0.75, 0.75],
        ]
    )
    species = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
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
    basis = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.5, 0.5],
        ]
    )
    species = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
    return box, basis, species


def _basis_zincblende(a, c=None):
    """B3: diamond with two distinct species. species 1 occupies the 4 fcc
    sites, species 2 the 4 tetrahedral sites."""
    box = a * np.eye(3)
    basis = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75],
            [0.25, 0.75, 0.75],
        ]
    )
    species = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
    return box, basis, species


def _basis_fluorite(a, c=None):
    """Fluorite (CaF2): 4 cation FCC sites + 8 anion sites at the corners
    of two interior cubes (species ratio 1:2)."""
    box = a * np.eye(3)
    basis = np.array(
        [
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
        ]
    )
    species = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32)
    return box, basis, species


def _basis_l1_2(a, c=None):
    """L1_2 (Ni3Al / Cu3Au): species 0 occupies the 3 face centers,
    species 1 occupies the corner."""
    box = a * np.eye(3)
    basis = np.array(
        [
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.0, 0.0, 0.0],
        ]
    )
    species = np.array([0, 0, 0, 1], dtype=np.int32)
    return box, basis, species


def _basis_perovskite(a, c=None):
    """Cubic perovskite ABO3. atomsk's ordering is **B, A, O**:
    species 0 at body centre, species 1 at corner, species 2 at face
    centres."""
    box = a * np.eye(3)
    basis = np.array(
        [
            [0.5, 0.5, 0.5],  # B
            [0.0, 0.0, 0.0],  # A
            [0.5, 0.0, 0.0],  # O
            [0.0, 0.5, 0.0],  # O
            [0.0, 0.0, 0.5],  # O
        ]
    )
    species = np.array([0, 1, 2, 2, 2], dtype=np.int32)
    return box, basis, species


def _hexagonal_box(a, c):
    """atomsk-compatible hexagonal primitive cell with 120° angle."""
    return np.array(
        [
            [a, 0.0, 0.0],
            [-0.5 * a, 0.5 * _SQRT3 * a, 0.0],
            [0.0, 0.0, c],
        ]
    )


def _basis_hcp(a, c):
    """atomsk-compatible 2-atom hexagonal primitive cell. The legacy
    orthogonal 4-atom supercell that mdapy used to emit is *not*
    selected via this entry — it is preserved only for the
    ``"legacy_orth"`` dispatch tag and accessible via
    ``_get_basispos_and_box_cubic`` for callers that depend on the old
    layout."""
    box = _hexagonal_box(a, c)
    basis = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0 / 3.0, 2.0 / 3.0, 0.5],
        ]
    )
    species = np.array([0, 1], dtype=np.int32)
    return box, basis, species


def _basis_wurtzite(a, c):
    """Wurtzite B4 (e.g. GaN): 4-atom hexagonal primitive."""
    box = _hexagonal_box(a, c)
    basis = np.array(
        [
            [1.0 / 3.0, 2.0 / 3.0, 0.0],
            [2.0 / 3.0, 1.0 / 3.0, 0.5],
            [1.0 / 3.0, 2.0 / 3.0, 3.0 / 8.0],
            [2.0 / 3.0, 1.0 / 3.0, 7.0 / 8.0],
        ]
    )
    species = np.array([0, 0, 1, 1], dtype=np.int32)
    return box, basis, species


def _basis_graphene(a, c):
    """Single-layer hexagonal honeycomb. Same primitive cell as graphite
    but with only the first ``z = 0`` layer (2 atoms / cell), and
    ``c`` chosen as a vacuum spacing so periodic images do not interact
    along z. Two-species form gives a hex-BN-style monolayer."""
    box = _hexagonal_box(a, c)
    basis = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0 / 3.0, 2.0 / 3.0, 0.0],
        ]
    )
    species = np.array([0, 1], dtype=np.int32)
    return box, basis, species


def _basis_graphite(a, c):
    """Hexagonal graphite (A9): two layers along ``c`` with AB stacking,
    4 atoms per primitive cell."""
    box = _hexagonal_box(a, c)
    basis = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5],
            [1.0 / 3.0, 2.0 / 3.0, 0.0],
            [2.0 / 3.0, 1.0 / 3.0, 0.5],
        ]
    )
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
    "sc": (_basis_sc, (1,), None),
    "fcc": (_basis_fcc, (1, 2), None),
    "bcc": (_basis_bcc, (1, 2), None),
    "diamond": (_basis_diamond, (1, 2), None),
    "cscl": (_basis_bcc_2sp, (2,), None),
    "rocksalt": (_basis_rocksalt, (2,), None),
    "zincblende": (_basis_zincblende, (2,), None),
    "fluorite": (_basis_fluorite, (2,), None),
    "l1_2": (_basis_l1_2, (2,), None),
    "perovskite": (_basis_perovskite, (3,), None),
    # HEXAGONAL — atomsk-compatible 120° primitive cell.
    "hcp": (_basis_hcp, (1, 2), np.sqrt(8 / 3)),
    "wurtzite": (_basis_wurtzite, (1, 2), np.sqrt(8 / 3)),
    "graphite": (_basis_graphite, (1, 2), None),  # c must be given
    "graphene": (_basis_graphene, (1, 2), None),  # c must be given
    # Hexagonal diamond (lonsdaleite) — alias for single-species wurtzite.
    "lonsdaleite": (_basis_wurtzite, (1,), np.sqrt(8 / 3)),
}

# Cubic structures supporting Miller-indexed orientation.
_MILLER_CUBIC = {
    "sc",
    "fcc",
    "bcc",
    "diamond",
    "cscl",
    "rocksalt",
    "zincblende",
    "fluorite",
    "l1_2",
    "perovskite",
}
# Hexagonal structures supporting Miller-Bravais [hkil] (or 3-index [uvw])
# orientation.
_MILLER_HEX = {"hcp", "wurtzite", "graphite", "graphene", "lonsdaleite"}
_MILLER_SUPPORTED = _MILLER_CUBIC | _MILLER_HEX


# Common synonyms — resolved to the canonical key used by
# ``_STRUCTURES``. Add new aliases here, not in the dispatch table itself.
_STRUCTURE_ALIASES = {
    "rs": "rocksalt",
    "nacl": "rocksalt",
    "b1": "rocksalt",
    "zb": "zincblende",
    "b3": "zincblende",
    "wz": "wurtzite",
    "b4": "wurtzite",
    "a9": "graphite",
    "b2": "cscl",
    "l12": "l1_2",
    "hex_diamond": "lonsdaleite",
    "hexagonal_diamond": "lonsdaleite",
    "diamond_hex": "lonsdaleite",
}


def _normalize_structure_name(structure: str) -> str:
    s = structure.lower().strip()
    return _STRUCTURE_ALIASES.get(s, s)


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


def _to_lower_triangular_box(new_lattice: np.ndarray) -> np.ndarray:
    """Rotate ``new_lattice`` (rows = lattice vectors) into atomsk's
    standard lower-triangular form: ``v1`` along ``+x``, ``v2`` in the
    ``+xy`` half-plane, ``v3`` with non-negative ``z``. Preserves all
    lengths and angles, so fractional coordinates in the original cell
    are also valid fractional coordinates of the returned cell.

    Equivalent to ``CONVMAT(|v1|,|v2|,|v3|, α, β, γ, H)`` in atomsk
    (called after orienting hexagonal cells in ``mode_create.f90``).
    For a right-handed input this is a proper rotation; for a
    left-handed input it is an improper rotation (a reflection composed
    with a rotation) — the returned box is always right-handed."""
    v1, v2, v3 = new_lattice[0], new_lattice[1], new_lattice[2]
    a = float(np.linalg.norm(v1))
    b = float(np.linalg.norm(v2))
    c = float(np.linalg.norm(v3))
    if a < 1e-12 or b < 1e-12 or c < 1e-12:
        raise ValueError("Lattice vectors must be non-zero.")
    cos_gamma = float(np.dot(v1, v2) / (a * b))
    cos_beta = float(np.dot(v3, v1) / (c * a))
    cos_alpha = float(np.dot(v2, v3) / (b * c))
    sin_gamma = float(np.sqrt(max(0.0, 1.0 - cos_gamma * cos_gamma)))
    if sin_gamma < 1e-12:
        raise ValueError("First two lattice vectors are colinear.")
    aligned = np.zeros((3, 3))
    aligned[0, 0] = a
    aligned[1, 0] = b * cos_gamma
    aligned[1, 1] = b * sin_gamma
    aligned[2, 0] = c * cos_beta
    aligned[2, 1] = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    z_sq = c * c - aligned[2, 0] ** 2 - aligned[2, 1] ** 2
    if z_sq < -1e-9:
        raise ValueError("Lattice angles are geometrically inconsistent.")
    aligned[2, 2] = float(np.sqrt(max(0.0, z_sq)))
    # Snap near-zero off-diagonals so downstream orthogonality checks
    # (e.g. _find_minimal_cell_with_species) treat genuinely orthogonal
    # cells as orthogonal.
    aligned[np.abs(aligned) < 1e-12] = 0.0
    return aligned


def _build_lattice_from_miller_hex(
    structure: str,
    miller1,
    miller2,
    miller3,
    a: float,
    c: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hexagonal counterpart of :func:`_build_lattice_from_miller`.

    Each Miller direction is given in 3-index ``[uvw]`` or 4-index
    ``[hkil]`` notation; the latter is converted via atomsk's
    ``HKIL2UVW`` rule.

    Matches atomsk's hexagonal-orient branch in ``mode_create.f90``:
    **neither orthogonality nor the right-hand rule is enforced** —
    only linear independence (non-zero triple product). When the chosen
    Miller frame is non-orthogonal, a triclinic cell in lower-triangular
    form is returned, identical to what atomsk's ``CONVMAT`` produces.
    """
    u1, v1, w1 = _hkil_to_uvw(miller1)
    u2, v2, w2 = _hkil_to_uvw(miller2)
    u3, v3, w3 = _hkil_to_uvw(miller3)

    s = _normalize_structure_name(structure)
    build_fn, _, _ = _STRUCTURES[s]
    box, basis, species = build_fn(a, c)
    # Build M with new vectors as COLUMNS, mirroring _build_transform_matrix.
    M = np.array(
        [
            [u1, u2, u3],
            [v1, v2, v3],
            [w1, w2, w3],
        ],
        dtype=int,
    )
    # Cartesian new lattice vectors. Row k = sum_j M[j, k] * box[j, :].
    new_lattice = M.T @ box

    # Linear-independence guard (atomsk's SCALAR_TRIPLE_PRODUCT check).
    triple = float(
        np.dot(np.cross(new_lattice[0], new_lattice[1]), new_lattice[2])
    )
    if abs(triple) < 1e-9:
        raise ValueError(
            "Hexagonal Miller directions must be linearly independent "
            f"(got {tuple(miller1)}, {tuple(miller2)}, {tuple(miller3)})."
        )

    new_basis, new_species = _find_atoms_in_new_cell(M, basis, species)
    aligned_lattice = _to_lower_triangular_box(new_lattice)
    return aligned_lattice, new_basis, new_species


def _build_lattice_from_miller(
    structure: str,
    miller1: Tuple[int, int, int],
    miller2: Tuple[int, int, int],
    miller3: Tuple[int, int, int],
    lattice_constant: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cubic Miller-indexed cell builder. Returns ``(box, basis_frac,
    species_idx)`` for the rotated rectangular cell.

    Matches atomsk's ``--create ... orient`` semantics: orthogonality is
    required, but **the right-hand rule is not** — atomsk accepts
    left-handed Miller frames and the alignment step is equivalent to a
    mirror reflection, which is a lattice symmetry of every supported
    cubic structure (FCC, BCC, diamond, etc.) so the output is still a
    valid crystal."""
    miller1 = _reduce_miller(miller1)
    miller2 = _reduce_miller(miller2)
    miller3 = _reduce_miller(miller3)

    if not _check_orthogonality_miller(miller1, miller2, miller3):
        raise ValueError(
            f"Miller indices must be orthogonal. Got {miller1}, {miller2}, {miller3}"
        )

    build_fn, _, _ = _STRUCTURES[_normalize_structure_name(structure)]
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
        for an ordered multi-species structure (NaCl, GaN, SrTiO3,
        ...). Each element must be a recognised symbol from
        ``mdapy.data.atomic_numbers``.

    structure : str
        Crystal type. Case-insensitive; common synonyms accepted
        (``"nacl"`` → ``"rocksalt"``, ``"zb"`` → ``"zincblende"``,
        ``"b2"`` → ``"cscl"``, ``"hex_diamond"`` → ``"lonsdaleite"``,
        ...). Supported structures and their allowed species counts:

        Cubic (Miller-rotatable, 3-index ``[h, k, l]``):

        - ``sc`` (1): simple cubic, e.g. α-Po
        - ``fcc`` (1 or 2): Cu, Al, Ni; or ordered FCC
        - ``bcc`` (1 or 2): Fe, W; B2 alloy with two species
        - ``diamond`` (1 or 2): C, Si; ``zincblende`` for two species
        - ``cscl`` / B2 (2): CsCl, NiAl
        - ``rocksalt`` / B1 (2): NaCl, MgO
        - ``zincblende`` / B3 (2): GaAs, ZnS
        - ``fluorite`` (2): CaF2, UO2
        - ``l1_2`` (2): Ni3Al, Cu3Au
        - ``perovskite`` (3, **B-A-O ordering**): SrTiO3 → ``("Ti","Sr","O")``

        Hexagonal (Miller-rotatable, 3-index ``[u, v, w]`` or 4-index
        ``[h, k, i, l]``):

        - ``hcp`` (1 or 2): Mg, Ti, Zr
        - ``wurtzite`` / B4 (1 or 2): GaN, ZnO
        - ``lonsdaleite`` / hex-diamond (1): hexagonal carbon polytype
        - ``graphite`` (1 or 2): graphite, hex-BN — ``c`` required
        - ``graphene`` (1 or 2): single-layer honeycomb — ``c`` required

    a : float
        Primary lattice constant in Å (cubic edge, hexagonal in-plane
        spacing).

    miller1, miller2, miller3 : sequence of int, optional
        Crystallographic axes for the three sides of the supercell.
        For cubic structures pass three integers ``[h, k, l]``; for
        hexagonal structures pass either Miller-Bravais
        ``[h, k, i, l]`` (with ``h + k + i == 0``) or 3-index
        ``[u, v, w]`` directly. When all three are ``None``, the
        canonical orientation is used.

        Cubic Miller indices must be mutually orthogonal in Cartesian
        space; left-handed frames are accepted and yield a mirror-image
        crystal (a lattice symmetry for every supported cubic
        structure). Hexagonal Miller indices only need to be linearly
        independent — non-orthogonal frames are returned as triclinic
        boxes in lower-triangular form. This matches atomsk's
        ``--create ... orient`` semantics.

    nx, ny, nz : int, default=1
        Replication counts along the (possibly Miller-rotated) axes.
        Total atom count = ``len(basis) * nx * ny * nz``.

    c : float, optional
        Out-of-plane lattice constant in Å for hexagonal structures.
        For ``hcp``, ``wurtzite`` and ``lonsdaleite`` defaults to
        ``a * sqrt(8/3)`` (ideal close-packing). For ``graphite`` and
        ``graphene`` it must be supplied — no sensible default exists
        (interlayer / vacuum spacing).

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

    Three-species perovskite (B, A, O ordering):

    >>> sto = build_crystal(("Ti", "Sr", "O"), "perovskite", a=3.905)

    Hexagonal — wurtzite GaN with explicit ``c``:

    >>> gan = build_crystal(("Ga", "N"), "wurtzite", a=3.19, c=5.18)

    Hexagonal diamond (lonsdaleite) — single-species wurtzite:

    >>> lons = build_crystal("C", "lonsdaleite", a=2.51, c=4.12)

    Single-layer graphene (vacuum-padded along c):

    >>> g = build_crystal("C", "graphene", a=2.46, c=20.0)

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
    # If the structure has a canonical c/a factor, fall back to it when
    # `c` is not supplied; otherwise the user must provide `c`.
    if c is None and c_default_factor is not None:
        c = a * float(c_default_factor)
    if c is None and s in ("graphite", "graphene"):
        raise ValueError(f"`{s}` requires an explicit `c` parameter.")

    # --- Build the (possibly Miller-rotated) unit cell + species index ---
    if miller1 is None and miller2 is None and miller3 is None:
        # Standard orientation
        old_box, old_pos, species_idx = build_fn(a, c)
        # Single-name broadcast → every basis atom is physically the
        # same element, so collapse the species index to zero. This is
        # what lets the elements column come out uniform AND lets the
        # Miller-path minimal-cell reduction below treat the cell as
        # single-species when the user is.
        if len(name_tuple) == 1:
            species_idx = np.zeros(len(species_idx), dtype=np.int32)
    else:
        if s not in _MILLER_SUPPORTED:
            raise ValueError(
                f"Miller orientation is not yet supported for structure " f"'{s}'."
            )
        if s in _MILLER_HEX:
            # Hexagonal: accept 3-index [uvw] or 4-index [hkil].
            for label, m in (
                ("miller1", miller1),
                ("miller2", miller2),
                ("miller3", miller3),
            ):
                if len(m) not in (3, 4):
                    raise ValueError(
                        f"Hexagonal Miller indices must be 3-index [uvw] "
                        f"or 4-index [hkil]. Got {label}={tuple(m)}."
                    )
            old_box, old_pos, species_idx = _build_lattice_from_miller_hex(
                s,
                miller1,
                miller2,
                miller3,
                a,
                c,
            )
        else:
            if len(miller1) != 3 or len(miller2) != 3 or len(miller3) != 3:
                raise ValueError(
                    f"Cubic structures require 3-index Miller indices [h,k,l]. "
                    f"Got miller1={miller1}, miller2={miller2}, miller3={miller3}"
                )
            old_box, old_pos, species_idx = _build_lattice_from_miller(
                s,
                miller1,
                miller2,
                miller3,
                a,
            )
        # Single-name broadcast → all atoms physically the same; treat
        # the cell as single-species so the minimal-cell reduction can
        # shrink to atomsk's primitive (matters for fcc/bcc/diamond
        # whose dispatch entries return an ordered-alloy basis).
        if len(name_tuple) == 1:
            species_idx = np.zeros(len(species_idx), dtype=np.int32)
        # Carry species through the minimal-cell reduction.
        old_box, old_pos, species_idx = _find_minimal_cell_with_species(
            old_box, old_pos, species_idx
        )

    # --- Replicate ---
    old_pos = old_pos @ old_box
    n_old = old_pos.shape[0]
    total = n_old * nx * ny * nz * 3
    new_pos = np.zeros(total, dtype=np.float64)
    _repeat_cell.repeat_cell(new_pos, old_box, old_pos, nx, ny, nz, get_num_threads())
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
    box: np.ndarray,
    basis: np.ndarray,
    species: np.ndarray,
    tolerance: float = 1e-6,
    max_search: int = 10,
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
                test_box = np.array([[a_ / nx, 0, 0], [0, b_ / ny, 0], [0, 0, c_ / nz]])

                # Atoms inside the proposed first sub-cell (with their
                # species). If two distinct species share the same site,
                # the sub-cell is invalid → skip.
                small_atoms = []
                small_species = []
                for atom, sp in zip(basis_w, species):
                    if (
                        atom[0] >= -tolerance
                        and atom[0] < 1.0 / nx - tolerance
                        and atom[1] >= -tolerance
                        and atom[1] < 1.0 / ny - tolerance
                        and atom[2] >= -tolerance
                        and atom[2] < 1.0 / nz - tolerance
                    ):
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
    c: Optional[float] = None,
    random_seed: Optional[int] = None,
) -> System:
    """
    Build a random high-entropy alloy (HEA) on a single sublattice.

    Parameters
    ----------
    element_list : tuple of str
        Element symbols, e.g. ``('Cu', 'Al', 'Mg')``. Must be unique.
    element_ratio : tuple of float
        Corresponding atomic ratios; must sum to 1.
    structure : str
        Single-species crystal type — typically ``'fcc'``, ``'bcc'`` or
        ``'hcp'``. Any structure from :func:`build_crystal` whose
        allowed species count includes 1 will work; only one sublattice
        is randomly substituted.
    a : float
        Primary lattice constant in Å.
    miller1, miller2, miller3 : sequence of int, optional
        Crystallographic axes (3-index for cubic, 3- or 4-index for
        hexagonal). See :func:`build_crystal` for the conventions.
    nx, ny, nz : int, default=1
        Replication counts along each box axis.
    c : float, optional
        Out-of-plane lattice constant for hexagonal structures.
    random_seed : int, optional
        Seed for the per-atom element assignment.

    Returns
    -------
    System
        Crystal with the ``element`` column randomly populated by
        ``element_list`` according to ``element_ratio``.

    Examples
    --------
    >>> sys_ = build_hea(
    ...     ("Cr", "Co", "Ni"), (0.1, 0.2, 0.7),
    ...     "fcc", 3.526, nx=5, ny=5, nz=5, random_seed=1,
    ... )
    """
    system = build_crystal(
        "X",
        structure,
        a,
        miller1,
        miller2,
        miller3,
        nx,
        ny,
        nz,
        c=c,
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
    assert len(set(element_list)) == len(
        element_list
    ), "Each element in element_list must be unique."
    assert len(element_list) == len(
        element_ratio
    ), "element_list and element_ratio must have the same length."
    assert (
        abs(np.sum(element_ratio) - 1.0) < 1e-6
    ), f"Element ratios must sum to 1 (got {np.sum(element_ratio):.6f})."
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
