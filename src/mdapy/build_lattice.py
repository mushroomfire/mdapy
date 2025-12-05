# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
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
    """Get basis positions and box for cubic/HCP/graphene structures (internal helper)."""
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
    transform_matrix: np.ndarray, basis_positions: np.ndarray
) -> np.ndarray:
    """Find all atoms in the new unit cell (internal helper)."""
    M = transform_matrix.astype(float)
    M_inv = np.linalg.inv(M)

    # Calculate search range
    det = abs(np.linalg.det(M))
    expected_atoms = int(np.round(det * len(basis_positions)))

    # Search range determined by matrix elements
    max_coef = int(np.max(np.abs(M)))
    search_range = max_coef + 1

    new_basis_list = []

    # Iterate through periodic replicas of original lattice
    for i in range(-search_range, search_range + 1):
        for j in range(-search_range, search_range + 1):
            for k in range(-search_range, search_range + 1):
                for basis_atom in basis_positions:
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

    new_basis = np.array(new_basis_list)

    # Verify atom count
    if len(new_basis) != expected_atoms:
        print(f"Warning: Expected {expected_atoms} atoms but found {len(new_basis)}")

    return new_basis


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


def _find_minimal_cell(
    box: np.ndarray, basis: np.ndarray, tolerance: float = 1e-6, max_search: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """Find the minimal periodic cell using brute force search (internal helper)."""
    # Verify box alignment
    off_diag = [box[0, 1], box[0, 2], box[1, 0], box[1, 2], box[2, 0], box[2, 1]]
    if any(abs(x) > tolerance for x in off_diag):
        raise ValueError("Box must be aligned to axes")

    a, b, c = box[0, 0], box[1, 1], box[2, 2]
    n_atoms = len(basis)

    if n_atoms == 0:
        return box, basis

    # Wrap basis to [0, 1)
    basis_wrapped = basis - np.floor(basis + tolerance)

    best_box = box.copy()
    best_basis = basis_wrapped.copy()
    min_atoms = n_atoms

    # Brute force: try ALL divisor combinations
    for nx in range(1, max_search + 1):
        for ny in range(1, max_search + 1):
            for nz in range(1, max_search + 1):
                if nx == 1 and ny == 1 and nz == 1:
                    continue

                n_div = nx * ny * nz
                if n_atoms % n_div != 0:
                    continue  # Must divide evenly

                expected_atoms = n_atoms // n_div
                if expected_atoms >= min_atoms:
                    continue  # Not better

                # Test this division
                test_box = np.array([[a / nx, 0, 0], [0, b / ny, 0], [0, 0, c / nz]])

                # Find atoms in first subcell [0, 1/nx) × [0, 1/ny) × [0, 1/nz)
                small_atoms = []
                for atom in basis_wrapped:
                    if (
                        atom[0] >= -tolerance
                        and atom[0] < 1.0 / nx - tolerance
                        and atom[1] >= -tolerance
                        and atom[1] < 1.0 / ny - tolerance
                        and atom[2] >= -tolerance
                        and atom[2] < 1.0 / nz - tolerance
                    ):
                        # Convert to small cell fractional coords
                        frac_small = atom * np.array([nx, ny, nz])
                        frac_small = frac_small - np.floor(frac_small + tolerance)

                        # Check duplicates
                        is_dup = False
                        for existing in small_atoms:
                            diff = frac_small - existing
                            diff = diff - np.round(diff)
                            if np.linalg.norm(diff) < tolerance:
                                is_dup = True
                                break
                        if not is_dup:
                            small_atoms.append(frac_small)

                if len(small_atoms) != expected_atoms:
                    continue

                # Verify: replicate and check
                replicated = []
                for i in range(nx):
                    for j in range(ny):
                        for k in range(nz):
                            for atom in small_atoms:
                                frac_orig = (atom + np.array([i, j, k])) / np.array(
                                    [nx, ny, nz]
                                )
                                frac_orig = frac_orig - np.floor(frac_orig + tolerance)
                                replicated.append(frac_orig)

                if len(replicated) != n_atoms:
                    continue

                # Match all atoms
                matched = [False] * n_atoms
                for rep in replicated:
                    for idx, orig in enumerate(basis_wrapped):
                        if matched[idx]:
                            continue
                        diff = rep - orig
                        diff = diff - np.round(diff)
                        if np.linalg.norm(diff) < tolerance:
                            matched[idx] = True
                            break

                if all(matched):
                    # Valid!
                    if expected_atoms < min_atoms:
                        min_atoms = expected_atoms
                        best_box = test_box.copy()
                        best_basis = np.array(small_atoms)

    return best_box, best_basis


def _build_lattice_from_miller(
    structure: str,
    miller1: Tuple[int, int, int],
    miller2: Tuple[int, int, int],
    miller3: Tuple[int, int, int],
    lattice_constant: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build lattice from Miller indices (internal helper)."""
    # Reduce Miller indices
    miller1 = _reduce_miller(miller1)
    miller2 = _reduce_miller(miller2)
    miller3 = _reduce_miller(miller3)

    # Check orthogonality and right-hand rule
    if not _check_orthogonality_miller(miller1, miller2, miller3):
        raise ValueError(
            f"Miller indices must be orthogonal. Got {miller1}, {miller2}, {miller3}"
        )

    if not _check_right_hand_miller(miller1, miller2, miller3):
        raise ValueError("Miller indices must satisfy right-hand rule")

    # Get original lattice (cubic)
    box, basis = _get_basispos_and_box_cubic(structure, lattice_constant)

    # Build transformation matrix
    M = _build_transform_matrix(miller1, miller2, miller3)

    # Calculate new lattice vectors
    new_lattice = box @ M.T

    # Find atoms in new cell
    new_basis = _find_atoms_in_new_cell(M, basis)

    # Align box to axes for rectangular shape
    aligned_lattice, aligned_basis = _align_box_to_axes(new_lattice, new_basis)

    return aligned_lattice, aligned_basis


def build_crystal(
    name: str,
    structure: str,
    a: float,
    miller1: Optional[Tuple[int, int, int]] = None,
    miller2: Optional[Tuple[int, int, int]] = None,
    miller3: Optional[Tuple[int, int, int]] = None,
    nx: int = 1,
    ny: int = 1,
    nz: int = 1,
    c_over_a: float = np.sqrt(8 / 3),
) -> System:
    """
    Build crystal structure with optional Miller indices orientation.

    This function creates a crystal structure with specified lattice type and orientation.
    It supports standard crystallographic structures (FCC, BCC, HCP, diamond, graphene) and
    allows custom orientations via Miller indices for cubic structures.

    Parameters
    ----------
    name : str
        Element symbol (e.g., 'Cu', 'Al', 'Mg', 'C').
    structure : str
        Crystal structure type. Supported values:

        - ``'fcc'``: Face-centered cubic
        - ``'bcc'``: Body-centered cubic
        - ``'diamond'``: Diamond cubic
        - ``'hcp'``: Hexagonal close-packed
        - ``'graphene'``: Graphene layer

    a : float
        Lattice constant in Angstroms.

        - For cubic structures (FCC/BCC/diamond): edge length of cubic unit cell
        - For HCP: basal plane lattice parameter
        - For graphene: C-C bond length

    miller1 : tuple of int, optional
        First Miller index (h, k, l) defining the x-axis direction.
        Only applicable for cubic structures. If None, uses standard orientation [1,0,0].
    miller2 : tuple of int, optional
        Second Miller index (h, k, l) defining the y-axis direction.
        Only applicable for cubic structures. If None, uses standard orientation [0,1,0].
    miller3 : tuple of int, optional
        Third Miller index (h, k, l) defining the z-axis direction.
        Only applicable for cubic structures. If None, uses standard orientation [0,0,1].

        .. note::
           The three Miller indices must be mutually orthogonal and satisfy the
           right-hand rule: miller1 × miller2 = miller3.

    nx : int, default=1
        Number of repetitions along x-axis direction.
    ny : int, default=1
        Number of repetitions along y-axis direction.
    nz : int, default=1
        Number of repetitions along z-axis direction.
    c_over_a : float, default=sqrt(8/3)
        Ratio of c to a for HCP structure. Default value gives ideal HCP packing.
        Ignored for other structures.

    Returns
    -------
    System
        A System object containing the built crystal structure with atomic positions
        and simulation box.

    Raises
    ------
    ValueError
        - If an unsupported structure type is specified
        - If Miller indices are provided for non-cubic structures
        - If Miller indices are not mutually orthogonal
        - If Miller indices don't satisfy the right-hand rule

    Examples
    --------
    Create a standard FCC copper crystal:

    >>> cu = build_crystal("Cu", "fcc", a=3.615, nx=5, ny=5, nz=5)

    Create an FCC crystal with custom Miller orientation:

    >>> cu_rotated = build_crystal(
    ...     "Cu",
    ...     "fcc",
    ...     a=3.615,
    ...     miller1=(1, -1, 0),
    ...     miller2=(1, 1, -2),
    ...     miller3=(1, 1, 1),
    ...     nx=5,
    ...     ny=5,
    ...     nz=5,
    ... )

    Create an HCP magnesium crystal:

    >>> mg = build_crystal("Mg", "hcp", a=3.21, c_over_a=1.624, nx=10, ny=10, nz=10)

    Create a graphene sheet:

    >>> graphene = build_crystal("C", "graphene", a=1.42, nx=20, ny=20, nz=1)

    Notes
    -----
    - For cubic structures with Miller indices, the function automatically finds
      the minimal periodic cell and aligns it to coordinate axes.
    - The resulting structure has an orthogonal simulation box aligned with
      the Cartesian coordinate system.
    - Atomic positions are generated by replicating the unit cell nx×ny×nz times.

    """
    if miller1 is None and miller2 is None and miller3 is None:
        # Standard orientation
        old_box, old_pos = _get_basispos_and_box_cubic(structure, a, c_over_a)
    else:
        # Custom Miller orientation
        if structure.lower() in ["fcc", "bcc", "diamond"]:
            # Cubic: 3-index Miller
            if len(miller1) != 3 or len(miller2) != 3 or len(miller3) != 3:
                raise ValueError(
                    f"Cubic structures require 3-index Miller indices [h,k,l]. "
                    f"Got miller1={miller1}, miller2={miller2}, miller3={miller3}"
                )
            old_box, old_pos = _build_lattice_from_miller(
                structure, miller1, miller2, miller3, a
            )
        else:
            raise ValueError(
                f"Miller indices are only supported for cubic structures (fcc, bcc, diamond). "
                f"Got structure='{structure}'"
            )

        old_box, old_pos = _find_minimal_cell(old_box, old_pos)

    old_pos = old_pos @ old_box
    n_old = old_pos.shape[0]
    total = n_old * nx * ny * nz * 3
    new_pos = np.zeros(total, dtype=np.float64)
    _repeat_cell.repeat_cell(new_pos, old_box, old_pos, nx, ny, nz)
    new_pos = new_pos.reshape((-1, 3))
    new_box = old_box * np.array([nx, ny, nz]).reshape((3, 1))
    data = pl.from_numpy(new_pos, schema=["x", "y", "z"]).with_columns(
        pl.lit(name).alias("element")
    )
    return System(data=data, box=Box(new_box))


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
        if type_counts[i] == 0 and element_ratio[i] > 0:
            type_counts[i] += 1

    type_counts[-1] = max(system.N - type_counts[:-1].sum(), 1)  # ensure total equals N

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
    system = build_partial_dislocation_fcc(
        "X", 3.526, 100, 60, 13, ["Cr", "Co", "Fe", "Ni"], [1 / 3, 1 / 3, 0, 1 / 3]
    )
    print("Fe" in system.data["element"])
    system.write_data("test.data", ["Cr", "Co", "Fe", "Ni"])
