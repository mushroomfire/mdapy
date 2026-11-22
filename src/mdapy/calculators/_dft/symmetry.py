# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The symmetry of a crystal for the plane-wave DFT. An operation takes the
fractional coordinates x to R x + t (R integer, rows of the cell as its
vectors); it takes atom I to atom J of the same kind (and first moment). By
them:
  - the k-points of a mesh reduce to the irreducible ones, k equivalent to R^T
    k and to -k, each weighted by the points it stands for;
  - the density is the average of rho(R x + t), on the sphere rho(m') =
    1/N sum rho(R^-T m') exp(2 pi i (R^-T m') . t);
  - the occupations of the projectors of atom J are the average of M B_I M^T,
    M the rotation of the real harmonics, Y(C^-1 v) = M^T Y(v) with C the
    operation on Cartesian vectors;
  - the forces on J the average of C F_I, the stress of C s C^T."""

from dataclasses import dataclass

import numpy as np

from mdapy.calculators._dft import angular


@dataclass(frozen=True)
class Symmetry:
    """The operations of a crystal.

    Attributes
    ----------
    rotations : numpy.ndarray
        (K, 3, 3) int32 R on fractional coordinates.
    translations : numpy.ndarray
        (K, 3) float64 t, fractions of the cell vectors.
    cartesian : numpy.ndarray
        (K, 3, 3) float64 C on Cartesian vectors.
    atoms : numpy.ndarray
        (K, N) int32 the atom J that each operation takes atom I to.
    """

    rotations: np.ndarray
    translations: np.ndarray
    cartesian: np.ndarray
    atoms: np.ndarray

    def __len__(self) -> int:
        return len(self.rotations)


def available() -> bool:
    """Whether spglib, which finds the operations, is installed."""
    try:
        import spglib  # noqa: F401
    except ImportError:
        return False
    return True


def operations(
    cell, fractions, kinds, magmoms=None, tolerance: float = 1e-5
) -> Symmetry:
    """The operations of a cell of rows cell (bohr) with atoms of kinds at
    fractions, those of first moments magmoms keeping them; tolerance in bohr,
    as spglib's symprec."""
    import spglib

    cell = np.asarray(cell, dtype=float)
    fractions = np.asarray(fractions, dtype=float)
    labels = np.asarray(kinds, dtype=np.int64)
    if magmoms is not None:
        pairs = [(int(k), round(float(m), 8)) for k, m in zip(labels, magmoms)]
        index = {p: i for i, p in enumerate(dict.fromkeys(pairs))}
        labels = np.array([index[p] for p in pairs], dtype=np.int64)
    found = spglib.get_symmetry((cell, fractions, labels), symprec=tolerance)
    if found is None:
        rotations, translations = np.eye(3, dtype=np.int32)[None], np.zeros((1, 3))
    else:
        rotations = np.asarray(found["rotations"], dtype=np.int32)
        translations = np.asarray(found["translations"], dtype=float)
    # x_cart = A^T x with A the rows of the cell
    lattice = cell.T
    cartesian = lattice @ rotations @ np.linalg.inv(lattice)
    n = len(fractions)
    atoms = np.zeros((len(rotations), n), dtype=np.int32)
    for k, (r, t) in enumerate(zip(rotations, translations)):
        images = fractions @ r.T + t
        apart = images[:, None, :] - fractions[None, :, :]
        off = np.linalg.norm((apart - np.round(apart)) @ cell, axis=2)
        atoms[k] = np.argmin(off, axis=1)
        if not (off[np.arange(n), atoms[k]] < 10 * tolerance).all():
            raise ValueError("an operation of spglib takes an atom to none")
        if not (labels[atoms[k]] == labels).all():
            raise ValueError("an operation of spglib takes an atom to another kind")
    return Symmetry(rotations, translations, cartesian, atoms)


def identity(atoms: int) -> Symmetry:
    """The identity alone, for a cell taken without symmetry."""
    return Symmetry(
        np.eye(3, dtype=np.int32)[None], np.zeros((1, 3)), np.eye(3)[None],
        np.arange(atoms, dtype=np.int32)[None],
    )  # fmt: skip


def irreducible(
    kpoints: np.ndarray, rotations: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The irreducible k-points (fractions of the reciprocal vectors) of a
    full mesh kpoints (K, 3), each the first of its star under R^T and -1
    met in the mesh, and their weights adding up to one."""
    kpoints = np.asarray(kpoints, dtype=float)

    def key(k):
        wrapped = k - np.floor(k + 1e-9)
        return tuple(np.round(wrapped * 1e6).astype(np.int64))

    where = {}
    for i, k in enumerate(kpoints):
        where.setdefault(key(k), i)
    owner = np.full(len(kpoints), -1)
    for i, k in enumerate(kpoints):
        if owner[i] >= 0:
            continue
        owner[i] = i
        for r in rotations:
            for sign in (1.0, -1.0):
                j = where.get(key(sign * (r.T @ k)))
                if j is not None and owner[j] < 0:
                    owner[j] = i
    first = np.unique(owner)
    weights = np.array([(owner == i).sum() for i in first], dtype=float)
    return kpoints[first], weights / weights.sum()


def harmonics(cartesian: np.ndarray, lmax: int = 3) -> list[list[np.ndarray]]:
    """For each operation C, the matrices M of l = 0 ... lmax with Y_l(C^-1 v)
    = M^T Y_l(v), Y_l the column of the real harmonics of l (angular's)."""
    rng = np.random.default_rng(0)
    v = rng.normal(size=(64, 3))
    y = angular.real_harmonics(lmax, v).T  # (64, (lmax + 1)^2)
    out = []
    for c in cartesian:
        rotated = angular.real_harmonics(lmax, v @ np.linalg.inv(c).T).T
        blocks = []
        for ell in range(lmax + 1):
            part = slice(ell * ell, (ell + 1) * (ell + 1))
            m, *_ = np.linalg.lstsq(y[:, part], rotated[:, part], rcond=None)
            blocks.append(np.ascontiguousarray(m))
        out.append(blocks)
    return out


def forces(symmetry: Symmetry, f: np.ndarray) -> np.ndarray:
    """(N, 3) the forces made symmetric: F_J the average of C F_I over the
    operations that take I to J."""
    f = np.asarray(f, dtype=float)
    out = np.zeros_like(f)
    for c, atoms in zip(symmetry.cartesian, symmetry.atoms):
        out[atoms] += f @ c.T
    return out / len(symmetry)


def tensor(symmetry: Symmetry, s: np.ndarray) -> np.ndarray:
    """(3, 3) a tensor made symmetric, the average of C s C^T."""
    s = np.asarray(s, dtype=float)
    return sum(c @ s @ c.T for c in symmetry.cartesian) / len(symmetry)


def payload(symmetry: Symmetry, lmax: int = 3) -> dict:
    """The operations for the GPU: rotations (K, 9) and their inverse
    transposes R^-T (K, 9) as integers, translations (K, 3), atoms (K, N)
    and the matrices of the harmonics (K, l, (2l+1)^2) flat."""
    inverse_t = np.array(
        [np.round(np.linalg.inv(r).T).astype(np.int32) for r in symmetry.rotations]
    )
    blocks = harmonics(symmetry.cartesian, lmax)
    flat = np.array([np.concatenate([b.ravel() for b in op]) for op in blocks])
    return {
        "rotations": np.ascontiguousarray(symmetry.rotations.reshape(-1, 9)),
        "inverse_transposes": np.ascontiguousarray(inverse_t.reshape(-1, 9)),
        "translations": np.ascontiguousarray(symmetry.translations),
        "atoms": np.ascontiguousarray(symmetry.atoms),
        "harmonics": np.ascontiguousarray(flat),
        "lmax": lmax,
    }
