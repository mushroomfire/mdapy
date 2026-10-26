# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The displacements of a straight dislocation in an anisotropic crystal,
by the sextic formalism of Stroh."""

import numpy as np
from numpy.typing import ArrayLike

from mdapy.errors import InvalidParameter

# the Voigt index of each pair of Cartesian indices
_VOIGT = np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]])

# roots of the sextic equation closer than this are taken as equal: the
# medium is then isotropic about the line, as far as it matters
_DEGENERATE = 1e-4


def stiffness(elastic: ArrayLike) -> np.ndarray:
    """The (6, 6) Voigt stiffness from (C11, C12, C44) of a cubic crystal or
    a (6, 6) matrix, checked to be symmetric and positive definite."""
    c = np.asarray(elastic, dtype=float)
    if c.shape == (3,):
        c11, c12, c44 = c
        c = np.zeros((6, 6))
        c[:3, :3] = c12
        c[np.arange(3), np.arange(3)] = c11
        c[np.arange(3, 6), np.arange(3, 6)] = c44
    elif c.shape != (6, 6):
        raise InvalidParameter(
            "elastic must be (C11, C12, C44) or a (6, 6) matrix, "
            f"not of shape {c.shape}"
        )
    if not np.isfinite(c).all() or not np.allclose(c, c.T, rtol=1e-6, atol=1e-9):
        raise InvalidParameter("elastic must be a finite symmetric matrix")
    if np.linalg.eigvalsh(c).min() <= 0:
        raise InvalidParameter("elastic must be positive definite")
    return 0.5 * (c + c.T)


def rotated(voigt: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """The (3, 3, 3, 3) stiffness tensor in the frame whose axes are the
    rows of rotation, from the Voigt stiffness in the crystal frame."""
    tensor = voigt[_VOIGT[:, :, None, None], _VOIGT[None, None, :, :]]
    r = rotation
    return np.einsum("ia,jb,kc,ld,abcd->ijkl", r, r, r, r, tensor)


class StraightDislocation:
    """The displacement field of straight dislocations along z in a medium
    of stiffness tensor (3, 3, 3, 3) in this frame."""

    def __init__(self, tensor: np.ndarray):
        q = tensor[:, 0, :, 0]
        r = tensor[:, 0, :, 1]
        t = tensor[:, 1, :, 1]
        t_inv = np.linalg.inv(t)
        n = np.block([[-t_inv @ r.T, t_inv], [r @ t_inv @ r.T - q, -r @ t_inv]])
        p, xi = np.linalg.eig(n)
        upper = np.argsort(-p.imag)[:3]  # the roots of positive imaginary part
        self.p = p[upper]
        a, b = xi[:3, upper], xi[3:, upper]
        gaps = np.abs(self.p[:, None] - self.p[None, :]) + np.eye(3)
        self.isotropic = bool(gaps.min() < _DEGENERATE)
        if self.isotropic:
            # Poisson's ratio of the Voigt average
            trace = np.einsum("iijj->", tensor)
            shear = (np.einsum("ijij->", tensor) - trace / 3) / 10
            bulk = trace / 9
            self.nu = (3 * bulk - 2 * shear) / (2 * (3 * bulk + shear))
        else:
            # B^T A + A^T B = I
            scale = np.sqrt(2 * np.einsum("ia,ia->a", a, b))
            self.a, self.b = a / scale, b / scale

    def __call__(
        self, x1: np.ndarray, x2: np.ndarray, burgers: np.ndarray, cut: float = np.pi
    ) -> np.ndarray:
        """(N, 3) displacements at points (x1, x2) from the line of a
        dislocation of Burgers vector (3,) in this frame, cut along the
        half-line from it at angle cut from x1, from -pi to pi: crossed
        counterclockwise about the line, the displacement drops by the
        Burgers vector."""
        theta = np.arctan2(x2, x1)
        beyond = theta > cut  # the sector from the cut to the principal one
        if self.isotropic:
            nu = self.nu
            angle = theta - 2 * np.pi * beyond
            r2 = x1 * x1 + x2 * x2
            log = (1 - 2 * nu) / (4 * (1 - nu)) * np.log(r2)
            cross = x1 * x2 / (2 * (1 - nu) * r2)
            squares = (x1 * x1 - x2 * x2) / (4 * (1 - nu) * r2)
            b1, b2, b3 = burgers
            u = np.empty((len(x1), 3))
            u[:, 0] = b1 * (angle + cross) + b2 * (log - squares)
            u[:, 1] = -b1 * (log + squares) + b2 * (angle - cross)
            u[:, 2] = b3 * angle
            return u / (2 * np.pi)
        z = x1[:, None] + self.p[None, :] * x2[:, None]
        logs = np.log(z)
        logs.imag -= 2 * np.pi * beyond[:, None]
        weights = self.b.T @ np.asarray(burgers, dtype=float)
        return ((logs * weights) @ self.a.T).imag / np.pi
