# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Choosing frames for training sets: principal components, farthest point
sampling, descriptors of whole structures and perturbed copies."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from mdapy.core.box import Box
from mdapy.core.result import Result
from mdapy.core.system import System
from mdapy.errors import InvalidParameter


@dataclass(frozen=True)
class PrincipalComponents(Result):
    """The principal components of points.

    Attributes
    ----------
    projection : numpy.ndarray
        (M, n) coordinates of the points along the components.
    components : numpy.ndarray
        (n, D) the components, unit vectors, by decreasing variance.
    variance : numpy.ndarray
        (n,) the variance along each component.
    ratio : numpy.ndarray
        (n,) the fraction of the total variance of each component.
    mean : numpy.ndarray
        (D,) the mean of the points.
    """

    projection: np.ndarray
    components: np.ndarray
    variance: np.ndarray
    ratio: np.ndarray
    mean: np.ndarray

    def transform(self, points: ArrayLike) -> np.ndarray:
        """(K, n) coordinates of other points along the components."""
        return (np.asarray(points, dtype=float) - self.mean) @ self.components.T


def principal_components(points: ArrayLike, n: int = 2) -> PrincipalComponents:
    """The principal components of points, such as descriptors.

    Parameters
    ----------
    points : array_like
        (M, D) points.
    n : int, optional
        Number of components kept, at most D. Default 2.

    Returns
    -------
    PrincipalComponents
        The projection of the points, the components and their variances;
        each component points so that its largest entry is positive.
    """
    x = np.asarray(points, dtype=float)
    if x.ndim != 2 or len(x) < 2:
        raise InvalidParameter("points must be (M, D) with two points or more")
    if not 1 <= n <= x.shape[1]:
        raise InvalidParameter(f"n must be from 1 to {x.shape[1]}, not {n!r}")
    mean = x.mean(axis=0)
    centered = x - mean
    variance, vectors = np.linalg.eigh(
        np.cov(centered, rowvar=False).reshape(x.shape[1], x.shape[1])
    )
    order = np.argsort(variance)[::-1]
    variance, vectors = variance[order], vectors[:, order]
    components = vectors[:, :n].T.copy()
    largest = np.argmax(np.abs(components), axis=1)
    components *= np.sign(components[np.arange(n), largest])[:, None]
    total = variance.sum()
    return PrincipalComponents(
        projection=centered @ components.T,
        components=components,
        variance=variance[:n].copy(),
        ratio=variance[:n] / total if total > 0 else np.zeros(n),
        mean=mean,
    )


def farthest_point_sampling(
    points: ArrayLike,
    count: int,
    *,
    start: int | None = 0,
    selected: ArrayLike | None = None,
) -> np.ndarray:
    """Points chosen one by one, each the farthest from those chosen before,
    such as the frames of a training set that cover the space of
    descriptors.

    Parameters
    ----------
    points : array_like
        (M, D) the candidates.
    count : int
        Number of points chosen.
    start : int or None, optional
        The first point chosen; None to start from the farthest from
        ``selected``. Default 0.
    selected : array_like, optional
        (K, D) points already chosen, such as the training set so far; the
        candidates are chosen far from them as well.

    Returns
    -------
    numpy.ndarray
        (count,) int64 indices of the points chosen, in order.
    """
    x = np.asarray(points, dtype=float)
    if x.ndim != 2:
        raise InvalidParameter("points must be (M, D)")
    m = len(x)
    if (
        isinstance(count, bool)
        or not isinstance(count, int | np.integer)
        or not 0 <= count <= m
    ):
        raise InvalidParameter(f"count must be an integer from 0 to {m}, not {count!r}")
    nearest = np.full(m, np.inf)
    if selected is not None:
        old = np.asarray(selected, dtype=float).reshape(-1, x.shape[1])
        for point in old:
            nearest = np.minimum(nearest, np.linalg.norm(x - point, axis=1))
    chosen: list[int] = []
    if count == 0:
        return np.zeros(0, dtype=np.int64)
    if start is None:
        if selected is None:
            raise InvalidParameter("start may be None only with selected points")
        first = int(np.argmax(nearest))
    else:
        if not 0 <= start < m:
            raise InvalidParameter(f"start must be from 0 to {m - 1}, not {start!r}")
        first = int(start)
    chosen.append(first)
    nearest = np.minimum(nearest, np.linalg.norm(x - x[first], axis=1))
    for _ in range(count - 1):
        following = int(np.argmax(nearest))
        chosen.append(following)
        nearest = np.minimum(nearest, np.linalg.norm(x - x[following], axis=1))
    return np.array(chosen, dtype=np.int64)


def structure_descriptors(frames: Sequence[System], calculator: Any) -> np.ndarray:
    """The descriptor of each frame: the mean of the descriptors of its
    atoms.

    Parameters
    ----------
    frames : sequence of System
        The frames.
    calculator : NEP
        A calculator with ``descriptors(system)``, such as
        :class:`mdapy.calculators.NEP`.

    Returns
    -------
    numpy.ndarray
        (F, D) one descriptor per frame.
    """
    return np.array(
        [np.asarray(calculator.descriptors(frame)).mean(axis=0) for frame in frames]
    )


def perturb(
    system: System,
    count: int,
    *,
    displacement: float = 0.05,
    strain: float = 0.03,
    style: str = "normal",
    seed: int | np.random.Generator | None = None,
) -> list[System]:
    """Copies of a system with its atoms displaced and its box strained at
    random, such as for a first training set.

    Parameters
    ----------
    system : System
        The system.
    count : int
        Number of copies.
    displacement : float, optional
        Size of the displacement of each atom in Å. Default 0.05.
    strain : float, optional
        Largest component of the symmetric strain of the box, applied with
        the atoms. Default 0.03.
    style : {"normal", "uniform", "const"}, optional
        Displacements with a root mean square of displacement; uniform in a
        ball of radius displacement; or exactly displacement long. Default
        normal.
    seed : int or numpy.random.Generator, optional
        Seed of the random numbers.

    Returns
    -------
    list of System
        The copies, without the energies and forces of the system.
    """
    if isinstance(count, bool) or not isinstance(count, int | np.integer) or count < 0:
        raise InvalidParameter(f"count must be a non-negative integer, not {count!r}")
    if not displacement >= 0 or not strain >= 0:
        raise InvalidParameter("displacement and strain must not be negative")
    if style not in ("normal", "uniform", "const"):
        raise InvalidParameter(f"style must be normal, uniform or const, not {style!r}")
    rng = np.random.default_rng(seed)
    n = len(system)
    box = system.box
    keep = [c for c in system.table.columns if c not in ("forces", "velocities")]
    base = system.with_table(system.table.select(keep)).with_info(
        energy=None, virial=None, stress=None
    )
    copies = []
    for _ in range(int(count)):
        e = rng.uniform(-strain, strain, size=(3, 3))
        deformation = np.eye(3) + 0.5 * (e + e.T)
        direction = rng.normal(size=(n, 3))
        if style == "normal":
            step = direction * (displacement / np.sqrt(3))
        else:
            unit = direction / np.linalg.norm(direction, axis=1, keepdims=True)
            length = (
                displacement * rng.random(n) ** (1 / 3)
                if style == "uniform"
                else np.full(n, displacement)
            )
            step = unit * length[:, None]
        positions = (
            (system["positions"] - box.origin) @ deformation.T + box.origin + step
        )
        strained = Box(box.matrix @ deformation.T, origin=box.origin, pbc=box.pbc)
        copies.append(base.with_box(strained).with_columns(positions=positions))
    return copies
