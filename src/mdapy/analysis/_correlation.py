# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The spatial correlation function of two properties of the atoms."""

from dataclasses import dataclass

import numpy as np

from mdapy.core.result import Result
from mdapy.core.system import System
from mdapy.errors import InvalidParameter
from mdapy.neighbor import NeighborList

# below this a length counts as zero, as in OVITO
_EPSILON = 1e-12


@dataclass(frozen=True)
class SpatialCorrelationResult(Result):
    """The spatial correlation function of two properties.

    Attributes
    ----------
    r : numpy.ndarray
        (bins,) centers of the distance bins in Å, up to half the shortest
        distance between opposite faces of the box.
    correlation : numpy.ndarray
        (bins,) the correlation function in each bin, from the grid.
    rdf : numpy.ndarray
        (bins,) the radial distribution function of the same grid.
    q : numpy.ndarray
        (wavevectors,) centers of the wavevector bins in 1/Å.
    reciprocal : numpy.ndarray
        (wavevectors,) the correlation function in reciprocal space.
    means : numpy.ndarray
        (2,) the mean of each property over the atoms.
    mean_squares : numpy.ndarray
        (2,) the mean of the square of each property.
    mean_product : float
        The mean over the atoms of the product of the two properties.
    neighbor_r : numpy.ndarray or None
        (neighbor bins,) centers of the bins of the direct summation in Å,
        when computed.
    neighbor_correlation : numpy.ndarray or None
        (neighbor bins,) the correlation function summed over the pairs of
        neighbors, when computed.
    neighbor_rdf : numpy.ndarray or None
        (neighbor bins,) the radial distribution function of the pairs of
        neighbors, when computed.
    """

    r: np.ndarray
    correlation: np.ndarray
    rdf: np.ndarray
    q: np.ndarray
    reciprocal: np.ndarray
    means: np.ndarray
    mean_squares: np.ndarray
    mean_product: float
    neighbor_r: np.ndarray | None = None
    neighbor_correlation: np.ndarray | None = None
    neighbor_rdf: np.ndarray | None = None


def _grid(fractional, values, n, pbc, window):
    """The sum of the values in the cells of the grid that the atoms fall
    in, weighted by a Hann window along the directions that are not
    periodic; the atoms of a NaN value left out."""
    keep = ~np.isnan(values)
    fractional, values = fractional[keep], values[keep]
    # the cell by truncation toward zero, then into the grid along the
    # periodic directions; atoms outside it along the others are left out
    cell = np.trunc(fractional * n).astype(np.int64)
    weight = np.ones(len(values))
    for e in range(3):
        if pbc[e]:
            cell[:, e] %= n[e]
        elif window:
            weight *= np.sqrt(2.0 / 3.0) * (
                1.0 - np.cos(2.0 * np.pi * fractional[:, e])
            )
    inside = ((cell >= 0) & (cell < n)).all(axis=1)
    flat = np.ravel_multi_index(tuple(cell[inside].T), tuple(n))
    total = np.bincount(flat, (weight * values)[inside], minlength=int(np.prod(n)))
    return total.reshape(n)


def _signed(n):
    """The indices of a grid of n points as the signed ones nearest zero."""
    return (np.arange(n) + n // 2) % n - n // 2


class SpatialCorrelation:
    """The spatial correlation function of two properties of the atoms,
    the mean of the first at one place times the second at a distance r
    from it, by fast Fourier transforms on a grid, as OVITO's spatial
    correlation function.

    Parameters
    ----------
    first : str
        A column of one value per atom, a component such as
        "velocities.x", or an expression over the columns.
    second : str, optional
        The other property; default the first, its autocorrelation.
    grid_spacing : float, optional
        The approximate size in Å of the cells of the grid. Default 3.
    window : bool, optional
        Weight the atoms by a Hann window along the directions of the box
        that are not periodic. Default True.
    direct : bool, optional
        Also sum the correlation over the pairs of neighbors closer than
        cutoff. Default False.
    cutoff : float, optional
        The distance in Å of the direct summation. Default 5.
    bins : int, optional
        The number of bins of the direct summation. Default 50.
    """

    def __init__(
        self,
        first: str,
        second: str | None = None,
        *,
        grid_spacing: float = 3.0,
        window: bool = True,
        direct: bool = False,
        cutoff: float = 5.0,
        bins: int = 50,
    ):
        if not grid_spacing > 0:
            raise InvalidParameter(
                f"grid_spacing must be positive, not {grid_spacing!r}"
            )
        if not cutoff > 0:
            raise InvalidParameter(f"cutoff must be positive, not {cutoff!r}")
        if isinstance(bins, bool) or int(bins) != bins or bins < 1:
            raise InvalidParameter(f"bins must be a positive integer, not {bins!r}")
        self.first = str(first)
        self.second = self.first if second is None else str(second)
        self.grid_spacing = float(grid_spacing)
        self.window = bool(window)
        self.direct = bool(direct)
        self.cutoff = float(cutoff)
        self.bins = int(bins)

    def __values(self, system: System, name: str) -> np.ndarray:
        values = np.asarray(system[name], dtype=np.float64)
        if values.shape != (len(system),):
            raise InvalidParameter(
                f"{name} must be one value per atom; choose a component, such as {name}.x"
            )
        return values

    def compute(self, system: System) -> SpatialCorrelationResult:
        """The correlation function of a system.

        Parameters
        ----------
        system : System
            The atoms, in a box of some volume.

        Returns
        -------
        SpatialCorrelationResult
            The correlation in real and reciprocal space and the moments of
            the two properties.

        Raises
        ------
        InvalidParameter
            For a property that is not one number per atom, no atoms, or a
            grid spacing larger than a quarter of the box.
        """
        n_atoms = len(system)
        if n_atoms == 0:
            raise InvalidParameter("the system has no atoms")
        first = self.__values(system, self.first)
        second = self.__values(system, self.second)
        box = system.box
        matrix = box.matrix
        lengths = np.linalg.norm(matrix, axis=1)
        n = np.maximum(1, (lengths / self.grid_spacing).astype(np.int64))
        # the distances between opposite faces of the box
        faces = 1.0 / np.linalg.norm(box.inverse, axis=0)
        shortest = faces.min()
        distance_bins = int(shortest / (2 * self.grid_spacing))
        wavevector_bins = int(shortest / (2 * self.grid_spacing))
        if distance_bins == 0:
            raise InvalidParameter("grid_spacing is too large for this box")
        fractional = box.to_fractional(system["positions"])
        pbc = box.pbc
        transforms = [
            np.fft.fftn(_grid(fractional, v, n, pbc, self.window))
            for v in (first, second, np.ones(n_atoms))
        ]
        product = transforms[0] * np.conj(transforms[1])
        density = np.abs(transforms[2]) ** 2
        signed = np.stack(np.meshgrid(*(_signed(k) for k in n), indexing="ij"), axis=-1)
        origin = (signed == 0).all(axis=-1)
        # reciprocal space: the wavevectors of the grid in bins of the
        # shortest reciprocal vector of the box, without 2 pi
        step = 1.0 / shortest
        wavevectors = np.linalg.norm(signed @ box.inverse.T, axis=-1)
        index = np.floor(wavevectors / step).astype(np.int64)
        used = ~origin & (index < wavevector_bins)
        counts = np.bincount(index[used], minlength=wavevector_bins)
        sums = np.bincount(index[used], product.real[used], minlength=wavevector_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            reciprocal = np.where(
                counts > 0, sums * box.volume / (n_atoms * n_atoms) / counts, 0.0
            )
        # real space: the transforms back, unnormalized, in bins of the
        # distances of the grid points from the origin
        size = int(np.prod(n))
        real = np.fft.ifftn(product).real * size
        pairs = np.fft.ifftn(density).real * size
        width = shortest / (2 * distance_bins)
        distances = np.linalg.norm((signed / n) @ matrix, axis=-1)
        index = np.floor(distances / width).astype(np.int64)
        used = ~origin & (index < distance_bins)
        counts = np.bincount(index[used], minlength=distance_bins)
        scale = 1.0 / (n_atoms * n_atoms)
        with np.errstate(invalid="ignore", divide="ignore"):
            correlation = np.bincount(index[used], real[used], minlength=distance_bins)
            rdf = np.bincount(index[used], pairs[used], minlength=distance_bins)
            correlation = np.where(counts > 0, correlation * scale / counts, 0.0)
            rdf = np.where(counts > 0, rdf * scale / counts, 0.0)
        found = {}
        if self.direct:
            found = self.__direct(system, first, second)
        return SpatialCorrelationResult(
            r=(np.arange(distance_bins) + 0.5) * width,
            correlation=correlation,
            rdf=rdf,
            q=2 * np.pi * (np.arange(wavevector_bins) + 0.5) * step,
            reciprocal=reciprocal,
            means=np.array([first.mean(), second.mean()]),
            mean_squares=np.array([(first**2).mean(), (second**2).mean()]),
            mean_product=float((first * second).mean()),
            **found,
        )

    def __direct(self, system, first, second):
        """The correlation and the pair distribution of the neighbors."""
        pairs = NeighborList.build(system, self.cutoff)
        width = (self.cutoff + _EPSILON) / self.bins
        index = np.minimum((pairs.distances / width).astype(np.int64), self.bins - 1)
        owner = np.repeat(np.arange(len(system)), np.diff(pairs.offsets))
        products = first[owner] * second[pairs.j]
        lower = np.arange(self.bins) * width
        shells = (lower + width) ** 3 - lower**3
        n = len(system)
        scale = 3 * system.box.volume / (4 * np.pi * n * n) / shells
        return {
            "neighbor_r": (np.arange(self.bins) + 0.5) * self.cutoff / self.bins,
            "neighbor_correlation": np.bincount(index, products, minlength=self.bins)
            * scale,
            "neighbor_rdf": np.bincount(index, minlength=self.bins) * scale,
        }

    def __repr__(self) -> str:
        return f"SpatialCorrelation({self.first!r}, {self.second!r})"
