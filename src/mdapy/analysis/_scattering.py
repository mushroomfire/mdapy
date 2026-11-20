# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""What X-rays and neutrons see: the total structure factor and the pair
distribution function, weighted by the scattering of each element."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from mdapy import _core
from mdapy.analysis._distribution import (
    METHODS,
    RadialDistributionFunction,
    StructureFactor,
)
from mdapy.core.result import Result
from mdapy.core.system import System
from mdapy.core.threads import _native_token
from mdapy.elements import neutron_scattering_lengths, xray_form_factors
from mdapy.errors import InvalidParameter, MissingProperty
from mdapy.neighbor import NeighborList

_native = _core.analysis

#: What scatters: X-rays, by the form factors of the atoms, or neutrons, by
#: the coherent scattering lengths of their nuclei.
RADIATIONS = ("xray", "neutron")


@dataclass(frozen=True)
class TotalScatteringResult(Result):
    """The total structure factor and the pair distribution function of a
    system, weighted by the scattering of its elements.

    Attributes
    ----------
    q : numpy.ndarray
        (bins,) float64 centers of the bins of |q| in 1/Å.
    s : numpy.ndarray
        (bins,) float64, the total structure factor S(q), which tends to 1 at
        large q; with the direct method NaN in bins without wave vectors.
    f : numpy.ndarray
        (bins,) float64, the total scattering F(q) = (S(q) - 1) <w>², in
        electrons² for X-rays and fm² for neutrons.
    partial : dict
        (bins,) float64 share of each pair of elements ``(a, b)`` in
        S(q) - 1, ``a`` not after ``b`` by atomic number, both orders
        counted; they sum to S(q) - 1.
    r : numpy.ndarray
        (r_bins,) float64 centers of the bins of distance in Å.
    g : numpy.ndarray
        (r_bins,) float64, the weighted radial distribution function.
    pdf : numpy.ndarray
        (r_bins,) float64, the reduced pair distribution function
        G(r) = 4π ρ r (g(r) - 1) in 1/Å².
    weights : dict
        The weight of each element: (bins,) float64 form factors in
        electrons for X-rays, a float64 scattering length in fm for
        neutrons; in real space X-rays weigh with their form factors at
        q = 0.
    radiation : str
        ``"xray"`` or ``"neutron"``.
    density : float64
        Number density of the atoms in 1/Å³.
    """

    q: np.ndarray
    s: np.ndarray
    f: np.ndarray
    partial: dict[tuple[str, str], np.ndarray]
    r: np.ndarray
    g: np.ndarray
    pdf: np.ndarray
    weights: dict[str, Any]
    radiation: str
    density: float

    def transform(
        self,
        r: Any = None,
        *,
        q_range: tuple[float, float] | None = None,
        lorch: bool = False,
    ) -> np.ndarray:
        """The reduced pair distribution function from the structure factor,
        G(r) = (2/π) ∫ q (S(q) - 1) sin(qr) dq over a range of q, as an
        experiment that measures S(q) only there gives it.

        Parameters
        ----------
        r : array_like, optional
            Distances in Å; default ``r`` of the result.
        q_range : tuple of float64, optional
            The range of q in 1/Å, ``(q_min, q_max)``; default all the bins.
        lorch : bool, optional
            Whether to damp the ends of the range with the function of Lorch,
            sin(πq/q_max) / (πq/q_max), against the ripples of the cut.
            Default False.

        Returns
        -------
        numpy.ndarray
            (len(r),) float64, G(r) in 1/Å².

        Raises
        ------
        InvalidParameter
            For a range outside the bins or not increasing.
        """
        r = self.r if r is None else np.asarray(r, dtype=np.float64)
        q, s = self.q, self.s
        low, high = (q[0], q[-1]) if q_range is None else map(float, q_range)
        if not q[0] - 1e-12 <= low < high <= q[-1] + 1e-12:
            raise InvalidParameter(
                f"q_range must be increasing and within {q[0]:.4g} to {q[-1]:.4g} 1/Å"
            )
        keep = (q >= low) & (q <= high) & np.isfinite(s)
        q, s = q[keep], s[keep]
        dq = self.q[1] - self.q[0] if len(self.q) > 1 else 1.0
        damping = np.sinc(q / high) if lorch else 1.0  # numpy's sinc has the π
        integrand = q * (s - 1.0) * damping
        return 2.0 / np.pi * np.sin(np.outer(r, q)) @ integrand * dq


class TotalScattering:
    """The total structure factor S(q) and the pair distribution function of
    a periodic system as X-rays or neutrons see them, its pairs of elements
    weighted by their scattering.

    Parameters
    ----------
    radiation : {"xray", "neutron"}
        What scatters: X-rays, weighted by the form factors of neutral atoms,
        or neutrons, by the coherent scattering lengths of the natural
        elements.
    q_max : float64
        Largest |q| in 1/Å; at most 75.4 for X-rays.
    bins : int32, optional
        Number of bins of |q|. Default 200.
    r_max : float64, optional
        Largest distance of the pair distribution function in Å. Default 10.
    r_bins : int32, optional
        Number of bins of distance. Default 500.
    method : {"direct", "debye"}, optional
        How the structure factor is computed, as ``StructureFactor`` does.
        Default "direct".
    cutoff : float64, optional
        Largest distance of the Debye method in Å, as ``StructureFactor``.
    weights : dict of str to float64, optional
        Weights that replace those of the radiation for some elements, as
        the scattering length of deuterium, 6.6681 fm, for the hydrogen of
        heavy water.
    """

    def __init__(
        self,
        radiation: str,
        q_max: float,
        bins: int = 200,
        *,
        r_max: float = 10.0,
        r_bins: int = 500,
        method: str = "direct",
        cutoff: float | None = None,
        weights: Mapping[str, float] | None = None,
    ):
        if radiation not in RADIATIONS:
            raise InvalidParameter(
                f"radiation must be one of {RADIATIONS}, not {radiation!r}"
            )
        if radiation == "xray" and q_max > 4 * np.pi * 6:
            raise InvalidParameter("q_max of X-rays must be at most 75.40 1/Å")
        if method not in METHODS:
            raise InvalidParameter(f"method must be one of {METHODS}, not {method!r}")
        self.radiation = radiation
        self.weights = dict(weights or {})
        self.__factor = StructureFactor(q_max, bins, method=method, cutoff=cutoff)
        self.__rdf = RadialDistributionFunction(r_max, r_bins)

    def compute(
        self,
        system: System,
        *,
        neighbors: NeighborList | None = None,
        progress: Any = None,
        cancel: Any = None,
    ) -> TotalScatteringResult:
        """The structure factor and the pair distribution function.

        Parameters
        ----------
        system : System
            The atoms, with an ``element`` column, in a box periodic in
            every direction.
        neighbors : NeighborList, optional
            A neighbor list of this system reaching ``r_max``, and the
            cutoff of the Debye method.

        Returns
        -------
        TotalScatteringResult
            The weighted functions.

        Raises
        ------
        MissingProperty
            Without an ``element`` column.
        InvalidParameter
            For an element without a weight, or a box that is not periodic.
        """
        if "element" not in system.table.columns:
            raise MissingProperty("total scattering needs an element column")
        factor = self.__factor.compute(
            system, neighbors=neighbors, progress=progress, cancel=cancel
        )
        rdf = self.__rdf.compute(system, neighbors=neighbors)
        symbols = system["element"]
        present = sorted(set(symbols.tolist()), key=lambda e: list(symbols).index(e))
        fractions = {e: float(np.mean(symbols == e)) for e in present}
        weights, at_zero = {}, {}
        for e in present:
            if e in self.weights:
                weights[e] = at_zero[e] = float(self.weights[e])
            elif self.radiation == "xray":
                weights[e] = xray_form_factors([e], factor.q)[0]
                at_zero[e] = float(xray_form_factors([e], 0.0)[0])
            else:
                weights[e] = at_zero[e] = float(neutron_scattering_lengths([e])[0])
        mean = sum(fractions[e] * np.asarray(weights[e]) for e in present)
        mean0 = sum(fractions[e] * at_zero[e] for e in present)
        if np.any(np.asarray(mean) == 0) or mean0 == 0:
            raise InvalidParameter("the mean weight of the atoms is zero")
        partial, fsum = {}, 0.0
        g = np.zeros_like(rdf.g)
        for (a, b), shape in factor.faber_ziman.items():
            both = 1.0 if a == b else 2.0
            wa, wb = np.asarray(weights[a]), np.asarray(weights[b])
            share = both * fractions[a] * fractions[b] * wa * wb * (shape - 1.0)
            fsum = fsum + share
            partial[a, b] = share / mean**2
            g += (
                both
                * fractions[a]
                * fractions[b]
                * at_zero[a]
                * at_zero[b]
                * rdf.pair(a, b)
            )
        g /= mean0**2
        density = len(system) / float(system.box.volume)
        s = 1.0 + fsum / mean**2
        return TotalScatteringResult(
            q=factor.q,
            s=s,
            f=fsum,
            partial=partial,
            r=rdf.r,
            g=g,
            pdf=4.0 * np.pi * density * rdf.r * (g - 1.0),
            weights=weights,
            radiation=self.radiation,
            density=density,
        )


@dataclass(frozen=True)
class PowderDiffractionResult(Result):
    """A powder diffraction pattern.

    Attributes
    ----------
    two_theta : numpy.ndarray
        (bins,) float64 centers of the bins of the angle 2θ in degrees.
    intensity : numpy.ndarray
        (bins,) float64 intensity of each bin, the squared amplitudes of the
        wave vectors in it per atom, in electrons² for X-rays and fm² for
        neutrons, with the factors asked for; broadened when asked.
    wavelength : float64
        Wavelength in Å.
    radiation : str
        ``"xray"`` or ``"neutron"``.
    """

    two_theta: np.ndarray
    intensity: np.ndarray
    wavelength: float
    radiation: str


class PowderDiffraction:
    """The powder diffraction pattern of a periodic system: the scattering of
    the wave vectors of its box, binned by the angle 2θ.

    Parameters
    ----------
    wavelength : float64, optional
        Wavelength in Å. Default 1.5406, Cu Kα1.
    two_theta : tuple of float64, optional
        Range of 2θ in degrees, ``(low, high)``, within 0 to 180. Default
        (10, 90).
    bins : int32, optional
        Number of bins of 2θ. Default 1600.
    radiation : {"xray", "neutron"}, optional
        X-rays, weighted by the form factors of neutral atoms, or neutrons,
        by the coherent scattering lengths. Default "xray".
    lorentz_polarization : bool, optional
        Whether to multiply by the Lorentz-polarization factor of a powder in
        an unpolarized beam, (1 + cos² 2θ) / (sin²θ cos θ), for neutrons
        without the polarization, 1 / (sin²θ cos θ). Default True.
    debye_waller : float64, optional
        An isotropic displacement factor B in Å², which multiplies by
        exp(-2B (sin θ / λ)²), for a crystal without thermal motion; the
        atoms of molecular dynamics carry their own. Default None.
    broadening : float64, optional
        Full width at half maximum of a Gaussian in degrees of 2θ that each
        bin is spread over. Default None, none.
    weights : dict of str to float64, optional
        Weights that replace those of the radiation for some elements.
    """

    def __init__(
        self,
        wavelength: float = 1.5406,
        two_theta: tuple[float, float] = (10.0, 90.0),
        bins: int = 1600,
        *,
        radiation: str = "xray",
        lorentz_polarization: bool = True,
        debye_waller: float | None = None,
        broadening: float | None = None,
        weights: Mapping[str, float] | None = None,
    ):
        if not wavelength > 0:
            raise InvalidParameter(f"wavelength must be positive, not {wavelength!r}")
        low, high = (float(v) for v in two_theta)
        if not 0.0 <= low < high <= 180.0:
            raise InvalidParameter(
                "two_theta must be increasing, within 0 to 180 degrees"
            )
        if bins < 1:
            raise InvalidParameter(f"bins must be at least 1, not {bins!r}")
        if radiation not in RADIATIONS:
            raise InvalidParameter(
                f"radiation must be one of {RADIATIONS}, not {radiation!r}"
            )
        if debye_waller is not None and not debye_waller >= 0:
            raise InvalidParameter("debye_waller must be at least 0")
        if broadening is not None and not broadening > 0:
            raise InvalidParameter("broadening must be positive")
        q_max = 4 * np.pi * np.sin(np.radians(high) / 2) / wavelength
        if radiation == "xray" and q_max > 4 * np.pi * 6:
            raise InvalidParameter("the X-ray form factors hold up to q = 75.40 1/Å")
        self.wavelength = float(wavelength)
        self.two_theta = (low, high)
        self.bins = int(bins)
        self.radiation = radiation
        self.lorentz_polarization = bool(lorentz_polarization)
        self.debye_waller = None if debye_waller is None else float(debye_waller)
        self.broadening = None if broadening is None else float(broadening)
        self.weights = dict(weights or {})
        self.__q_max = q_max

    def compute(
        self,
        system: System,
        *,
        progress: Any = None,
        cancel: Any = None,
    ) -> PowderDiffractionResult:
        """The pattern of a system.

        Parameters
        ----------
        system : System
            The atoms, with an ``element`` column, in a box periodic in
            every direction.

        Returns
        -------
        PowderDiffractionResult
            The pattern.

        Raises
        ------
        MissingProperty
            Without an ``element`` column.
        InvalidParameter
            For an element without a weight, or a box that is not periodic.
        """
        if "element" not in system.table.columns:
            raise MissingProperty("powder diffraction needs an element column")
        box = system.box
        if not np.all(box.pbc):
            raise InvalidParameter(
                "powder diffraction needs a box periodic in every direction"
            )
        symbols = system["element"]
        names = sorted(set(symbols.tolist()))
        index = np.searchsorted(np.array(names), symbols).astype(np.int32)
        lengths, rho_re, rho_im = _native.wave_densities(
            np.ascontiguousarray(system["positions"]),
            np.ascontiguousarray(box.matrix),
            index,
            len(names),
            self.__q_max,
            progress,
            _native_token(cancel),
        )
        # the weight of each element at the length of each vector
        weights = np.empty((len(names), len(lengths)))
        for t, e in enumerate(names):
            if e in self.weights:
                weights[t] = float(self.weights[e])
            elif self.radiation == "xray":
                weights[t] = xray_form_factors([e], lengths)[0]
            else:
                weights[t] = float(neutron_scattering_lengths([e])[0])
        real = np.einsum("tv,vt->v", weights, rho_re)
        imaginary = np.einsum("tv,vt->v", weights, rho_im)
        # per atom, each vector and its opposite
        intensity_v = 2.0 * (real * real + imaginary * imaginary) / len(system)
        theta = np.arcsin(np.clip(lengths * self.wavelength / (4 * np.pi), 0.0, 1.0))
        if self.lorentz_polarization:
            factor = 1.0 / (np.sin(theta) ** 2 * np.cos(theta))
            if self.radiation == "xray":
                factor *= 1.0 + np.cos(2 * theta) ** 2
            intensity_v = intensity_v * factor
        if self.debye_waller is not None:
            s = lengths / (4 * np.pi)
            intensity_v = intensity_v * np.exp(-2 * self.debye_waller * s**2)
        low, high = self.two_theta
        edges = np.linspace(low, high, self.bins + 1)
        centers = 0.5 * (edges[1:] + edges[:-1])
        angle = np.degrees(2 * theta)
        which = np.searchsorted(edges, angle, side="right") - 1
        inside = (which >= 0) & (which < self.bins)
        intensity = np.bincount(which[inside], intensity_v[inside], self.bins)
        if self.broadening is not None:
            sigma = self.broadening / (2 * np.sqrt(2 * np.log(2)))
            kernel = np.exp(-0.5 * ((centers[:, None] - centers[None, :]) / sigma) ** 2)
            kernel /= sigma * np.sqrt(2 * np.pi)
            intensity = kernel @ intensity * (edges[1] - edges[0])
        return PowderDiffractionResult(
            two_theta=centers,
            intensity=intensity,
            wavelength=self.wavelength,
            radiation=self.radiation,
        )
