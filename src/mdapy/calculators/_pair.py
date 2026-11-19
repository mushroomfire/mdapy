# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Pair potentials of a formula with coefficients for each pair of elements:
Lennard-Jones and Morse, as LAMMPS pair_style lj/cut and morse."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from mdapy import _core
from mdapy.calculators import _gpu
from mdapy.calculators._base import Calculator, CalculatorResult, _element_indices
from mdapy.core.system import System
from mdapy.core.threads import _native_token
from mdapy.elements import SYMBOLS
from mdapy.errors import InvalidParameter
from mdapy.neighbor import NeighborList
from mdapy.neighbor._neighbor_list import _given_or_built

_native = _core.calculators
_MIXING = ("geometric", "arithmetic", "sixthpower")


def _element(name: Any) -> str:
    if not isinstance(name, str) or name not in SYMBOLS:
        raise InvalidParameter(f"{name!r} is not an element")
    return name


def _numbers(values: Any, sizes: tuple[int, ...], what: str) -> tuple[float, ...]:
    """The numbers of an entry, finite, as many as one of sizes."""
    try:
        numbers = tuple(float(v) for v in values)
    except (TypeError, ValueError):
        raise InvalidParameter(f"{what} must be numbers, not {values!r}") from None
    if len(numbers) not in sizes or not all(np.isfinite(numbers)):
        count = " or ".join(str(s) for s in sizes)
        raise InvalidParameter(f"{what} must be {count} finite numbers, not {values!r}")
    return numbers


def _entries(
    parameters: Mapping[Any, Any], sizes: tuple[int, ...], pairs_only: bool
) -> tuple[dict[str, tuple[float, ...]], dict[tuple[str, str], tuple[float, ...]]]:
    """The entries of single elements and of pairs, each pair in both orders."""
    if not isinstance(parameters, Mapping) or not parameters:
        raise InvalidParameter("parameters must be a mapping with at least one entry")
    singles: dict[str, tuple[float, ...]] = {}
    pairs: dict[tuple[str, str], tuple[float, ...]] = {}
    for key, values in parameters.items():
        if isinstance(key, str) and not pairs_only:
            singles[_element(key)] = _numbers(values, sizes, f"the parameters of {key}")
            continue
        if isinstance(key, str):
            key = (key, key)
        if not isinstance(key, Sequence) or isinstance(key, str) or len(key) != 2:
            raise InvalidParameter(
                f"a key of parameters must be an element or a pair of elements, not {key!r}"
            )
        a, b = _element(key[0]), _element(key[1])
        numbers = _numbers(values, sizes, f"the parameters of {a}–{b}")
        for pair in ((a, b), (b, a)):
            if pair in pairs and pairs[pair] != numbers:
                raise InvalidParameter(f"{a}–{b} is given twice with other parameters")
            pairs[pair] = numbers
    return singles, pairs


def _elements(
    singles: Iterable[str], pairs: Iterable[tuple[str, str]]
) -> tuple[str, ...]:
    """The elements, in the order they first appear."""
    found: dict[str, None] = {}
    for name in singles:
        found[name] = None
    for a, b in pairs:
        found[a] = None
        found[b] = None
    return tuple(found)


def _cutoff(value: Any, what: str = "cutoff") -> float:
    try:
        cutoff = float(value)
    except (TypeError, ValueError):
        raise InvalidParameter(f"{what} must be a number, not {value!r}") from None
    if not cutoff > 0 or not np.isfinite(cutoff):
        raise InvalidParameter(f"{what} must be positive, not {value!r}")
    return cutoff


class _PairCalculator(Calculator):
    """A pair potential computed natively from tables of coefficients."""

    _style: str
    _name: str

    def _build(
        self,
        elements: tuple[str, ...],
        cutoffs: np.ndarray,
        offset: np.ndarray,
        coefficients: np.ndarray,
    ) -> None:
        self._any_atom = elements is None
        self.elements = elements
        self.cutoffs = cutoffs
        self.cutoff = float(cutoffs.max())
        tables = (
            np.ascontiguousarray((cutoffs**2).ravel()),
            np.ascontiguousarray(offset.ravel()),
            np.ascontiguousarray(coefficients.ravel()),
        )
        self.__potential = _native.make_pair(self._style, *tables)
        self.__gpu: _gpu.OnDevice | None = None
        if self.device == "cuda":
            self.__gpu = _gpu.OnDevice(
                _gpu.cuda().make_pair(
                    self._style, len(cutoffs), *tables, self.precision
                ),
                self.cutoff,
            )

    def _species(self, system: System) -> np.ndarray:
        """The index of each atom in the tables, (N,) int32."""
        if self._any_atom:  # one kind of atom, whatever its element
            return np.zeros(len(system), dtype=np.int32)
        return _element_indices(system, self.elements, self._name)

    @property
    def _on_device(self) -> _gpu.OnDevice | None:
        """The potential on the GPU, for its molecular dynamics there."""
        return self.__gpu

    def compute(
        self,
        system: System,
        properties: Iterable[str] = ("energy", "forces"),
        *,
        neighbors: NeighborList | None = None,
        progress: Any = None,
        cancel: Any = None,
    ) -> CalculatorResult:
        wanted = self._check_properties(properties)
        species = self._species(system)
        if self.__gpu is not None:  # with a neighbor list of its own on the GPU
            return self.__gpu.compute(system, species, wanted)
        neighbors = _given_or_built(system, self.cutoff, neighbors)
        with neighbors._reading():
            energies, forces, virials = _native.compute_pair(
                self.__potential,
                species,
                np.ascontiguousarray(system["positions"]),
                np.ascontiguousarray(system.box.matrix),
                neighbors.offsets,
                neighbors.j,
                neighbors.shifts,
                progress,
                _native_token(cancel),
            )
        return CalculatorResult(
            energy=float(energies.sum()),
            atomic_energies=energies,
            forces=forces,
            virial=virials.sum(axis=0),
            atomic_virials=virials,
            volume=system.box.volume,
        )


class LennardJones(_PairCalculator):
    """Lennard-Jones potential, E = 4 ε ((σ/r)^12 − (σ/r)^6) below a cutoff,
    as LAMMPS ``pair_style lj/cut`` computes it.

    Parameters
    ----------
    parameters : mapping or sequence of float64
        For an element, (ε, σ) or (ε, σ, cutoff) of its pairs with itself, in
        eV and Å; the pairs of two such elements are mixed. For a pair of
        elements, a tuple such as ("Cu", "Ni"), the same numbers of that pair,
        taken as they are. Numbers alone, (ε, σ) or (ε, σ, cutoff): one kind
        of atom, whatever its element, for systems with no elements.
    cutoff : float64
        Cutoff in Å of the pairs given none.
    mixing : {"geometric", "arithmetic", "sixthpower"}, optional
        How ε, σ and the cutoffs of two elements mix, as ``pair_modify mix``
        of LAMMPS: geometric means; ε geometric and σ and the cutoff
        arithmetic (Lorentz–Berthelot); or the sixth-power rule. Default
        "geometric".
    shift : bool, optional
        Take off each pair the energy at its cutoff, so that it goes to zero
        there, as ``pair_modify shift yes``. Default False.
    device : {"cpu", "cuda"}, optional
        Where it computes: on the processor, or on an NVIDIA GPU with a
        neighbor list of its own. Default "cpu".
    precision : {"mixed", "double"}, optional
        On the GPU: single precision for the terms of each pair and double
        for the positions and the sums, or double throughout. Default
        "mixed" on the GPU, "double" on the processor, which has no other.

    Attributes
    ----------
    elements : tuple of str or None
        The elements, in the order of the parameters; None for one kind of
        atom of any element.
    cutoffs : numpy.ndarray
        (E, E) float64 cutoff of each pair of elements in Å.
    cutoff : float64
        The largest cutoff in Å.
    """

    _style = "lj/cut"
    _name = "LennardJones"

    def __init__(
        self,
        parameters: Mapping[Any, Sequence[float]],
        cutoff: float,
        *,
        mixing: str = "geometric",
        shift: bool = False,
        device: str = "cpu",
        precision: str | None = None,
    ):
        self.device, self.precision = _gpu.resolve(device, precision)
        elements, any_atom, epsilon, sigma, cut = _mixed_tables(
            parameters, cutoff, mixing
        )
        coefficients = np.stack(
            [
                48.0 * epsilon * sigma**12.0,
                24.0 * epsilon * sigma**6.0,
                4.0 * epsilon * sigma**12.0,
                4.0 * epsilon * sigma**6.0,
            ],
            axis=-1,
        )
        ratio = sigma / cut
        offset = (
            4.0 * epsilon * (ratio**12.0 - ratio**6.0)
            if shift
            else np.zeros_like(epsilon)
        )
        self.mixing = mixing
        self.shift = bool(shift)
        self.epsilon = epsilon
        self.sigma = sigma
        self._build(None if any_atom else elements, cut, offset, coefficients)

    def __repr__(self) -> str:
        return (
            f"LennardJones(elements={self.elements}, cutoff={self.cutoff}, "
            f"mixing={self.mixing!r}, shift={self.shift}{_where(self)})"
        )


class UhlenbeckFord(_PairCalculator):
    """Uhlenbeck-Ford model, E = −ε ln(1 − exp(−r²/σ²)) below a cutoff, as
    LAMMPS ``pair_style ufm`` computes it: a soft repulsion of finite range
    whose free energy is known, the reference liquid of free energies.

    Parameters
    ----------
    parameters : mapping or sequence of float64
        For an element, (ε, σ) or (ε, σ, cutoff) of its pairs with itself, in
        eV and Å; the pairs of two such elements are mixed. For a pair of
        elements, a tuple such as ("Cu", "Ni"), the same numbers of that pair,
        taken as they are. Numbers alone, (ε, σ) or (ε, σ, cutoff): any atom,
        whatever its element.
    cutoff : float64
        Cutoff in Å of the pairs given none.
    mixing : {"geometric", "arithmetic", "sixthpower"}, optional
        How ε, σ and the cutoffs of two elements mix, as ``pair_modify mix``
        of LAMMPS. Default "geometric".
    shift : bool, optional
        Take off each pair the energy at its cutoff, as ``pair_modify shift
        yes``. Default False.
    device : {"cpu", "cuda"}, optional
        Where it computes: on the processor, or on an NVIDIA GPU with a
        neighbor list of its own. Default "cpu".
    precision : {"mixed", "double"}, optional
        On the GPU: single precision for the terms of each pair and double
        for the positions and the sums, or double throughout. Default
        "mixed" on the GPU, "double" on the processor, which has no other.

    Attributes
    ----------
    elements : tuple of str or None
        The elements, in the order of the parameters; None for any atom.
    cutoffs : numpy.ndarray
        (E, E) float64 cutoff of each pair of elements in Å.
    cutoff : float64
        The largest cutoff in Å.
    """

    _style = "ufm"
    _name = "UhlenbeckFord"
    # its forces are pairwise, and their virials split in halves
    _heat_flux = True

    def __init__(
        self,
        parameters: Mapping[Any, Sequence[float]] | Sequence[float],
        cutoff: float,
        *,
        mixing: str = "geometric",
        shift: bool = False,
        device: str = "cpu",
        precision: str | None = None,
    ):
        self.device, self.precision = _gpu.resolve(device, precision)
        elements, any_atom, epsilon, sigma, cut = _mixed_tables(
            parameters, cutoff, mixing
        )
        coefficients = np.stack(
            [
                2.0 * epsilon / sigma**2,
                1.0 / sigma**2,
                epsilon,
                np.zeros_like(epsilon),
            ],
            axis=-1,
        )
        offset = (
            -epsilon * np.log(1.0 - np.exp(-((cut / sigma) ** 2)))
            if shift
            else np.zeros_like(epsilon)
        )
        self.mixing = mixing
        self.shift = bool(shift)
        self.epsilon = epsilon
        self.sigma = sigma
        self._build(None if any_atom else elements, cut, offset, coefficients)

    def __repr__(self) -> str:
        return (
            f"UhlenbeckFord(elements={self.elements}, cutoff={self.cutoff}, "
            f"mixing={self.mixing!r}, shift={self.shift}{_where(self)})"
        )


def _where(calculator: Calculator) -> str:
    """The device and the precision of a calculator on the GPU, for repr."""
    if calculator.device == "cpu":
        return ""
    return f", device={calculator.device!r}, precision={calculator.precision!r}"


def _mix(
    first: tuple[float, float, float], second: tuple[float, float, float], rule: str
) -> tuple[float, float, float]:
    """ε, σ and the cutoff of a pair from those of its two elements, by the
    rules of LAMMPS."""
    (e1, s1, c1), (e2, s2, c2) = first, second
    if rule == "geometric":
        return np.sqrt(e1 * e2), np.sqrt(s1 * s2), np.sqrt(c1 * c2)
    if rule == "arithmetic":
        return np.sqrt(e1 * e2), 0.5 * (s1 + s2), 0.5 * (c1 + c2)
    eps = 2.0 * np.sqrt(e1 * e2) * s1**3 * s2**3 / (s1**6 + s2**6)
    return (
        eps,
        (0.5 * (s1**6 + s2**6)) ** (1.0 / 6.0),
        (0.5 * (c1**6 + c2**6)) ** (1.0 / 6.0),
    )


def _mixed_tables(
    parameters: Any, cutoff: float, mixing: str
) -> tuple[tuple[str, ...], bool, np.ndarray, np.ndarray, np.ndarray]:
    """The elements, whether one kind of atom stands for any, and the (E, E)
    energies, lengths and cutoffs of the pairs, mixed by the rules of LAMMPS,
    of parameters of (energy, length) or (energy, length, cutoff)."""
    if mixing not in _MIXING:
        raise InvalidParameter(f"mixing must be one of {list(_MIXING)}, not {mixing!r}")
    default = _cutoff(cutoff)
    any_atom = not isinstance(parameters, Mapping)
    if any_atom:  # one kind of atom, of any element
        parameters = {"X": parameters}
    singles, pairs = _entries(parameters, (2, 3), pairs_only=False)
    elements = _elements(singles, pairs)
    e = len(elements)
    epsilon, sigma, cut = np.zeros((e, e)), np.zeros((e, e)), np.zeros((e, e))
    own = {}
    for name, numbers in singles.items():
        if (name, name) in pairs and pairs[name, name] != numbers:
            raise InvalidParameter(
                f"{name} is given alone and as {name}–{name} with other parameters"
            )
        own[name] = (
            numbers[0],
            numbers[1],
            numbers[2] if len(numbers) == 3 else default,
        )
    for a, b in pairs:
        if a == b and a not in own:
            numbers = pairs[a, b]
            own[a] = (
                numbers[0],
                numbers[1],
                numbers[2] if len(numbers) == 3 else default,
            )
    for i, a in enumerate(elements):
        for j, b in enumerate(elements):
            if (a, b) in pairs:
                numbers = pairs[a, b]
                eps, sig = numbers[0], numbers[1]
                rc = numbers[2] if len(numbers) == 3 else default
            elif a == b and a in own:  # as given, not mixed with itself
                eps, sig, rc = own[a]
            elif a in own and b in own:
                eps, sig, rc = _mix(own[a], own[b], mixing)
            else:
                missing = b if a in own else a
                raise InvalidParameter(
                    f"{a}–{b} has no parameters: give {missing} its own, or the pair"
                )
            if not eps >= 0 or not sig > 0:
                raise InvalidParameter(
                    f"{a}–{b}: epsilon must not be negative and sigma must be positive"
                )
            epsilon[i, j], sigma[i, j], cut[i, j] = eps, sig, _cutoff(rc)
    return elements, any_atom, epsilon, sigma, cut


class Morse(_PairCalculator):
    """Morse potential, E = D0 (exp(−2α (r − r0)) − 2 exp(−α (r − r0)))
    below a cutoff, as LAMMPS ``pair_style morse`` computes it.

    Parameters
    ----------
    parameters : mapping
        For each pair of elements, a tuple such as ("Cu", "Ni"), or an element
        for its pairs with itself: (D0, α, r0) or (D0, α, r0, cutoff), in eV,
        1/Å and Å. Every pair of the elements needs its own, as Morse
        parameters do not mix.
    cutoff : float64
        Cutoff in Å of the pairs given none.
    shift : bool, optional
        Take off each pair the energy at its cutoff, so that it goes to zero
        there, as ``pair_modify shift yes``. Default False.
    device : {"cpu", "cuda"}, optional
        Where it computes: on the processor, or on an NVIDIA GPU with a
        neighbor list of its own. Default "cpu".
    precision : {"mixed", "double"}, optional
        On the GPU: single precision for the terms of each pair and double
        for the positions and the sums, or double throughout. Default
        "mixed" on the GPU, "double" on the processor, which has no other.

    Attributes
    ----------
    elements : tuple of str
        The elements, in the order of the parameters.
    cutoffs : numpy.ndarray
        (E, E) float64 cutoff of each pair of elements in Å.
    cutoff : float64
        The largest cutoff in Å.
    """

    _style = "morse"
    _name = "Morse"

    def __init__(
        self,
        parameters: Mapping[Any, Sequence[float]],
        cutoff: float,
        *,
        shift: bool = False,
        device: str = "cpu",
        precision: str | None = None,
    ):
        self.device, self.precision = _gpu.resolve(device, precision)
        default = _cutoff(cutoff)
        _, pairs = _entries(parameters, (3, 4), pairs_only=True)
        elements = _elements((), pairs)
        e = len(elements)
        table = np.zeros((e, e, 4))
        cut = np.zeros((e, e))
        for i, a in enumerate(elements):
            for j, b in enumerate(elements):
                if (a, b) not in pairs:
                    raise InvalidParameter(
                        f"{a}–{b} has no parameters: every pair of the elements needs its own"
                    )
                numbers = pairs[a, b]
                d0, alpha, r0 = numbers[:3]
                if not d0 >= 0 or not alpha > 0 or not r0 > 0:
                    raise InvalidParameter(
                        f"{a}–{b}: D0 must not be negative, alpha and r0 must be positive"
                    )
                table[i, j] = (d0, alpha, r0, 2.0 * d0 * alpha)
                cut[i, j] = _cutoff(numbers[3] if len(numbers) == 4 else default)
        d0, alpha, r0 = table[..., 0], table[..., 1], table[..., 2]
        alpha_dr = -alpha * (cut - r0)
        offset = (
            d0 * (np.exp(2.0 * alpha_dr) - 2.0 * np.exp(alpha_dr))
            if shift
            else np.zeros((e, e))
        )
        self.shift = bool(shift)
        self._build(elements, cut, offset, table)

    def __repr__(self) -> str:
        return (
            f"Morse(elements={self.elements}, cutoff={self.cutoff}, "
            f"shift={self.shift}{_where(self)})"
        )
