# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Neuroevolution potentials (NEP4) in the nep.txt files of GPUMD."""

import os
from collections.abc import Iterable
from typing import Any

import numpy as np

from mdapy._core import calculators as _native
from mdapy.calculators._base import (
    Calculator,
    CalculatorResult,
    _element_indices,
)
from mdapy.core.system import System
from mdapy.core.threads import _native_token
from mdapy.elements import SYMBOLS
from mdapy.errors import FileFormatError
from mdapy.neighbor import NeighborList
from mdapy.neighbor._neighbor_list import _given_or_built

#: e^2 / (4 pi epsilon_0) in eV Å, as GPUMD has it.
_COULOMB = 14.399645
#: 1 / 0.46848 Å, the inverse ZBL screening length per Z^0.23, rounded as in GPUMD.
_INVERSE_SCREENING = 2.134563
#: The universal ZBL screening function, sum of a exp(-b x).
_UNIVERSAL = (0.18175, 3.1998, 0.50986, 0.94229, 0.28022, 0.4029, 0.02817, 0.20162)

#: Covalent radii in Å by atomic number (index 0 unused) that GPUMD uses for
#: the typewise ZBL cutoff, for H to Pu.
COVALENT_RADII = np.array(
    [0.0]
    + [
        0.426667, 0.613333, 1.6, 1.25333, 1.02667, 1.0, 0.946667, 0.84, 0.853333,
        0.893333, 1.86667, 1.66667, 1.50667, 1.38667, 1.46667, 1.36, 1.32, 1.28,
        2.34667, 2.05333, 1.77333, 1.62667, 1.61333, 1.46667, 1.42667, 1.38667,
        1.33333, 1.32, 1.34667, 1.45333, 1.49333, 1.45333, 1.53333, 1.46667, 1.52,
        1.56, 2.52, 2.22667, 1.96, 1.85333, 1.76, 1.65333, 1.53333, 1.50667,
        1.50667, 1.44, 1.53333, 1.64, 1.70667, 1.68, 1.68, 1.64, 1.76, 1.74667,
        2.78667, 2.34667, 2.16, 1.96, 2.10667, 2.09333, 2.08, 2.06667, 2.01333,
        2.02667, 2.01333, 2.0, 1.98667, 1.98667, 1.97333, 2.04, 1.94667, 1.82667,
        1.74667, 1.64, 1.57333, 1.54667, 1.48, 1.49333, 1.50667, 1.76, 1.73333,
        1.73333, 1.81333, 1.74667, 1.84, 1.89333, 2.68, 2.41333, 2.22667, 2.10667,
        2.02667, 2.04, 2.05333, 2.06667,
    ]
)  # fmt: skip


class _Header:
    """The keyword lines at the top of a nep.txt file."""

    def __init__(self, path: str | os.PathLike, lines: list[str]):
        self.name = os.fspath(path)
        self.lines = lines
        self.line = 0

    def error(self, message: str) -> FileFormatError:
        return FileFormatError(message, self.name, self.line)

    def next(self, keyword: str | None, counts: tuple[int, ...]) -> list[str]:
        while self.line < len(self.lines):
            self.line += 1
            words = self.lines[self.line - 1].split()
            if words:
                break
        else:
            raise FileFormatError("unexpected end of file", self.name, self.line + 1)
        if keyword is not None and words[0] != keyword:
            raise self.error(f"expected the {keyword} line")
        if len(words) - 1 not in counts:
            expected = " or ".join(str(c) for c in counts)
            raise self.error(f"the {words[0]} line needs {expected} values")
        return words[1:]

    def numbers(self, words: list[str], kind: type) -> list[Any]:
        try:
            return [kind(w) for w in words]
        except ValueError:
            raise self.error(f"expected {kind.__name__} values") from None


def _read_nep(path: str | os.PathLike) -> dict[str, Any]:
    """The parameters of a nep4 or nep4_zbl file, with the coefficients as
    (type, type, n, k) arrays and the weights as (type, neuron, ...)."""
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    header = _Header(path, lines)
    while header.line < len(lines) and not lines[header.line].split():
        header.line += 1
    words = lines[header.line].split() if header.line < len(lines) else [""]
    header.line += 1
    if words[0] not in ("nep4", "nep4_zbl"):
        raise header.error(
            f"unsupported model {words[0]!r}: mdapy reads nep4 and nep4_zbl"
        )
    zbl = words[0] == "nep4_zbl"
    try:
        count = int(words[1])
    except (ValueError, IndexError):
        raise header.error("expected the number of elements") from None
    elements = tuple(words[2:])
    if count < 1 or len(elements) != count:
        raise header.error(f"expected {count} element names, found {len(elements)}")
    for element in elements:
        if element not in SYMBOLS[1:]:
            raise header.error(f"unknown element {element!r}")

    model: dict[str, Any] = {"elements": elements, "zbl": None}
    if zbl:
        values = header.numbers(header.next("zbl", (2, 3)), float)
        if values[0] == 0.0 and values[1] == 0.0:
            model["zbl"] = {"kind": "flexible"}
        else:
            if not 0.0 <= values[0] < values[1]:
                raise header.error("the ZBL cutoffs need 0 <= inner < outer")
            model["zbl"] = {
                "kind": "typewise" if len(values) == 3 else "universal",
                "inner": values[0],
                "outer": values[1],
                "factor": values[2] if len(values) == 3 else 0.0,
            }
    words = header.next("cutoff", (4, 2 * count + 2))
    cutoffs = header.numbers(words[:-2], float)
    header.numbers(words[-2:], int)  # neighbor counts that GPUMD reserves
    radial = np.array(cutoffs[0::2] * (count if len(cutoffs) == 2 else 1))
    angular = np.array(cutoffs[1::2] * (count if len(cutoffs) == 2 else 1))
    if not ((radial > 0).all() and (angular > 0).all()):
        raise header.error("cutoffs must be positive")
    n_radial, n_angular = header.numbers(header.next("n_max", (2,)), int)
    k_radial, k_angular = header.numbers(header.next("basis_size", (2,)), int)
    l_max, l_4body, l_5body = header.numbers(header.next("l_max", (3,)), int)
    neurons, _ = header.numbers(header.next("ANN", (2,)), int)
    if min(n_radial, n_angular, k_radial, k_angular) < 0 or neurons < 1:
        raise header.error("sizes must not be negative and neurons positive")
    if not (
        1 <= l_max <= 8
        and l_4body in (0, 2)
        and l_5body in (0, 1)
        and (l_4body == 2 or l_5body == 0)
        and (l_4body == 0 or l_max >= 2)
    ):
        raise header.error(
            "l_max needs 1 to 8, 0 or 2 for four-body and 0 or 1 for five-body "
            "terms, five-body terms only with four-body terms"
        )
    terms = l_max + (l_4body == 2) + (l_5body == 1)
    dim = n_radial + 1 + (n_angular + 1) * terms

    try:
        values = np.array(" ".join(lines[header.line :]).split(), dtype=np.float64)
    except ValueError:
        for number, text in enumerate(lines[header.line :], header.line + 1):
            try:
                [float(w) for w in text.split()]
            except ValueError:
                raise FileFormatError("expected numbers", header.name, number) from None
        raise  # pragma: no cover
    pairs = count * (count + 1) // 2
    sizes = {
        "w0": (count, neurons, dim),
        "b0": (count, neurons),
        "w1": (count, neurons),
    }
    expected = (
        count * neurons * (dim + 2)
        + 1
        + count
        * count
        * ((n_radial + 1) * (k_radial + 1) + (n_angular + 1) * (k_angular + 1))
        + dim
        + (
            10 * pairs
            if model["zbl"] is not None and model["zbl"]["kind"] == "flexible"
            else 0
        )
    )
    if values.size != expected:
        raise FileFormatError(
            f"expected {expected} parameters after the header, found {values.size}",
            header.name,
        )
    per_type = values[: count * neurons * (dim + 2)].reshape(count, neurons * (dim + 2))
    model["w0"] = per_type[:, : neurons * dim].reshape(sizes["w0"]).copy()
    model["b0"] = per_type[:, neurons * dim : neurons * (dim + 1)].copy()
    model["w1"] = per_type[:, neurons * (dim + 1) :].copy()
    at = count * neurons * (dim + 2)
    model["b1"] = float(values[at])
    at += 1
    for name, n, k in (
        ("c_radial", n_radial, k_radial),
        ("c_angular", n_angular, k_angular),
    ):
        size = count * count * (n + 1) * (k + 1)
        # the file has index ((n * (k + 1) + k) * types + t1) * types + t2
        block = values[at : at + size].reshape(n + 1, k + 1, count, count)
        model[name] = np.ascontiguousarray(block.transpose(2, 3, 0, 1))
        at += size
    model["q_scaler"] = values[at : at + dim].copy()
    at += dim
    if model["zbl"] is not None and model["zbl"]["kind"] == "flexible":
        table = values[at:].reshape(pairs, 10)
        if not ((table[:, 0] >= 0) & (table[:, 0] < table[:, 1])).all():
            raise FileFormatError(
                "the flexible ZBL cutoffs need 0 <= inner < outer", header.name
            )
        model["zbl"]["table"] = table.copy()
    model.update(
        rc_radial=radial,
        rc_angular=angular,
        n_radial=n_radial,
        n_angular=n_angular,
        k_radial=k_radial,
        k_angular=k_angular,
        l_max=l_max,
        four_body=l_4body == 2,
        five_body=l_5body == 1,
        neurons=neurons,
        dim=dim,
    )
    return model


def _zbl_table(model: dict[str, Any]) -> np.ndarray:
    """(types, types, 12) ZBL parameters of each pair, or an empty array."""
    zbl = model["zbl"]
    count = len(model["elements"])
    if zbl is None:
        return np.zeros(0)
    numbers = [SYMBOLS.index(e) for e in model["elements"]]
    table = np.zeros((count, count, 12))
    for a in range(count):
        for b in range(count):
            za, zb = numbers[a], numbers[b]
            if zbl["kind"] == "flexible":
                low, high = min(a, b), max(a, b)
                row = zbl["table"][low * count - low * (low - 1) // 2 + high - low]
                inner, outer, screening = row[0], row[1], row[2:]
            else:
                inner, outer, screening = zbl["inner"], zbl["outer"], _UNIVERSAL
                if zbl["kind"] == "typewise":
                    covalent = (COVALENT_RADII[za] + COVALENT_RADII[zb]) * zbl["factor"]
                    inner, outer = 0.0, min(outer, covalent)
            table[a, b, :4] = (
                inner,
                outer,
                _COULOMB * za * zb,
                (za**0.23 + zb**0.23) * _INVERSE_SCREENING,
            )
            table[a, b, 4:] = screening
    return table


class NEP(Calculator):
    """Neuroevolution potential of a GPUMD ``nep.txt`` file (nep4, nep4_zbl).

    Parameters
    ----------
    path : str or os.PathLike
        The potential file.

    Attributes
    ----------
    elements : tuple of str
        The elements of the potential, in the order of the file.
    cutoff : float
        The largest cutoff radius in Å.
    """

    def __init__(self, path: str | os.PathLike):
        model = _read_nep(path)
        self.path = os.fspath(path)
        self.elements = model["elements"]
        self.cutoff = float(max(model["rc_radial"].max(), model["rc_angular"].max()))
        self.__model = _native.make_nep(
            [
                len(self.elements),
                model["n_radial"],
                model["n_angular"],
                model["k_radial"],
                model["k_angular"],
                model["l_max"],
                int(model["four_body"]),
                int(model["five_body"]),
                model["neurons"],
            ],
            model["rc_radial"],
            model["rc_angular"],
            model["w0"].ravel(),
            model["b0"].ravel(),
            model["w1"].ravel(),
            model["b1"],
            model["c_radial"].ravel(),
            model["c_angular"].ravel(),
            model["q_scaler"],
            _zbl_table(model).ravel(),
        )

    def compute(
        self,
        system: System,
        properties: Iterable[str] = ("energy", "forces"),
        *,
        neighbors: NeighborList | None = None,
        progress: Any = None,
        cancel: Any = None,
    ) -> CalculatorResult:
        self._check_properties(properties)
        species = _element_indices(system, self.elements, "NEP")
        neighbors = _given_or_built(system, self.cutoff, neighbors)
        with neighbors._reading():
            energies, forces, virials = _native.compute_nep(
                self.__model,
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
        )

    def descriptors(
        self,
        system: System,
        *,
        neighbors: NeighborList | None = None,
        progress: Any = None,
        cancel: Any = None,
    ) -> np.ndarray:
        """The descriptors of each atom, scaled as the network sees them.

        Parameters
        ----------
        system : System
            The atoms, with an ``element`` column.
        neighbors : NeighborList, optional
            A neighbor list of this system with at least ``cutoff``.

        Returns
        -------
        numpy.ndarray
            (N, dim) descriptors: the radial ones, then the angular ones by
            l and n, then the four-body and five-body ones by n.
        """
        species = _element_indices(system, self.elements, "NEP")
        neighbors = _given_or_built(system, self.cutoff, neighbors)
        with neighbors._reading():
            return _native.nep_descriptors(
                self.__model,
                species,
                np.ascontiguousarray(system["positions"]),
                np.ascontiguousarray(system.box.matrix),
                neighbors.offsets,
                neighbors.j,
                neighbors.shifts,
                progress,
                _native_token(cancel),
            )

    def __repr__(self) -> str:
        return f"NEP({self.path!r}, elements={self.elements}, cutoff={self.cutoff})"
