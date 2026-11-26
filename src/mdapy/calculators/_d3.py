# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The DFT-D3 dispersion correction of Grimme et al., J. Chem. Phys. 132, 154104
(2010), with the damping of Becke and Johnson (Grimme, Ehrlich and Goerigk, J.
Comput. Chem. 32, 1456 (2011)) or zero damping, and the three-body term of
Axilrod, Teller and Muto, as s-dftd3 computes them."""

import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from mdapy import _core
from mdapy.calculators import _d3_tables, _gpu
from mdapy.calculators._base import (
    WITHOUT_VIRIAL,
    Calculator,
    CalculatorResult,
    _element_indices,
)
from mdapy.core.system import System
from mdapy.core.threads import _native_token
from mdapy.elements import SYMBOLS
from mdapy.errors import InvalidParameter, MissingProperty
from mdapy.neighbor import NeighborList
from mdapy.neighbor._neighbor_list import _given_or_built

_native = _core.calculators

# The bohr in Å and the hartree in eV, CODATA 2022: the tables are in atomic
# units.
_BOHR = 0.529177210544
_HARTREE = 27.211386245981

# The cutoffs of s-dftd3: 60 bohr for the pairs, 40 bohr for the coordination
# numbers and the triangles.
CUTOFF = 60.0 * _BOHR
CN_CUTOFF = 40.0 * _BOHR
THREE_BODY_CUTOFF = 40.0 * _BOHR

DAMPINGS = ("bj", "zero")
# On the GPU D3 finds its pairs itself, from cells of its own: the list the
# device keeps of its atoms, which sorts them, needs no more than this, in Å.
_DEVICE_LIST = 1.0
# The exponent of zero damping, and of the three-body damping plus 2.
_ALPHA = 14.0

# For each functional, (s6, s8, a1, a2 in bohr) of Becke–Johnson damping and
# (s6, s8, rs6, rs8) of zero damping, as published: Grimme et al. (2010) and
# Grimme, Ehrlich and Goerigk (2011) for PBE, revPBE, PBE0, B3LYP, BLYP, BP86
# and TPSS; Goerigk and Grimme, PCCP 13, 6670 (2011) for PBEsol; the tables of
# the Grimme group for RPBE and PBEsol with zero damping; Moellmann and Grimme,
# J. Phys. Chem. C 118, 7615 (2014) for HSE06; Brandenburg et al., PRB 94,
# 115144 (2016) for SCAN; Ehlert et al., J. Chem. Phys. 154, 061101 (2021) for
# r²SCAN.
FUNCTIONALS: dict[str, dict[str, tuple[float, float, float, float]]] = {
    "pbe": {"bj": (1.0, 0.7875, 0.4289, 4.4407), "zero": (1.0, 0.722, 1.217, 1.0)},
    "pbesol": {"bj": (1.0, 2.9491, 0.4466, 6.1742), "zero": (1.0, 0.612, 1.345, 1.0)},
    "revpbe": {"bj": (1.0, 2.3550, 0.5238, 3.5016), "zero": (1.0, 1.010, 0.923, 1.0)},
    "rpbe": {"bj": (1.0, 0.8318, 0.1820, 4.0094), "zero": (1.0, 0.514, 0.872, 1.0)},
    "pbe0": {"bj": (1.0, 1.2177, 0.4145, 4.8593), "zero": (1.0, 0.928, 1.287, 1.0)},
    "hse06": {"bj": (1.0, 2.310, 0.383, 5.685), "zero": (1.0, 0.109, 1.129, 1.0)},
    "b3lyp": {"bj": (1.0, 1.9889, 0.3981, 4.4211), "zero": (1.0, 1.703, 1.261, 1.0)},
    "blyp": {"bj": (1.0, 2.6996, 0.4298, 4.2359), "zero": (1.0, 1.682, 1.094, 1.0)},
    "bp86": {"bj": (1.0, 3.2822, 0.3946, 4.8516), "zero": (1.0, 1.683, 1.139, 1.0)},
    "tpss": {"bj": (1.0, 1.9435, 0.4535, 4.4752), "zero": (1.0, 1.105, 1.166, 1.0)},
    "scan": {"bj": (1.0, 0.0, 0.5380, 5.4200), "zero": (1.0, 0.0, 1.324, 1.0)},
    "r2scan": {"bj": (1.0, 0.78981345, 0.49484001, 5.73083694)},
}
_NAMES = {"bj": ("s6", "s8", "a1", "a2"), "zero": ("s6", "s8", "rs6", "rs8")}
# what a parameter defaults to when no functional gives it
_DEFAULTS = {"s6": 1.0, "rs8": 1.0}


def _number(name: str, value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise InvalidParameter(f"{name} must be a number, not {value!r}") from None
    if not math.isfinite(number):
        raise InvalidParameter(f"{name} must be finite, not {value!r}")
    return number


def _functional_key(functional: Any) -> str:
    if not isinstance(functional, str):
        raise InvalidParameter(f"functional must be a name or None, not {functional!r}")
    key = functional.lower().replace("-", "").replace("_", "")
    if key not in FUNCTIONALS:
        raise InvalidParameter(
            f"D3 has no parameters for {functional!r}; it has {sorted(FUNCTIONALS)}, "
            "or give the numbers of the damping"
        )
    return key


def _parameters(
    functional: str | None, damping: str, given: dict[str, Any]
) -> dict[str, float]:
    """The numbers of the damping: those given, the others of the functional."""
    names = _NAMES[damping]
    for name, value in given.items():
        if value is not None and name not in names:
            raise InvalidParameter(
                f"{name} is a parameter of the other damping, not of {damping!r}"
            )
    table: dict[str, float] = dict(_DEFAULTS)
    if functional is not None:
        key = _functional_key(functional)
        if damping not in FUNCTIONALS[key]:
            raise InvalidParameter(
                f"{functional!r} has no parameters of {damping!r} damping; it has "
                f"{sorted(FUNCTIONALS[key])}"
            )
        table.update(zip(names, FUNCTIONALS[key][damping], strict=True))
    for name in names:
        if given.get(name) is not None:
            table[name] = _number(name, given[name])
    missing = [name for name in names if name not in table]
    if missing:
        raise InvalidParameter(
            f"without a functional, give {', '.join(missing)} of {damping!r} damping"
        )
    parameters = {name: table[name] for name in names}
    if damping == "bj":
        if parameters["a1"] < 0 or parameters["a2"] < 0:
            raise InvalidParameter("a1 and a2 must not be negative")
        if parameters["a1"] == 0 and parameters["a2"] == 0:
            raise InvalidParameter("a1 and a2 cannot both be zero")
    elif parameters["rs6"] <= 0 or parameters["rs8"] <= 0:
        raise InvalidParameter("rs6 and rs8 must be positive")
    return parameters


def _cutoff(name: str, value: Any) -> float:
    number = _number(name, value)
    if not number > 0:
        raise InvalidParameter(f"{name} must be positive, not {value!r}")
    return number


def _blocks() -> tuple[np.ndarray, np.ndarray]:
    """The number of references of each element and the first C6 of each pair
    of elements i >= j in the packed table."""
    counts = np.array([len(r) for r in _d3_tables.REFERENCE_CN], dtype=np.int64)
    i, j = np.tril_indices(_d3_tables.MAX_Z)
    starts = np.concatenate([[0], np.cumsum(counts[i] * counts[j])])
    first = np.zeros((_d3_tables.MAX_Z, _d3_tables.MAX_Z), dtype=np.int64)
    first[i, j] = starts[:-1]
    return counts, first


def _tables(numbers: Sequence[int]) -> dict[str, np.ndarray]:
    """The tables of make_d3 for the elements of these atomic numbers, in Å and
    eV."""
    counts, first = _blocks()
    packed = _d3_tables.reference_c6()
    index = np.asarray(numbers) - 1
    species = len(index)
    reference_cn = np.zeros((species, 7))
    for k, z in enumerate(index):
        values = _d3_tables.REFERENCE_CN[z]
        reference_cn[k, : len(values)] = values
    blocks = []
    for zi in index:
        for zj in index:
            a, b = (zi, zj) if zi >= zj else (zj, zi)
            block = packed[first[a, b] : first[a, b] + counts[a] * counts[b]]
            block = block.reshape(counts[a], counts[b])
            blocks.append((block if zi >= zj else block.T).ravel())
    r4r2 = np.array([_d3_tables.R4_OVER_R2[z] for z in index])
    return {
        "references": counts[index].astype(np.int32),
        "reference_cn": reference_cn.ravel(),
        "rcov": 4.0 / 3.0 * np.array([_d3_tables.COVALENT_RADII[z] for z in index]),
        "r4r2": np.sqrt(0.5 * r4r2 * np.sqrt(index + 1.0)) * _BOHR,
        "vdw": np.ascontiguousarray(
            _d3_tables.vdw_radii()[np.ix_(index, index)]
        ).ravel(),
        "c6": np.concatenate(blocks) * (_HARTREE * _BOHR**6),
    }


class D3(Calculator):
    """The DFT-D3 dispersion correction of Grimme et al., with Becke–Johnson
    or zero damping and the three-body term of Axilrod, Teller and Muto, as
    s-dftd3 computes it; added to another calculator with `Overlay`.

    Parameters
    ----------
    functional : str or None, optional
        The density functional whose parameters to take: "pbe", "pbesol",
        "revpbe", "rpbe", "pbe0", "hse06", "b3lyp", "blyp", "bp86", "tpss",
        "scan" or "r2scan" (Becke–Johnson damping only), in any case; None
        with all the numbers of the damping given. Default "pbe".
    damping : {"bj", "zero"}, optional
        The damping at short distances: of Becke and Johnson, VASP's
        ``IVDW = 12``, or zero damping, VASP's ``IVDW = 11``. Default "bj".
    three_body : bool, optional
        Add the three-body term of Axilrod, Teller and Muto, with s9 = 1.
        Default False.
    cutoff : float64, optional
        Cutoff of the pairs in Å, at least the other two. Default 60 bohr,
        31.75 Å.
    cn_cutoff : float64, optional
        Cutoff of the coordination numbers in Å. Default 40 bohr, 21.17 Å.
    three_body_cutoff : float64, optional
        Cutoff of each side of the triangles of the three-body term in Å.
        Default 40 bohr, 21.17 Å.
    s6, s8, a1, a2 : float64, optional
        The numbers of Becke–Johnson damping, a2 in bohr as published; given,
        they replace those of the functional.
    rs6, rs8 : float64, optional
        The numbers of zero damping with s6 and s8; given, they replace those
        of the functional.
    elements : sequence of str, optional
        The elements, in the order of the species of an `Overlay` on the GPU.
        Default any element from H to Lr.
    device : {"cpu", "cuda"}, optional
        Where it computes: on the processor, or on an NVIDIA GPU. Default
        "cpu".
    precision : {"mixed", "double"}, optional
        On the GPU: single precision for the terms of each pair and double
        for the positions and the sums, or double throughout. Default
        "mixed" on the GPU, "double" on the processor, which has no other.

    Attributes
    ----------
    functional : str or None
        The functional, in lower case.
    damping : str
        "bj" or "zero".
    three_body : bool
        Whether the three-body term is added.
    parameters : dict of str to float64
        s6, s8, a1 and a2 (in bohr) of Becke–Johnson damping, or s6, s8, rs6
        and rs8 of zero damping, and s9.
    cutoff : float64
        Cutoff of the pairs in Å, the largest of the three.
    cn_cutoff : float64
        Cutoff of the coordination numbers in Å.
    three_body_cutoff : float64
        Cutoff of the triangles in Å.
    elements : tuple of str or None
        The elements given, or None for any.
    """

    _takes_wider_lists = True

    def __init__(
        self,
        functional: str | None = "pbe",
        damping: str = "bj",
        *,
        three_body: bool = False,
        cutoff: float = CUTOFF,
        cn_cutoff: float = CN_CUTOFF,
        three_body_cutoff: float = THREE_BODY_CUTOFF,
        s6: float | None = None,
        s8: float | None = None,
        a1: float | None = None,
        a2: float | None = None,
        rs6: float | None = None,
        rs8: float | None = None,
        elements: Sequence[str] | None = None,
        device: str = "cpu",
        precision: str | None = None,
    ):
        self.device, self.precision = _gpu.resolve(device, precision)
        if damping not in DAMPINGS:
            raise InvalidParameter(
                f"damping must be one of {DAMPINGS}, not {damping!r}"
            )
        if not isinstance(three_body, bool | np.bool_):
            raise InvalidParameter(
                f"three_body must be True or False, not {three_body!r}"
            )
        given = {"s6": s6, "s8": s8, "a1": a1, "a2": a2, "rs6": rs6, "rs8": rs8}
        parameters = _parameters(functional, damping, given)
        self.cutoff = _cutoff("cutoff", cutoff)
        self.cn_cutoff = _cutoff("cn_cutoff", cn_cutoff)
        self.three_body_cutoff = _cutoff("three_body_cutoff", three_body_cutoff)
        for name in ("cn_cutoff", "three_body_cutoff")[: 1 + bool(three_body)]:
            if getattr(self, name) > self.cutoff:
                raise InvalidParameter(
                    f"{name} {getattr(self, name)} Å is beyond the cutoff of the pairs, "
                    f"{self.cutoff} Å"
                )
        if elements is None:
            self.elements = None
            numbers = list(range(1, _d3_tables.MAX_Z + 1))
        else:
            names = (elements,) if isinstance(elements, str) else tuple(elements)
            if not names or len(set(names)) != len(names):
                raise InvalidParameter("elements must name each element once")
            for name in names:
                if name not in SYMBOLS[1 : _d3_tables.MAX_Z + 1]:
                    raise InvalidParameter(
                        f"D3 has no references for {name!r}: it covers H to Lr"
                    )
            self.elements = names
            numbers = [SYMBOLS.index(name) for name in names]
        self.functional = None if functional is None else _functional_key(functional)
        self.damping = damping
        self.three_body = bool(three_body)
        parameters["s9"] = 1.0 if self.three_body else 0.0
        self.parameters = parameters
        tables = _tables(numbers)
        p = parameters
        bj = damping == "bj"
        arguments = (
            damping,
            tables["references"],
            tables["reference_cn"],
            tables["rcov"],
            tables["r4r2"],
            tables["vdw"],
            tables["c6"],
            p["s6"],
            p["s8"],
            p["s9"],
            p["a1"] if bj else 0.0,
            p["a2"] * _BOHR if bj else 0.0,
            0.0 if bj else p["rs6"],
            0.0 if bj else p["rs8"],
            _ALPHA,
            _HARTREE,
            self.cutoff,
            self.cn_cutoff,
            self.three_body_cutoff,
        )
        self.__potential = _native.make_d3(*arguments)
        self.__gpu: _gpu.OnDevice | None = None
        if self.device == "cuda":
            self.__gpu = _gpu.OnDevice(
                _gpu.cuda().make_d3(*arguments, self.precision), _DEVICE_LIST
            )

    @property
    def _on_device(self) -> _gpu.OnDevice | None:
        """The potential on the GPU, for its molecular dynamics there."""
        return self.__gpu

    def _species(self, system: System) -> np.ndarray:
        """The index of each atom in the tables, (N,) int32."""
        if self.elements is not None:
            return _element_indices(system, self.elements, "D3")
        if "element" not in system.table.columns:
            raise MissingProperty("D3 needs the element column")
        numbers = system.table["element"].to_physical().to_numpy()
        outside = (numbers < 1) | (numbers > _d3_tables.MAX_Z)
        if outside.any():
            missing = sorted({SYMBOLS[z] for z in numbers[outside]})
            raise InvalidParameter(
                f"D3 has no references for {missing}: it covers H to Lr"
            )
        return (numbers - 1).astype(np.int32)

    def compute(
        self,
        system: System,
        properties: Iterable[str] = ("energy", "forces"),
        *,
        neighbors: NeighborList | None = None,
        progress: Any = None,
        cancel: Any = None,
    ) -> CalculatorResult:
        properties = tuple(properties)
        wanted = self._check_properties(properties)
        species = self._species(system)
        if self.__gpu is not None:  # with its own pairs on the GPU
            return self.__gpu.compute(system, species, wanted)
        virial = WITHOUT_VIRIAL not in properties
        neighbors = _given_or_built(system, self.cutoff, neighbors)
        with neighbors._reading():
            energies, forces, virials = _native.compute_d3(
                self.__potential,
                species,
                np.ascontiguousarray(system["positions"]),
                np.ascontiguousarray(system.box.matrix),
                neighbors.offsets,
                neighbors.j,
                neighbors.shifts,
                progress,
                _native_token(cancel),
                virial,
            )
        return CalculatorResult(
            energy=float(energies.sum()),
            atomic_energies=energies,
            forces=forces,
            virial=virials.sum(axis=0) if virials is not None else None,
            atomic_virials=virials,
            volume=system.box.volume,
        )

    def __repr__(self) -> str:
        where = (
            f", device='cuda', precision={self.precision!r}"
            if self.device == "cuda"
            else ""
        )
        return (
            f"D3({self.functional!r}, damping={self.damping!r}, "
            f"three_body={self.three_body}, cutoff={self.cutoff}, "
            f"cn_cutoff={self.cn_cutoff}{where})"
        )
