# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Chemical elements: symbols, atomic numbers and standard atomic weights.

Atomic weights are the IUPAC 2016 standard atomic weights; for elements
without stable isotopes they are the mass number of the longest-lived
isotope.
"""

from collections.abc import Iterable

import numpy as np
import polars as pl

from mdapy.errors import InvalidParameter

__all__ = [
    "SYMBOLS",
    "ATOMIC_MASSES",
    "ELEMENT_DTYPE",
    "atomic_numbers",
    "atomic_masses",
    "element_from_mass",
]

# Generated from IUPAC 2016 data; index = atomic number, 0 = unknown ("X").
SYMBOLS = (
    "X",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
)

_ATOMIC_MASSES = (
    0.0,
    1.008,
    4.002602,
    6.94,
    9.0121831,
    10.81,
    12.011,
    14.007,
    15.999,
    18.998403163,
    20.1797,
    22.98976928,
    24.305,
    26.9815385,
    28.085,
    30.973761998,
    32.06,
    35.45,
    39.948,
    39.0983,
    40.078,
    44.955908,
    47.867,
    50.9415,
    51.9961,
    54.938044,
    55.845,
    58.933194,
    58.6934,
    63.546,
    65.38,
    69.723,
    72.63,
    74.921595,
    78.971,
    79.904,
    83.798,
    85.4678,
    87.62,
    88.90584,
    91.224,
    92.90637,
    95.95,
    97.90721,
    101.07,
    102.9055,
    106.42,
    107.8682,
    112.414,
    114.818,
    118.71,
    121.76,
    127.6,
    126.90447,
    131.293,
    132.90545196,
    137.327,
    138.90547,
    140.116,
    140.90766,
    144.242,
    144.91276,
    150.36,
    151.964,
    157.25,
    158.92535,
    162.5,
    164.93033,
    167.259,
    168.93422,
    173.054,
    174.9668,
    178.49,
    180.94788,
    183.84,
    186.207,
    190.23,
    192.217,
    195.084,
    196.966569,
    200.592,
    204.38,
    207.2,
    208.9804,
    208.98243,
    209.98715,
    222.01758,
    223.01974,
    226.02541,
    227.02775,
    232.0377,
    231.03588,
    238.02891,
    237.04817,
    244.06421,
    243.06138,
    247.07035,
    247.07031,
    251.07959,
    252.083,
    257.09511,
    258.09843,
    259.101,
    262.11,
    267.122,
    268.126,
    271.134,
    270.133,
    269.1338,
    278.156,
    281.165,
    281.166,
    285.177,
    286.182,
    289.19,
    289.194,
    293.204,
    293.208,
    294.214,
)

ATOMIC_MASSES = np.array(_ATOMIC_MASSES, dtype=np.float64)
"""Standard atomic weights in amu, indexed by atomic number (index 0 is 0.0)."""
ATOMIC_MASSES.flags.writeable = False

ELEMENT_DTYPE = pl.Enum(SYMBOLS)
"""polars dtype of the ``element`` column. Its physical value is the atomic number."""

_ATOMIC_NUMBER = {symbol: z for z, symbol in enumerate(SYMBOLS)}


def atomic_numbers(symbols: Iterable[str]) -> np.ndarray:
    """Atomic numbers of chemical symbols.

    Parameters
    ----------
    symbols : iterable of str
        Chemical symbols, case-sensitive (``"Cu"``, not ``"CU"``). ``"X"``
        stands for an unknown element and maps to 0.

    Returns
    -------
    numpy.ndarray
        Integer array of atomic numbers, one per symbol.

    Raises
    ------
    InvalidParameter
        If a symbol is not a chemical symbol.
    """
    try:
        return np.array([_ATOMIC_NUMBER[s] for s in symbols], dtype=np.int64)
    except (KeyError, TypeError) as exc:
        raise InvalidParameter(f"unknown chemical symbol: {exc.args[0]!r}") from None


def atomic_masses(elements: Iterable[str] | Iterable[int]) -> np.ndarray:
    """Standard atomic weights of elements.

    Parameters
    ----------
    elements : iterable of str or int
        Chemical symbols or atomic numbers (1-118).

    Returns
    -------
    numpy.ndarray
        Atomic weights in amu, one per element.

    Raises
    ------
    InvalidParameter
        If an element is unknown, including ``"X"`` and 0, which have no mass.
    """
    elements = list(elements)
    if elements and isinstance(elements[0], str):
        z = atomic_numbers(elements)
    else:
        z = np.asarray(elements, dtype=np.int64).reshape(-1)
    if z.size and (z.min() < 1 or z.max() >= len(SYMBOLS)):
        bad = z[(z < 1) | (z >= len(SYMBOLS))][0]
        raise InvalidParameter(f"no atomic weight for element {bad}")
    return ATOMIC_MASSES[z].copy()


# Elements with at least one stable isotope: Z = 1-83 except Tc (43) and Pm (61).
_STABLE = tuple(z for z in range(1, 84) if z not in (43, 61))


def element_from_mass(
    mass: float,
    tolerance: float = 0.01,
    candidates: Iterable[str] | None = None,
) -> str:
    """The element whose standard atomic weight matches a mass.

    Parameters
    ----------
    mass : float
        Mass in amu.
    tolerance : float, optional
        Largest accepted difference in amu between ``mass`` and the atomic
        weight. Default 0.01.
    candidates : iterable of str, optional
        Chemical symbols to choose from. Default: the elements with at least
        one stable isotope (H to Bi, without Tc and Pm). Radioactive elements
        such as Po, whose atomic weights can lie within 0.01 amu of another
        element's, are only matched when listed here.

    Returns
    -------
    str
        Chemical symbol of the matching element.

    Raises
    ------
    InvalidParameter
        If no candidate, or more than one, lies within ``tolerance``.
    """
    z = np.array(_STABLE) if candidates is None else atomic_numbers(candidates)
    diff = np.abs(ATOMIC_MASSES[z] - float(mass))
    matches = z[diff <= tolerance]
    if len(matches) != 1:
        found = [SYMBOLS[i] for i in matches]
        raise InvalidParameter(
            f"mass {mass} amu matches {len(matches)} elements within "
            f"{tolerance} amu {found}; specify the elements explicitly"
        )
    return SYMBOLS[int(matches[0])]
