# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Chemical elements: symbols, atomic numbers, standard atomic weights and
radii.

Atomic weights are the IUPAC 2016 standard atomic weights; for elements
without stable isotopes they are the mass number of the longest-lived
isotope. Covalent radii are those of Cordero et al. (2008), van der Waals
radii those of Alvarez (2013).
"""

from collections.abc import Iterable
from math import nan

import numpy as np
import polars as pl

from mdapy.errors import InvalidParameter

__all__ = [
    "SYMBOLS",
    "ATOMIC_MASSES",
    "ELEMENT_DTYPE",
    "COVALENT_RADII",
    "VDW_RADII",
    "atomic_numbers",
    "atomic_masses",
    "covalent_radii",
    "vdw_radii",
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

# Covalent radii in Å of Cordero et al. (2008), sp3 carbon, low-spin
# manganese, iron and cobalt; van der Waals radii in Å of Alvarez (2013);
# NaN where they give none. Index = atomic number.
_COVALENT_RADII = (
    nan,  # X
    0.31,  # H
    0.28,  # He
    1.28,  # Li
    0.96,  # Be
    0.84,  # B
    0.76,  # C
    0.71,  # N
    0.66,  # O
    0.57,  # F
    0.58,  # Ne
    1.66,  # Na
    1.41,  # Mg
    1.21,  # Al
    1.11,  # Si
    1.07,  # P
    1.05,  # S
    1.02,  # Cl
    1.06,  # Ar
    2.03,  # K
    1.76,  # Ca
    1.7,  # Sc
    1.6,  # Ti
    1.53,  # V
    1.39,  # Cr
    1.39,  # Mn
    1.32,  # Fe
    1.26,  # Co
    1.24,  # Ni
    1.32,  # Cu
    1.22,  # Zn
    1.22,  # Ga
    1.2,  # Ge
    1.19,  # As
    1.2,  # Se
    1.2,  # Br
    1.16,  # Kr
    2.2,  # Rb
    1.95,  # Sr
    1.9,  # Y
    1.75,  # Zr
    1.64,  # Nb
    1.54,  # Mo
    1.47,  # Tc
    1.46,  # Ru
    1.42,  # Rh
    1.39,  # Pd
    1.45,  # Ag
    1.44,  # Cd
    1.42,  # In
    1.39,  # Sn
    1.39,  # Sb
    1.38,  # Te
    1.39,  # I
    1.4,  # Xe
    2.44,  # Cs
    2.15,  # Ba
    2.07,  # La
    2.04,  # Ce
    2.03,  # Pr
    2.01,  # Nd
    1.99,  # Pm
    1.98,  # Sm
    1.98,  # Eu
    1.96,  # Gd
    1.94,  # Tb
    1.92,  # Dy
    1.92,  # Ho
    1.89,  # Er
    1.9,  # Tm
    1.87,  # Yb
    1.87,  # Lu
    1.75,  # Hf
    1.7,  # Ta
    1.62,  # W
    1.51,  # Re
    1.44,  # Os
    1.41,  # Ir
    1.36,  # Pt
    1.36,  # Au
    1.32,  # Hg
    1.45,  # Tl
    1.46,  # Pb
    1.48,  # Bi
    1.4,  # Po
    1.5,  # At
    1.5,  # Rn
    2.6,  # Fr
    2.21,  # Ra
    2.15,  # Ac
    2.06,  # Th
    2.0,  # Pa
    1.96,  # U
    1.9,  # Np
    1.87,  # Pu
    1.8,  # Am
    1.69,  # Cm
    nan,  # Bk
    nan,  # Cf
    nan,  # Es
    nan,  # Fm
    nan,  # Md
    nan,  # No
    nan,  # Lr
    nan,  # Rf
    nan,  # Db
    nan,  # Sg
    nan,  # Bh
    nan,  # Hs
    nan,  # Mt
    nan,  # Ds
    nan,  # Rg
    nan,  # Cn
    nan,  # Nh
    nan,  # Fl
    nan,  # Mc
    nan,  # Lv
    nan,  # Ts
    nan,  # Og
)
_VDW_RADII = (
    nan,  # X
    1.2,  # H
    1.43,  # He
    2.12,  # Li
    1.98,  # Be
    1.91,  # B
    1.77,  # C
    1.66,  # N
    1.5,  # O
    1.46,  # F
    1.58,  # Ne
    2.5,  # Na
    2.51,  # Mg
    2.25,  # Al
    2.19,  # Si
    1.9,  # P
    1.89,  # S
    1.82,  # Cl
    1.83,  # Ar
    2.73,  # K
    2.62,  # Ca
    2.58,  # Sc
    2.46,  # Ti
    2.42,  # V
    2.45,  # Cr
    2.45,  # Mn
    2.44,  # Fe
    2.4,  # Co
    2.4,  # Ni
    2.38,  # Cu
    2.39,  # Zn
    2.32,  # Ga
    2.29,  # Ge
    1.88,  # As
    1.82,  # Se
    1.86,  # Br
    2.25,  # Kr
    3.21,  # Rb
    2.84,  # Sr
    2.75,  # Y
    2.52,  # Zr
    2.56,  # Nb
    2.45,  # Mo
    2.44,  # Tc
    2.46,  # Ru
    2.44,  # Rh
    2.15,  # Pd
    2.53,  # Ag
    2.49,  # Cd
    2.43,  # In
    2.42,  # Sn
    2.47,  # Sb
    1.99,  # Te
    2.04,  # I
    2.06,  # Xe
    3.48,  # Cs
    3.03,  # Ba
    2.98,  # La
    2.88,  # Ce
    2.92,  # Pr
    2.95,  # Nd
    nan,  # Pm
    2.9,  # Sm
    2.87,  # Eu
    2.83,  # Gd
    2.79,  # Tb
    2.87,  # Dy
    2.81,  # Ho
    2.83,  # Er
    2.79,  # Tm
    2.8,  # Yb
    2.74,  # Lu
    2.63,  # Hf
    2.53,  # Ta
    2.57,  # W
    2.49,  # Re
    2.48,  # Os
    2.41,  # Ir
    2.29,  # Pt
    2.32,  # Au
    2.45,  # Hg
    2.47,  # Tl
    2.6,  # Pb
    2.54,  # Bi
    nan,  # Po
    nan,  # At
    nan,  # Rn
    nan,  # Fr
    nan,  # Ra
    2.8,  # Ac
    2.93,  # Th
    2.88,  # Pa
    2.71,  # U
    2.82,  # Np
    2.81,  # Pu
    2.83,  # Am
    3.05,  # Cm
    3.4,  # Bk
    3.05,  # Cf
    2.7,  # Es
    nan,  # Fm
    nan,  # Md
    nan,  # No
    nan,  # Lr
    nan,  # Rf
    nan,  # Db
    nan,  # Sg
    nan,  # Bh
    nan,  # Hs
    nan,  # Mt
    nan,  # Ds
    nan,  # Rg
    nan,  # Cn
    nan,  # Nh
    nan,  # Fl
    nan,  # Mc
    nan,  # Lv
    nan,  # Ts
    nan,  # Og
)

COVALENT_RADII = np.array(_COVALENT_RADII, dtype=np.float64)
"""Covalent radii in Å (Cordero et al. 2008), indexed by atomic number; NaN
where there is none."""
COVALENT_RADII.flags.writeable = False

VDW_RADII = np.array(_VDW_RADII, dtype=np.float64)
"""Van der Waals radii in Å (Alvarez 2013), indexed by atomic number; NaN
where there is none."""
VDW_RADII.flags.writeable = False

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


def _radii(
    table: np.ndarray, elements: Iterable[str] | Iterable[int], kind: str
) -> np.ndarray:
    elements = list(elements)
    if elements and isinstance(elements[0], str):
        z = atomic_numbers(elements)
    else:
        z = np.asarray(elements, dtype=np.int64).reshape(-1)
    if z.size and (z.min() < 0 or z.max() >= len(SYMBOLS)):
        bad = z[(z < 0) | (z >= len(SYMBOLS))][0]
        raise InvalidParameter(f"no element {bad}")
    radii = table[z]
    if np.isnan(radii).any():
        raise InvalidParameter(f"no {kind} radius for {SYMBOLS[z[np.isnan(radii)][0]]}")
    return radii


def covalent_radii(elements: Iterable[str] | Iterable[int]) -> np.ndarray:
    """Covalent radii of elements (Cordero et al. 2008).

    Parameters
    ----------
    elements : iterable of str or int
        Chemical symbols or atomic numbers.

    Returns
    -------
    numpy.ndarray
        Radii in Å, one per element.

    Raises
    ------
    InvalidParameter
        If an element is unknown or has no covalent radius, as ``"X"``.
    """
    return _radii(COVALENT_RADII, elements, "covalent")


def vdw_radii(elements: Iterable[str] | Iterable[int]) -> np.ndarray:
    """Van der Waals radii of elements (Alvarez 2013).

    Parameters
    ----------
    elements : iterable of str or int
        Chemical symbols or atomic numbers.

    Returns
    -------
    numpy.ndarray
        Radii in Å, one per element.

    Raises
    ------
    InvalidParameter
        If an element is unknown or has no van der Waals radius.
    """
    return _radii(VDW_RADII, elements, "van der Waals")


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
