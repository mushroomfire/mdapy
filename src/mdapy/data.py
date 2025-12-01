# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

"""

This module provides fundamental atomic data used throughout **MDAPY**, including
chemical symbols, atomic numbers, van der Waals radii, and atomic masses (copy from `ase.data <https://ase-lib.org/ase/data.html>`_).
All data are stored as immutable NumPy arrays or Python dictionaries for fast lookup
and consistent use across simulations and analysis.

The data originate from reliable references such as IUPAC atomic weight tables and
peer-reviewed compilations of van der Waals radii.

References
----------
- S. Alvarez, *A cartography of the van der Waals territories*, Dalton Trans., 2013, 42, 8617–8636.
  DOI: 10.1039/C3DT50599E
- J. Meija et al., *Atomic weights of the elements 2013 (IUPAC Technical Report)*,
  Pure and Applied Chemistry, 88(3), 265–291 (2016). DOI: 10.1515/pac-2015-0305

Attributes
----------
chemical_symbols : list[str]
    List of all chemical element symbols in periodic table order.
    The first entry "X" is a placeholder for undefined species.

atomic_numbers : dict[str, int]
    A mapping from element symbol to its atomic number, e.g. ``atomic_numbers["Fe"] == 26``.

vdw_radii : np.ndarray
    NumPy array of van der Waals radii (in Å).
    Values are based on Alvarez (2013). Some elements have ``np.nan`` where data are unavailable.
    The array is read-only.

atomic_masses : np.ndarray
    NumPy array of standard atomic weights (in unified atomic mass units, u).
    Values follow the IUPAC 2013 report (Meija et al., 2016).
    For elements without stable isotopes, the mass of the most stable isotope is used.
    The array is read-only.

Examples
--------
>>> from mdapy import data
>>> data.atomic_numbers["C"]
6
>>> data.chemical_symbols[8]
'O'
>>> data.vdw_radii[8]
1.5
>>> data.atomic_masses[26]
55.845
"""

import numpy as np
from typing import List, Dict


chemical_symbols: List[str] = [
    # 0
    "X",
    # 1
    "H",
    "He",
    # 2
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    # 3
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    # 4
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
    # 5
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
    # 6
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
    # 7
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
]

atomic_numbers: Dict[str, int] = {
    symbol: Z for Z, symbol in enumerate(chemical_symbols)
}

vdw_radii: np.ndarray = np.array(
    [
        np.nan,  # X
        1.2,  # H
        1.43,  # He [larger uncertainty]
        2.12,  # Li
        1.98,  # Be
        1.91,  # B
        1.77,  # C
        1.66,  # N
        1.5,  # O
        1.46,  # F
        1.58,  # Ne [larger uncertainty]
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
        np.nan,  # Pm
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
        np.nan,  # Po
        np.nan,  # At
        np.nan,  # Rn
        np.nan,  # Fr
        np.nan,  # Ra
        2.8,  # Ac [larger uncertainty]
        2.93,  # Th
        2.88,  # Pa [larger uncertainty]
        2.71,  # U
        2.82,  # Np
        2.81,  # Pu
        2.83,  # Am
        3.05,  # Cm [larger uncertainty]
        3.4,  # Bk [larger uncertainty]
        3.05,  # Cf [larger uncertainty]
        2.7,  # Es [larger uncertainty]
        np.nan,  # Fm
        np.nan,  # Md
        np.nan,  # No
        np.nan,  # Lr
    ]
)
vdw_radii.flags.writeable = False

atomic_masses: np.ndarray = np.array(
    [
        1.0,  # X
        1.008,  # H [1.00784, 1.00811]
        4.002602,  # He
        6.94,  # Li [6.938, 6.997]
        9.0121831,  # Be
        10.81,  # B [10.806, 10.821]
        12.011,  # C [12.0096, 12.0116]
        14.007,  # N [14.00643, 14.00728]
        15.999,  # O [15.99903, 15.99977]
        18.998403163,  # F
        20.1797,  # Ne
        22.98976928,  # Na
        24.305,  # Mg [24.304, 24.307]
        26.9815385,  # Al
        28.085,  # Si [28.084, 28.086]
        30.973761998,  # P
        32.06,  # S [32.059, 32.076]
        35.45,  # Cl [35.446, 35.457]
        39.948,  # Ar
        39.0983,  # K
        40.078,  # Ca
        44.955908,  # Sc
        47.867,  # Ti
        50.9415,  # V
        51.9961,  # Cr
        54.938044,  # Mn
        55.845,  # Fe
        58.933194,  # Co
        58.6934,  # Ni
        63.546,  # Cu
        65.38,  # Zn
        69.723,  # Ga
        72.630,  # Ge
        74.921595,  # As
        78.971,  # Se
        79.904,  # Br [79.901, 79.907]
        83.798,  # Kr
        85.4678,  # Rb
        87.62,  # Sr
        88.90584,  # Y
        91.224,  # Zr
        92.90637,  # Nb
        95.95,  # Mo
        97.90721,  # 98Tc
        101.07,  # Ru
        102.90550,  # Rh
        106.42,  # Pd
        107.8682,  # Ag
        112.414,  # Cd
        114.818,  # In
        118.710,  # Sn
        121.760,  # Sb
        127.60,  # Te
        126.90447,  # I
        131.293,  # Xe
        132.90545196,  # Cs
        137.327,  # Ba
        138.90547,  # La
        140.116,  # Ce
        140.90766,  # Pr
        144.242,  # Nd
        144.91276,  # 145Pm
        150.36,  # Sm
        151.964,  # Eu
        157.25,  # Gd
        158.92535,  # Tb
        162.500,  # Dy
        164.93033,  # Ho
        167.259,  # Er
        168.93422,  # Tm
        173.054,  # Yb
        174.9668,  # Lu
        178.49,  # Hf
        180.94788,  # Ta
        183.84,  # W
        186.207,  # Re
        190.23,  # Os
        192.217,  # Ir
        195.084,  # Pt
        196.966569,  # Au
        200.592,  # Hg
        204.38,  # Tl [204.382, 204.385]
        207.2,  # Pb
        208.98040,  # Bi
        208.98243,  # 209Po
        209.98715,  # 210At
        222.01758,  # 222Rn
        223.01974,  # 223Fr
        226.02541,  # 226Ra
        227.02775,  # 227Ac
        232.0377,  # Th
        231.03588,  # Pa
        238.02891,  # U
        237.04817,  # 237Np
        244.06421,  # 244Pu
        243.06138,  # 243Am
        247.07035,  # 247Cm
        247.07031,  # 247Bk
        251.07959,  # 251Cf
        252.0830,  # 252Es
        257.09511,  # 257Fm
        258.09843,  # 258Md
        259.1010,  # 259No
        262.110,  # 262Lr
        267.122,  # 267Rf
        268.126,  # 268Db
        271.134,  # 271Sg
        270.133,  # 270Bh
        269.1338,  # 269Hs
        278.156,  # 278Mt
        281.165,  # 281Ds
        281.166,  # 281Rg
        285.177,  # 285Cn
        286.182,  # 286Nh
        289.190,  # 289Fl
        289.194,  # 289Mc
        293.204,  # 293Lv
        293.208,  # 293Ts
        294.214,  # 294Og
    ]
)
atomic_masses.flags.writeable = False

"""
The elemental radius and colors come from OVITO for visualization. The original statements are as below:

https://gitlab.com/stuko/ovito/-/blob/master/src/ovito/particles/objects/ParticleType.cpp#L225-329

// Define default names, colors, and radii for some predefined particle types.
//
// Van der Waals radii have been adopted from the VMD software, which adopted them from A. Bondi, J. Phys. Chem., 68, 441 - 452, 1964,
// except the value for H, which was taken from R.S. Rowland & R. Taylor, J. Phys. Chem., 100, 7384 - 7391, 1996.
// For radii that are not available in either of these publications use r = 2.0.
// The radii for ions (Na, K, Cl, Ca, Mg, and Cs) are based on the CHARMM27 Rmin/2 parameters for (SOD, POT, CLA, CAL, MG, CES).
//
// Colors and covalent radii of elements marked with '//' have been adopted from OpenBabel.

Here we time 2 for radius.

Here is the code to extract the radius and color information:

res = [i.split(', ')[:5] for i in context.split('ParticleType::PredefinedChemicalType')[2:]]
ele_rgb, ele_radius = {}, {}
for i in res:
    ele, r, g, b, size = i
    #print(ele)
    ele = ele.split('(')[-1].split('"')[1]
    if '/' in r:
        L, R = r.split('(')[1].split('/')
        r = float(L[:-1]) / float(R[:-1])
        L, R = g.split('/')
        g = float(L[:-1]) / float(R[:-1])
        L, R = b.split('/')
        b = float(L[:-1]) / float(R[:-2])
    else:
        r = float(r.split()[1].replace('f', ''))
        g = float(g.split()[0].replace('f', ''))
        b = float(b.split()[0][:-1].replace('f', ''))
    size = float(size[:-1])
    ele_rgb[ele] = [int(r*255), int(g*255), int(b*255)]
    ele_radius[ele] = size
"""

ele_radius: Dict[str, float] = {
    "X": 2.0,
    "H": 0.92,
    "He": 2.44,
    "Li": 3.14,
    "Be": 2.94,
    "B": 4.02,
    "C": 1.54,
    "N": 1.48,
    "O": 1.48,
    "F": 1.48,
    "Ne": 1.48,
    "Na": 3.82,
    "Mg": 3.2,
    "Al": 2.86,
    "Si": 2.36,
    "P": 2.14,
    "S": 2.1,
    "Cl": 2.04,
    "Ar": 2.12,
    "K": 4.06,
    "Ca": 3.94,
    "Sc": 3.4,
    "Ti": 2.94,
    "V": 3.06,
    "Cr": 2.58,
    "Mn": 2.78,
    "Fe": 2.52,
    "Co": 2.5,
    "Ni": 2.5,
    "Cu": 2.56,
    "Zn": 2.74,
    "Ga": 3.06,
    "Ge": 2.44,
    "As": 2.38,
    "Se": 2.4,
    "Br": 2.4,
    "Kr": 3.96,
    "Rb": 4.4,
    "Sr": 4.3,
    "Y": 3.64,
    "Zr": 3.2,
    "Nb": 2.94,
    "Mo": 3.08,
    "Tc": 2.94,
    "Ru": 2.92,
    "Rh": 2.84,
    "Pd": 2.74,
    "Ag": 2.9,
    "Cd": 2.88,
    "In": 2.84,
    "Sn": 2.78,
    "Sb": 2.78,
    "Te": 2.76,
    "I": 2.78,
    "Xe": 2.8,
    "Cs": 4.88,
    "Ba": 4.3,
    "La": 4.14,
    "Ce": 4.08,
    "Pr": 4.06,
    "Nd": 4.02,
    "Pm": 3.98,
    "Sm": 3.96,
    "Eu": 3.96,
    "Gd": 3.92,
    "Tb": 3.88,
    "Dy": 3.84,
    "Ho": 3.84,
    "Er": 3.78,
    "Tm": 3.8,
    "Yb": 3.74,
    "Lu": 3.74,
    "Hf": 3.5,
    "Ta": 3.4,
    "W": 3.24,
    "Re": 3.02,
    "Os": 2.88,
    "Ir": 2.82,
    "Pt": 2.78,
    "Au": 2.88,
    "Hg": 2.64,
    "Tl": 2.9,
    "Pb": 2.94,
    "Bi": 2.92,
    "Po": 2.8,
    "At": 3.0,
    "Rn": 3.0,
    "Fr": 5.2,
}

ele_rgb: Dict[str, List[int]] = {
    "X": [255, 255, 255],
    "H": [255, 255, 255],
    "He": [217, 255, 255],
    "Li": [204, 128, 255],
    "Be": [193, 255, 0],
    "B": [255, 181, 181],
    "C": [144, 144, 144],
    "N": [48, 80, 248],
    "O": [255, 13, 13],
    "F": [127, 178, 255],
    "Ne": [178, 226, 244],
    "Na": [171, 92, 242],
    "Mg": [138, 255, 0],
    "Al": [191, 166, 166],
    "Si": [240, 200, 160],
    "P": [255, 127, 0],
    "S": [178, 178, 0],
    "Cl": [30, 239, 30],
    "Ar": [127, 209, 226],
    "K": [142, 63, 211],
    "Ca": [61, 255, 0],
    "Sc": [229, 229, 229],
    "Ti": [191, 194, 199],
    "V": [165, 165, 170],
    "Cr": [138, 153, 199],
    "Mn": [155, 122, 198],
    "Fe": [224, 102, 51],
    "Co": [240, 144, 160],
    "Ni": [80, 208, 80],
    "Cu": [200, 128, 51],
    "Zn": [125, 128, 176],
    "Ga": [194, 143, 143],
    "Ge": [102, 143, 143],
    "As": [188, 127, 226],
    "Se": [255, 160, 0],
    "Br": [165, 40, 40],
    "Kr": [92, 184, 209],
    "Rb": [112, 45, 175],
    "Sr": [0, 255, 38],
    "Y": [102, 152, 142],
    "Zr": [0, 255, 0],
    "Nb": [76, 178, 118],
    "Mo": [84, 181, 181],
    "Tc": [58, 158, 158],
    "Ru": [35, 142, 142],
    "Rh": [10, 124, 140],
    "Pd": [0, 105, 133],
    "Ag": [224, 224, 255],
    "Cd": [255, 216, 142],
    "In": [165, 117, 114],
    "Sn": [102, 127, 127],
    "Sb": [158, 99, 181],
    "Te": [211, 122, 0],
    "I": [147, 0, 147],
    "Xe": [66, 158, 175],
    "Cs": [86, 22, 142],
    "Ba": [0, 201, 0],
    "La": [112, 211, 255],
    "Ce": [255, 255, 198],
    "Pr": [216, 255, 198],
    "Nd": [198, 255, 198],
    "Pm": [163, 255, 198],
    "Sm": [142, 255, 198],
    "Eu": [96, 255, 198],
    "Gd": [68, 255, 198],
    "Tb": [48, 255, 198],
    "Dy": [30, 255, 198],
    "Ho": [0, 255, 155],
    "Er": [0, 229, 117],
    "Tm": [0, 211, 81],
    "Yb": [0, 191, 56],
    "Lu": [0, 170, 35],
    "Hf": [76, 193, 255],
    "Ta": [76, 165, 255],
    "W": [33, 147, 214],
    "Re": [38, 124, 170],
    "Os": [38, 102, 150],
    "Ir": [22, 84, 135],
    "Pt": [229, 216, 173],
    "Au": [255, 209, 35],
    "Hg": [181, 181, 193],
    "Tl": [165, 84, 76],
    "Pb": [87, 89, 97],
    "Bi": [158, 79, 181],
    "Po": [170, 91, 0],
    "At": [117, 79, 68],
    "Rn": [66, 130, 150],
    "Fr": [66, 0, 102],
}

struc_rgb: Dict[str, List[int]] = {
    "Other": [243, 243, 243],
    "FCC": [102, 255, 102],
    "HCP": [255, 102, 102],
    "BCC": [102, 102, 255],
    "ICO": [243, 204, 51],
    "Cubic diamond": [19, 160, 254],
    "Cubic diamond (1st neighbor)": [0, 254, 245],
    "Cubic diamond (2nd neighbor)": [126, 254, 181],
    "Hexagonal diamond": [254, 137, 0],
    "Hexagonal diamond (1st neighbor)": [254, 220, 0],
    "Hexagonal diamond (2nd neighbor)": [204, 229, 81],
    "Simple cubic": [160, 20, 254],
    "Graphene": [160, 120, 254],
    "Hexagonal ice": [0, 230, 230],
    "Cubic ice": [255, 193, 5],
    "Interfacial ice": [128, 30, 102],
    "Hydrate": [255, 76, 25],
    "Interfacial hydrate": [25, 255, 25],
}

type_rgb: Dict[int, List[int]] = {
    1: [255, 102, 102],
    2: [102, 102, 255],
    3: [255, 186, 25],
    4: [120, 120, 120],
    5: [214, 188, 39],
    6: [255, 102, 255],
    7: [179, 0, 255],
    8: [51, 255, 255],
    9: [102, 255, 102],
}

ele_dict: Dict[str, int] = {
    "X": 16777215,
    "H": 16777215,
    "He": 14286847,
    "Li": 13402367,
    "Be": 12713728,
    "B": 16758197,
    "C": 9474192,
    "N": 3166456,
    "O": 16715021,
    "F": 8368895,
    "Ne": 11723508,
    "Na": 11230450,
    "Mg": 9109248,
    "Al": 12560038,
    "Si": 15780000,
    "P": 16744192,
    "S": 11710976,
    "Cl": 2027294,
    "Ar": 8376802,
    "K": 9322451,
    "Ca": 4062976,
    "Sc": 15066597,
    "Ti": 12567239,
    "V": 10855850,
    "Cr": 9083335,
    "Mn": 10189510,
    "Fe": 14706227,
    "Co": 15765664,
    "Ni": 5296208,
    "Cu": 13140019,
    "Zn": 8224944,
    "Ga": 12750735,
    "Ge": 6721423,
    "As": 12353506,
    "Se": 16752640,
    "Br": 10823720,
    "Kr": 6076625,
    "Rb": 7351727,
    "Sr": 65318,
    "Y": 6723726,
    "Zr": 65280,
    "Nb": 5026422,
    "Mo": 5551541,
    "Tc": 3841694,
    "Ru": 2330254,
    "Rh": 687244,
    "Pd": 27013,
    "Ag": 14737663,
    "Cd": 16767118,
    "In": 10843506,
    "Sn": 6717311,
    "Sb": 10380213,
    "Te": 13859328,
    "I": 9633939,
    "Xe": 4365999,
    "Cs": 5641870,
    "Ba": 51456,
    "La": 7394303,
    "Ce": 16777158,
    "Pr": 14221254,
    "Nd": 13041606,
    "Pm": 10747846,
    "Sm": 9371590,
    "Eu": 6356934,
    "Gd": 4521926,
    "Tb": 3211206,
    "Dy": 2031558,
    "Ho": 65435,
    "Er": 58741,
    "Tm": 54097,
    "Yb": 48952,
    "Lu": 43555,
    "Hf": 5030399,
    "Ta": 5023231,
    "W": 2200534,
    "Re": 2522282,
    "Os": 2516630,
    "Ir": 1463431,
    "Pt": 15063213,
    "Au": 16765219,
    "Hg": 11908545,
    "Tl": 10835020,
    "Pb": 5724513,
    "Bi": 10375093,
    "Po": 11164416,
    "At": 7688004,
    "Rn": 4358806,
    "Fr": 4325478,
}

struc_dict: Dict[str, int] = {
    "Other": 15987699,
    "FCC": 6750054,
    "HCP": 16737894,
    "BCC": 6711039,
    "ICO": 15977523,
    "Cubic diamond": 1286398,
    "Cubic diamond (1st neighbor)": 65269,
    "Cubic diamond (2nd neighbor)": 8322741,
    "Hexagonal diamond": 16681216,
    "Hexagonal diamond (1st neighbor)": 16702464,
    "Hexagonal diamond (2nd neighbor)": 13428049,
    "Simple cubic": 10491134,
    "Graphene": 10516734,
    "Hexagonal ice": 59110,
    "Cubic ice": 16761093,
    "Interfacial ice": 8396390,
    "Hydrate": 16731161,
    "Interfacial hydrate": 1703705,
}

type_dict: Dict[int, int] = {
    1: 16737894,
    2: 6711039,
    3: 16759321,
    4: 7895160,
    5: 14072871,
    6: 16738047,
    7: 11731199,
    8: 3407871,
    9: 6750054,
}
