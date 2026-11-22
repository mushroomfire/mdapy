# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Calculators: energies, forces and virials of systems."""

from mdapy.calculators._adp import ADP
from mdapy.calculators._base import PROPERTIES, Calculator, CalculatorResult
from mdapy.calculators._dft.calculator import DFT
from mdapy.calculators._eam import EAM
from mdapy.calculators._lammps import LAMMPS
from mdapy.calculators._meam import MEAM
from mdapy.calculators._nep import NEP
from mdapy.calculators._overlay import Overlay
from mdapy.calculators._pair import ZBL, LennardJones, Morse, UhlenbeckFord
from mdapy.calculators._springs import Springs
from mdapy.calculators._three_body import StillingerWeber, Tersoff
from mdapy.calculators._vasp import VASP

__all__ = [
    "ADP",
    "Calculator",
    "CalculatorResult",
    "DFT",
    "EAM",
    "LAMMPS",
    "LennardJones",
    "MEAM",
    "Morse",
    "Overlay",
    "PROPERTIES",
    "Springs",
    "StillingerWeber",
    "Tersoff",
    "UhlenbeckFord",
    "NEP",
    "VASP",
    "ZBL",
]
