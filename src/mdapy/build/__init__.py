# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Structure builder: crystals, alloys and defects."""

from mdapy.build._alloy import random_alloy
from mdapy.build._crystal import STRUCTURES, build_crystal
from mdapy.build._defects import (
    add_interstitials,
    add_vacancies,
    add_vacuum,
    slip,
    substitute,
)
from mdapy.build._dislocation import edge_dislocation, insert_dislocation
from mdapy.build._graphene import add_graphene
from mdapy.build._polycrystal import build_polycrystal, random_rotations, twin_lamellae
from mdapy.build._sqs import sqs_alloy

__all__ = [
    "STRUCTURES",
    "add_graphene",
    "add_interstitials",
    "add_vacancies",
    "add_vacuum",
    "build_crystal",
    "build_polycrystal",
    "edge_dislocation",
    "insert_dislocation",
    "random_alloy",
    "random_rotations",
    "slip",
    "sqs_alloy",
    "substitute",
    "twin_lamellae",
]
