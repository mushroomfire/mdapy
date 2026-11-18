# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Potential files of LAMMPS for the tests of the native calculators:
Tersoff and Stillinger–Weber files from published parameters and their
mixing rules, with entries that take the less common branches, and ADP files
of smooth functions."""

import itertools

import numpy as np

import mdapy

from .eam_files import CUTOFF, DR, DRHO, NR, NRHO, PARAMETERS, smooth, tables

# Tersoff, Phys. Rev. B 39, 5566 (1989): A B lambda1 lambda2 beta n c d h R D
TERSOFF = {
    "Si": (
        1830.8,
        471.18,
        2.4799,
        1.7322,
        1.1e-6,
        0.78734,
        1.0039e5,
        16.217,
        -0.59825,
        2.85,
        0.15,
    ),
    "C": (
        1393.6,
        346.74,
        3.4879,
        2.2119,
        1.5724e-7,
        0.72751,
        3.8049e4,
        4.3484,
        -0.57058,
        1.95,
        0.15,
    ),
    # an element of its own for the less common branches: m = 1, lambda3 > 0
    "Ge": (
        1769.0,
        419.23,
        2.4451,
        1.7047,
        9.0166e-7,
        0.75627,
        1.0643e5,
        15.652,
        -0.43884,
        2.95,
        0.15,
    ),
}
CHI = {("Si", "C"): 0.9776, ("Si", "Ge"): 1.00061, ("C", "Ge"): 0.95}


def tersoff_entry(i, j, k):
    """The 14 numbers of the entry i j k: the two-body parameters of the pair
    i j mixed by Tersoff's rules, the three-body ones of i, the cutoff of i k."""
    a_i, b_i, l1_i, l2_i, beta, n, c, d, h, r_i, d_i = TERSOFF[i]
    a_j, b_j, l1_j, l2_j, *_, r_j, d_j = TERSOFF[j]
    _, _, _, _, _, _, _, _, _, r_k, d_k = TERSOFF[k]
    chi = 1.0 if i == j else CHI.get((i, j), CHI.get((j, i)))
    m, lam3 = (1.0, 1.3) if i == "Ge" else (3.0, 0.0)
    if j == k:
        a, b = np.sqrt(a_i * a_j), chi * np.sqrt(b_i * b_j)
        l1, l2 = 0.5 * (l1_i + l1_j), 0.5 * (l2_i + l2_j)
        r, dd = np.sqrt(r_i * r_j), 0.5 * (d_i + d_j)
    else:  # the three-body terms of k use its cutoff from i alone
        a = b = l1 = l2 = 0.0
        r, dd = np.sqrt(r_i * r_k), 0.5 * (d_i + d_k)
    if j != k:
        beta, n = 0.0, 0.0
    return (m, 1.0, lam3, c, d, h, n, beta, l2, b, r, dd, l1, a)


def write_tersoff(path, elements=("Si", "C"), entry=tersoff_entry):
    lines = ["# DATE: 2026-11-17 UNITS: metal CONTRIBUTOR: MDAPY tests", ""]
    for i, j, k in itertools.product(elements, repeat=3):
        numbers = entry(i, j, k)
        lines.append(f"{i} {j} {k} " + " ".join(repr(float(v)) for v in numbers[:7]))
        lines.append("    " + " ".join(repr(float(v)) for v in numbers[7:]))
    path.write_text("\n".join(lines) + "\n")
    return path


# Stillinger and Weber, Phys. Rev. B 31, 5262 (1985), Si: epsilon sigma a
# lambda gamma cos(theta0) A B p q tol; two more elements of their own
SW = {
    "Si": (
        2.1683,
        2.0951,
        1.80,
        21.0,
        1.20,
        -1 / 3,
        7.049556277,
        0.6022245584,
        4.0,
        0.0,
        0.0,
    ),
    "Ge": (
        1.93,
        2.181,
        1.80,
        31.0,
        1.20,
        -1 / 3,
        7.049556277,
        0.6022245584,
        4.0,
        0.0,
        0.0,
    ),
    "C": (2.5, 1.6, 1.85, 18.0, 1.1, -0.30, 6.5, 0.7, 4.5, 1.0, 0.0),
}


def sw_entry(i, j, k):
    """The 11 numbers of the entry i j k: two-body numbers of the pair i j as
    means, three-body ones of the centre i, its strength the geometric mean
    of those of j and k."""
    pi, pj, pk = np.array(SW[i]), np.array(SW[j]), np.array(SW[k])
    two = 0.5 * (pi + pj)
    epsilon = np.sqrt(pi[0] * pj[0]) if j == k else np.sqrt(pj[0] * pk[0])
    sigma = two[1] if j == k else 0.5 * (pj[1] + pk[1])
    return (
        epsilon,
        sigma,
        two[2],
        pi[3],
        two[4],
        pi[5],
        two[6],
        two[7],
        two[8],
        two[9],
        two[10],
    )


def write_sw(path, elements=("Si", "Ge"), entry=sw_entry, tol=0.0):
    lines = ["# DATE: 2026-11-17 UNITS: metal CONTRIBUTOR: MDAPY tests", ""]
    for i, j, k in itertools.product(elements, repeat=3):
        numbers = list(entry(i, j, k))
        numbers[10] = tol
        lines.append(f"{i} {j} {k} " + " ".join(repr(float(v)) for v in numbers))
    path.write_text("\n".join(lines) + "\n")
    return path


def dipole(a, b, r):
    """u(r) of a pair, smooth to zero at the cutoff."""
    scale = 0.05 if a == b else 0.08
    return scale * np.exp(-1.2 * (r - 2.5)) * smooth(r)


def quadrupole(a, b, r):
    """w(r) of a pair, smooth to zero at the cutoff."""
    scale = -0.02 if a == b else 0.03
    return scale * np.exp(-0.8 * (r - 2.5)) * smooth(r)


def write_adp(path, elements=("Cu", "Ni"), per_line=5):
    data = tables(elements)
    r = np.arange(NR) * DR

    def block(values):
        values = list(values)
        return "\n".join(
            " ".join(repr(float(v)) for v in values[k : k + per_line])
            for k in range(0, len(values), per_line)
        )

    lines = [
        "synthetic angular-dependent potential for the mdapy tests",
        "the EAM of eam_files with smooth dipole and quadrupole functions",
        "",
        f"{len(elements)} {' '.join(elements)}",
        f"{NRHO} {DRHO!r} {NR} {DR!r} {CUTOFF!r}",
    ]
    for index, element in enumerate(elements):
        z = mdapy.elements.atomic_numbers([element])[0]
        lines.append(f"{z} {PARAMETERS[element]['mass']!r} 3.6 fcc")
        lines.append(block(data["embedding"][index]))
        lines.append(block(data["density"][index]))
    lines.extend(block(values) for values in data["pair"])
    for function in (dipole, quadrupole):
        for i, a in enumerate(elements):
            for b in elements[: i + 1]:
                lines.append(block(function(a, b, r)))
    path.write_text("\n".join(lines) + "\n")
    return path
