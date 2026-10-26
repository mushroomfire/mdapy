# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Synthetic EAM potentials for the tests, and a NumPy reference."""

import itertools
import math

import numpy as np

import mdapy

CUTOFF = 5.0
NR, DR = 1000, 0.0055  # tables reach beyond the cutoff
NRHO, DRHO = 1000, 0.05

PARAMETERS = {
    "Cu": {
        "f": 1.0,
        "beta": 5.0,
        "re": 2.55,
        "A": 0.04,
        "rho_e": 12.0,
        "E0": 3.5,
        "mass": 63.546,
    },
    "Ni": {
        "f": 1.2,
        "beta": 5.5,
        "re": 2.49,
        "A": 0.05,
        "rho_e": 14.0,
        "E0": 4.4,
        "mass": 58.6934,
    },
}


def smooth(r):
    return np.where(r < CUTOFF, (1 - (r / CUTOFF) ** 2) ** 2, 0.0)


def density(element, r):
    p = PARAMETERS[element]
    return p["f"] * np.exp(-p["beta"] * (r / p["re"] - 1)) * smooth(r)


def embedding(element, rho):
    p = PARAMETERS[element]
    return p["A"] * (rho - p["rho_e"]) ** 2 - p["E0"]


def pair(a, b, r):
    depth = 0.3 if a == b else 0.35
    alpha = 1.4
    r0 = 0.5 * (PARAMETERS[a]["re"] + PARAMETERS[b]["re"])
    return (
        depth
        * (np.exp(-2 * alpha * (r - r0)) - 2 * np.exp(-alpha * (r - r0)))
        * smooth(r)
    )


def tables(elements):
    r = np.arange(NR) * DR
    rho = np.arange(NRHO) * DRHO
    return {
        "embedding": [embedding(e, rho) for e in elements],
        "density": [density(e, r) for e in elements],
        "pair": [
            r * pair(a, b, r) for i, a in enumerate(elements) for b in elements[: i + 1]
        ],
    }


def write_setfl(path, elements=("Cu", "Ni"), per_line=5):
    data = tables(elements)

    def block(values):
        values = list(values)
        return "\n".join(
            " ".join(repr(float(v)) for v in values[k : k + per_line])
            for k in range(0, len(values), per_line)
        )

    lines = [
        "synthetic potential for the mdapy tests",
        "Morse pairs with smooth cutoffs",
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
    path.write_text("\n".join(lines) + "\n")
    return path


# ------------------------------------------------- NumPy reference of LAMMPS


def spline(f, delta):
    """The seven coefficients of LAMMPS pair eam, rows 1..n (row 0 unused)."""
    n = len(f)
    s = np.zeros((n + 1, 7))
    s[1:, 6] = f
    s[1, 5] = s[2, 6] - s[1, 6]
    s[2, 5] = 0.5 * (s[3, 6] - s[1, 6])
    s[n - 1, 5] = 0.5 * (s[n, 6] - s[n - 2, 6])
    s[n, 5] = s[n, 6] - s[n - 1, 6]
    for m in range(3, n - 1):
        s[m, 5] = (
            (s[m - 2, 6] - s[m + 2, 6]) + 8.0 * (s[m + 1, 6] - s[m - 1, 6])
        ) / 12.0
    for m in range(1, n):
        s[m, 4] = 3.0 * (s[m + 1, 6] - s[m, 6]) - 2.0 * s[m, 5] - s[m + 1, 5]
        s[m, 3] = s[m, 5] + s[m + 1, 5] - 2.0 * (s[m + 1, 6] - s[m, 6])
    s[1:, 2] = s[1:, 5] / delta
    s[1:, 1] = 2.0 * s[1:, 4] / delta
    s[1:, 0] = 3.0 * s[1:, 3] / delta
    return s


def evaluate(s, delta, x):
    n = len(s) - 1
    p = x / delta + 1.0
    m = max(1, min(int(p), n - 1))
    p = min(p - m, 1.0)
    c = s[m]
    return ((c[3] * p + c[4]) * p + c[5]) * p + c[6], (c[0] * p + c[1]) * p + c[2]


def reference(system, elements=("Cu", "Ni")):
    """Energies, forces and virials by brute force over periodic images."""
    data = tables(elements)
    f_s = [spline(t, DRHO) for t in data["embedding"]]
    rho_s = [spline(t, DR) for t in data["density"]]
    pair_s = [spline(t, DR) for t in data["pair"]]
    species = [elements.index(e) for e in system.table["element"].to_list()]
    x, box = system["positions"], system.box
    reach = [
        math.ceil(CUTOFF / t) + 1 if p else 0 for t, p in zip(box.thickness, box.pbc)
    ]
    images = [
        np.array(s) @ box.matrix
        for s in itertools.product(*(range(-k, k + 1) for k in reach))
    ]
    n = len(system)
    neighbors = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for shift in images:
                r = x[j] - x[i] + shift
                d = math.sqrt(r @ r)
                if 0 < d < CUTOFF:
                    neighbors[i].append((j, r, d))
    rho = [
        sum(evaluate(rho_s[species[j]], DR, d)[0] for j, _, d in neighbors[i])
        for i in range(n)
    ]
    energies, slopes = np.zeros(n), np.zeros(n)
    rho_max = (NRHO - 1) * DRHO
    for i in range(n):
        value, slope = evaluate(f_s[species[i]], DRHO, rho[i])
        energies[i] = value + (slope * (rho[i] - rho_max) if rho[i] > rho_max else 0.0)
        slopes[i] = slope
    forces, virials = np.zeros((n, 3)), np.zeros((n, 3, 3))
    for i in range(n):
        for j, r, d in neighbors[i]:
            a, b = max(species[i], species[j]), min(species[i], species[j])
            z2, z2p = evaluate(pair_s[a * (a + 1) // 2 + b], DR, d)
            _, rho_ip = evaluate(rho_s[species[i]], DR, d)
            _, rho_jp = evaluate(rho_s[species[j]], DR, d)
            phi = z2 / d
            psip = slopes[i] * rho_jp + slopes[j] * rho_ip + (z2p / d - phi / d)
            fpair = -psip / d
            forces[i] += -r * fpair
            virials[i] += 0.5 * np.outer(r, r) * fpair
            energies[i] += 0.5 * phi
    return energies, forces, virials


def alloy(seed=0, repeat=2, pbc=True, jitter=0.15, a=3.6):
    """A perturbed fcc alloy with lattice constant a (the potential's fcc Cu
    has its minimum near 3.3 Å)."""
    rng = np.random.default_rng(seed)
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    cells = np.array(list(itertools.product(range(repeat), repeat=3)))
    fractional = (cells[:, None, :] + basis[None]).reshape(-1, 3) / repeat
    box = mdapy.Box(a * repeat, pbc=pbc)
    positions = box.to_cartesian(fractional) + rng.normal(
        scale=jitter, size=(len(fractional), 3)
    )
    return mdapy.System(
        positions, box, elements=rng.choice(["Cu", "Ni"], len(fractional))
    )
