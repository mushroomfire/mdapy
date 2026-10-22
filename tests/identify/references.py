# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Plain Python references of common neighbor analysis and the
centro-symmetry parameter, written from their definitions."""

import itertools

import numpy as np

import mdapy

NAMES = ("other", "fcc", "hcp", "bcc", "ico")


def neighbor_vectors(system, k):
    """The k nearest neighbor vectors of each atom, nearest first."""
    nl = mdapy.NeighborList.build(system, num_neighbors=k)
    return nl.vectors.reshape(len(system), k, 3)


def signatures(vectors, cutoff):
    """CNA signature (common, bonds, longest chain) of each neighbor."""
    m = len(vectors)
    bonded = {
        (a, b)
        for a, b in itertools.combinations(range(m), 2)
        if np.linalg.norm(vectors[a] - vectors[b]) <= cutoff
    }

    def linked(a, b):
        return (min(a, b), max(a, b)) in bonded

    result = []
    for j in range(m):
        common = [k for k in range(m) if k != j and linked(j, k)]
        bonds = [(a, b) for a, b in itertools.combinations(common, 2) if linked(a, b)]
        # group the bonds that share atoms
        longest, left = 0, set(bonds)
        while left:
            group = {left.pop()}
            grew = True
            while grew:
                atoms = {x for bond in group for x in bond}
                joined = {bond for bond in left if atoms & set(bond)}
                left -= joined
                group |= joined
                grew = bool(joined)
            longest = max(longest, len(group))
        result.append((len(common), len(bonds), longest))
    return sorted(result)


def classify(sigs):
    if sigs == [(4, 2, 1)] * 12:
        return "fcc"
    if sigs == [(4, 2, 1)] * 6 + [(4, 2, 2)] * 6:
        return "hcp"
    if sigs == [(5, 5, 5)] * 12:
        return "ico"
    if sigs == [(4, 4, 4)] * 6 + [(6, 6, 6)] * 8:
        return "bcc"
    return "other"


def adaptive_cna(system):
    """Structure name of each atom of a periodic system."""
    names = []
    for v in neighbor_vectors(system, 14):
        d = np.linalg.norm(v, axis=1)
        local = (1 + np.sqrt(2)) / 2 * d[:12].mean()
        name = classify(signatures(v[:12], local))
        if name == "other":
            local = (
                (1 + np.sqrt(2)) / 2 * (2 / np.sqrt(3) * d[:8].sum() + d[8:].sum()) / 14
            )
            name = classify(signatures(v, local))
            name = name if name == "bcc" else "other"
        names.append(name)
    return names


def centro_symmetry(system, k):
    """Sum of the k / 2 smallest |r_a + r_b|^2 over pairs of the k nearest."""
    values = []
    for v in neighbor_vectors(system, k):
        pairs = sorted(
            float(np.sum((v[a] + v[b]) ** 2))
            for a, b in itertools.combinations(range(k), 2)
        )
        values.append(sum(pairs[: k // 2]))
    return np.array(values)


def diamond(system):
    """Diamond type of each atom: the CNA of its twelve second neighbors,
    when it is among the four nearest of each of its neighbors."""
    nl = mdapy.NeighborList.build(system, num_neighbors=4)
    v = nl.vectors.reshape(len(system), 4, 3)
    j = nl.j.reshape(len(system), 4)
    names = []
    for i in range(len(system)):
        second, whole = [], True
        for a in range(4):
            reach = v[i, a] + v[j[i, a]]
            back = np.argmin(np.linalg.norm(reach, axis=1))
            # the atom must be among the four nearest of each of its neighbors
            whole = whole and np.linalg.norm(reach[back]) < 1e-2
            second += [reach[b] for b in range(4) if b != back]
        if not whole:
            names.append("other")
            continue
        second = np.array(second)
        local = (1 + np.sqrt(2)) / 2 * np.linalg.norm(second, axis=1).mean()
        name = classify(signatures(second, local))
        names.append(
            {"fcc": "cubic_diamond", "hcp": "hexagonal_diamond"}.get(name, "other")
        )
    return names


def chill_plus(system, cutoff):
    """CHILL+ types, with the q_3m correlations from the addition theorem:
    sum_m q_3m(i) q_3m(j)* is 7 / (4 pi) times the sum of P_3 over the
    bonds of i and j."""
    nl = mdapy.NeighborList.build(system, cutoff).within(cutoff)
    bonds = []
    for i in range(len(system)):
        v = nl.vectors[nl.offsets[i] : nl.offsets[i + 1]]
        bonds.append(v / np.linalg.norm(v, axis=1, keepdims=True))

    def overlap(i, j):
        if len(bonds[i]) == 0 or len(bonds[j]) == 0:
            return 0.0
        x = bonds[i] @ bonds[j].T
        return (0.5 * (5 * x**3 - 3 * x)).sum()  # P_3

    names = []
    for i in range(len(system)):
        eclipsed = staggered = 0
        for j in nl.j[nl.offsets[i] : nl.offsets[i + 1]]:
            scale = np.sqrt(overlap(i, i) * overlap(j, j))
            c = overlap(i, j) / scale if scale > 0 else 0.0
            eclipsed += -0.35 < c < 0.25
            staggered += c < -0.8
        name = "other"
        if nl.counts[i] == 4:
            if eclipsed == 4:
                name = "hydrate"
            elif eclipsed == 3:
                name = "interfacial_hydrate"
            elif staggered == 4:
                name = "cubic_ice"
            elif staggered == 3 and eclipsed == 1:
                name = "hexagonal_ice"
            elif staggered in (2, 3):
                name = "interfacial_ice"
        names.append(name)
    return names


def ackland_jones(system, cutoff=6.0):
    """Ackland-Jones types from all neighbors within 1.55 r0^2, as in the
    paper, and the largest number of them, n1, of any atom."""
    nl = mdapy.NeighborList.build(system, cutoff)
    bins = [-np.inf, -0.945, -0.915, -0.755, -0.195, 0.195, 0.245, 0.795, np.inf]
    types, largest = [], 0
    for i in range(len(system)):
        v = nl.vectors[nl.offsets[i] : nl.offsets[i + 1]]
        r2 = (v * v).sum(1)
        order = np.argsort(r2, kind="stable")
        v, r2 = v[order], r2[order]
        r0 = r2[:6].mean()
        n0, n1 = int((r2 < 1.45 * r0).sum()), int((r2 < 1.55 * r0).sum())
        largest = max(largest, n1)
        u = v[:n0] / np.sqrt(r2[:n0])[:, None]
        chi = np.histogram((u @ u.T)[np.triu_indices(n0, 1)], bins)[0]
        cp = abs(1 - chi[6] / 24)
        below = chi[5] + chi[6] - chi[4]
        bcc = 0.35 * chi[4] / below if below else np.inf
        fcc = 0.61 * (abs(chi[0] + chi[1] - 6) + chi[2]) / 6
        hcp = (abs(chi[0] - 3) + abs(chi[0] + chi[1] + chi[2] + chi[3] - 9)) / 12
        if chi[0] == 7:
            bcc = 0.0
        elif chi[0] == 6:
            fcc = 0.0
        elif chi[0] <= 3:
            hcp = 0.0
        if chi[7] > 0:
            name = "other"
        elif chi[4] < 3:
            name = "ico" if 11 <= n1 <= 13 else "other"
        elif bcc <= cp:
            name = "bcc" if n1 >= 11 else "other"
        elif not 11 <= n1 <= 12:
            name = "other"
        else:
            name = "fcc" if fcc < hcp else "hcp"
        types.append(NAMES.index(name))
    return np.array(types), largest
