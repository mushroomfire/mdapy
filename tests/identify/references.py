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
    nl = mdapy.NeighborList.nearest(system, k)
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
