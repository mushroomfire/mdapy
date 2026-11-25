# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The augmentation charges of an ultrasoft pseudopotential as the GPU takes
them: the projector functions a = (p, m) of an atom in the order of the
projectors and of m from -l to l, their pairs a <= b, and for each pair the
terms (L, M) of Q_ab(G) = sum_L (-i)^L Q^L_pq(|G|) sum_M c(a, b, L M) Y_LM(G)
whose Gaunt coefficient c is not zero, with the radial table of each."""

import numpy as np

from . import angular, radial


def functions(pp) -> list[tuple[int, int, int]]:
    """(p, l, m) of the projector functions of an atom."""
    return [(p, beta.angular, m) for p, beta in enumerate(pp.projectors)
            for m in range(-beta.angular, beta.angular + 1)]  # fmt: skip


def terms(pp, qmax: float, volume: float) -> dict:
    """The augmentation of pp for the GPU: functions, pair_start, term_lm,
    term_l, term_table, term_coef, lmax, points and tables (tables, points)."""
    f = functions(pp)
    keys, tables = radial.augmentation_tables(pp, qmax, volume)
    where = {k: i for i, k in enumerate(keys)}
    lmax = pp.lmax
    gaunt = angular.gaunt(lmax)
    start, lm, ls, table, coef = [0], [], [], [], []
    for a, (pa, la, ma) in enumerate(f):
        for pb, lb, mb in f[a:]:
            key = (min(pa, pb), max(pa, pb))
            for ell in range(abs(la - lb), la + lb + 1, 2):
                if (*key, ell) not in where:
                    continue
                for m in range(-ell, ell + 1):
                    c = gaunt[
                        angular.index(la, ma),
                        angular.index(lb, mb),
                        angular.index(ell, m),
                    ]
                    if abs(c) > 1e-12:
                        lm.append(angular.index(ell, m))
                        ls.append(ell)
                        table.append(where[(*key, ell)])
                        coef.append(float(c))
            start.append(len(lm))
    return {
        "functions": len(f), "pair_start": start, "term_lm": lm, "term_l": ls,
        "term_table": table, "term_coef": coef, "lmax": 2 * lmax,
        "points": tables.shape[1], "tables": np.ascontiguousarray(tables),
    }  # fmt: skip


def pairs(pp) -> list[tuple[int, int]]:
    """The pairs (a, b), a <= b, of the projector functions, in their order."""
    n = len(functions(pp))
    return [(a, b) for a in range(n) for b in range(a, n)]
