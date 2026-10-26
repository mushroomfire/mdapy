# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Synthetic NEP potentials with random parameters for the tests."""

import numpy as np


def write_nep(
    path,
    elements=("Cu", "Ni"),
    *,
    zbl=None,
    cutoffs=((5.0, 4.0),),
    n_max=(4, 3),
    basis=(5, 4),
    l_max=(4, 2, 1),
    neurons=6,
    seed=0,
):
    """Writes a nep.txt file and returns its path.

    zbl is None, "universal", "typewise" or "flexible"; cutoffs is one
    (radial, angular) pair for all elements or one per element.
    """
    rng = np.random.default_rng(seed)
    types = len(elements)
    terms = l_max[0] + (l_max[1] == 2) + (l_max[2] == 1)
    dim = n_max[0] + 1 + (n_max[1] + 1) * terms
    lines = [f"nep4{'_zbl' if zbl else ''} {types} {' '.join(elements)}"]
    if zbl == "universal":
        lines.append("zbl 1.0 2.2")
    elif zbl == "typewise":
        lines.append("zbl 1.0 2.4 0.8")
    elif zbl == "flexible":
        lines.append("zbl 0 0")
    lines.append(
        "cutoff "
        + " ".join(f"{r} {a}" for r, a in cutoffs)
        + " 100 40"  # neighbor counts, not used by mdapy
    )
    lines.append(f"n_max {n_max[0]} {n_max[1]}")
    lines.append(f"basis_size {basis[0]} {basis[1]}")
    lines.append(f"l_max {l_max[0]} {l_max[1]} {l_max[2]}")
    lines.append(f"ANN {neurons} 0")
    values = []
    for _ in range(types):
        values += list(rng.normal(scale=1 / np.sqrt(dim), size=neurons * dim))
        values += list(rng.normal(scale=0.1, size=neurons))  # b0
        values += list(rng.normal(scale=0.5, size=neurons))  # w1
    values.append(0.3)  # b1
    for n, k in zip(n_max[:2], basis):
        values += list(rng.normal(scale=0.4, size=types * types * (n + 1) * (k + 1)))
    values += list(rng.uniform(0.2, 1.0, size=dim))  # q_scaler
    if zbl == "flexible":
        for _ in range(types * (types + 1) // 2):
            inner, outer = rng.uniform(0.6, 1.0), rng.uniform(1.8, 2.3)
            screening = np.array(
                [0.18175, 3.1998, 0.50986, 0.94229, 0.28022, 0.4029, 0.02817, 0.20162]
            )
            values += [inner, outer] + list(screening * rng.uniform(0.8, 1.2, size=8))
    lines += [repr(float(v)) for v in values]
    path.write_text("\n".join(lines) + "\n")
    return path
