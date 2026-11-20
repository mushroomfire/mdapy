# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Run under mpirun: the bricks of simulation/_domain.py hold each atom on
one process, and as ghosts exactly the periodic images of the atoms within
the reach of each brick. Prints "ok" on the first process, or raises.

    mpirun -np 4 python tests/mpi/check_bricks.py
"""

import itertools

import numpy as np
from mpi4py import MPI

from mdapy.simulation import _domain as domain

comm = MPI.COMM_WORLD
rng = np.random.default_rng(7)
for case, (h, pbc, reach) in enumerate(
    [
        (np.diag([20.0, 18.0, 16.0]), (True, True, True), 3.0),
        (
            np.array([[20.0, 0, 0], [4.0, 18.0, 0], [-3.0, 2.5, 16.0]]),
            (True, True, True),
            2.5,
        ),
        (np.diag([20.0, 18.0, 30.0]), (True, True, False), 3.0),
        (np.diag([24.0, 24.0, 24.0]), (False, False, False), 3.0),
    ]
):
    n = 900
    lo = np.array([-1.0, 2.0, 0.5])
    s = rng.random((n, 3))
    x = lo + s @ h
    ids = np.arange(n, dtype=np.int64)
    try:
        bricks = domain.Bricks(comm, h, lo, pbc, reach)
    except ValueError as error:
        if comm.Get_rank() == 0:
            print(f"case {case}: {error}")
        continue
    # the atoms of this brick
    f = bricks.fractional(x)
    brick = np.clip(np.floor(f * bricks.counts), 0, bricks.counts - 1).astype(int)
    mine = (brick == bricks.at).all(axis=1)
    fields = {
        "positions": x[mine],
        "image": np.zeros((mine.sum(), 3), np.int64),
        "id": ids[mine],
    }
    # move every atom a little, some across the boundaries, then migrate
    width = (1.0 / np.linalg.norm(np.linalg.inv(h), axis=0)) / bricks.counts
    step = rng.normal(0, 0.15, (n, 3)) * np.minimum(width.min() / 4, 1.0)
    fields["positions"] = fields["positions"] + step[fields["id"]]
    moved = x + step
    fields = bricks.migrate(fields)
    comm.Barrier()
    assert not comm.iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG), (
        f"case {case}: a message left by migrate"
    )
    owned = comm.allgather(fields["id"])
    every = np.concatenate(owned)
    assert sorted(every.tolist()) == list(range(n)), (
        f"case {case}: {len(every)} atoms, {len(set(every.tolist()))} distinct"
    )
    # every atom's unwrapped place is kept
    np.testing.assert_allclose(
        fields["positions"] + fields["image"] @ h, moved[fields["id"]], atol=1e-9
    )
    # ghosts against all images
    xg, ig = bricks.borders(fields["positions"], fields["id"])
    comm.Barrier()
    assert not comm.iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG), (
        f"case {case}: a message left by borders"
    )
    local = bricks.local
    assert local == len(fields["id"])
    lower, upper = bricks.at / bricks.counts, (bricks.at + 1) / bricks.counts
    frac = bricks.fraction
    wrapped = comm.allgather(fields["positions"])
    wids = comm.allgather(fields["id"])
    allx, alli = np.concatenate(wrapped), np.concatenate(wids)
    expect = []
    ranges = [(-1, 0, 1) if p else (0,) for p in pbc]
    for shift in itertools.product(*ranges):
        y = allx + np.array(shift) @ h
        g = bricks.fractional(y)
        open_lower = np.where(
            ~np.asarray(pbc) & (bricks.at == 0), -np.inf, lower - frac
        )
        open_upper = np.where(
            ~np.asarray(pbc) & (bricks.at == bricks.counts - 1), np.inf, upper + frac
        )
        near = ((g >= open_lower) & (g < open_upper)).all(axis=1)
        # the atoms of this brick, whose own places are not ghosts; along a
        # direction that is not periodic the end bricks own what left the box
        own = np.isin(alli, fields["id"]) & (np.array(shift) == 0).all()
        for k in np.flatnonzero(near & ~own):
            expect.append((int(alli[k]), *np.round(y[k], 9)))
    got = [(int(i), *np.round(p, 9)) for i, p in zip(ig[local:], xg[local:])]
    assert sorted(got) == sorted(expect), (
        f"case {case} rank {comm.Get_rank()}: {len(got)} ghosts, expected {len(expect)}"
    )
    # the ghosts follow their atoms
    # (drawn for every atom, so that all processes keep the same random numbers)
    nudge = rng.normal(0, 0.01, (n, 3))
    followed = bricks.follow(
        np.concatenate([xg[:local] + nudge[ig[:local]], xg[local:]])
    )
    np.testing.assert_allclose(
        followed[local:], xg[local:] + nudge[ig[local:]], atol=1e-12
    )
    assert bricks.total(local) == n
    # no message left behind for the next case
    comm.Barrier()
    assert not comm.iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG), (
        f"case {case}: a message left"
    )
if comm.Get_rank() == 0:
    print("ok", comm.Get_size(), "processes")
