# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Processes of MPI stood in for by threads, for tests without MPI: each has
a communicator with the calls of mpi4py's that MDAPY makes, every message
pickled as mpi4py pickles it, and MDAPY's dynamics in it takes that
communicator for its world."""

import pickle
import queue
import threading
from typing import Any, Callable

import numpy as np

from mdapy.simulation import _domain

# seconds a process waits for the others before the test fails
TIMEOUT = 300


class _World:
    def __init__(self, size: int):
        self.size = size
        self.mail = {(a, b): queue.Queue() for a in range(size) for b in range(size)}
        # the messages of buffers of each tag, apart from the others
        self.tagged: dict = {}
        self.lock = threading.Lock()
        self.barrier = threading.Barrier(size, timeout=TIMEOUT)
        self.slots: list = [None] * size

    def box(self, source: int, dest: int, tag: int) -> queue.Queue:
        with self.lock:
            return self.tagged.setdefault((source, dest, tag), queue.Queue())


class _Done:
    """A request of mpi4py's that is done when it is waited for."""

    def __init__(self, wait: Callable[[], None] = lambda: None):
        self.__wait = wait

    def Wait(self) -> None:  # noqa: N802
        self.__wait()


class Comm:
    """The communicator of one process."""

    def __init__(self, world: _World, rank: int):
        self.__world, self.__rank = world, rank

    def Get_size(self) -> int:  # noqa: N802, as mpi4py names it
        return self.__world.size

    def Get_rank(self) -> int:  # noqa: N802
        return self.__rank

    def send(self, value: Any, dest: int) -> None:
        self.__world.mail[(self.__rank, dest)].put(pickle.dumps(value))

    def recv(self, source: int) -> Any:
        return pickle.loads(
            self.__world.mail[(source, self.__rank)].get(timeout=TIMEOUT)
        )

    def sendrecv(self, value: Any, dest: int, source: int) -> Any:
        self.send(value, dest)
        return self.recv(source)

    def Sendrecv(  # noqa: N802
        self, sendbuf: Any, dest: Any = None, recvbuf: Any = None, source: Any = None
    ) -> None:
        """As mpi4py's, of buffers, None the process that is not."""
        if dest is not None:
            self.send(np.array(sendbuf), dest)
        if source is not None:
            recvbuf[...] = self.recv(source)

    def Isend(self, sendbuf: Any, dest: int, tag: int = 0) -> _Done:  # noqa: N802
        """As mpi4py's: sent at once, through the box of its tag."""
        self.__world.box(self.__rank, dest, tag).put(pickle.dumps(np.array(sendbuf)))
        return _Done()

    def Irecv(self, recvbuf: Any, source: int, tag: int = 0) -> _Done:  # noqa: N802
        """As mpi4py's: received when waited for."""
        box = self.__world.box(source, self.__rank, tag)

        def wait() -> None:
            recvbuf[...] = pickle.loads(box.get(timeout=TIMEOUT))

        return _Done(wait)

    def Allreduce(self, sendbuf: Any, recvbuf: Any, op: str) -> None:  # noqa: N802
        """As mpi4py's, of buffers, the operation "sum" or "max"."""
        parts = self.allgather(np.array(sendbuf))
        recvbuf[...] = np.sum(parts, axis=0) if op == "sum" else np.max(parts, axis=0)

    def allgather(self, value: Any) -> list:
        world = self.__world
        world.barrier.wait()
        world.slots[self.__rank] = pickle.dumps(value)
        world.barrier.wait()
        out = [pickle.loads(slot) for slot in world.slots]
        world.barrier.wait()
        return out

    def allreduce(self, value: Any) -> Any:
        parts = self.allgather(value)
        total = parts[0]
        for part in parts[1:]:
            total = total + part
        return total

    def bcast(self, value: Any, root: int = 0) -> Any:
        return self.allgather(value)[root]

    def Barrier(self) -> None:  # noqa: N802
        self.__world.barrier.wait()


class Alone:
    """A communicator of one process, as mpi4py's MPI.COMM_SELF."""

    def Get_size(self) -> int:  # noqa: N802
        return 1

    def Get_rank(self) -> int:  # noqa: N802
        return 0


_here = threading.local()


def world() -> Comm | None:
    """The communicator of the process this thread stands for; None in the
    thread of the test, one process."""
    return getattr(_here, "comm", None)


def run(size: int, task: Callable[[int], Any], monkeypatch: Any) -> list:
    """task(rank) on size processes, each a thread, MDAPY's dynamics in each
    split among them; the results in the order of the ranks. An error of
    one process is raised, the others stopped."""
    monkeypatch.setattr(_domain, "world", world)
    shared = _World(size)
    results: list = [None] * size
    errors: list = [None] * size

    def main(rank: int) -> None:
        _here.comm = Comm(shared, rank)
        try:
            results[rank] = task(rank)
        except BaseException as error:  # noqa: BLE001, raised below
            errors[rank] = error
            shared.barrier.abort()

    threads = [threading.Thread(target=main, args=(r,)) for r in range(size)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    first = next(
        (
            e
            for e in errors
            if e is not None and not isinstance(e, threading.BrokenBarrierError)
        ),
        next((e for e in errors if e is not None), None),
    )
    if first is not None:
        raise first
    return results
