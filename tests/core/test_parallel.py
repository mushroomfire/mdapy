# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.
"""Behaviour of the compiled thread pool, exercised through test-only kernels."""

import multiprocessing
import os
import signal
import sys
import threading
import time

import numpy as np
import pytest

import mdapy
from mdapy import _core
from mdapy.errors import Cancelled

kernels = _core.testing


def chunked_sum(values, grain):
    """Reference: sequential sum per chunk, folded in chunk order."""
    if grain <= 0:
        grain = _core.core.default_grain(len(values))
    total = 0.0
    for begin in range(0, len(values), grain):
        part = 0.0
        for v in values[begin : begin + grain].tolist():
            part += v
        total += part
    return total


@pytest.fixture(scope="module")
def values():
    return np.random.default_rng(42).normal(size=100_003) * 1e3


# --- results -----------------------------------------------------------------


@pytest.mark.parametrize("grain", [0, 1, 7, 1000, 100_003, 10**9])
def test_sum_matches_reference_bit_for_bit(values, grain):
    mdapy.set_num_threads(4)
    assert kernels.parallel_sum(values, grain=grain) == chunked_sum(values, grain)


@pytest.mark.parametrize("grain", [0, 3, 997])
def test_sum_is_identical_for_any_thread_count(values, grain):
    results = set()
    for n in (1, 2, 3, 4, 8):
        mdapy.set_num_threads(n)
        results.add(kernels.parallel_sum(values, grain=grain))
    assert len(results) == 1


@pytest.mark.parametrize("n", [0, 1, 2, 511, 512, 513, 10_000])
@pytest.mark.parametrize("threads", [1, 4])
def test_square_in_place(n, threads):
    mdapy.set_num_threads(threads)
    x = np.arange(n, dtype=np.float64)
    kernels.parallel_square(x)
    np.testing.assert_array_equal(x, np.arange(n, dtype=np.float64) ** 2)


def test_square_rejects_read_only_array():
    x = np.ones(10)
    x.flags.writeable = False
    with pytest.raises(TypeError):
        kernels.parallel_square(x)


@pytest.mark.parametrize("n", [0, 1, 511, 512, 513, 10**6, 10**9])
def test_default_grain_depends_only_on_n(n):
    grain = _core.core.default_grain(n)
    assert grain >= 1
    assert (n + grain - 1) // grain <= 512


@pytest.mark.parametrize("threads", [1, 4])
def test_nested_regions(threads):
    mdapy.set_num_threads(threads)
    n_outer, n_inner = 37, 53
    expected = (n_outer * (n_outer - 1) // 2) * (n_inner * (n_inner - 1) // 2)
    assert kernels.nested_sum(n_outer, n_inner) == expected


def test_thread_count_is_respected():
    mdapy.set_num_threads(1)
    assert kernels.count_threads(8, 0.002) == 1
    mdapy.set_num_threads(4)
    used = kernels.count_threads(32, 0.01)
    assert 2 <= used <= 4


# --- errors -----------------------------------------------------------------


@pytest.mark.parametrize("threads", [1, 4])
def test_exception_in_kernel_propagates(threads):
    mdapy.set_num_threads(threads)
    with pytest.raises(RuntimeError, match="failure at item 37"):
        kernels.fail_at(100, 37)
    # the pool is still usable afterwards
    assert kernels.parallel_sum(np.ones(1000)) == 1000.0


# --- cancellation ------------------------------------------------------------


def test_pre_cancelled_token_stops_immediately():
    token = mdapy.CancelToken()
    token.cancel()
    start = time.perf_counter()
    with pytest.raises(Cancelled):
        kernels.sleep_items(1000, 0.01, cancel=token._native)
    assert time.perf_counter() - start < 1.0


@pytest.mark.parametrize("threads", [1, 4])
def test_cancel_from_progress_callback(threads):
    mdapy.set_num_threads(threads)
    token = mdapy.CancelToken()
    calls = []

    def progress(done, total):
        calls.append(done)
        token.cancel()

    start = time.perf_counter()
    with pytest.raises(Cancelled):
        kernels.sleep_items(2000, 0.005, progress=progress, cancel=token._native)
    assert time.perf_counter() - start < 2.0  # full run would take >= 2.5 s
    assert calls and calls[-1] < 2000


def test_cancel_from_another_thread():
    mdapy.set_num_threads(4)
    token = mdapy.CancelToken()
    timer = threading.Timer(0.1, token.cancel)
    timer.start()
    start = time.perf_counter()
    try:
        with pytest.raises(Cancelled):
            kernels.sleep_items(4000, 0.005, cancel=token._native)
    finally:
        timer.cancel()
    assert time.perf_counter() - start < 2.0  # full run would take >= 5 s


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal delivery")
def test_ctrl_c_interrupts_a_running_kernel():
    mdapy.set_num_threads(2)
    timer = threading.Timer(0.2, os.kill, args=(os.getpid(), signal.SIGINT))
    timer.start()
    start = time.perf_counter()
    try:
        with pytest.raises(KeyboardInterrupt):
            kernels.sleep_items(4000, 0.005)
    finally:
        timer.cancel()
    assert time.perf_counter() - start < 2.0  # full run would take >= 10 s


# --- progress ----------------------------------------------------------------


@pytest.mark.parametrize("threads", [1, 4])
def test_progress_reports_monotonic_counts_and_finishes_at_total(threads):
    mdapy.set_num_threads(threads)
    calls = []
    kernels.sleep_items(60, 0.005, progress=lambda d, t: calls.append((d, t)))
    assert calls[-1] == (60, 60)
    assert all(t == 60 for _, t in calls)
    done = [d for d, _ in calls]
    assert done == sorted(done)
    assert len(calls) >= 2  # at least one report before the end


def test_progress_exception_propagates_and_stops_work():
    mdapy.set_num_threads(4)

    def progress(done, total):
        raise ValueError("stop here")

    start = time.perf_counter()
    with pytest.raises(ValueError, match="stop here"):
        kernels.sleep_items(2000, 0.005, progress=progress)
    assert time.perf_counter() - start < 2.0


def test_progress_exception_on_final_report_propagates():
    mdapy.set_num_threads(1)

    def progress(done, total):
        if done == total:
            raise ValueError("final")

    with pytest.raises(ValueError, match="final"):
        kernels.parallel_sum(np.ones(10), progress=progress)


# --- concurrency ---------------------------------------------------------------


def test_gil_is_released_during_kernels():
    mdapy.set_num_threads(2)
    worker = threading.Thread(target=kernels.sleep_items, args=(40, 0.01))
    ticks = 0
    worker.start()
    while worker.is_alive():
        ticks += 1
        time.sleep(0.001)
    worker.join()
    assert ticks > 20  # the main thread kept running Python code


def test_concurrent_calls_from_python_threads(values):
    mdapy.set_num_threads(4)
    expected = chunked_sum(values, 0)
    results = [None] * 6

    def run(i):
        results[i] = kernels.parallel_sum(values)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(results))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [expected] * len(results)


def _sum_in_child(queue):
    queue.put(_core.testing.parallel_sum(np.arange(10_000, dtype=np.float64)))


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(), reason="needs fork"
)
# Forking a process that has started threads is what this test is about.
@pytest.mark.filterwarnings("ignore:.*fork.*:DeprecationWarning")
def test_pool_works_in_forked_child():
    mdapy.set_num_threads(4)
    kernels.parallel_sum(np.ones(100_000))  # start the pool in the parent
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    child = ctx.Process(target=_sum_in_child, args=(queue,))
    child.start()
    child.join(timeout=60)
    assert child.exitcode == 0
    assert queue.get(timeout=10) == float(sum(range(10_000)))
