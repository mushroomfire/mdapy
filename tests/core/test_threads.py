# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import os
import subprocess
import sys

import pytest

import mdapy
from mdapy.core.threads import _init_from_environment, _native_token
from mdapy.errors import InvalidParameter

BAD_COUNTS = [0, -1, 1.5, True, "2", None]


def test_default_is_positive():
    assert mdapy.get_num_threads() >= 1


@pytest.mark.parametrize("n", [1, 2, 7])
def test_set_and_get(n):
    mdapy.set_num_threads(n)
    assert mdapy.get_num_threads() == n


@pytest.mark.parametrize("n", BAD_COUNTS)
def test_set_rejects_non_positive_integers(n):
    before = mdapy.get_num_threads()
    with pytest.raises(InvalidParameter, match="positive integer"):
        mdapy.set_num_threads(n)
    assert mdapy.get_num_threads() == before


def test_context_manager_restores_previous_value():
    mdapy.set_num_threads(3)
    with mdapy.num_threads(1):
        assert mdapy.get_num_threads() == 1
    assert mdapy.get_num_threads() == 3


def test_context_manager_restores_on_error():
    mdapy.set_num_threads(3)
    with pytest.raises(RuntimeError):
        with mdapy.num_threads(2):
            raise RuntimeError("boom")
    assert mdapy.get_num_threads() == 3


@pytest.mark.parametrize("n", BAD_COUNTS)
def test_context_manager_rejects_bad_values_before_changing_anything(n):
    mdapy.set_num_threads(3)
    with pytest.raises(InvalidParameter):
        with mdapy.num_threads(n):
            pass  # pragma: no cover
    assert mdapy.get_num_threads() == 3


@pytest.mark.parametrize("value, expected", [("3", 3), (" 5 ", 5), ("1", 1)])
def test_environment_value_is_applied(value, expected):
    mdapy.set_num_threads(7)
    _init_from_environment({"MDAPY_NUM_THREADS": value})
    assert mdapy.get_num_threads() == expected


@pytest.mark.parametrize("environ", [{}, {"MDAPY_NUM_THREADS": ""}, {"OTHER": "2"}])
def test_missing_environment_value_changes_nothing(environ):
    mdapy.set_num_threads(7)
    _init_from_environment(environ)
    assert mdapy.get_num_threads() == 7


@pytest.mark.parametrize("value", ["abc", "0", "-2", "1.5"])
def test_invalid_environment_value_is_rejected(value):
    mdapy.set_num_threads(7)
    with pytest.raises(InvalidParameter, match="MDAPY_NUM_THREADS"):
        _init_from_environment({"MDAPY_NUM_THREADS": value})
    assert mdapy.get_num_threads() == 7


def _import_in_subprocess(env_value):
    env = dict(os.environ)
    env.pop("MDAPY_NUM_THREADS", None)
    if env_value is not None:
        env["MDAPY_NUM_THREADS"] = env_value
    code = "import mdapy; print(mdapy.get_num_threads())"
    return subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )


@pytest.mark.parametrize("value, expected", [("3", 3), (" 5 ", 5)])
def test_environment_variable_sets_initial_value(value, expected):
    result = _import_in_subprocess(value)
    assert result.returncode == 0, result.stderr
    assert int(result.stdout) == expected


@pytest.mark.parametrize("value", [None, ""])
def test_unset_or_empty_environment_variable_uses_all_cpus(value):
    result = _import_in_subprocess(value)
    assert result.returncode == 0, result.stderr
    if hasattr(os, "sched_getaffinity"):
        assert int(result.stdout) == len(os.sched_getaffinity(0))
    else:
        assert int(result.stdout) == (os.cpu_count() or 1)


LIMIT_TO_ONE_CPU = {
    "linux": "import os; os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})",
    "win32": (
        "import ctypes\n"
        "kernel = ctypes.WinDLL('kernel32')\n"
        "kernel.GetCurrentProcess.restype = ctypes.c_void_p\n"
        "kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]\n"
        "assert kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), 1)"
    ),
}


@pytest.mark.skipif(
    sys.platform not in LIMIT_TO_ONE_CPU, reason="a process cannot be limited to CPUs"
)
def test_the_default_follows_the_cpus_the_process_may_use():
    # a job of a cluster on a few CPUs of a large node, with threads for the node
    env = {k: v for k, v in os.environ.items() if k != "MDAPY_NUM_THREADS"}
    code = LIMIT_TO_ONE_CPU[sys.platform] + "\n"
    code += "import mdapy; print(mdapy.get_num_threads())"
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert int(result.stdout) == 1


@pytest.mark.parametrize("value", ["abc", "0", "-2", "1.5"])
def test_invalid_environment_variable_fails_import(value):
    result = _import_in_subprocess(value)
    assert result.returncode != 0
    assert "InvalidParameter" in result.stderr
    assert "MDAPY_NUM_THREADS" in result.stderr


def test_cancel_token_state():
    token = mdapy.CancelToken()
    assert not token.cancelled
    assert repr(token) == "CancelToken(cancelled=False)"
    token.cancel()
    assert token.cancelled
    token.cancel()  # idempotent
    assert token.cancelled
    token.reset()
    assert not token.cancelled


def test_native_token_conversion():
    assert _native_token(None) is None
    token = mdapy.CancelToken()
    assert _native_token(token) is token._native
    with pytest.raises(InvalidParameter, match="CancelToken"):
        _native_token(object())


@pytest.mark.skipif(not hasattr(os, "fork"), reason="the system has no fork")
def test_a_forked_process_is_warned_of_polars():
    code = (
        "import os, mdapy\n"
        "pid = os.fork()\n"
        "os._exit(0) if pid == 0 else os.waitpid(pid, 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "RuntimeWarning: this process was forked" in result.stderr
    assert 'get_context("spawn")' in result.stderr


def test_a_process_started_by_spawn_is_not_warned(tmp_path):
    script = tmp_path / "spawned.py"
    script.write_text(
        "import multiprocessing\n"
        "import mdapy\n"
        "def threads(_):\n"
        "    return mdapy.get_num_threads()\n"
        "if __name__ == '__main__':\n"
        "    with multiprocessing.get_context('spawn').Pool(1) as pool:\n"
        "        print(pool.map(threads, [0]))\n"
    )
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "forked" not in result.stderr
