# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Control of the threads used by mdapy's compiled kernels."""

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

from mdapy import _core
from mdapy.errors import InvalidParameter

__all__ = [
    "CancelToken",
    "get_num_threads",
    "set_num_threads",
    "num_threads",
]

ENV_NUM_THREADS = "MDAPY_NUM_THREADS"
"""Environment variable read at import to set the initial number of threads."""


def _validate(n: object, name: str) -> int:
    if isinstance(n, bool) or not isinstance(n, int):
        raise InvalidParameter(f"{name} must be a positive integer, got {n!r}")
    if n < 1:
        raise InvalidParameter(f"{name} must be a positive integer, got {n}")
    return n


def get_num_threads() -> int:
    """Number of threads used by compiled kernels.

    Returns
    -------
    int
        The current setting. Defaults to the number of logical CPUs, or to
        the ``MDAPY_NUM_THREADS`` environment variable if it was set when
        mdapy was imported.
    """
    return _core.core.get_num_threads()


def set_num_threads(n: int) -> None:
    """Set the number of threads used by compiled kernels.

    The setting is global to the process and takes effect at the next
    computation.

    Parameters
    ----------
    n : int
        Number of threads, at least 1.

    Raises
    ------
    InvalidParameter
        If ``n`` is not a positive integer.
    """
    _core.core.set_num_threads(_validate(n, "n"))


@contextmanager
def num_threads(n: int) -> Iterator[None]:
    """Use ``n`` threads inside a ``with`` block.

    The previous setting is restored when the block exits, also on error.
    Like :func:`set_num_threads`, the setting is global to the process.

    Parameters
    ----------
    n : int
        Number of threads, at least 1.

    Raises
    ------
    InvalidParameter
        If ``n`` is not a positive integer.
    """
    n = _validate(n, "n")
    previous = get_num_threads()
    set_num_threads(n)
    try:
        yield
    finally:
        set_num_threads(previous)


class CancelToken:
    """Flag for stopping a running computation from another thread.

    Pass the token to a computation that accepts ``cancel=``. Calling
    :meth:`cancel` from any thread makes that computation stop at the next
    chunk of work and raise :class:`mdapy.errors.Cancelled`.
    """

    def __init__(self) -> None:
        self._native = _core.core.CancelToken()

    def cancel(self) -> None:
        """Request cancellation."""
        self._native.cancel()

    def reset(self) -> None:
        """Clear the request so the token can be used again."""
        self._native.reset()

    @property
    def cancelled(self) -> bool:
        """Whether cancellation has been requested."""
        return self._native.cancelled

    def __repr__(self) -> str:
        return f"CancelToken(cancelled={self.cancelled})"


def _native_token(cancel: "CancelToken | None"):
    """The compiled token behind ``cancel``, for passing to kernels."""
    if cancel is None:
        return None
    if not isinstance(cancel, CancelToken):
        raise InvalidParameter(
            f"cancel must be a mdapy.CancelToken or None, got {type(cancel).__name__}"
        )
    return cancel._native


def _init_from_environment(environ: Mapping[str, str] = os.environ) -> None:
    """Apply ``MDAPY_NUM_THREADS`` from ``environ``, if set."""
    value = environ.get(ENV_NUM_THREADS)
    if value is None or value.strip() == "":
        return
    try:
        n = int(value)
    except ValueError:
        n = None
    if n is None or n < 1:
        raise InvalidParameter(
            f"environment variable {ENV_NUM_THREADS} must be a positive integer, "
            f"got {value!r}"
        )
    set_num_threads(n)


_init_from_environment()
