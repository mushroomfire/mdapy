# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

"""
Unified parallelism control for mdapy.

mdapy uses several parallel backends:
    * OpenMP inside the C++ extension modules
    * Polars' internal thread pool
    * Python multiprocessing (e.g. pigz)
    * Tachyon's POSIX thread pool (rendering)

They are all driven by the environment variable ``MDAPY_NUM_THREADS``.

Usage
-----
Set the env var **before** importing mdapy::

    import os
    os.environ["MDAPY_NUM_THREADS"] = "1"  # serial
    import mdapy

Rules
-----
* ``MDAPY_NUM_THREADS`` must be a positive integer (> 0) — otherwise mdapy
  raises ``ValueError`` at import time.
* If the value exceeds ``os.cpu_count()`` a warning is emitted (oversubscribing
  usually hurts performance).
* If unset, all parallel regions default to ``os.cpu_count()`` (use all cores).
* Polars locks its thread pool at import time — that's why mdapy mirrors the
  value to ``POLARS_MAX_THREADS`` in ``mdapy/__init__.py`` *before* polars is
  imported. This is the only env var mdapy writes.
* OpenMP threads are passed explicitly into each C++ extension function via a
  ``num_threads(nt)`` clause, so mdapy never relies on ``OMP_NUM_THREADS``.
  We deliberately do **not** mutate ``OMP_NUM_THREADS`` to avoid affecting
  other OpenMP libraries (torch, sklearn, scipy) running in the same process.
"""

from __future__ import annotations

import os


def get_num_threads() -> int:
    """Resolve the thread count to use for any mdapy parallel region.

    Reads ``MDAPY_NUM_THREADS`` (validated at mdapy import time). Falls back
    to ``os.cpu_count()`` when the env var is unset.
    """
    env = os.environ.get("MDAPY_NUM_THREADS")
    if env is not None:
        return int(env)
    return os.cpu_count() or 1
