# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Moved in 2.2 to the submodules of ``mdapy.properties``: ``phonons``,
``elastic``, ``transport`` and ``thermodynamics``. The names of 2.1 are found
here in 2.2 and 2.3, with a ``DeprecationWarning`` that names their new place;
this module goes in 2.4."""

from typing import Any

from mdapy._moved import forward

__all__: list[str] = []


def __getattr__(name: str) -> Any:
    return forward(__name__, name)
