# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""The base class of the results of analyses."""

from dataclasses import dataclass, fields

import numpy as np


@dataclass(frozen=True)
class Result:
    """Results of an analysis: immutable, with read-only arrays.

    ``System.with_columns(result)`` adds the per-atom values of
    :meth:`columns` to a system.
    """

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
            elif isinstance(value, dict):
                for item in value.values():
                    if isinstance(item, np.ndarray):
                        item.flags.writeable = False

    def columns(self) -> dict[str, np.ndarray]:
        """Per-atom values by column name; empty for results of a whole system."""
        return {}
