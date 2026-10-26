# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

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
