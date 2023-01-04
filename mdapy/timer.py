# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

from datetime import datetime
from functools import wraps


def timer(function):
    """Decorators function for timing."""

    @wraps(function)
    def timer(*args, **kwargs):
        start = datetime.now()
        result = function(*args, **kwargs)
        end = datetime.now()
        print(f"{function.__name__} finished. Time costs {end-start}.")
        return result

    return timer
