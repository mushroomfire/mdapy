from datetime import datetime
from functools import wraps


def timer(function):
    @wraps(function)
    def timer(*args, **kwargs):
        start = datetime.now()
        result = function(*args, **kwargs)
        end = datetime.now()
        print(f"{function.__name__} finished. Time costs {end-start}.")
        return result

    return timer
