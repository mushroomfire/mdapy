from datetime import datetime
from pyAnalysis.screen_output.custom_print import print_color, Color
from functools import wraps

def timer(function):

    @wraps(function)
    def timer(*args, **kwargs):
        start = datetime.now()
        result = function(*args, **kwargs)
        end = datetime.now()
        print_color(f'{function.__name__} finished. Time costs {end-start}.', fg = Color.BLUE.value)
        return result
    return timer