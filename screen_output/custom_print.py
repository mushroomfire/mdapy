from enum import Enum


class Color(Enum):
    BLACK = 30
    RED = 31
    GREEN = 32
    YELLOW = 33
    BLUE = 34
    MAGENTA = 35
    CYAN = 36
    WHITE = 37


def print_color(text: str, fg: Color = Color.BLACK.value, bg: Color = Color.WHITE.value, N=80):
    bg += 10
    text = text + (80-len(text))*' '
    print(f'\033[{fg};{bg}m{text}\033[0m')
    
    

