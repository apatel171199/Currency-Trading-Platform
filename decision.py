from enum import Enum, auto


class Decision(Enum):
    BUY = auto()
    SELL = auto()
    WAIT = auto()