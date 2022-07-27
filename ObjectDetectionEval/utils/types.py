from typing import Tuple
from enum import Enum, auto


Coordinates = Tuple[float, float, float, float]


class BoxFormat(Enum):
    """
    The bounding box coordinate system.

    LTRB: xmin, ymin, xmax, ymax
    LTWH: xmin, ymin, width, height
    XYWH: xmid, ymid, width, height
    """
    LTRB = auto()
    LTWH = auto()
    XYWH = auto()

    @classmethod
    def from_string(cls, string: str) -> "BoxFormat":
        if string == "ltrb":
            return BoxFormat.LTRB
        elif string == "ltwh":
            return BoxFormat.LTWH
        elif string == "xywh":
            return BoxFormat.XYWH
        else:
            raise ValueError(f"Invalid BoxFormat string '{string}'")


class RecallSteps(Enum):

    ELEVEN = auto()
    ALL = auto()