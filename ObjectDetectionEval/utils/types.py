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


class RecallSteps(Enum):

    ELEVEN = auto()
    ALL = auto()