from .errors import ParsingError

from enum import Enum, auto
from typing import Mapping, Union, Tuple
import xml.etree.ElementTree as et


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


class BoundingBox:
    """Represents a bounding box with a label and an optional confidence score.

    The bounding box coordinates are specified by the top-left corner (xmin, ymin)
    and the bottom-right corner (xmax, ymax). Coordinates are absolute (i.e. in pixels).
    
    ```
        xmin   xmid   xmax
    ymin ╆╍╍╍╍╍╍┿╍╍╍╍╍╍┪
         ╏      ┆      ╏
    ymid ╂┄┄┄┄┄┄┼┄┄┄┄┄┄┨
         ╏      ┆      ╏
    ymax ┺╍╍╍╍╍╍┴╍╍╍╍╍╍┛
    ```
    
    Use the '.create(...)' classmethod to create a BoundingBox using a different coordinate
    system."""

    __slots__ = ("label", "_xmin", "_ymin", "_xmax", "_ymax", "_confidence")

    def __init__(self,
        label: str, 
        xmin: float, 
        ymin: float, 
        xmax: float,
        ymax: float, *,
        confidence: float = None
    ) -> None:
        assert xmin <= xmax, "'xmax' must be greater than 'xmin'"
        assert ymin <= ymax, "'ymax' must be greater than 'ymin'"

        if confidence: 
            assert 0.0 <= confidence <= 1.0, \
                f"Confidence ({confidence}) should be in 0...1"

        self.label = label
        self._xmin = xmin
        self._ymin = ymin
        self._xmax = xmax
        self._ymax = ymax
        self._confidence = confidence

    @property
    def confidence(self) -> float:
        return self._confidence

    @confidence.setter
    def confidence(self, confidence: float):
        assert 0.0 <= confidence <= 1.0, \
                f"Confidence ({confidence}) should be in 0...1"
        self._confidence = confidence

    @property
    def xmin(self) -> float:
        return self._xmin

    @property
    def ymin(self) -> float:
        return self._ymin

    @property
    def xmax(self) -> float:
        return self._xmax

    @property
    def ymax(self) -> float:
        return self._ymax

    @property
    def xmid(self) -> float:
        return (self._xmin + self._xmax) / 2.0

    @property
    def ymid(self) -> float: 
        return (self._ymin + self._ymax) / 2.0
    
    @property
    def width(self) -> float: 
        return self._xmax - self._xmin

    @property
    def height(self) -> float: 
        return self._ymax - self._ymin

    @property
    def area(self) -> float:
        return self.width * self.height

    def area_in(self, range_: "tuple[float, float]") -> bool:
        lower_bound, upper_bound = range_
        return lower_bound <= self.area <= upper_bound

    @property
    def pascal_area(self) -> int:
        width = int(self._xmax) - int(self._xmin) + 1
        height = int(self._ymax) - int(self._ymin) + 1
        return width * height

    def iou(self, other: "BoundingBox") -> float:
        """Intersection over Union computed with float coordinates."""
        xmin = max(self._xmin, other._xmin)
        ymin = max(self._ymin, other._ymin)
        xmax = min(self._xmax, other._xmax)
        ymax = min(self._ymax, other._ymax)

        if xmax < xmin or ymax < ymin:
            return 0.0

        intersection = (xmax - xmin) * (ymax - ymin)
        union = self.area + other.area - intersection

        if union == 0.0:
            return 1.0

        return intersection / union

    def pascal_iou(self, other: "BoundingBox") -> float:
        """Intersection over Union computed with integer coordinates."""
        xmin = max(int(self._xmin), int(other._xmin))
        ymin = max(int(self._ymin), int(other._ymin))
        xmax = min(int(self._xmax), int(other._xmax))
        ymax = min(int(self._ymax), int(other._ymax))

        if xmax < xmin or ymax < ymin:
            return 0.0

        intersection = (xmax - xmin + 1) * (ymax - ymin + 1)
        union = self.pascal_area + other.pascal_area - intersection

        if union == 0:
            return 1.0

        return intersection / union

    @property
    def is_detection(self) -> bool:
        return self._confidence is not None

    @property
    def is_ground_truth(self) -> bool:
        return self._confidence is None

    @staticmethod
    def rel_to_abs(coords: Coordinates, size: "tuple[int, int]") -> Coordinates:
        a, b, c, d = coords
        w, h = size
        return a*w, b*h, c*w, d*h
    
    @staticmethod
    def abs_to_rel(coords: Coordinates, size: "tuple[int, int]") -> Coordinates:
        a, b, c, d = coords
        w, h = size
        return a/w, b/h, c/w, d/h

    @staticmethod
    def ltwh_to_ltrb(coords: Coordinates) -> Coordinates:
        xmin, ymin, width, height = coords
        return xmin, ymin, xmin + width, ymin + height

    @staticmethod
    def xywh_to_ltrb(coords: Coordinates) -> Coordinates:
        xmid, ymid, width, height = coords
        w_h, h_h = width/2, height/2
        return xmid-w_h, ymid-h_h, xmid+w_h, ymid+h_h

    @property
    def ltrb(self) -> Coordinates:
        return self._xmin, self._ymin, self._xmax, self._ymax

    @property
    def ltwh(self) -> Coordinates:
        return self._xmin, self._ymin, self.width, self.height

    @property
    def xywh(self) -> Coordinates:
        return self.xmid, self.ymid, self.width, self.height

    @classmethod
    def create(cls, *,
        label: str, 
        coords: Coordinates, 
        confidence: float = None,
        box_format = BoxFormat.LTRB,
        relative = False,
        image_size: "tuple[int, int]" = None, 
    ) -> "BoundingBox":
        if relative:
            assert image_size is not None, "For relative coordinates image_size should be provided"
            coords = cls.rel_to_abs(coords, image_size)

        if box_format is BoxFormat.LTWH:
            coords = cls.ltwh_to_ltrb(coords)
        elif box_format is BoxFormat.XYWH:
            coords = cls.xywh_to_ltrb(coords)
        elif box_format is BoxFormat.LTRB:
            pass
        else:
            raise ValueError(f"Unknown BoxFormat '{box_format}'")

        return cls(label, *coords, confidence=confidence)

    @staticmethod
    def from_txt(
        string: str,
        box_format = BoxFormat.LTRB,
        relative = False,
        image_size: "tuple[int, int]" = None,
        separator: str = " "
    ) -> "BoundingBox":
        values = string.strip().split(separator)

        if len(values) == 5:
            label, *coords = values  
            confidence = None 
        elif len(values) == 6:
            label, confidence, *coords = values
        else:
            raise ParsingError(f"line '{string}' should have 5 or 6 values separated by whitespaces, not {len(values)}")

        try:
            coords = (float(c) for c in coords)
            if confidence is not None:
                confidence = float(confidence)
        except ValueError as e:
            raise ParsingError(f"{e} in line '{string}'")

        return BoundingBox.create(
            label=label, 
            coords=coords, 
            confidence=confidence, 
            box_format=box_format, 
            relative=relative, 
            image_size=image_size)

    @staticmethod
    def from_yolo(string: str, image_size: "tuple[int, int]") -> "BoundingBox":
        return BoundingBox.from_txt(string, 
            box_format=BoxFormat.XYWH, 
            relative=True, 
            image_size=image_size, 
            separator=" ")

    @staticmethod
    def from_xml(node: et.Element) -> "BoundingBox":
        try:
            label = node.findtext("name")
            box_node = node.find("bndbox")
            coords = (float(box_node.findtext(c)) 
                for c in ("xmin", "ymin", "xmax", "ymax"))
        except et.ParseError as e:
            line, _ = e.position
            raise ParsingError(f"syntax error at line {line}")
        except ValueError as e:
            raise ParsingError(f"{e}")

        return BoundingBox(label, *coords)

    @staticmethod
    def from_labelme(node: dict) -> "BoundingBox":
        # TODO: Add error handling
        # TODO: Handle if 'shape_type' is not rectangle
        label = str(node["label"])
        (xmin, ymin), (xmax, ymax) = node["points"]
        coords = (float(c) for c in (xmin, ymin, xmax, ymax))
        return BoundingBox(label, *coords)

    @staticmethod
    def from_cvat(node: et.Element) -> "BoundingBox":
        # TODO: Add error handling
        label = node.attrib["label"]
        coords = (float(node.attrib[c]) for c in ("xtl", "ytl", "xbr", "ybr"))
        return BoundingBox(label, *coords)

    def to_txt(self, 
        label_to_id: Mapping[str, Union[float, str]] = None,
        box_format: BoxFormat = BoxFormat.LTRB, 
        relative = False, 
        image_size: "tuple[int, int]" = None,
        separator: str = " "
    ) -> str:
        if box_format is BoxFormat.LTRB:
            coords = self.ltrb
        elif box_format is BoxFormat.XYWH:
            coords = self.xywh
        elif box_format is BoxFormat.LTWH:
            coords = self.ltwh
        else:
            raise ValueError(f"Unknown BoxFormat '{box_format}'")
        
        if relative:
            assert image_size is not None, "For relative coordinates `image_size` should be provided"
            coords = BoundingBox.abs_to_rel(coords, image_size)

        label = self.label
        if label_to_id is not None:
            label = label_to_id[label]

        if self.is_ground_truth:
            return separator.join(f"{v}" for v in (label, *coords))
        else:
            return separator.join(f"{v}" for v in (label, self._confidence, *coords))

    def to_yolo(self,
        image_size: "tuple[int, int]",
        label_to_id: Mapping[str, Union[float, str]] = None
    ) -> str:
        return self.to_txt(
            label_to_id=label_to_id, 
            box_format=BoxFormat.XYWH, 
            relative=True, 
            image_size=image_size, 
            separator=" ")

    def to_labelme(self) -> dict:
        xmin, ymin, xmax, ymax = self.ltrb
        return {
            "label": self.label, 
            "points": [[xmin, ymin], [xmax, ymax]], 
            "shape_type": "rectangle"}

    def to_xml(self) -> et.Element:
        obj_node = et.Element("object")
        et.SubElement(obj_node, "name").text = self.label
        box_node = et.SubElement(obj_node, "bndbox")
        for tag, coord in zip(("xmin", "ymin", "xmax", "ymax"), self.ltrb):
            et.SubElement(box_node, tag).text = f"{coord}"
        return obj_node

    def to_cvat(self) -> et.Element:
        xtl, ytl, xbr, ybr = self.ltrb
        return et.Element("box", attrib={
            "label": self.label,
            "xtl": f"{xtl}",
            "ytl": f"{ytl}",
            "xbr": f"{xbr}",
            "ybr": f"{ybr}",
        })

    def __repr__(self) -> str:
        return f"BoundingBox(label: {self.label}, xmin: {self._xmin}, ymin: {self._ymin}, xmax: {self._xmax}, ymax: {self._ymax}, confidence: {self._confidence})"