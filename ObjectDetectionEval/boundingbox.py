from .utils import *
import xml.etree.ElementTree as et


class BoundingBox:

    """
    You are responsible for ensuring that xmin <= xmax and ymin <= ymax
    at every moment.
    """

    __slots__ = ("label", "xmin", "ymin", "xmax", "ymax", "confidence")

    def __init__(self, 
        label: str, 
        xmin: float, 
        ymin: float, 
        xmax: float, 
        ymax: float,
        confidence: float = None
    ) -> None:
        assert xmin <= xmax, "xmax must be greater than xmin"
        assert ymin <= ymax, "ymax must be greater than ymin"

        if confidence: 
            assert 0.0 <= confidence <= 1.0, \
                f"Confidence ({confidence}) should be in 0...1"

        self.label = label
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.confidence = confidence    

    @property
    def xmid(self) -> float: 
        return (self.xmin + self.xmax) / 2.0

    @property
    def ymid(self) -> float: 
        return (self.ymin + self.ymax) / 2.0
    
    @property
    def width(self) -> float: 
        return self.xmax - self.xmin

    @property
    def height(self) -> float: 
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        return self.width * self.height

    def iou(self, other: "BoundingBox") -> float:
        xmin = max(self.xmin, other.xmin)
        ymin = max(self.ymin, other.ymin)
        xmax = min(self.xmax, other.xmax)
        ymax = min(self.ymax, other.ymax)

        if xmax < xmin or ymax < ymin:
            return 0.0

        intersection = (xmax - xmin) * (ymax - ymin)
        union = self.area + other.area - intersection

        if union == 0.0:
            return 1.0

        return intersection / union

    @property
    def is_detection(self) -> bool:
        return self.confidence is not None

    @property
    def is_ground_truth(self) -> bool:
        return self.confidence is None

    @staticmethod
    def rel_to_abs(coords: Coordinates, size: "tuple[int, int]") -> Coordinates:
        a, b, c, d = coords
        w, h = size
        return a*w, b*h, c*w, d*h

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
        return self.xmin, self.ymin, self.xmax, self.ymax

    @property
    def ltwh(self) -> Coordinates:
        return self.xmin, self.ymin, self.width, self.height

    @property
    def xywh(self) -> Coordinates:
        return self.xmid, self.ymid, self.width, self.height

    @staticmethod
    def create(
        label: str, 
        coords: Coordinates, 
        confidence: float = None,
        box_format = BoxFormat.LTRB,
        relative = False,
        image_size: "tuple[int, int]" = None, 
    ) -> "BoundingBox":
        if relative:
            assert image_size is not None
            coords = BoundingBox.rel_to_abs(coords, image_size)

        if box_format is BoxFormat.LTWH:
            coords = BoundingBox.ltwh_to_ltrb(coords)
        elif box_format is BoxFormat.XYWH:
            coords = BoundingBox.xywh_to_ltrb(coords)
        elif box_format is BoxFormat.LTRB:
            pass
        else:
            raise ValueError(f"Unknown BoxFormat '{box_format}'")

        return BoundingBox(label, *coords, confidence)

    @staticmethod
    def from_txt(
        string: str,
        box_format = BoxFormat.LTRB,
        relative = False,
        image_size: "tuple[int, int]" = None,
        separator: str = None
    ) -> "BoundingBox":
        values = string.split(separator)

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

        return BoundingBox.create(label, coords, confidence, box_format, relative, image_size)

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

        return BoundingBox(label, *coords, confidence=None)

    @staticmethod
    def from_labelme(node: dict) -> "BoundingBox":
        # TODO: Add error handling
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

    def __repr__(self) -> str:
        return f"BoundingBox(xmin: {self.xmin}, ymin: {self.ymin}, xmax: {self.xmax}, ymax: {self.ymax}, confidence: {self.confidence})"