import xml.etree.ElementTree as et
from enum import Enum, auto
from typing import Any, Mapping, Optional, Tuple, Union

from .errors import ParsingError

Coordinates = Tuple[float, float, float, float]
"""
The four raw coordinates of a `BoundingBox` with no assigned format. The coordinates at indices
0 and 2 must be on the X-axis (horizontal) and the ones at indices 1 and 3 must be on 
the Y-axis (vertical).
"""


class BoxFormat(Enum):
    """
    The coordinate format of a `BoundingBox`, i.e. the order and defining coordinates.

    It can be one of:

    * `BoxFormat.LTRB`: [xmin, ymin, xmax, ymax],
    * `BoxFormat.LTWH`: [xmin, ymin, width, height],
    * `BoxFormat.XYWH`: [xmid, ymid, width, height].

    See the `BoundingBox` documentation for more detail on the coordinate format.
    """

    LTRB = auto()
    """[xmin, ymin, xmax, ymax]"""

    LTWH = auto()
    """[xmin, ymin, width, height]"""

    XYWH = auto()
    """[xmid, ymid, width, height]"""

    @classmethod
    def from_string(cls, string: str) -> "BoxFormat":
        """
        Create a `BoxFormat` from a raw string.

        The raw string can be either "ltrb", "ltwh" or "xywh".
        """
        if string == "ltrb":
            return BoxFormat.LTRB
        elif string == "ltwh":
            return BoxFormat.LTWH
        elif string == "xywh":
            return BoxFormat.XYWH
        else:
            raise ValueError(f"Invalid BoxFormat string '{string}'")


class BoundingBox:
    """
    A rectangular bounding box with a label and an optional confidence score. Coordinates are
    absolute (i.e. in pixels) and stored internally in `BoxFormat.LTRB` format.

    Spatial layout:

    ```
        xmin   xmid   xmax
    ymin ╆╍╍╍╍╍╍┿╍╍╍╍╍╍┪
         ╏      ┆      ╏
    ymid ╂┄┄┄┄┄┄┼┄┄┄┄┄┄┨
         ╏      ┆      ╏
    ymax ┺╍╍╍╍╍╍┴╍╍╍╍╍╍┛
    ```
    """

    __slots__ = ("label", "_xmin", "_ymin", "_xmax", "_ymax", "_confidence")

    def __init__(
        self,
        *,
        label: str,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
        confidence: Optional[float] = None,
    ) -> None:
        """
        Create a `BoundingBox` from the top-left and bottom-right corner coordinates.

        Use `BoundingBox.create()` to instantiate a `BoundingBox` from other
        coordinate formats.
        """
        assert xmin <= xmax, "`xmax` must be greater than `xmin`."
        assert ymin <= ymax, "`ymax` must be greater than `ymin`."

        if confidence is not None:
            assert (
                0.0 <= confidence <= 1.0
            ), f"Confidence ({confidence}) should be in [0, 1]."

        self.label = label
        self._xmin = xmin
        self._ymin = ymin
        self._xmax = xmax
        self._ymax = ymax
        self._confidence = confidence

    @property
    def confidence(self) -> Optional[float]:
        """
        The bounding box optional confidence score. If present, it is a `float`
        between 0 and 1.
        """
        return self._confidence

    @confidence.setter
    def confidence(self, confidence: Optional[float]):
        if confidence is not None:
            assert (
                0.0 <= confidence <= 1.0
            ), f"Confidence ({confidence}) should be in [0, 1]."
        self._confidence = confidence

    @property
    def xmin(self) -> float:
        """The `x` value in pixels of the bounding box top-left corner (horizontal axis)."""
        return self._xmin

    @property
    def ymin(self) -> float:
        """The `y` value in pixels of the bounding box top-left corner (vertical axis)."""
        return self._ymin

    @property
    def xmax(self) -> float:
        """The `x` value in pixels of the bounding box bottom-right corner (horizontal axis)."""
        return self._xmax

    @property
    def ymax(self) -> float:
        """The `y` value in pixels of the bounding box top-left corner (vertical axis)."""
        return self._ymax

    @property
    def xmid(self) -> float:
        """The `x` value in pixels of the bounding box center point (horizontal axis)."""
        return (self._xmin + self._xmax) / 2.0

    @property
    def ymid(self) -> float:
        """The `y` value in pixels of the bounding box center point (vertical axis)."""
        return (self._ymin + self._ymax) / 2.0

    @property
    def width(self) -> float:
        """The bounding box width in pixels."""
        return self._xmax - self._xmin

    @property
    def height(self) -> float:
        """The bounding box height in pixels."""
        return self._ymax - self._ymin

    @property
    def area(self) -> float:
        """The area of the bounding box in pixels."""
        return self.width * self.height

    def _area_in(self, range_: "tuple[float, float]") -> bool:
        """Returns `True` if the bounding box area is in a given range."""
        lower_bound, upper_bound = range_
        return lower_bound <= self.area <= upper_bound

    @property
    def pascal_area(self) -> int:
        """The bounding box PascalVOC area in pixels."""
        width = int(self._xmax) - int(self._xmin) + 1
        height = int(self._ymax) - int(self._ymin) + 1

        return width * height

    def iou(self, other: "BoundingBox") -> float:
        """The Intersection over Union (IoU) between two bounding boxes."""
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
        """The Pascal VOC Intersection over Union (IoU) between two bounding boxes."""
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
        """Return `True` if the bounding box confidence score is not `None`."""
        return self._confidence is not None

    @property
    def is_ground_truth(self) -> bool:
        """Return `True` if the bounding box confidence score is `None`."""
        return self._confidence is None

    @staticmethod
    def rel_to_abs(coords: Coordinates, size: "tuple[int, int]") -> Coordinates:
        """
        Convert coordinates from relative (between 0 and 1) to absolute (pixels) form.

        The coordinates at indices 0 and 2 must be on the X-axis (horizontal) and the ones
        at indices 1 and 3 must be on the Y-axis (vertical).

        The image size should be a `(width, height)` tuple expressed in pixels.
        """
        a, b, c, d = coords
        w, h = size
        return a * w, b * h, c * w, d * h

    @staticmethod
    def abs_to_rel(coords: Coordinates, size: "tuple[int, int]") -> Coordinates:
        """
        Convert coordinates from absolute (pixels) to relative (between 0 and 1) form.

        The coordinates at indices 0 and 2 must be on the X-axis (horizontal) and the ones
        at indices 1 and 3 must be on the Y-axis (vertical).

        The image size should be a `(width, height)` tuple expressed in pixels.
        """
        a, b, c, d = coords
        w, h = size
        return a / w, b / h, c / w, d / h

    @staticmethod
    def ltwh_to_ltrb(coords: Coordinates) -> Coordinates:
        """
        Convert coordinates from `BoxFormat.LTWH` to `BoxFormat.LTRB` format.

        The coordinates can be either in relative (between 0 and 1) or absolute (pixels) form.
        """
        xmin, ymin, width, height = coords
        return xmin, ymin, xmin + width, ymin + height

    @staticmethod
    def xywh_to_ltrb(coords: Coordinates) -> Coordinates:
        """
        Convert coordinates from `BoxFormat.XYWH` to `BoxFormat.LTRB` format.

        The coordinates can be either in relative (between 0 and 1)  or absolute (pixels) form.
        """
        xmid, ymid, width, height = coords
        w_h, h_h = width / 2, height / 2
        return xmid - w_h, ymid - h_h, xmid + w_h, ymid + h_h

    @property
    def ltrb(self) -> Coordinates:
        """The bounding box coordinates in `BoxFormat.LTRB` format and absolute form (pixels)."""
        return self._xmin, self._ymin, self._xmax, self._ymax

    @property
    def ltwh(self) -> Coordinates:
        """The bounding box coordinates in `BoxFormat.LTWH` format and absolute form (pixels)."""
        return self._xmin, self._ymin, self.width, self.height

    @property
    def xywh(self) -> Coordinates:
        """The bounding box coordinates in `BoxFormat.XYWH` format and absolute form (pixels)."""
        return self.xmid, self.ymid, self.width, self.height

    @classmethod
    def create(
        cls,
        *,
        label: str,
        coords: Coordinates,
        confidence: Optional[float] = None,
        box_format=BoxFormat.LTRB,
        relative=False,
        image_size: Optional["tuple[int, int]"] = None,
    ) -> "BoundingBox":
        """
        Create a `BoundingBox` from different coordinate formats.

        The image size should be provided if the coordinates are given in the relative form
        (values between 0 and 1).
        """
        if relative:
            assert (
                image_size is not None
            ), "For relative coordinates `image_size` should be provided."
            coords = cls.rel_to_abs(coords, image_size)

        if box_format is BoxFormat.LTWH:
            coords = cls.ltwh_to_ltrb(coords)
        elif box_format is BoxFormat.XYWH:
            coords = cls.xywh_to_ltrb(coords)
        elif box_format is BoxFormat.LTRB:
            pass
        else:
            raise ValueError(f"Unknown BoxFormat '{box_format}'.")

        xmin, ymin, xmax, ymax = coords

        return cls(
            label=label,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            confidence=confidence,
        )

    @staticmethod
    def from_txt(
        string: str,
        *,
        box_format=BoxFormat.LTRB,
        relative=False,
        image_size: Optional["tuple[int, int]"] = None,
        separator: Optional[str] = None,
        conf_last: bool = False,
    ) -> "BoundingBox":
        values = string.strip().split(separator)

        if len(values) == 5:
            label, *coords = values
            confidence = None
        elif len(values) == 6:
            if conf_last:
                label, *coords, confidence = values
            else:
                label, confidence, *coords = values
        else:
            raise ParsingError("Syntax error in txt annotation file.")

        try:
            coords = tuple(float(c) for c in coords)
            if confidence is not None:
                confidence = float(confidence)
        except ValueError:
            raise ParsingError("Syntax error in txt annotation file.")

        return BoundingBox.create(
            label=label,
            coords=coords,
            confidence=confidence,
            box_format=box_format,
            relative=relative,
            image_size=image_size,
        )

    @staticmethod
    def from_yolo(
        string: str, *, image_size: "tuple[int, int]", conf_last: bool = False
    ) -> "BoundingBox":
        return BoundingBox.from_txt(
            string,
            box_format=BoxFormat.XYWH,
            relative=True,
            image_size=image_size,
            separator=None,
            conf_last=conf_last,
        )

    @staticmethod
    def from_yolo_darknet(
        string: str,
        *,
        image_size: "tuple[int, int]",
    ) -> "BoundingBox":
        return BoundingBox.from_yolo(string, image_size=image_size, conf_last=False)

    @staticmethod
    def from_yolo_v5(
        string: str,
        *,
        image_size: "tuple[int, int]",
    ) -> "BoundingBox":
        return BoundingBox.from_yolo(string, image_size=image_size, conf_last=True)

    @staticmethod
    def from_yolo_v7(
        string: str,
        *,
        image_size: "tuple[int, int]",
    ) -> "BoundingBox":
        return BoundingBox.from_yolo_v5(string, image_size=image_size)

    @staticmethod
    def from_xml(node: et.Element) -> "BoundingBox":
        label = node.findtext("name")
        box_node = node.find("bndbox")

        if label is None or box_node is None:
            raise ValueError("Syntax error in imagenet annotation format")

        l, t, r, b = (
            box_node.findtext("xmin"),
            box_node.findtext("ymin"),
            box_node.findtext("xmax"),
            box_node.findtext("ymax"),
        )

        if (l is None) or (t is None) or (r is None) or (b is None):
            raise ValueError("Syntax error in imagenet annotation format")

        try:
            coords = tuple(float(c) for c in (l, t, r, b))
        except ValueError:
            raise ParsingError("Syntax error in imagenet annotation format")

        return BoundingBox.create(label=label, coords=tuple(coords))

    @staticmethod
    def from_labelme(node: dict) -> "BoundingBox":
        try:
            label = str(node["label"])
            xs, ys = zip(*node["points"])
            xmin, ymin = min(xs), min(ys)
            xmax, ymax = max(xs), max(ys)
            coords = tuple(float(c) for c in (xmin, ymin, xmax, ymax))
        except (ValueError, KeyError):
            raise ParsingError("Syntax error in labelme annotation file.")

        return BoundingBox.create(label=label, coords=coords)

    @staticmethod
    def from_cvat(node: et.Element) -> "BoundingBox":
        label = node.get("label")
        l, t, r, b = node.get("xtl"), node.get("ytl"), node.get("xbr"), node.get("ybr")

        if (label is None) or (l is None) or (t is None) or (r is None) or (b is None):
            raise ParsingError("Syntax error in CVAT annotation file.")

        try:
            coords = tuple(float(c) for c in (l, t, r, b))
        except ValueError:
            raise ParsingError("Syntax error in CVAT annotation file.")

        return BoundingBox.create(label=label, coords=coords)

    @staticmethod
    def from_via_json(
        region: dict, *, label_key: str = "label_id", confidence_key: str = "confidence"
    ) -> "BoundingBox":
        try:
            region_attrs = region["region_attributes"]
            label = region_attrs[label_key]
            confidence = region_attrs.get(confidence_key)

            shape_attrs = region["shape_attributes"]
            xmin, ymin = shape_attrs["x"], shape_attrs["y"]
            width, height = shape_attrs["width"], shape_attrs["height"]
        except KeyError:
            raise ParsingError("Syntax error in VIA JSON annotation file.")

        return BoundingBox.create(
            label=label,
            coords=(xmin, ymin, width, height),
            confidence=confidence,
            box_format=BoxFormat.LTWH,
        )

    @staticmethod
    def from_yolo_seg(
        string: str,
        *,
        image_size: "tuple[int, int]",
    ) -> "BoundingBox":
        values = string.strip().split()

        if len(values) < 7:
            raise ParsingError(
                "Syntax error in yolo_seg annotation file. There should be at least 7 values."
            )
        elif len(values) % 2 != 1:
            raise ParsingError(
                "Syntax error in yolo_seg annotation file. There should be an odd number of values."
            )
        else:
            label = str(values[0])
            coords_x = tuple(float(value) for value in values[1::2])
            coords_y = tuple(float(value) for value in values[2::2])
            coords = min(coords_x), min(coords_y), max(coords_x), max(coords_y)

        return BoundingBox.create(
            label=label,
            coords=coords,
            box_format=BoxFormat.LTRB,
            relative=True,
            image_size=image_size,
        )

    def to_txt(
        self,
        *,
        label_to_id: Optional[Mapping[str, Union[int, str]]] = None,
        box_format: BoxFormat = BoxFormat.LTRB,
        relative=False,
        image_size: Optional["tuple[int, int]"] = None,
        separator: str = " ",
        conf_last: bool = False,
    ) -> str:
        assert (
            "\n" not in separator
        ), "The newline character '\\n' cannot be used as the separator character."

        if box_format is BoxFormat.LTRB:
            coords = self.ltrb
        elif box_format is BoxFormat.XYWH:
            coords = self.xywh
        elif box_format is BoxFormat.LTWH:
            coords = self.ltwh
        else:
            raise ValueError(f"Unknown BoxFormat '{box_format}'")

        if relative:
            assert (
                image_size is not None
            ), "For relative coordinates, `image_size` should be provided."
            coords = BoundingBox.abs_to_rel(coords, image_size)

        label = self.label
        if label_to_id is not None:
            label = label_to_id[label]

        if isinstance(label, str):
            assert separator not in label, (
                f"The box label '{label}' contains the character '{separator}' which is the same "
                "as the separtor character used for BoundingBox representation in TXT/YOLO format. "
                "This will corrupt the saved annotation file and likely make it unreadable. "
                "Use another character in the label name or `label_to_id` mapping, e.g. use and "
                "underscore instead of a whitespace."
            )

        if self.is_ground_truth:
            line = (label, *coords)
        elif conf_last:
            line = (label, *coords, self._confidence)
        else:
            line = (label, self._confidence, *coords)

        return separator.join(f"{v}" for v in line)

    def to_yolo(
        self,
        *,
        image_size: "tuple[int, int]",
        label_to_id: Optional[Mapping[str, Union[int, str]]] = None,
        conf_last: bool = False,
    ) -> str:
        return self.to_txt(
            label_to_id=label_to_id,
            box_format=BoxFormat.XYWH,
            relative=True,
            image_size=image_size,
            separator=" ",
            conf_last=conf_last,
        )

    def to_yolo_darknet(
        self,
        *,
        image_size: "tuple[int, int]",
        label_to_id: Optional[Mapping[str, Union[int, str]]] = None,
    ) -> str:
        return self.to_yolo(
            image_size=image_size, label_to_id=label_to_id, conf_last=False
        )

    def to_yolo_v5(
        self,
        *,
        image_size: "tuple[int, int]",
        label_to_id: Optional[Mapping[str, Union[int, str]]] = None,
    ) -> str:
        return self.to_yolo(
            image_size=image_size, label_to_id=label_to_id, conf_last=True
        )

    def to_yolo_v7(
        self,
        *,
        image_size: "tuple[int, int]",
        label_to_id: Optional[Mapping[str, Union[int, str]]] = None,
    ) -> str:
        return self.to_yolo_v5(image_size=image_size, label_to_id=label_to_id)

    def to_labelme(self) -> dict:
        xmin, ymin, xmax, ymax = self.ltrb

        return {
            "label": self.label,
            "points": [[xmin, ymin], [xmax, ymax]],
            "shape_type": "rectangle",
        }

    def to_xml(self) -> et.Element:
        obj_node = et.Element("object")
        et.SubElement(obj_node, "name").text = self.label
        box_node = et.SubElement(obj_node, "bndbox")

        for tag, coord in zip(("xmin", "ymin", "xmax", "ymax"), self.ltrb):
            et.SubElement(box_node, tag).text = f"{coord}"

        return obj_node

    def to_cvat(self) -> et.Element:
        xtl, ytl, xbr, ybr = self.ltrb
        return et.Element(
            "box",
            attrib={
                "label": self.label,
                "xtl": f"{xtl}",
                "ytl": f"{ytl}",
                "xbr": f"{xbr}",
                "ybr": f"{ybr}",
            },
        )

    def to_via_json(
        self, *, label_key: str = "label_id", confidence_key: str = "confidence"
    ) -> dict:
        assert (
            label_key != confidence_key
        ), f"Label key '{label_key}' and confidence key '{confidence_key}' should be different."

        shape_attributes = {
            "name": "rect",
            "x": self.xmin,
            "y": self.ymin,
            "width": self.width,
            "height": self.height,
        }

        region_attributes: dict[str, Any] = {label_key: self.label}

        if self.confidence is not None:
            region_attributes[confidence_key] = self.confidence

        return {
            "shape_attributes": shape_attributes,
            "region_attributes": region_attributes,
        }

    def __eq__(self, other):
        if not isinstance(other, BoundingBox):
            raise NotImplementedError

        return (
            self.label == other.label
            and self.xmin == other.xmin
            and self.ymin == other.ymin
            and self.xmax == other.xmax
            and self.ymax == other.ymax
            and self.confidence == other.confidence
        )

    def __repr__(self) -> str:
        return (
            f"BoundingBox(label: {self.label}, xmin: {self._xmin}, ymin: {self._ymin}, "
            f"xmax: {self._xmax}, ymax: {self._ymax}, confidence: {self._confidence})"
        )
