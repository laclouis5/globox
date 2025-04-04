import json
import xml.etree.ElementTree as et
from pathlib import Path
from typing import Mapping, Optional, Union
from warnings import warn

from .atomic import open_atomic
from .boundingbox import BoundingBox, BoxFormat
from .errors import FileParsingError, ParsingError
from .file_utils import PathLike


class Annotation:
    """
    The bounding boxes associated with a uniquely identified image.
    """

    __slots__ = ("_image_id", "_image_size", "boxes")

    def __init__(
        self,
        image_id: str,
        image_size: Optional["tuple[int, int]"] = None,
        boxes: Optional["list[BoundingBox]"] = None,
    ) -> None:
        """
        Create an `Annotation` for an image identified with a unique `str` tag and with
        the provided list of bounding boxes.

        The image size in pixels ((width, height) tuple) can be optionally specified and is required
        for some export formats. This value can be queried from an image file with the
        `get_image_size()` function if it cannot be retreived from the annotation file.
        """
        if image_size is not None:
            img_w, img_h = image_size
            assert (
                img_w > 0 and img_h > 0
            ), f"Image size '({img_h}, {img_w})' should be positive."
            assert (
                int(img_w) == img_w and int(img_h) == img_h
            ), f"Image size '({img_h}, {img_w})' components should be integers."

        self._image_id = image_id
        self._image_size = image_size
        self.boxes = boxes or []

    @property
    def image_id(self) -> str:
        """The unique identifier of the image for this annotation."""
        return self._image_id

    @property
    def image_size(self) -> "Optional[tuple[int, int]]":
        """The image size in pixels ((width, height) tuple) if present."""
        return self._image_size

    @image_size.setter
    def image_size(self, image_size: "Optional[tuple[int, int]]"):
        if image_size is not None:
            img_w, img_h = image_size
            assert (
                img_w > 0 and img_h > 0
            ), f"Image size '({img_h}, {img_w})' should be positive."
            assert (
                int(img_w) == img_w and int(img_h) == img_h
            ), f"Image size '({img_h}, {img_w})' components should be integers."
        self._image_size = image_size

    @property
    def image_width(self) -> Optional[int]:
        """The image width in pixels."""
        return self.image_size[0] if self.image_size is not None else None

    @property
    def image_height(self) -> Optional[int]:
        """The image height in pixels."""
        return self.image_size[1] if self.image_size is not None else None

    def add(self, box: BoundingBox):
        """Add a bounding box to the image annotation."""
        self.boxes.append(box)

    def map_labels(self, mapping: Mapping[str, str]) -> "Annotation":
        """
        Update all the bounding box labels according to the provided dictionary which maps former
        names to new names. If a label name is not present in the dictionary keys, then it won't
        be updated.
        """
        for box in self.boxes:
            if box.label in mapping.keys():
                box.label = mapping[box.label]
        return self

    def _labels(self) -> "set[str]":
        """The set of the different label names present in the annotation."""
        return {b.label for b in self.boxes}

    @staticmethod
    def from_txt(
        file_path: PathLike,
        *,
        image_id: Optional[str] = None,
        image_extension: str = ".jpg",
        box_format: BoxFormat = BoxFormat.LTRB,
        relative: bool = False,
        image_size: Optional["tuple[int, int]"] = None,
        separator: Optional[str] = None,
        conf_last: bool = False,
    ) -> "Annotation":
        path = Path(file_path).expanduser().resolve()

        if image_id is None:
            assert image_extension.startswith(
                "."
            ), f"Image extension '{image_extension}' should start with a dot."
            image_id = path.with_suffix(image_extension).name

        try:
            lines = path.read_text().splitlines()
        except OSError:
            raise FileParsingError(path, reason="cannot read file")

        try:
            boxes = [
                BoundingBox.from_txt(
                    l,
                    box_format=box_format,
                    relative=relative,
                    image_size=image_size,
                    separator=separator,
                    conf_last=conf_last,
                )
                for l in lines
            ]
        except ParsingError as e:
            raise FileParsingError(path, e.reason)

        return Annotation(image_id, image_size, boxes)

    @staticmethod
    def _from_yolo(
        file_path: PathLike,
        *,
        image_size: "tuple[int, int]",
        image_id: Optional[str] = None,
        image_extension: str = ".jpg",
        conf_last: bool = False,
    ) -> "Annotation":
        return Annotation.from_txt(
            file_path,
            image_id=image_id,
            image_extension=image_extension,
            box_format=BoxFormat.XYWH,
            relative=True,
            image_size=image_size,
            separator=None,
            conf_last=conf_last,
        )

    @staticmethod
    def from_yolo(
        file_path: PathLike,
        *,
        image_size: "tuple[int, int]",
        image_id: Optional[str] = None,
        image_extension: str = ".jpg",
        conf_last: bool = False,
    ) -> "Annotation":
        warn(
            "'from_yolo' is deprecated. Please use `from_yolo_darknet` or `from_yolo_v5`",
            category=DeprecationWarning,
            stacklevel=2,
        )

        return Annotation._from_yolo(
            file_path,
            image_size=image_size,
            image_id=image_id,
            image_extension=image_extension,
            conf_last=conf_last,
        )

    @staticmethod
    def from_yolo_darknet(
        file_path: PathLike,
        *,
        image_size: "tuple[int, int]",
        image_id: Optional[str] = None,
        image_extension: str = ".jpg",
    ) -> "Annotation":
        return Annotation._from_yolo(
            file_path,
            image_size=image_size,
            image_id=image_id,
            image_extension=image_extension,
            conf_last=False,
        )

    @staticmethod
    def from_yolo_v5(
        file_path: PathLike,
        *,
        image_size: "tuple[int, int]",
        image_id: Optional[str] = None,
        image_extension: str = ".jpg",
    ) -> "Annotation":
        return Annotation._from_yolo(
            file_path,
            image_size=image_size,
            image_id=image_id,
            image_extension=image_extension,
            conf_last=True,
        )

    @staticmethod
    def from_yolo_v7(
        file_path: PathLike,
        *,
        image_size: "tuple[int, int]",
        image_id: Optional[str] = None,
        image_extension: str = ".jpg",
    ) -> "Annotation":
        return Annotation.from_yolo_v5(
            file_path,
            image_size=image_size,
            image_id=image_id,
            image_extension=image_extension,
        )

    @staticmethod
    def from_xml(file_path: PathLike) -> "Annotation":
        path = Path(file_path).expanduser().resolve()

        try:
            with path.open() as f:
                root = et.parse(f).getroot()
        except (OSError, et.ParseError):
            raise ParsingError("Syntax error in imagenet annotation file.")

        image_id = root.findtext("filename")
        size_node = root.find("size")

        if (image_id is None) or (size_node is None):
            raise ParsingError("Syntax error in imagenet annotation file.")

        width = size_node.findtext("width")
        height = size_node.findtext("height")

        if (width is None) or (height is None):
            raise ParsingError("Syntax error in imagenet annotation file.")

        try:
            image_size = int(width), int(height)
        except ValueError:
            raise ParsingError("Syntax error in imagenet annotation file.")

        boxes = [BoundingBox.from_xml(n) for n in root.iter("object")]

        return Annotation(image_id, image_size, boxes)

    @staticmethod
    def from_pascal_voc(file_path: PathLike) -> "Annotation":
        return Annotation.from_xml(file_path)

    @staticmethod
    def from_imagenet(file_path: PathLike) -> "Annotation":
        return Annotation.from_xml(file_path)

    @staticmethod
    def from_labelme(file_path: PathLike, include_poly: bool = False) -> "Annotation":
        path = Path(file_path).expanduser().resolve()

        shape_types = {"rectangle"}
        if include_poly:
            shape_types.add("polygon")

        try:
            with path.open() as f:
                content = json.load(f)
                if "imageData" in content:
                    del content["imageData"]
        except (OSError, json.JSONDecodeError):
            raise ParsingError("Syntax error in labelme annotation file.")

        try:
            image_id = str(content["imagePath"])
            width = int(content["imageWidth"])
            height = int(content["imageHeight"])
            boxes = [
                BoundingBox.from_labelme(n)
                for n in content["shapes"]
                if n["shape_type"] in shape_types
            ]
        except (KeyError, ValueError):
            raise ParsingError("Syntax error in labelme annotation file.")

        return Annotation(image_id, image_size=(width, height), boxes=boxes)

    @staticmethod
    def _from_coco_partial(node: dict) -> "Annotation":
        try:
            image_id = str(node["file_name"])
            image_size = int(node["width"]), int(node["height"])
        except (ValueError, KeyError):
            raise ParsingError("Syntax error in COCO annotation file.")

        return Annotation(image_id, image_size)

    @staticmethod
    def from_cvat(node: et.Element) -> "Annotation":
        image_id = node.get("name")
        width, height = node.get("width"), node.get("height")

        if (image_id is None) or (width is None) or (height is None):
            raise ParsingError("Syntax error in CVAT annotation file.")

        try:
            img_size = int(width), int(height)
        except ValueError:
            raise ParsingError("Syntax error in CVAT annotation file.")

        boxes = [BoundingBox.from_cvat(n) for n in node.iter("box")]

        return Annotation(image_id, image_size=img_size, boxes=boxes)

    @staticmethod
    def from_via_json(
        annotation: dict,
        *,
        label_key: str = "label_id",
        confidence_key: str = "confidence",
        image_size: "Optional[tuple[int, int]]" = None,
    ) -> "Annotation":
        try:
            filename = annotation["filename"]
            regions = annotation["regions"]

            bboxes = [
                BoundingBox.from_via_json(
                    region, label_key=label_key, confidence_key=confidence_key
                )
                for region in regions
                if region["shape_attributes"]["name"] == "rect"
            ]
        except KeyError:
            raise ParsingError("Syntax error in VIA JSON annotation file.")

        return Annotation(
            image_id=filename,
            image_size=image_size,
            boxes=bboxes,
        )

    @staticmethod
    def from_yolo_seg(
        file_path: PathLike,
        *,
        image_id: Optional[str] = None,
        image_extension: str = ".jpg",
        image_size: Optional["tuple[int, int]"] = None,
    ) -> "Annotation":
        path = Path(file_path).expanduser().resolve()

        if image_id is None:
            assert image_extension.startswith(
                "."
            ), f"Image extension '{image_extension}' should start with a dot."
            image_id = path.with_suffix(image_extension).name

        try:
            lines = path.read_text().splitlines()
        except OSError:
            raise FileParsingError(path, reason="cannot read file")

        try:
            boxes = [
                BoundingBox.from_yolo_seg(
                    l,
                    image_size=image_size,
                )
                for l in lines
            ]
        except ParsingError as e:
            raise FileParsingError(path, e.reason)

        return Annotation(image_id, image_size, boxes)

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
        image_size = image_size or self.image_size

        return "\n".join(
            box.to_txt(
                label_to_id=label_to_id,
                box_format=box_format,
                relative=relative,
                image_size=image_size,
                separator=separator,
                conf_last=conf_last,
            )
            for box in self.boxes
        )

    def _to_yolo(
        self,
        *,
        label_to_id: Optional[Mapping[str, Union[int, str]]] = None,
        image_size: Optional["tuple[int, int]"] = None,
        conf_last: bool = False,
    ) -> str:
        image_size = image_size or self.image_size

        if image_size is None:
            raise ValueError(
                "Either `image_size` shoud be provided as argument or stored in the Annotation "
                "object for conversion to YOLO format."
            )

        return "\n".join(
            box.to_yolo(
                image_size=image_size, label_to_id=label_to_id, conf_last=conf_last
            )
            for box in self.boxes
        )

    def to_yolo(
        self,
        *,
        label_to_id: Optional[Mapping[str, Union[int, str]]] = None,
        image_size: Optional["tuple[int, int]"] = None,
        conf_last: bool = False,
    ) -> str:
        warn(
            "'to_yolo' is deprecated. Please use `to_yolo_darknet` or `to_yolo_v5`",
            category=DeprecationWarning,
            stacklevel=2,
        )

        return self._to_yolo(
            label_to_id=label_to_id, image_size=image_size, conf_last=conf_last
        )

    def to_yolo_darknet(
        self,
        *,
        label_to_id: Optional[Mapping[str, Union[int, str]]] = None,
        image_size: Optional["tuple[int, int]"] = None,
    ) -> str:
        return self._to_yolo(
            label_to_id=label_to_id, image_size=image_size, conf_last=False
        )

    def to_yolo_v5(
        self,
        *,
        label_to_id: Optional[Mapping[str, Union[int, str]]] = None,
        image_size: Optional["tuple[int, int]"] = None,
    ) -> str:
        return self._to_yolo(
            label_to_id=label_to_id, image_size=image_size, conf_last=True
        )

    def to_yolo_v7(
        self,
        *,
        label_to_id: Optional[Mapping[str, Union[int, str]]] = None,
        image_size: Optional["tuple[int, int]"] = None,
    ) -> str:
        return self.to_yolo_v5(label_to_id=label_to_id, image_size=image_size)

    def save_txt(
        self,
        path: PathLike,
        *,
        label_to_id: Optional[Mapping[str, Union[int, str]]] = None,
        box_format: BoxFormat = BoxFormat.LTRB,
        relative: bool = False,
        image_size: Optional["tuple[int, int]"] = None,
        separator: str = " ",
        conf_last: bool = False,
    ):
        content = self.to_txt(
            label_to_id=label_to_id,
            box_format=box_format,
            relative=relative,
            image_size=image_size,
            separator=separator,
            conf_last=conf_last,
        )

        with open_atomic(path, "w") as f:
            f.write(content)

    def _save_yolo(
        self,
        path: PathLike,
        *,
        label_to_id: Optional[Mapping[str, Union[int, str]]] = None,
        image_size: Optional["tuple[int, int]"] = None,
        conf_last: bool = False,
    ):
        content = self._to_yolo(
            label_to_id=label_to_id, image_size=image_size, conf_last=conf_last
        )

        with open_atomic(path, "w") as f:
            f.write(content)

    def save_yolo(
        self,
        path: PathLike,
        *,
        label_to_id: Optional[Mapping[str, Union[int, str]]] = None,
        image_size: Optional["tuple[int, int]"] = None,
        conf_last: bool = False,
    ):
        warn(
            "'save_yolo' is deprecated. Please use `save_yolo_darknet` or `save_yolo_v5`",
            category=DeprecationWarning,
            stacklevel=2,
        )

        self._save_yolo(
            path, label_to_id=label_to_id, image_size=image_size, conf_last=conf_last
        )

    def save_yolo_darknet(
        self,
        path: PathLike,
        *,
        label_to_id: Optional[Mapping[str, Union[int, str]]] = None,
        image_size: Optional["tuple[int, int]"] = None,
    ):
        self._save_yolo(
            path, label_to_id=label_to_id, image_size=image_size, conf_last=False
        )

    def save_yolo_v5(
        self,
        path: PathLike,
        *,
        label_to_id: Optional[Mapping[str, Union[int, str]]] = None,
        image_size: Optional["tuple[int, int]"] = None,
    ):
        self._save_yolo(
            path, label_to_id=label_to_id, image_size=image_size, conf_last=True
        )

    def save_yolo_v7(
        self,
        path: PathLike,
        *,
        label_to_id: Optional[Mapping[str, Union[int, str]]] = None,
        image_size: Optional["tuple[int, int]"] = None,
    ):
        self.save_yolo_v5(path, label_to_id=label_to_id, image_size=image_size)

    def to_labelme(self, *, image_size: Optional["tuple[int, int]"] = None) -> dict:
        image_size = image_size or self.image_size
        assert (
            image_size is not None
        ), "An image size should be provided either by argument or by `self.image_size`."

        return {
            "imagePath": self.image_id,
            "imageWidth": image_size[0],
            "imageHeight": image_size[1],
            "imageData": None,
            "shapes": [b.to_labelme() for b in self.boxes],
        }

    def save_labelme(
        self, path: PathLike, *, image_size: Optional["tuple[int, int]"] = None
    ):
        content = self.to_labelme(image_size=image_size)
        with open_atomic(path, "w") as f:
            json.dump(content, fp=f, allow_nan=False)

    def to_xml(self, *, image_size: Optional["tuple[int, int]"] = None) -> et.Element:
        image_size = image_size or self.image_size
        assert (
            image_size is not None
        ), "An image size should be provided either by argument or by `self.image_size`."

        ann_node = et.Element("annotation")
        et.SubElement(ann_node, "filename").text = self.image_id

        size_node = et.SubElement(ann_node, "size")
        et.SubElement(size_node, "width").text = f"{image_size[0]}"
        et.SubElement(size_node, "height").text = f"{image_size[1]}"

        for box in self.boxes:
            ann_node.append(box.to_xml())

        return ann_node

    def to_pascal_voc(
        self, *, image_size: Optional["tuple[int, int]"] = None
    ) -> et.Element:
        return self.to_xml(image_size=image_size)

    def to_imagenet(
        self, *, image_size: Optional["tuple[int, int]"] = None
    ) -> et.Element:
        return self.to_xml(image_size=image_size)

    def save_xml(
        self, path: PathLike, *, image_size: Optional["tuple[int, int]"] = None
    ):
        content = self.to_xml(image_size=image_size)
        content = et.tostring(content, encoding="unicode")

        with open_atomic(path, "w") as f:
            f.write(content)

    def save_pascal_voc(
        self, path: PathLike, *, image_size: Optional["tuple[int, int]"] = None
    ):
        self.save_xml(path, image_size=image_size)

    def save_imagenet(
        self, path: PathLike, *, image_size: Optional["tuple[int, int]"] = None
    ):
        self.save_xml(path, image_size=image_size)

    def to_cvat(self, *, image_size: Optional["tuple[int, int]"] = None) -> et.Element:
        image_size = image_size or self.image_size
        assert (
            image_size is not None
        ), "An image size should be provided either by argument or by `self.image_size`."

        img_node = et.Element(
            "image",
            attrib={
                "name": self.image_id,
                "width": f"{image_size[0]}",
                "height": f"{image_size[1]}",
            },
        )

        img_node.extend(box.to_cvat() for box in self.boxes)

        return img_node

    def to_via_json(
        self,
        *,
        image_folder: PathLike,
        label_key: str = "label_id",
        confidence_key: str = "confidence",
    ) -> dict:
        path = Path(image_folder).expanduser().resolve()

        assert path.is_dir(), f"Filepath '{path}' is not a folder or does not exist."

        image_id = self.image_id
        image_path = path / image_id
        file_size = image_path.stat().st_size

        regions = [
            box.to_via_json(label_key=label_key, confidence_key=confidence_key)
            for box in self.boxes
        ]

        return {"filename": image_id, "size": file_size, "regions": regions}

    def __repr__(self) -> str:
        return (
            f"Annotation(image_id: {self.image_id}, image_size: {self.image_size}, "
            f"boxes: {self.boxes})"
        )
