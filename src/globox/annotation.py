from .boundingbox import BoundingBox, BoxFormat
from .errors import ParsingError, FileParsingError
from .atomic import open_atomic

from pathlib import Path
from typing import Mapping, Optional, Union
import xml.etree.ElementTree as et
import json


class Annotation: 
    """Stores bounding box annotations for one image.

    The image should be uniquely identified by the `image_id` str. The
    image size is necessary for some operations."""

    def __init__(self, 
        image_id: str, 
        image_size: Optional["tuple[int, int]"] = None, 
        boxes: "list[BoundingBox]" = None
    ) -> None:
        if image_size is not None:
            img_w, img_h = image_size
            assert img_w > 0 and img_h > 0
            assert int(img_w) == img_w and int(img_h) == img_h

        self.image_id = image_id
        self.image_size = image_size
        self.boxes = boxes or []

    @property
    def image_width(self) -> Optional[int]:
        return self.image_size[0]

    @property
    def image_height(self) -> Optional[int]:
        return self.image_size[1]

    def add(self, box: BoundingBox):
        self.boxes.append(box)

    def map_labels(self, mapping: Mapping[str, str]) -> "Annotation":
        """Change all the bounding box annotation labels according to
        the provided mapping (act as a translation)."""
        for box in self.boxes:
            box.label = mapping[box.label]
        return self

    def _labels(self) -> "set[str]":
        return {b.label for b in self.boxes}

    @staticmethod
    def from_txt(
        file_path: Path,
        image_id: str,
        box_format: BoxFormat = BoxFormat.LTRB,
        relative = False,
        image_size: "tuple[int, int]" = None,
        separator: str = " ",
        conf_last: bool = False,
    ) -> "Annotation":
        try:
            lines = file_path.read_text().splitlines()
        except OSError:
            raise FileParsingError(file_path, reason="cannot read file")

        try:
            boxes = [
                BoundingBox.from_txt(l, 
                    box_format=box_format, 
                    relative=relative, 
                    image_size=image_size, 
                    separator=separator, 
                    conf_last=conf_last
                )
                for l in lines
            ]
        except ParsingError as e:
            raise FileParsingError(file_path, e.reason)

        return Annotation(image_id, image_size, boxes)

    @staticmethod
    def from_yolo(
        file_path: Path,
        image_id: str,
        image_size: "tuple[int, int]",
        conf_last: bool = False,
    ):
        return Annotation.from_txt(file_path, 
            image_id=image_id,
            box_format=BoxFormat.XYWH, 
            relative=True, 
            image_size=image_size, 
            separator=" ",
            conf_last=conf_last
        )

    @staticmethod
    def from_xml(file_path: Path) -> "Annotation":
        try:
            with file_path.open() as f:
                root = et.parse(f).getroot()
            
            image_id = root.findtext("filename")
            size_node = root.find("size")
            width = size_node.findtext("width")
            height = size_node.findtext("height")
            image_size = int(width), int(height) 
            boxes = [BoundingBox.from_xml(n) for n in root.iter("object")]
        except OSError:
            raise FileParsingError(file_path, reason="cannot read file")
        except et.ParseError as e:
            line, _ = e.position
            raise FileParsingError(file_path, reason=f"syntax error at line {line}")
        except ParsingError as e:
            raise FileParsingError(file_path, reason=e.reason)
        except ValueError as e:
            raise ParsingError(f"{e}")

        return Annotation(image_id, image_size, boxes)
        
    @staticmethod
    def from_labelme(file_path: Path) -> "Annotation":
        # TODO: Add error handling.
        with file_path.open() as f:
            content = json.load(f)
            if "imageData" in content: 
                del content["imageData"]

        image_id = str(content["imagePath"])
        width = int(content["imageWidth"])
        height = int(content["imageHeight"])
        boxes = [BoundingBox.from_labelme(n) for n in content["shapes"]
            if n["shape_type"] == "rectangle"]
        
        return Annotation(image_id, (width, height), boxes)

    @staticmethod
    def _from_coco_partial(node: dict) -> "Annotation":
        # TODO: Add error handling
        image_id = str(node["file_name"])
        image_size = int(node["width"]), int(node["height"])
        return Annotation(image_id, image_size)

    @staticmethod
    def from_cvat(node: et.Element) -> "Annotation":
        # TODO: Add error handling
        attribs = node.attrib
        image_id = attribs["name"]
        image_size = int(attribs["width"]), int(attribs["height"])
        boxes = [BoundingBox.from_cvat(n) for n in node.iter("box")]
        return Annotation(image_id, image_size, boxes)

    def to_txt(self, *,
        label_to_id: Mapping[str, Union[int, str]] = None,
        box_format: BoxFormat = BoxFormat.LTRB, 
        relative = False,
        image_size: "tuple[int, int]" = None, 
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
                conf_last=conf_last
            ) for box in self.boxes
        )

    def to_yolo(self, *,
        label_to_id: Mapping[str, Union[int, str]] = None,
        image_size: "tuple[int, int]" = None,
        conf_last: bool = False,
    ) -> str:
        image_size = image_size or self.image_size
        
        return "\n".join(
            box.to_yolo(
                image_size=image_size, 
                label_to_id=label_to_id,
                conf_last=conf_last
            ) for box in self.boxes
        )

    def save_txt(self, path: Path, *,
        label_to_id: Mapping[str, Union[int, str]] = None,
        box_format: BoxFormat = BoxFormat.LTRB, 
        relative = False, 
        image_size: "tuple[int, int]" = None,
        separator: str = " ",
        conf_last: bool = False,
    ):
        content = self.to_txt(
            label_to_id=label_to_id, 
            box_format=box_format, 
            relative=relative, 
            image_size=image_size, 
            separator=separator,
            conf_last=conf_last
        )

        with open_atomic(path, "w") as f:
            f.write(content)

    def save_yolo(self, path: Path, *,
        label_to_id: Mapping[str, Union[int, str]] = None,
        image_size: "tuple[int, int]" = None,
        conf_last: bool = False,
    ):
        content = self.to_yolo(
            label_to_id=label_to_id, 
            image_size=image_size,
            conf_last=conf_last
        )
        
        with open_atomic(path, "w") as f:
            f.write(content)

    def to_labelme(self, *, image_size: "tuple[int, int]" = None) -> dict:
        image_size = image_size or self.image_size
        assert image_size is not None, "An image size should be provided either by argument or by `self.image_size`."

        return {
            "imagePath": self.image_id,
            "imageWidth": image_size[0],
            "imageHeight": image_size[1],
            "imageData": None,
            "shapes": [b.to_labelme() for b in self.boxes]}

    def save_labelme(self, path: Path, *, image_size: "tuple[int, int]" = None):
        content = self.to_labelme(image_size=image_size)
        with open_atomic(path, "w") as f:
            json.dump(content, fp=f, allow_nan=False)

    def to_xml(self, *, image_size: "tuple[int, int]" = None) -> et.Element:
        image_size = image_size or self.image_size
        assert image_size is not None, "An image size should be provided either by argument or by `self.image_size`."

        ann_node = et.Element("annotation")
        et.SubElement(ann_node, "filename").text = self.image_id

        size_node = et.SubElement(ann_node, "size")
        et.SubElement(size_node, "width").text = f"{image_size[0]}"
        et.SubElement(size_node, "height").text = f"{image_size[1]}"

        for box in self.boxes:
            ann_node.append(box.to_xml())

        return ann_node

    def save_xml(self, path: Path, *, image_size: "tuple[int, int]" = None):
        content = self.to_xml(image_size=image_size)
        content = et.tostring(content, encoding="unicode")
        
        with open_atomic(path, "w") as f:
            f.write(content)

    def to_cvat(self, *, image_size: "tuple[int, int]" = None) -> et.Element:
        image_size = image_size or self.image_size
        assert image_size is not None, "An image size should be provided either by argument or by `self.image_size`."

        img_node = et.Element("image", attrib={
            "name": self.image_id,
            "width": f"{image_size[0]}",
            "height": f"{image_size[1]}",
        })

        img_node.extend(box.to_cvat() for box in self.boxes)

        return img_node

    def to_via_json(self, *, 
        image_folder: Path,
        label_key: str = "label_id", 
        confidence_key: str = "confidence"
    ) -> dict:
        assert image_folder.is_dir()

        image_id = self.image_id
        image_path = image_folder / image_id
        file_size = image_path.stat().st_size

        regions = [
            box.to_via_json(label_key=label_key, confidence_key=confidence_key)
            for box in self.boxes
        ]

        return {
            "filename": image_id,
            "size": file_size,
            "regions": regions
        }

    def __repr__(self) -> str:
        return f"Annotation(image_id: {self.image_id}, image_size: {self.image_size}, boxes: {self.boxes})"