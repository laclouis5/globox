from .utils import *
from .boundingbox import BoundingBox

from typing import Mapping
import xml.etree.ElementTree as et
import json

# from PIL import Image, ImageDraw

class Annotation: 
    """
    image_id: an identifier that uniquely identifies an image. A ground-truth
    annotation and a prediction annotation for the same image should have 
    the same `image_id`. This can be the full image path, the image name
    of an image number. `image_name` should be fine.
    """

    def __init__(self, 
        image_id: str, 
        image_size: "tuple[int, int]", 
        boxes: "list[BoundingBox]" = None
    ) -> None:
        img_w, img_h = image_size
        assert img_w >= 0 and img_h >= 0
        assert int(img_w) == img_w and int(img_h) == img_h

        self.image_id = image_id
        self.image_size = image_size
        self.boxes = boxes or []

    @property
    def image_width(self) -> int:
        return self.image_size[0]

    @property
    def image_height(self) -> int:
        return self.image_size[1]

    def add(self, box: BoundingBox):
        self.boxes.append(box)

    def map_labels(self, mapping: Mapping[str, str]) -> "Annotation":
        for box in self.boxes:
            box.label = mapping[box.label]
        return self

    def _labels(self) -> "set[str]":
        return {b.label for b in self.boxes}

    @staticmethod
    def empty() -> "Annotation":
        return Annotation(image_id="", image_size=(0, 0))

    @staticmethod
    def from_txt(
        file_path: Path,
        image_id: str,
        box_format: BoxFormat = BoxFormat.LTRB,
        relative = False,
        image_size: "tuple[int, int]" = None,
        separator: str = " "
    ) -> "Annotation":
        try:
            lines = file_path.read_text().splitlines()
        except OSError:
            raise FileParsingError(file_path, reason="cannot read file")

        try:
            boxes = [BoundingBox.from_txt(
                l, box_format, relative, image_size, separator)
                for l in lines]
        except ParsingError as e:
            raise FileParsingError(file_path, e.reason)

        return Annotation(image_id, image_size, boxes)

    @staticmethod
    def from_yolo(
        file_path: Path,
        image_id: str,
        image_size: "tuple[int, int]",
    ):
        return Annotation.from_txt(file_path, 
            image_id=image_id,
            box_format=BoxFormat.XYWH, 
            relative=True, 
            image_size=image_size, 
            separator=" ")

    @staticmethod
    def from_xml(file_path: Path) -> "Annotation":
        try:
            with file_path.open() as f:
                root = et.parse(f).getroot()
            
            image_id = root.findtext("filename")
            size_node = root.find("size")
            width = size_node.findtext("width")
            height = size_node.findtext("height")
            image_size = float(width), float(height) 
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
        image_id = node.attrib["name"]
        image_size = int(node.attrib["width"]), int(node.attrib["height"])
        boxes = [BoundingBox.from_cvat(n) for n in node.iter("box")]
        return Annotation(image_id, image_size, boxes)

    def to_txt(self, 
        label_to_id: Mapping[str, Union[float, str]] = None,
        box_format: BoxFormat = BoxFormat.LTRB, 
        relative = False, 
        separator: str = " "
    ) -> str:
        return "\n".join(b.to_txt(label_to_id, box_format, relative, self.image_size, separator)
            for b in self.boxes)

    def to_yolo(self, label_to_id: Mapping[str, Union[float, str]] = None) -> str:
        return "\n".join(b.to_yolo(self.image_size, label_to_id) 
            for b in self.boxes)

    def save_txt(self, 
        path: Path,
        label_to_id: Mapping[str, Union[float, str]] = None,
        box_format: BoxFormat = BoxFormat.LTRB, 
        relative = False, 
        separator: str = " "
    ):
        content = self.to_txt(label_to_id, box_format, relative, separator)
        path.write_text(content)

    def save_yolo(self, path: Path, label_to_id: Mapping[str, Union[float, str]] = None):
        content = self.to_yolo(label_to_id)
        path.write_text(content)

    def to_labelme(self) -> dict:
        return {
            "imagePath": self.image_id,
            "imageWidth": self.image_width,
            "imageHeight": self.image_height,
            "imageData": None,
            "shapes": [b.to_labelme() for b in self.boxes]}

    def save_labelme(self, path: Path):
        content = self.to_labelme()
        path.write_text(json.dumps(content, allow_nan=False, indent=2))

    def to_xml(self) -> et.Element:
        ann_node = et.Element("annotation")
        et.SubElement(ann_node, "filename").text = self.image_id

        size_node = et.SubElement(ann_node, "size")
        et.SubElement(size_node, "width").text = f"{self.image_width}"
        et.SubElement(size_node, "height").text = f"{self.image_height}"

        for box in self.boxes:
            ann_node.append(box.to_xml())

        return ann_node

    def save_xml(self, path: Path):
        content = self.to_xml()
        et.indent(content)
        content = et.tostring(content, encoding="unicode")
        path.write_text(content)

    def to_cvat(self) -> et.Element:
        img_node = et.Element("image")
        img_node.attrib["name"] = self.image_id
        img_node.attrib["width"] = f"{self.image_width}"
        img_node.attrib["height"] = f"{self.image_height}"

        for box in self.boxes:
            img_node.append(box.to_cvat())

        return img_node

    # def draw(self, img: Image):
    #     draw = ImageDraw.Draw(img)
    #     for box in self.boxes:
    #         draw.rectangle(box.ltrb, outline="black", width=5)

    def __repr__(self) -> str:
        return f"Annotation(image_id: {self.image_id}, image_size: {self.image_size}, boxes: {self.boxes})"