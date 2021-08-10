from .utils import *
from .boundingbox import *
from typing import Mapping
import json

from PIL import Image, ImageDraw


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
        assert img_w > 0 and img_h > 0
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

    def map_labels(self, mapping: Mapping[str, str]) -> None:
        for box in self.boxes:
            box.label = mapping[box.label]

    def _labels(self) -> "set[str]":
        return {b.label for b in self.boxes}

    @staticmethod
    def from_txt(
        file_path: Path,
        image_path: Path,    
        box_format: BoxFormat = BoxFormat.LTRB,
        relative = False,
        separator: str = None
    ) -> "Annotation":
        try:
            lines = file_path.read_text().splitlines()
        except OSError:
            raise FileParsingError(file_path, reason="cannot read file")

        try:
            image_size = get_image_size(image_path)
        except UnknownImageFormat:
            raise FileParsingError(
                file_path, 
                reason=f"unable to read image file '{image_path}' \
                    to get the image size")

        try:
            boxes = [BoundingBox.from_txt(
                l, box_format, relative, image_size, separator)
                for l in lines]
        except ParsingError as e:
            raise FileParsingError(file_path, e.reason)

        return Annotation(image_path.name, image_size, boxes)

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
            del content["imageData"]

        image_id = str(content["imagePath"])
        width = int(content["imageWidth"])
        height = int(content["imageHeight"])
        boxes = [BoundingBox.from_labelme(n) for n in content["shapes"]
            if n["shape_type"] == "rectangle"]
        
        return Annotation(image_id, (width, height), boxes)

    @staticmethod
    def _from_coco_partial(node) -> "Annotation":
        # TODO: Add error handling
        image_id = str(node["file_name"])
        image_size = int(node["width"]), int(node["height"])
        return Annotation(image_id, image_size)

    @staticmethod
    def from_cvat(node: et.Element) -> "Annotation":
        image_id = node.attrib["name"]
        image_size = int(node.attrib["width"]), int(node.attrib["height"])
        boxes = [BoundingBox.from_cvat(n) for n in node.iter("box")]
        return Annotation(image_id, image_size, boxes)

    def __repr__(self) -> str:
        return f"Annotation(image_id: {self.image_id}, image_size: {self.image_size}, boxes: {self.boxes})"

    def draw(self, img: Image) -> None:
        draw = ImageDraw.Draw(img)
        for box in self.boxes:
            draw.rectangle(box.ltrb, outline="black", width=5)