from .utils import *
from .boundingbox import BoundingBox
from .annotation import Annotation

from typing import Dict, Callable, Iterator, Mapping, TypeVar
import json
from csv import DictReader, DictWriter
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import lxml.etree as et
from rich.table import Table
from rich import print as rprint


T = TypeVar("T")


class AnnotationSet:

    def __init__(self, annotations: Iterable[Annotation] = None, override = False):
        """TODO: Add optional addition of labels found during
        parsing, for instance COCO names and YOLO `.names`.
        Could also add a (lazy) computed accessor that 
        runs through all boxes to get labels.
        """
        self._annotations: Dict[str, Annotation] = {}

        if annotations is None:
            return 

        for annotation in annotations:
            self.add(annotation, override)

    def __getitem__(self, image_id) -> Annotation:
        return self._annotations[image_id]

    def __len__(self) -> int:
        return len(self._annotations)

    def __iter__(self):
        yield from self._annotations.values()

    def items(self):
        return self._annotations.items()

    def __contains__(self, annotation: Annotation) -> bool:
        return annotation.image_id in self._annotations.keys()

    def add(self, annotation: Annotation, override = False):
        if not override:
            assert annotation.image_id not in self.image_ids, \
                "image_id already in the set (set 'override' to True to remove this assertion)"
        self._annotations[annotation.image_id] = annotation

    def update(self, other: "AnnotationSet", override = False) -> "AnnotationSet":
        if not override:
            assert self.image_ids.isdisjoint(other.image_ids), \
                "some image ids are already in the set (set 'override' to True to remove this assertion)"
        self._annotations.update(other._annotations)
        return self

    def __iadd__(self, other: "AnnotationSet") -> "AnnotationSet":
        return self.update(other)

    def __add__(self, other: "AnnotationSet") -> "AnnotationSet":
        return AnnotationSet().update(self).update(other)

    def map_labels(self, mapping: Mapping[str, str]) -> "AnnotationSet":
        for annotation in self:
            annotation.map_labels(mapping)
        return self

    @property
    def image_ids(self):
        return self._annotations.keys()

    @property
    def all_boxes(self) -> Iterator[BoundingBox]:
        for annotation in self:
            yield from annotation.boxes

    def _labels(self) -> "set[str]":
        return {b.label for b in self.all_boxes}

    @staticmethod
    def from_iter(
        parser: Callable[[T], Annotation],
        it: Iterable[T],
    ) -> "AnnotationSet":
        annotations = ThreadPoolExecutor().map(parser, it)
        return AnnotationSet(annotations)

    @staticmethod
    def from_folder(
        folder: Path, 
        file_extension: str,
        file_parser: Callable[[Path], Annotation]
    ) -> "AnnotationSet":
        assert folder.is_dir()
        annotations = AnnotationSet.from_iter(file_parser, glob(folder, file_extension))
        return AnnotationSet(annotations)

    @staticmethod
    def from_txt(
        folder: Path,
        image_folder: Path = None,
        box_format = BoxFormat.LTRB,
        relative = False,
        file_extension: str = ".txt",
        image_extension: str = ".jpg",
        separator: str = " "
    ) -> "AnnotationSet":
        # TODO: Add error handling
        if image_folder is None:
            image_folder = folder

        assert folder.is_dir()
        assert image_folder.is_dir()
        assert image_extension.startswith(".")

        def _get_annotation(file: Path) -> Annotation:
            image_path = image_folder / file.with_suffix(image_extension).name

            try:
                image_size = get_image_size(image_path)
            except UnknownImageFormat:
                raise FileParsingError(image_path, 
                    reason=f"unable to read image file '{image_path}' to get the image size")
            
            return Annotation.from_txt(
                file_path=file,
                image_id=image_path.name,
                box_format=box_format,
                relative=relative,
                image_size=image_size,
                separator=separator)

        return AnnotationSet.from_folder(folder, file_extension, _get_annotation)

    @staticmethod
    def from_yolo(
        folder: Path, 
        image_folder: Path = None, 
        image_extension = ".jpg",
    ) -> "AnnotationSet":
        return AnnotationSet.from_txt(folder, image_folder, 
            box_format=BoxFormat.XYWH, 
            relative=True, 
            image_extension=image_extension)

    @staticmethod
    def from_xml(folder: Path) -> "AnnotationSet":
        return AnnotationSet.from_folder(folder, ".xml", Annotation.from_xml)      

    @staticmethod
    def from_openimage(file_path: Path, image_folder: Path) -> "AnnotationSet":
        # TODO: Add error handling.
        annotations = AnnotationSet()

        with file_path.open(newline="") as f:
            reader = DictReader(f)

            for row in reader:
                image_id = row["ImageID"]
                label = row["LabelName"]
                coords = [float(row[r]) for r in ("XMin", "YMin", "XMax", "YMax")]
                confidence = row.get("Confidence")
                
                if confidence is not None and confidence != "":
                    confidence = float(confidence)
                else:
                    confidence = None

                if image_id not in annotations.image_ids:
                    image_path = image_folder / image_id
                    image_size = get_image_size(image_path)
                    annotations.add(Annotation(image_id, image_size))
                
                annotation = annotations[image_id]
                coords = BoundingBox.rel_to_abs(coords, annotation.image_size)
                annotation.add(BoundingBox(label, *coords, confidence))

        return annotations

    @staticmethod
    def from_labelme(folder: Path) -> "AnnotationSet":
        return AnnotationSet.from_folder(folder, ".json", Annotation.from_labelme)

    @staticmethod
    def from_coco(file_path: Path) -> "AnnotationSet":
        # TODO: Add error handling
        with file_path.open() as f:
            content = json.load(f)

        id_to_label = {int(d["id"]): str(d["name"]) 
            for d in content["categories"]}
        id_to_annotation = {int(d["id"]): Annotation._from_coco_partial(d)
                for d in content["images"]}

        for element in content["annotations"]:
            annotation = id_to_annotation[int(element["image_id"])]
            label = id_to_label[int(element["category_id"])]
            coords = (float(c) for c in element["bbox"])
            coords = BoundingBox.ltwh_to_ltrb(coords)
            confidence = element.get("score")
    
            if confidence is not None:
                confidence = float(confidence)

            annotation.add(BoundingBox(label, *coords, confidence))

        return AnnotationSet(id_to_annotation.values())

    @staticmethod
    def from_cvat(file_path: Path) -> "AnnotationSet":
        # TODO: Add error handling.
        with file_path.open() as f:
            root = et.parse(f).getroot()  
        return AnnotationSet.from_iter(Annotation.from_cvat, root.iter("image"))

    def save_txt(self, 
        save_dir: Path,
        label_to_id: Mapping[str, Union[float, str]] = None,
        box_format: BoxFormat = BoxFormat.LTRB, 
        relative: bool = False, 
        separator: str = " ",
        file_extension: str = ".txt"
    ):
        save_dir.mkdir(exist_ok=True)

        def _save(annotation: Annotation):
            image_id = Path(annotation.image_id)
            path = save_dir / image_id.with_suffix(file_extension)
            annotation.save_txt(path, label_to_id, box_format, relative, separator)

        ThreadPoolExecutor().map(_save, self)

    def save_yolo(self, save_dir: Path, label_to_id: Mapping[str, Union[float, str]] = None):
        save_dir.mkdir(exist_ok=True)

        def _save(annotation: Annotation):
            path = save_dir / Path(annotation.image_id).with_suffix(".txt")
            annotation.save_yolo(path, label_to_id)

        ThreadPoolExecutor().map(_save, self)

    def save_labelme(self, save_dir: Path):
        save_dir.mkdir(exist_ok=True)
        
        def _save(annotation: Annotation):
            path = save_dir / Path(annotation.image_id).with_suffix(".json")
            annotation.save_labelme(path)

        ThreadPoolExecutor().map(_save, self)

    def save_xml(self, save_dir: Path):
        save_dir.mkdir(exist_ok=True)
        
        def _save(annotation: Annotation):
            path = save_dir / Path(annotation.image_id).with_suffix(".xml")
            annotation.save_xml(path)

        ThreadPoolExecutor().map(_save, self)

    def to_coco(self) -> dict:
        labels = sorted(self._labels())
        label_to_id = {l: i for i, l in enumerate(labels)}
        imageid_to_id = {n: i for i, n in enumerate(self.image_ids)}
        
        annotations = []
        for annotation in self:
            for idx, box in enumerate(annotation.boxes):
                box_annotation = {
                    "iscrowd": 0, "ignore": 0,
                    "image_id": imageid_to_id[annotation.image_id],
                    "bbox": box.ltwh,
                    "category_id": label_to_id[box.label],
                    "id": idx}

                if box.is_detection:
                    box_annotation["score"] = box.confidence

                annotations.append(box_annotation)

        images = [{
            "id": imageid_to_id[a.image_id], 
            "file_name": a.image_id, 
            "width": a.image_width, 
            "height": a.image_height} for a in self]

        categories = [{"supercategory": "none", "id": label_to_id[l], "name": l} for l in labels]

        return {"images": images, "annotations": annotations, "categories": categories}

    def save_coco(self, path: Path):
        if path.suffix == "":
            path = path.with_suffix(".json")
        assert path.suffix == ".json"
        content = json.dumps(self.to_coco(), allow_nan=False)
        path.write_text(content)

    def save_openimage(self, path: Path):
        if path.suffix == "":
            path = path.with_suffix(".csv")
        assert path.suffix == ".csv"
        with path.open("w", newline="") as f:
            fields = (
                "ImageID", "Source", "LabelName", "Confidence", 
                "XMin", "XMax", "YMin", "YMax", "IsOccluded", 
                "IsTruncated", "IsGroupOf", "IsDepiction", "IsInside")
            writer = DictWriter(f, fieldnames=fields, restval="")
            writer.writeheader()

            for annotation in self:
                for box in annotation.boxes:
                    xmin, ymin, xmax, ymax = BoundingBox.abs_to_rel(box.ltrb, annotation.image_size)
                    
                    row = {
                        "ImageID": annotation.image_id,
                        "LabelName": box.label,
                        "XMin": xmin, "XMax": xmax, "YMin": ymin, "YMax": ymax}
                    
                    if box.is_detection:
                        row["Confidence"] = box.confidence
                    
                    writer.writerow(row)

    def to_cvat(self) -> et.Element:
        ann_node = et.Element("annotations")
        for annotation in self:
            ann_node.append(annotation.to_cvat())
        return ann_node

    def save_cvat(self, path: Path):
        if path.suffix == "":
            path = path.with_suffix(".xml")
        assert path.suffix == ".xml"
        content = et.tostring(self.to_cvat(), encoding="unicode", pretty_print=True)
        path.write_text(content)

    @staticmethod
    def parse_names_file(path: Path) -> "dict[str, str]":
        # TODO: Add error handling
        return {str(i): v for i, v in enumerate(path.read_text().splitlines())}

    def show_stats(self):
        box_by_label = defaultdict(int)
        im_by_label = defaultdict(int)

        for annotation in self:
            for box in annotation.boxes:
                box_by_label[box.label] += 1
            
            labels = annotation._labels()

            for label in labels:
                im_by_label[label] += 1
            
            if len(labels) == 0:
                im_by_label["<empty_image>"] += 1
            
        tot_box = sum(box_by_label.values())
        tot_im = len(self)

        table = Table(title="Database Stats", show_footer=True)
        table.add_column("Label", footer="Total")
        table.add_column("Images", footer=f"{tot_im}", justify="right")
        table.add_column("Boxes", footer=f"{tot_box}", justify="right")

        for label in sorted(im_by_label.keys()):
            nb_im = im_by_label[label]
            nb_box = box_by_label[label]
            table.add_row(label, f"{nb_im}", f"{nb_box}")

        rprint(table)

    def __repr__(self) -> str:
        return f"AnnotationSet(annotations: {self._annotations})"