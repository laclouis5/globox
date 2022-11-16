from .boundingbox import BoundingBox, BoxFormat
from .annotation import Annotation
from .errors import UnknownImageFormat, FileParsingError
from .file_utils import glob
from .image_utils import get_image_size
from .atomic import open_atomic
from .thread_utils import thread_map

from typing import Dict, Callable, Iterator, Mapping, TypeVar, Iterable, Union
import csv
from pathlib import Path
import xml.etree.ElementTree as et
from collections import defaultdict
import json
from tqdm import tqdm

T = TypeVar("T")
D = TypeVar("D")


class AnnotationSet:
    """Represents a set of annotations.
        
        This class efficiently stores annotations for several images (a validation 
        database for instance) with a fast lookup by `image_id`. It behaves like a 
        set or dictionary and has similar methods (contains, update, iterator, + 
        operator)."""

    def __init__(self, annotations: Iterable[Annotation] = None, override = False):
        # TODO: Add optional addition of labels found during
        # parsing, for instance COCO names and YOLO `.names`.
        # Could also add a (lazy) computed accessor that 
        # runs through all boxes to get labels.

        self._annotations: Dict[str, Annotation] = {}

        if annotations is not None:
            for annotation in annotations:
                self.add(annotation, override)

        self._id_to_label: "dict[int, str]" = None
        self._id_to_imageid: "dict[int, str]" = None

    def __getitem__(self, image_id: str) -> Annotation:
        return self._annotations[image_id]

    def get(self, image_id: str, default: D = None) -> Union[Annotation, D]:
        return self._annotations.get(image_id, default)

    def __len__(self) -> int:
        return len(self._annotations)

    def __iter__(self):
        yield from self._annotations.values()

    def items(self):
        return self._annotations.items()

    def __contains__(self, annotation: Annotation) -> bool:
        return annotation.image_id in self._annotations.keys()

    def add(self, annotation: Annotation, override = False):
        """Add an annotation to the set. The annotation 'image_id'
        should not already be present in the set.
        
        Parameters:
        - annotation: the annotation to add.
        - override: set to true to override any annotation with the same
        'image_id' already present in the set with the annotation."""

        if not override:
            assert annotation.image_id not in self.image_ids, \
                f"The annotation with id '{annotation.image_id}' is already present in the set (set `override` to True to remove this assertion)."
        self._annotations[annotation.image_id] = annotation

    def update(self, other: "AnnotationSet", override = False) -> "AnnotationSet":
        """Add annotations from another set to this set.
        
        Parameters:
        - other: the set of annotations to add.
        - override: set to true to override any annotation with the same
        'image_id' already present in the set.
        
        Returns:
        - the input set."""

        if not override:
            assert self.image_ids.isdisjoint(other.image_ids), \
                "some image ids are already in the set (set 'override' to True to remove this assertion)."
        self._annotations.update(other._annotations)
        return self

    def __ior__(self, other: "AnnotationSet") -> "AnnotationSet":
        return self.update(other)

    def __or__(self, other: "AnnotationSet") -> "AnnotationSet":
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
        """An iterator of all the bounding boxes."""
        for annotation in self:
            yield from annotation.boxes

    def nb_boxes(self) -> int:
        return sum(len(ann.boxes) for ann in self)

    def _labels(self) -> "set[str]":
        return {b.label for b in self.all_boxes}

    @staticmethod
    def from_iter(
        parser: Callable[[T], Annotation],
        iterable: Iterable[T], *,
        verbose: bool = False
    ) -> "AnnotationSet":
        annotations = thread_map(parser, iterable, desc="Parsing", verbose=verbose)
        return AnnotationSet(annotations)

    @staticmethod
    def from_folder(folder: Path, *,
        extension: str,
        parser: Callable[[Path], Annotation],
        recursive=False,
        verbose: bool = False
    ) -> "AnnotationSet":
        assert folder.is_dir()
        files = list(glob(folder, extension, recursive=recursive))
        return AnnotationSet.from_iter(parser, files, verbose=verbose)

    @staticmethod
    def from_txt(folder: Path, *,
        image_folder: Path = None,
        box_format = BoxFormat.LTRB,
        relative = False,
        file_extension: str = ".txt",
        image_extension: str = ".jpg",
        separator: str = " ",
        conf_last: bool = False,
        verbose: bool = False,
    ) -> "AnnotationSet":
        """This method won't try to retreive the image sizes by default. Specify `image_folder` if you need them. `image_folder` is required when `relative` is True."""
        # TODO: Add error handling
        assert folder.is_dir()
        assert image_extension.startswith(".")

        if relative:
            assert image_folder is not None, "When `relative` is set to True, `image_folder` must be provided to read image sizes."
        if image_folder is not None:
            assert image_folder.is_dir()

        def _get_annotation(file: Path) -> Annotation:
            if image_folder is not None:
                image_path = image_folder / file.with_suffix(image_extension).name
                try:
                    image_size = get_image_size(image_path)
                except UnknownImageFormat:
                    raise FileParsingError(image_path, 
                        reason=f"Unable to read image file '{image_path}' to get the image size.")
            else:
                image_size = None
            
            return Annotation.from_txt(
                file_path=file,
                image_id=file.with_suffix(image_extension).name,
                box_format=box_format,
                relative=relative,
                image_size=image_size,
                separator=separator,
                conf_last=conf_last,
            )

        return AnnotationSet.from_folder(folder, 
            extension=file_extension, 
            parser=_get_annotation, 
            verbose=verbose,
        )

    @staticmethod
    def from_yolo(folder: Path, *,
        image_folder: Path = None, 
        image_extension = ".jpg",
        conf_last: bool = False,
        verbose: bool = False
    ) -> "AnnotationSet":
        return AnnotationSet.from_txt(folder, 
            image_folder=image_folder, 
            box_format=BoxFormat.XYWH, 
            relative=True, 
            image_extension=image_extension,
            conf_last=conf_last,
            verbose=verbose,
        )

    @staticmethod
    def from_xml(folder: Path, verbose: bool = False) -> "AnnotationSet":
        return AnnotationSet.from_folder(folder, 
            extension=".xml", 
            parser=Annotation.from_xml,
            verbose=verbose)      

    @staticmethod
    def from_openimage(file_path: Path, *, 
        image_folder: Path,
        verbose: bool = False
    ) -> "AnnotationSet":
        assert file_path.is_file() and file_path.suffix == ".csv", f"OpenImage annotation file {file_path} must be a csv file."
        assert image_folder.is_dir(), f"Image folder {image_folder} must be a valid directory."
        
        # TODO: Add error handling.
        annotations = AnnotationSet()

        with file_path.open(newline="") as f:
            reader = csv.DictReader(f)

            for row in tqdm(reader, desc="Parsing", disable=not verbose):
                image_id = row["ImageID"]
                label = row["LabelName"]
                coords = (float(row[r]) for r in ("XMin", "YMin", "XMax", "YMax"))
                confidence = row.get("Confidence")
                
                if confidence is not None and confidence != "":
                    confidence = float(confidence)
                else:
                    confidence = None

                if image_id not in annotations.image_ids:
                    image_path = image_folder / image_id
                    image_size = get_image_size(image_path)
                    annotations.add(Annotation(image_id=image_id, image_size=image_size))
                
                annotation = annotations[image_id]
                annotation.add(
                    BoundingBox.create(
                        label=label, 
                        coords=coords, 
                        confidence=confidence, 
                        relative=True, 
                        image_size=annotation.image_size))

        return annotations

    @staticmethod
    def from_labelme(folder: Path, verbose: bool = False) -> "AnnotationSet":
        return AnnotationSet.from_folder(folder, 
            extension=".json", 
            parser=Annotation.from_labelme,
            verbose=verbose)

    @staticmethod
    def from_coco(file_path: Path, verbose: bool = False) -> "AnnotationSet":
        assert file_path.is_file() and file_path.suffix == ".json", f"COCO annotation file {file_path} must be a json file."

        # TODO: Add error handling
        with file_path.open() as f:
            content = json.load(f)

        id_to_label = {int(d["id"]): str(d["name"]) 
            for d in content["categories"]}
        id_to_annotation = {int(d["id"]): Annotation._from_coco_partial(d)
                for d in content["images"]}
        
        for element in tqdm(content["annotations"], desc="Parsing", disable=not verbose):
            annotation = id_to_annotation[int(element["image_id"])]
            label = id_to_label[int(element["category_id"])]
            coords = (float(c) for c in element["bbox"])
            confidence = element.get("score")
    
            if confidence is not None:
                confidence = float(confidence)

            annotation.add(BoundingBox.create(label=label, coords=coords, confidence=confidence, box_format=BoxFormat.LTWH))

        annotation_set = AnnotationSet(id_to_annotation.values())
        annotation_set._id_to_label = id_to_label
        annotation_set._id_to_imageid = {idx: ann.image_id for idx, ann in id_to_annotation.items()}

        return annotation_set

    def from_results(self, file_path: Path, verbose: bool = False) -> "AnnotationSet":
        assert file_path.is_file() and file_path.suffix == ".json", f"COCO annotation file {file_path} must be a json file."

        id_to_label = self._id_to_label
        id_to_imageid = self._id_to_imageid

        assert id_to_label is not None and id_to_imageid is not None, "The AnnotationSet instance should have been created with `AnnotationSet.from_coco()` or should have `self.id_to_label` and `self.id_to_image_id` populated. If not the case use the static method `AnnotationSet.from_coco_results()` instead."

        id_to_annotation = {}

        with file_path.open() as f:
            annotations = json.load(f)
        
        # TODO: Factorize this with `Self.from_coco()`?
        for element in tqdm(annotations, desc="Parsing", disable=not verbose):
            image_id = id_to_imageid[element["image_id"]]
            gt_ann = self[image_id]

            if image_id not in id_to_annotation:
                annotation = Annotation(image_id, (gt_ann.image_width, gt_ann.image_height))
                id_to_annotation[image_id] = annotation
            else:
                annotation = id_to_annotation[image_id]

            label = id_to_label[int(element["category_id"])]
            coords = (float(c) for c in element["bbox"])
            confidence = float(element["score"])

            annotation.add(BoundingBox.create(label=label, coords=coords, confidence=confidence, box_format=BoxFormat.LTWH))     

        annotation_set = AnnotationSet(id_to_annotation.values())
        annotation_set._id_to_label = id_to_label
        annotation_set._id_to_imageid = id_to_imageid
        return annotation_set

    @staticmethod
    def from_coco_results(file_path: Path, *,
        id_to_label: "dict[int, str]", 
        id_to_imageid: "dict[int, str]",
        verbose: bool = False
    ) -> "AnnotationSet":
        assert file_path.is_file() and file_path.suffix == ".json", f"COCO annotation file {file_path} must be a json file."

        id_to_annotation = {}

        with file_path.open() as f:
            annotations = json.load(f)
        
        # TODO: Factorize this with `Self.from_coco()`?
        for element in tqdm(annotations, desc="Parsing", disable=not verbose):
            image_id = id_to_imageid[element["image_id"]]

            if image_id not in id_to_annotation:
                annotation = Annotation(image_id)
                id_to_annotation[image_id] = annotation
            else:
                annotation = id_to_annotation[image_id]

            label = id_to_label[int(element["category_id"])]
            coords = (float(c) for c in element["bbox"])
            confidence = float(element["score"])

            annotation.add(BoundingBox.create(label=label, coords=coords, confidence=confidence, box_format=BoxFormat.LTWH))      

        annotation_set = AnnotationSet(id_to_annotation.values())
        annotation_set._id_to_label = id_to_label
        annotation_set._id_to_imageid = id_to_imageid
        return annotation_set

    @staticmethod
    def from_cvat(file_path: Path, verbose: bool = False) -> "AnnotationSet":
        assert file_path.is_file() and file_path.suffix == ".xml", f"CVAT annotation file {file_path} must be a xml file."
        
        # TODO: Add error handling.
        with file_path.open() as f:
            root = et.parse(f).getroot()
        image_nodes = list(root.iter("image"))
        return AnnotationSet.from_iter(Annotation.from_cvat, image_nodes, verbose=verbose)

    def save_from_it(self, save_fn: Callable[[Annotation], None], *, verbose: bool = False):        
        thread_map(save_fn, self, desc="Saving", verbose=verbose)

    def save_txt(self, save_dir: Path, *,
        label_to_id: Mapping[str, Union[int, str]] = None,
        box_format: BoxFormat = BoxFormat.LTRB, 
        relative: bool = False, 
        separator: str = " ",
        file_extension: str = ".txt",
        conf_last: bool = False,
        verbose: bool = False,
    ):
        save_dir.mkdir(exist_ok=True)

        def _save(annotation: Annotation):
            image_id = annotation.image_id
            path = (save_dir / image_id).with_suffix(file_extension)
            
            annotation.save_txt(path, 
                label_to_id=label_to_id, 
                box_format=box_format, 
                relative=relative,
                separator=separator,
                conf_last=conf_last
            )

        self.save_from_it(_save, verbose=verbose)

    def save_yolo(self, save_dir: Path, *, 
        label_to_id: Mapping[str, Union[int, str]] = None,
        conf_last: bool = False,
        verbose: bool = False,
    ):
        save_dir.mkdir(exist_ok=True)

        def _save(annotation: Annotation):
            path = save_dir / Path(annotation.image_id).with_suffix(".txt")
            annotation.save_yolo(path, label_to_id=label_to_id, conf_last=conf_last)

        self.save_from_it(_save, verbose=verbose)

    def save_labelme(self, save_dir: Path, *, verbose: bool = False):
        save_dir.mkdir(exist_ok=True)
        
        def _save(annotation: Annotation):
            path = save_dir / Path(annotation.image_id).with_suffix(".json")
            annotation.save_labelme(path)

        self.save_from_it(_save, verbose=verbose)

    def save_xml(self, save_dir: Path, *, verbose: bool = False):
        save_dir.mkdir(exist_ok=True)
        
        def _save(annotation: Annotation):
            path = save_dir / Path(annotation.image_id).with_suffix(".xml")
            annotation.save_xml(path)

        self.save_from_it(_save, verbose=verbose)

    def to_coco(self, *,
        label_to_id: "dict[str, int]" = None, 
        imageid_to_id: "dict[str, int]" = None,
        auto_ids: bool = False,
        verbose: bool = False
    ) -> dict:
        native_ids = (label_to_id or self._id_to_label) and (imageid_to_id or self._id_to_imageid)
        assert native_ids or auto_ids, "For COCO, mappings from labels and image ids to integer ids are required. They can be provided either by argument or automatically by the `AnnotationSet` instance if it was created with `AnnotationSet.from_coco()` or `AnnotationSet.from_coco_results()`. You can also set `auto_ids` to True to automatically create image and label ids (warning: this could cause unexpected compatibility issues with other COCO datasets)."

        if native_ids:  # native_ids takes precedence over auto_ids
            label_to_id = label_to_id or {v: k for k, v in self._id_to_label.items()}
            imageid_to_id = imageid_to_id or {v: k for k, v in self._id_to_imageid.items()}
        else:
            label_to_id = {l: i for i, l in enumerate(sorted(self._labels()))}
            imageid_to_id = {im: i for i, im in enumerate(sorted(self.image_ids))}

        annotations = []
        ann_id_count = 0
        for annotation in tqdm(self, desc="Saving", disable=not verbose):
            for box in annotation.boxes:
                box_annotation = {
                    "iscrowd": 0, "ignore": 0,
                    "image_id": imageid_to_id[annotation.image_id],
                    "bbox": box.ltwh,
                    "category_id": label_to_id[box.label],
                    "id": ann_id_count}

                if box.is_detection:
                    box_annotation["score"] = box.confidence

                annotations.append(box_annotation)

                ann_id_count += 1

        images = [{
            "id": imageid_to_id[a.image_id], 
            "file_name": a.image_id, 
            "width": a.image_width, 
            "height": a.image_height} for a in self]

        categories = [{"supercategory": "none", "id": i, "name": l} for l, i in label_to_id.items()]

        return {"images": images, "annotations": annotations, "categories": categories}

    def save_coco(self, path: Path, *,
        label_to_id: "dict[str, int]" = None, 
        imageid_to_id: "dict[str, int]" = None,
        auto_ids: bool = False,
        verbose: bool = False
    ):
        if path.suffix == "":
            path = path.with_suffix(".json")

        assert path.suffix == ".json"
        
        content = self.to_coco(
            label_to_id=label_to_id, 
            imageid_to_id=imageid_to_id, 
            auto_ids=auto_ids, 
            verbose=verbose)

        with open_atomic(path, "w") as f:
            json.dump(content, fp=f, allow_nan=False)

    def save_openimage(self, path: Path, *, verbose: bool = False):
        if path.suffix == "":
            path = path.with_suffix(".csv")

        assert path.suffix == ".csv"

        fields = (
            "ImageID", "Source", "LabelName", "Confidence", 
            "XMin", "XMax", "YMin", "YMax", "IsOccluded", 
            "IsTruncated", "IsGroupOf", "IsDepiction", "IsInside")
    
        with open_atomic(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, restval="")
            writer.writeheader()

            for annotation in tqdm(self, desc="Saving", disable=not verbose):
                for box in annotation.boxes:
                    label = box.label
                    
                    assert "," not in label, f"The box label '{label}' contains the character ',' which is the same as the separtor character used for BoundingBox representation in OpenImage format (CSV). This will corrupt the saved annotation file and likely make it unreadable. Use another character in the label name, e.g. use and underscore instead of a comma."

                    xmin, ymin, xmax, ymax = BoundingBox.abs_to_rel(coords=box.ltrb, size=annotation.image_size)
                    
                    row = {
                        "ImageID": annotation.image_id,
                        "LabelName": label,
                        "XMin": xmin, "XMax": xmax, "YMin": ymin, "YMax": ymax}
                    
                    if box.is_detection:
                        row["Confidence"] = box.confidence
                    
                    writer.writerow(row)

    def to_cvat(self, *, verbose: bool = False) -> et.Element:
        def _create_node(annotation: Annotation) -> et.Element:
            return annotation.to_cvat()

        sub_nodes = thread_map(_create_node, self, desc="Saving", verbose=verbose)
        node = et.Element("annotations")
        node.extend(sub_nodes)
        return node

    def save_cvat(self, path: Path, *, verbose: bool = False):
        if path.suffix == "":
            path = path.with_suffix(".xml")
        assert path.suffix == ".xml"
        content = self.to_cvat(verbose=verbose)
        content = et.tostring(content, encoding="unicode")
        with open_atomic(path, "w") as f:
            f.write(content)

    def to_via_json(self, path: Path, *, 
        image_folder: Path,
        label_key: str = "label_id", 
        confidence_key: str = "confidence",
        verbose: bool = False
    ) -> dict:
        if path.suffix == "":
            path = path.with_suffix(".json")

        assert path.suffix == ".json"

        output = {}
        for annotation in tqdm(self, desc="Saving", disable=not verbose):
            ann_dict = annotation.to_via_json(
                image_folder=image_folder,
                label_key=label_key,
                confidence_key=confidence_key
            )

            key = f"{ann_dict['filename']}{ann_dict['size']}"
            output[key] = ann_dict

        return output

    def save_via_json(self, path: Path, *, 
        image_folder: Path,
        label_key: str = "label_id", 
        confidence_key: str = "confidence",
        verbose: bool = False
    ):
        output = self.to_via_json(path,
            image_folder=image_folder,
            label_key=label_key,
            confidence_key=confidence_key,
            verbose=verbose
        )

        with open_atomic(path, "w") as f:
            json.dump(output, fp=f)

    @staticmethod
    def parse_names_file(path: Path) -> "dict[str, str]":
        """Parse .names file.
        
        Parameters:
        - path: the path to the .names file.

        Returns: 
        - A dictionary mapping label number with label names."""
        # TODO: Add error handling
        return {str(i): v for i, v in enumerate(path.read_text().splitlines())}

    @staticmethod
    def parse_mid_file(path: Path) -> "dict[str, str]":
        with path.open() as f:
            reader = csv.DictReader(f, fieldnames=("LabelName", "DisplayName"))
            return {l["LabelName"]: l["DisplayName"] for l in reader}

    def show_stats(self, *, verbose: bool = False):
        from rich.table import Table
        from rich import print as rprint

        box_by_label = defaultdict(int)
        im_by_label = defaultdict(int)

        for annotation in tqdm(self, desc="Stats", disable=not verbose):
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