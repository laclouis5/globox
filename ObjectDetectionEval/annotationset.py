from .utils import *
from .boundingbox import *
from .annotation import *
from typing import Dict, Callable, Iterator
from csv import DictReader


class AnnotationSet:

    def __init__(self, annotations: Iterable[Annotation] = None, override = False):
        """TODO: Add optional addition of labels found during
        parsing, for instance COCO names and YOLO `.names`.
        could also add a (lazy) computed accessor that 
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

    def add(self, annotation: Annotation, override = False):
        if not override:
            assert annotation.image_id not in self.image_ids, \
                "image_id already in the set (set 'override' to True \
                to remove this assertion)"
        self._annotations[annotation.image_id] = annotation

    def map_labels(self, mapping: Mapping[str, str]):
        for annotation in self:
            annotation.map_labels(mapping)

    def update(self, other: "AnnotationSet", override = False):
        if not override:
            assert self.image_ids.isdisjoint(other.image_ids), \
                "some image ids are already in the set (set 'override' to \
                True to remove this assertion)"

        self._annotations.update(other._annotations)

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
    def from_folder(
        folder: Path, 
        file_extension: str,
        file_parser: Callable[[Path], Annotation]
    ) -> "AnnotationSet":
        assert folder.is_dir()
        assert file_extension.startswith(".")

        return AnnotationSet(file_parser(p) for p in glob(folder, file_extension))

    @staticmethod
    def from_txt(
        folder: Path,
        image_folder: Path = None,
        box_format = BoxFormat.LTRB,
        relative = False,
        id_to_label: Mapping[str, str] = None,
        file_extension: str = ".txt",
        image_extension: str = ".jpg",
        separator: str = None
    ) -> "AnnotationSet":
        if image_folder is None:
            image_folder = folder

        assert folder.is_dir()
        assert image_folder.is_dir()
        assert image_extension.startswith(".")

        files = glob(folder, file_extension)
        annotations = AnnotationSet()

        for file in files:
            image_path = image_folder / (file.stem + image_extension)
            annotation = Annotation.from_txt(
                file, image_path, box_format, relative, separator)
            annotations.add(annotation)

        if id_to_label is not None:
            annotations.map_labels(id_to_label)

        return annotations

    @staticmethod
    def from_yolo(
        folder: Path, 
        image_folder: Path = None, 
        names_file: Path = None,
        image_extension = ".jpg",
    ) -> "AnnotationSet":
        id_to_label = AnnotationSet.parse_names_file(names_file)
        return AnnotationSet.from_txt(folder, image_folder, 
            box_format=BoxFormat.XYWH, 
            relative=True, 
            id_to_label=id_to_label, 
            image_extension=image_extension)

    @staticmethod
    def from_xml(folder: Path, file_extension: str = ".xml") -> "AnnotationSet":
        return AnnotationSet.from_folder(
            folder, file_extension, Annotation.from_xml)      

    @staticmethod
    def from_openimage(file_path: Path, image_folder: Path) -> "AnnotationSet":
        # TODO: Add error handling.
        annotations = AnnotationSet()

        with file_path.open(newline="") as f:
            reader = DictReader(f)

            for row in reader:
                image_id = row["ImageID"]
                label = row["LabelName"]
                coords = (row[r] for r in ("XMin", "YMin", "XMax", "YMax"))
                coords = [float(r.replace(",", ".")) for r in coords]
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
    def from_labelme(
        folder: Path, 
        file_extension: str = ".json"
    ) -> "AnnotationSet":
        return AnnotationSet.from_folder(
            folder, file_extension, Annotation.from_labelme)

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
        
        return AnnotationSet(Annotation.from_cvat(n) 
            for n in root.iter("image"))

    @staticmethod
    def parse_names_file(path: Path) -> "dict[str, str]":
        return {str(i): v for i, v in enumerate(path.read_text().splitlines())}

    def __repr__(self) -> str:
        return f"AnnotationSet(annotations: {self._annotations})"