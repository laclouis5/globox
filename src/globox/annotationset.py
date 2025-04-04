import csv
import json
import xml.etree.ElementTree as et
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    TypeVar,
    Union,
)
from warnings import warn

from tqdm import tqdm

from .annotation import Annotation
from .atomic import open_atomic
from .boundingbox import BoundingBox, BoxFormat
from .errors import ParsingError, UnknownImageFormat
from .file_utils import PathLike, glob
from .image_utils import IMAGE_EXTENSIONS, get_image_size
from .thread_utils import thread_map

T = TypeVar("T")


class AnnotationSet:
    """
    A set of annotations of multiple and distinct images, most commonly refered to a 'dataset'.
    """

    def __init__(
        self,
        annotations: Optional[Iterable[Annotation]] = None,
        *,
        override=False,
    ):
        """
        Create an `AnnotationSet` from multiple image annotations. Each annotation should be unique,
        i.e. multiple annotations for a single image (as idendified by its `image_id`) is not
        allowed.

        Parameters:

        * `annotation`: an iterable of image annotations.
        * `override`: if `True`, image annotations not unique are allowed and only the last one in
        the iterator will be kept, else an error is thrown.
        """
        # TODO: Add optional addition of labels found during
        # parsing, for instance COCO names and YOLO `.names`.
        # Could also add a (lazy) computed accessor that
        # runs through all boxes to get labels.

        self._annotations: Dict[str, Annotation] = {}

        if annotations is not None:
            for annotation in annotations:
                self.add(annotation, override=override)

        self._id_to_label: Optional["dict[Any, str]"] = None
        self._id_to_imageid: Optional["dict[Any, str]"] = None

    def __getitem__(self, image_id: str) -> Annotation:
        """
        Get the image annotation with the corresponding `image_id`. Will raise an exception
        if the image ID is not present in the dataset.
        """
        return self._annotations[image_id]

    def get(self, image_id: str) -> Optional[Annotation]:
        """
        Get the image annotation with the corresponding `image_id`, if present in the dataset
        (else `None` is returned).
        """
        return self._annotations.get(image_id)

    def __len__(self) -> int:
        """The number of annotations in the dataset."""
        return len(self._annotations)

    def __iter__(self):
        yield from self._annotations.values()

    def items(self):
        """A view on the image annotation items (key-value pairs)."""
        return self._annotations.items()

    def __contains__(self, annotation: Annotation) -> bool:
        """Return `True` if a given annotation is present in the dataset, else `False`."""
        return annotation.image_id in self._annotations.keys()

    def add(self, annotation: Annotation, *, override=False):
        """
        Add an annotation to the dataset.

        Parameters:

        * `annotation`: the annotation to add.
        * `override`: set to `True` if the annotation may already be in the dataset and the former
        it should be replaced by the new one. If `False` and the annotation is already in the
        dataset, an error is thrown.
        """

        if not override:
            assert annotation.image_id not in self.image_ids, (
                f"The annotation with id '{annotation.image_id}' is already present in the set "
                "(set `override` to True to remove this assertion)."
            )
        self._annotations[annotation.image_id] = annotation

    def update(self, other: "AnnotationSet", *, override=False) -> "AnnotationSet":
        """
        Add annotations from another datasetset to this one.

        Parameters:

        * `other`: the annotations to add.
        * `override`: if `True`, image annotations in `other` that aren't unique are allowed and
        only the last one in the iterator will be kept, else an error is thrown.
        """

        if not override:
            assert self.image_ids.isdisjoint(other.image_ids), (
                "some image ids are already in the set (set 'override' to True to remove "
                "this assertion)."
            )
        self._annotations.update(other._annotations)
        return self

    def __ior__(self, other: "AnnotationSet") -> "AnnotationSet":
        return self.update(other)

    def __or__(self, other: "AnnotationSet") -> "AnnotationSet":
        return AnnotationSet().update(self).update(other)

    def map_labels(self, mapping: Mapping[str, str]) -> "AnnotationSet":
        """
        Update all the bounding box labels according to the provided dictionary which maps former
        names to new names. If a label name is not present in the dictionary keys, then it won't
        be updated.
        """
        for annotation in self:
            annotation.map_labels(mapping)
        return self

    @property
    def image_ids(self):
        """A view on the set of image IDs of this dataset."""
        return self._annotations.keys()

    @property
    def all_boxes(self) -> Iterator[BoundingBox]:
        """An iterator of all the bounding boxes of the dataset."""
        for annotation in self:
            yield from annotation.boxes

    def nb_boxes(self) -> int:
        """The number of bounding boxes in the dataset."""
        return sum(len(ann.boxes) for ann in self)

    def _labels(self) -> "set[str]":
        """The set of the different label names present in the dataset."""
        return {b.label for b in self.all_boxes}

    @staticmethod
    def from_iter(
        parser: Callable[[T], Annotation],
        iterable: Iterable[T],
        *,
        verbose: bool = False,
    ) -> "AnnotationSet":
        annotations = thread_map(parser, iterable, desc="Parsing", verbose=verbose)
        return AnnotationSet(annotations)

    @staticmethod
    def from_folder(
        folder: PathLike,
        *,
        extension: str,
        parser: Callable[[Path], Annotation],
        recursive=False,
        verbose: bool = False,
    ) -> "AnnotationSet":
        folder = Path(folder).expanduser().resolve()

        assert (
            folder.is_dir()
        ), f"Filepath '{folder}' is not a folder or does not exist."

        files = list(glob(folder, extension, recursive=recursive))
        return AnnotationSet.from_iter(parser, files, verbose=verbose)

    @staticmethod
    def from_txt(
        folder: PathLike,
        *,
        image_folder: Optional[PathLike] = None,
        box_format=BoxFormat.LTRB,
        relative=False,
        file_extension: str = ".txt",
        image_extension: str = ".jpg",
        separator: Optional[str] = None,
        conf_last: bool = False,
        verbose: bool = False,
    ) -> "AnnotationSet":
        """This method won't try to retreive the image sizes by default. Specify `image_folder` if you need them.
        `image_folder` is required when `relative` is True."""
        # TODO: Add error handling

        folder = Path(folder).expanduser().resolve()

        assert folder.is_dir()
        assert image_extension.startswith(".")

        if relative:
            assert (
                image_folder is not None
            ), "When `relative` is set to True, `image_folder` must be provided to read image sizes."

        if image_folder is not None:
            image_folder = Path(image_folder).expanduser().resolve()
            assert image_folder.is_dir()

        def _get_annotation(file: Path) -> Annotation:
            if image_folder is not None:
                image_path: Path | None = None

                for image_ext in IMAGE_EXTENSIONS:
                    image_id = file.with_suffix(image_ext).name
                    path = image_folder / image_id  # type: ignore

                    if path.is_file():
                        image_path = path
                        break

                assert (
                    image_path is not None
                ), f"Image {file.name} does not exist, unable to read the image size."

                image_id = image_path.name

                try:
                    image_size = get_image_size(image_path)
                except UnknownImageFormat:
                    raise ParsingError(
                        f"Unable to read image size of file {image_path}. "
                        f"The file may be corrupted or the file format not supported."
                    )
            else:
                image_size = None
                image_id = file.with_suffix(image_extension).name

            return Annotation.from_txt(
                file_path=file,
                image_id=image_id,
                box_format=box_format,
                relative=relative,
                image_size=image_size,
                separator=separator,
                conf_last=conf_last,
            )

        return AnnotationSet.from_folder(
            folder,
            extension=file_extension,
            parser=_get_annotation,
            verbose=verbose,
        )

    @staticmethod
    def _from_yolo(
        folder: PathLike,
        *,
        image_folder: PathLike,
        image_extension=".jpg",
        conf_last: bool = False,
        verbose: bool = False,
    ) -> "AnnotationSet":
        return AnnotationSet.from_txt(
            folder,
            image_folder=image_folder,
            box_format=BoxFormat.XYWH,
            relative=True,
            image_extension=image_extension,
            separator=None,
            conf_last=conf_last,
            verbose=verbose,
        )

    @staticmethod
    def from_yolo(
        folder: PathLike,
        *,
        image_folder: PathLike,
        image_extension=".jpg",
        conf_last: bool = False,
        verbose: bool = False,
    ) -> "AnnotationSet":
        warn(
            "'from_yolo' is deprecated. Please use `from_yolo_darknet` or `from_yolo_v5`",
            category=DeprecationWarning,
            stacklevel=2,
        )

        return AnnotationSet._from_yolo(
            folder,
            image_folder=image_folder,
            image_extension=image_extension,
            conf_last=conf_last,
            verbose=verbose,
        )

    @staticmethod
    def from_yolo_darknet(
        folder: PathLike,
        *,
        image_folder: PathLike,
        image_extension=".jpg",
        verbose: bool = False,
    ) -> "AnnotationSet":
        return AnnotationSet._from_yolo(
            folder,
            image_folder=image_folder,
            image_extension=image_extension,
            conf_last=False,
            verbose=verbose,
        )

    @staticmethod
    def from_yolo_v5(
        folder: PathLike,
        *,
        image_folder: PathLike,
        image_extension=".jpg",
        verbose: bool = False,
    ) -> "AnnotationSet":
        return AnnotationSet._from_yolo(
            folder,
            image_folder=image_folder,
            image_extension=image_extension,
            conf_last=True,
            verbose=verbose,
        )

    @staticmethod
    def from_yolo_v7(
        folder: PathLike,
        *,
        image_folder: PathLike,
        image_extension=".jpg",
        verbose: bool = False,
    ) -> "AnnotationSet":
        return AnnotationSet.from_yolo_v5(
            folder,
            image_folder=image_folder,
            image_extension=image_extension,
            verbose=verbose,
        )

    @staticmethod
    def from_xml(folder: PathLike, *, verbose: bool = False) -> "AnnotationSet":
        return AnnotationSet.from_folder(
            folder, extension=".xml", parser=Annotation.from_xml, verbose=verbose
        )

    @staticmethod
    def from_pascal_voc(folder: PathLike, *, verbose: bool = False) -> "AnnotationSet":
        return AnnotationSet.from_xml(folder, verbose=verbose)

    @staticmethod
    def from_imagenet(folder: PathLike, *, verbose: bool = False) -> "AnnotationSet":
        return AnnotationSet.from_xml(folder, verbose=verbose)

    @staticmethod
    def from_openimage(
        file_path: PathLike,
        *,
        image_folder: PathLike,
        verbose: bool = False,
    ) -> "AnnotationSet":
        file_path = Path(file_path).expanduser().resolve()
        assert (
            file_path.is_file() and file_path.suffix == ".csv"
        ), f"OpenImage annotation file {file_path} must be a csv file."

        image_folder = Path(image_folder).expanduser().resolve()
        assert (
            image_folder.is_dir()
        ), f"Image folder {image_folder} must be a valid directory."

        # TODO: Error handling.
        # OSError, DictReader error, Key/Value Error

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
                    annotations.add(
                        Annotation(image_id=image_id, image_size=image_size)
                    )

                annotation = annotations[image_id]
                annotation.add(
                    BoundingBox.create(
                        label=label,
                        coords=tuple(coords),
                        confidence=confidence,
                        relative=True,
                        image_size=annotation.image_size,
                    )
                )

        return annotations

    @staticmethod
    def from_labelme(
        folder: PathLike, *, include_poly: bool = False, verbose: bool = False
    ) -> "AnnotationSet":
        parser = partial(Annotation.from_labelme, include_poly=include_poly)
        return AnnotationSet.from_folder(
            folder, extension=".json", parser=parser, verbose=verbose
        )

    @staticmethod
    def from_coco(file_path: PathLike, *, verbose: bool = False) -> "AnnotationSet":
        file_path = Path(file_path).expanduser().resolve()
        assert (
            file_path.is_file() and file_path.suffix == ".json"
        ), f"COCO annotation file {file_path} must be a json file."

        # TODO: Error handling.
        # OSError, JsonDecoderError, Key/ValueError
        with file_path.open() as f:
            content = json.load(f)

        id_to_label = {d["id"]: str(d["name"]) for d in content["categories"]}

        id_to_annotation = {
            d["id"]: Annotation._from_coco_partial(d) for d in content["images"]
        }

        elements = content["annotations"]

        for element in tqdm(elements, desc="Parsing", disable=not verbose):
            annotation = id_to_annotation[element["image_id"]]
            label = id_to_label[int(element["category_id"])]
            coords = tuple(float(c) for c in element["bbox"])
            confidence = element.get("score")

            if confidence is not None:
                confidence = float(confidence)

            annotation.add(
                BoundingBox.create(
                    label=label,
                    coords=coords,
                    confidence=confidence,
                    box_format=BoxFormat.LTWH,
                )
            )

        annotation_set = AnnotationSet(id_to_annotation.values())
        annotation_set._id_to_label = id_to_label
        annotation_set._id_to_imageid = {
            idx: ann.image_id for idx, ann in id_to_annotation.items()
        }

        return annotation_set

    def from_results(
        self, file_path: PathLike, *, verbose: bool = False
    ) -> "AnnotationSet":
        file_path = Path(file_path).expanduser().resolve()
        # TODO: Error handling.
        assert (
            file_path.is_file() and file_path.suffix == ".json"
        ), f"COCO annotation file {file_path} must be a json file."

        id_to_label = self._id_to_label
        id_to_imageid = self._id_to_imageid

        assert id_to_label is not None and id_to_imageid is not None, (
            "The AnnotationSet instance should have been created with `AnnotationSet.from_coco()` "
            "or should have `self.id_to_label` and `self.id_to_image_id` populated. If not the "
            "case use the static method `AnnotationSet.from_coco_results()` instead."
        )

        id_to_annotation = {}

        with file_path.open() as f:
            annotations = json.load(f)

        # TODO: Factorize this with `Self.from_coco()`?
        for element in tqdm(annotations, desc="Parsing", disable=not verbose):
            image_id = id_to_imageid[element["image_id"]]
            gt_ann = self[image_id]

            if image_id not in id_to_annotation:
                annotation = Annotation(image_id, image_size=gt_ann.image_size)
                id_to_annotation[image_id] = annotation
            else:
                annotation = id_to_annotation[image_id]

            label = id_to_label[int(element["category_id"])]
            coords = tuple(float(c) for c in element["bbox"])
            confidence = float(element["score"])

            annotation.add(
                BoundingBox.create(
                    label=label,
                    coords=coords,
                    confidence=confidence,
                    box_format=BoxFormat.LTWH,
                )
            )

        annotation_set = AnnotationSet(id_to_annotation.values())
        annotation_set._id_to_label = id_to_label
        annotation_set._id_to_imageid = id_to_imageid

        return annotation_set

    @staticmethod
    def from_coco_results(
        file_path: PathLike,
        *,
        id_to_label: "dict[int, str]",
        id_to_imageid: "dict[int, str]",
        verbose: bool = False,
    ) -> "AnnotationSet":
        # TODO: Error handling.

        file_path = Path(file_path).expanduser().resolve()
        assert (
            file_path.is_file() and file_path.suffix == ".json"
        ), f"COCO annotation file {file_path} must be a json file."

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
            coords = tuple(float(c) for c in element["bbox"])
            confidence = float(element["score"])

            annotation.add(
                BoundingBox.create(
                    label=label,
                    coords=coords,
                    confidence=confidence,
                    box_format=BoxFormat.LTWH,
                )
            )

        annotation_set = AnnotationSet(id_to_annotation.values())
        annotation_set._id_to_label = id_to_label
        annotation_set._id_to_imageid = id_to_imageid

        return annotation_set

    @staticmethod
    def from_cvat(file_path: PathLike, *, verbose: bool = False) -> "AnnotationSet":
        file_path = Path(file_path).expanduser().resolve()
        assert (
            file_path.is_file() and file_path.suffix == ".xml"
        ), f"CVAT annotation file {file_path} must be a xml file."

        # TODO: Error handling.
        with file_path.open() as f:
            root = et.parse(f).getroot()

        image_nodes = list(root.iter("image"))

        return AnnotationSet.from_iter(
            Annotation.from_cvat, image_nodes, verbose=verbose
        )

    @staticmethod
    def from_via_json(
        file_path: PathLike,
        *,
        label_key: str = "label_id",
        confidence_key: str = "confidence",
        image_folder: Optional[PathLike] = None,
    ) -> "AnnotationSet":
        file_path = Path(file_path).expanduser().resolve()

        if image_folder is not None:
            image_folder = Path(image_folder).expanduser().resolve()

            if not image_folder.is_dir():
                raise ParsingError("Invalid `image_folder`: not a directory.")

        if not file_path.is_file() or not file_path.suffix == ".json":
            raise ParsingError(
                f"VIA JSON annotation file {file_path} must be a valid json file."
            )

        with file_path.open() as f:
            content: dict = json.load(f)

        img_anns: dict = content.get("_via_img_metadata", content)

        annotations = AnnotationSet()

        for img_ann in img_anns.values():
            annotation = Annotation.from_via_json(
                img_ann, label_key=label_key, confidence_key=confidence_key
            )

            if image_folder is not None:
                img_path = image_folder / annotation.image_id
                image_size = get_image_size(img_path)
                annotation.image_size = image_size

            annotations.add(annotation)

        return annotations

    @staticmethod
    def from_yolo_seg(
        folder: PathLike,
        *,
        image_folder: Optional[PathLike] = None,
        relative=False,
        file_extension: str = ".txt",
        image_extension: str = ".jpg",
        verbose: bool = False,
    ) -> "AnnotationSet":
        """This method won't try to retreive the image sizes by default. Specify `image_folder` if you need them.
        `image_folder` is required when `relative` is True."""
        # TODO: Add error handling

        folder = Path(folder).expanduser().resolve()

        assert folder.is_dir()
        assert image_extension.startswith(".")

        if relative:
            assert (
                image_folder is not None
            ), "When `relative` is set to True, `image_folder` must be provided to read image sizes."

        if image_folder is not None:
            image_folder = Path(image_folder).expanduser().resolve()
            assert image_folder.is_dir()

        def _get_annotation(file: Path) -> Annotation:
            if image_folder is not None:
                image_path: Path | None = None

                for image_ext in IMAGE_EXTENSIONS:
                    image_id = file.with_suffix(image_ext).name
                    path = image_folder / image_id  # type: ignore

                    if path.is_file():
                        image_path = path
                        break

                assert (
                    image_path is not None
                ), f"Image {file.name} does not exist, unable to read the image size."

                image_id = image_path.name

                try:
                    image_size = get_image_size(image_path)
                except UnknownImageFormat:
                    raise ParsingError(
                        f"Unable to read image size of file {image_path}. "
                        f"The file may be corrupted or the file format not supported."
                    )
            else:
                image_size = None
                image_id = file.with_suffix(image_extension).name

            return Annotation.from_yolo_seg(
                file_path=file, image_id=image_id, image_size=image_size
            )

        return AnnotationSet.from_folder(
            folder,
            extension=file_extension,
            parser=_get_annotation,
            verbose=verbose,
        )

    def save_from_it(
        self, save_fn: Callable[[Annotation], None], *, verbose: bool = False
    ):
        thread_map(save_fn, self, desc="Saving", verbose=verbose)

    def save_txt(
        self,
        save_dir: PathLike,
        *,
        label_to_id: Optional[Mapping[str, Union[int, str]]] = None,
        box_format: BoxFormat = BoxFormat.LTRB,
        relative: bool = False,
        separator: str = " ",
        file_extension: str = ".txt",
        conf_last: bool = False,
        verbose: bool = False,
    ):
        save_dir = Path(save_dir).expanduser().resolve()
        save_dir.mkdir(exist_ok=True)

        def _save(annotation: Annotation):
            image_id = annotation.image_id
            path = (save_dir / image_id).with_suffix(file_extension)

            annotation.save_txt(
                path,
                label_to_id=label_to_id,
                box_format=box_format,
                relative=relative,
                separator=separator,
                conf_last=conf_last,
            )

        self.save_from_it(_save, verbose=verbose)

    def _save_yolo(
        self,
        save_dir: PathLike,
        *,
        label_to_id: Optional[Mapping[str, Union[int, str]]] = None,
        conf_last: bool = False,
        verbose: bool = False,
    ):
        save_dir = Path(save_dir).expanduser().resolve()
        save_dir.mkdir(exist_ok=True)

        def _save(annotation: Annotation):
            path = save_dir / Path(annotation.image_id).with_suffix(".txt")
            annotation._save_yolo(path, label_to_id=label_to_id, conf_last=conf_last)

        self.save_from_it(_save, verbose=verbose)

    def save_yolo(
        self,
        save_dir: PathLike,
        *,
        label_to_id: Optional[Mapping[str, Union[int, str]]] = None,
        conf_last: bool = False,
        verbose: bool = False,
    ):
        warn(
            "'save_yolo' is deprecated. Please use `save_yolo_darknet` or `save_yolo_v5`",
            category=DeprecationWarning,
            stacklevel=2,
        )

        self._save_yolo(
            save_dir, label_to_id=label_to_id, conf_last=conf_last, verbose=verbose
        )

    def save_yolo_darknet(
        self,
        save_dir: PathLike,
        *,
        label_to_id: Optional[Mapping[str, Union[int, str]]] = None,
        verbose: bool = False,
    ):
        self._save_yolo(
            save_dir, label_to_id=label_to_id, conf_last=False, verbose=verbose
        )

    def save_yolo_v5(
        self,
        save_dir: PathLike,
        *,
        label_to_id: Optional[Mapping[str, Union[int, str]]] = None,
        verbose: bool = False,
    ):
        self._save_yolo(
            save_dir, label_to_id=label_to_id, conf_last=True, verbose=verbose
        )

    def save_yolo_v7(
        self,
        save_dir: PathLike,
        *,
        label_to_id: Optional[Mapping[str, Union[int, str]]] = None,
        verbose: bool = False,
    ):
        self.save_yolo_v5(save_dir, label_to_id=label_to_id, verbose=verbose)

    def save_labelme(self, save_dir: PathLike, *, verbose: bool = False):
        save_dir = Path(save_dir).expanduser().resolve()
        save_dir.mkdir(exist_ok=True)

        def _save(annotation: Annotation):
            path = save_dir / Path(annotation.image_id).with_suffix(".json")
            annotation.save_labelme(path)

        self.save_from_it(_save, verbose=verbose)

    def save_xml(self, save_dir: PathLike, *, verbose: bool = False):
        save_dir = Path(save_dir).expanduser().resolve()
        save_dir.mkdir(exist_ok=True)

        def _save(annotation: Annotation):
            path = save_dir / Path(annotation.image_id).with_suffix(".xml")
            annotation.save_xml(path)

        self.save_from_it(_save, verbose=verbose)

    def save_pascal_voc(self, save_dir: PathLike, *, verbose: bool = False):
        self.save_xml(save_dir, verbose=verbose)

    def save_imagenet(self, save_dir: PathLike, *, verbose: bool = False):
        self.save_xml(save_dir, verbose=verbose)

    def to_coco(
        self,
        *,
        label_to_id: Optional["dict[str, int]"] = None,
        imageid_to_id: Optional["dict[str, int]"] = None,
        auto_ids: bool = False,
        verbose: bool = False,
    ) -> dict:
        if (label_to_id is not None) and (imageid_to_id is not None):
            pass
        elif (self._id_to_label is not None) and (self._id_to_imageid is not None):
            label_to_id = {v: k for k, v in self._id_to_label.items()}
            imageid_to_id = {v: k for k, v in self._id_to_imageid.items()}
        elif auto_ids:
            label_to_id = {l: i for i, l in enumerate(sorted(self._labels()))}
            imageid_to_id = {im: i for i, im in enumerate(sorted(self.image_ids))}
        else:
            # TODO: Convert to ConversionError.
            raise ValueError(
                "For COCO, mappings from labels and image ids to integer ids are required. "
                "They can be provided either by argument or automatically by the `AnnotationSet` "
                "instance if it was created with `AnnotationSet.from_coco()` or "
                "`AnnotationSet.from_coco_results()`. You can also set `auto_ids` to True to "
                "automatically create image and label ids (warning: this could cause unexpected "
                "compatibility issues with other COCO datasets)."
            )

        annotations = []
        ann_id_count = 0
        for annotation in tqdm(self, desc="Saving", disable=not verbose):
            for box in annotation.boxes:
                box_annotation = {
                    "iscrowd": 0,
                    "ignore": 0,
                    "image_id": imageid_to_id[annotation.image_id],
                    "bbox": box.ltwh,
                    "area": box.area,
                    "segmentation": [],
                    "category_id": label_to_id[box.label],
                    "id": ann_id_count,
                }

                if box.is_detection:
                    box_annotation["score"] = box.confidence

                annotations.append(box_annotation)

                ann_id_count += 1

        images = [
            {
                "id": imageid_to_id[a.image_id],
                "file_name": a.image_id,
                "width": a.image_width,
                "height": a.image_height,
            }
            for a in self
        ]

        categories = [
            {"supercategory": "none", "id": i, "name": l}
            for l, i in label_to_id.items()
        ]

        return {"images": images, "annotations": annotations, "categories": categories}

    def save_coco(
        self,
        path: PathLike,
        *,
        label_to_id: Optional["dict[str, int]"] = None,
        imageid_to_id: Optional["dict[str, int]"] = None,
        auto_ids: bool = False,
        verbose: bool = False,
    ):
        path = Path(path).expanduser().resolve()

        if path.suffix == "":
            path = path.with_suffix(".json")

        assert path.suffix == ".json", f"Path '{path}' suffix should be '.json'."

        content = self.to_coco(
            label_to_id=label_to_id,
            imageid_to_id=imageid_to_id,
            auto_ids=auto_ids,
            verbose=verbose,
        )

        with open_atomic(path, "w") as f:
            json.dump(content, fp=f, allow_nan=False)

    def save_openimage(self, path: PathLike, *, verbose: bool = False):
        path = Path(path).expanduser().resolve()

        if path.suffix == "":
            path = path.with_suffix(".csv")

        assert path.suffix == ".csv", f"Path '{path}' suffix should be '.csv'."

        fields = (
            "ImageID",
            "Source",
            "LabelName",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        )

        with open_atomic(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, restval="")
            writer.writeheader()

            for annotation in tqdm(self, desc="Saving", disable=not verbose):
                image_id = annotation.image_id
                image_size = annotation.image_size

                if image_size is None:
                    raise ValueError(
                        "The image size should be present in the annotation for `save_openimage`. "
                        "One should parse the annotations specifying the image folder or populate "
                        "the `image_size` attribute."
                    )

                for box in annotation.boxes:
                    label = box.label

                    if "," in label:
                        raise ValueError(
                            f"The box label '{label}' contains the character ',' which is the same "
                            "as the separtor character used for BoundingBox representation in "
                            "OpenImage format (CSV). This will corrupt the saved annotation file "
                            "and likely make it unreadable. Use another character in the label "
                            "name, e.g. use and underscore instead of a comma."
                        )

                    xmin, ymin, xmax, ymax = BoundingBox.abs_to_rel(
                        coords=box.ltrb, size=image_size
                    )

                    row = {
                        "ImageID": image_id,
                        "LabelName": label,
                        "XMin": xmin,
                        "XMax": xmax,
                        "YMin": ymin,
                        "YMax": ymax,
                    }

                    if box.confidence is not None:
                        row["Confidence"] = box.confidence

                    writer.writerow(row)  # type: ignore

    def to_cvat(self, *, verbose: bool = False) -> et.Element:
        def _create_node(annotation: Annotation) -> et.Element:
            return annotation.to_cvat()

        sub_nodes = thread_map(_create_node, self, desc="Saving", verbose=verbose)
        node = et.Element("annotations")
        node.extend(sub_nodes)

        return node

    def save_cvat(self, path: PathLike, *, verbose: bool = False):
        path = Path(path).expanduser().resolve()

        if path.suffix == "":
            path = path.with_suffix(".xml")

        assert path.suffix == ".xml", f"Path '{path}' suffix should be '.xml'."

        content = self.to_cvat(verbose=verbose)
        content = et.tostring(content, encoding="unicode")

        with open_atomic(path, "w") as f:
            f.write(content)

    def to_via_json(
        self,
        *,
        image_folder: Path,
        label_key: str = "label_id",
        confidence_key: str = "confidence",
        verbose: bool = False,
    ) -> dict:
        output = {}

        for annotation in tqdm(self, desc="Saving", disable=not verbose):
            ann_dict = annotation.to_via_json(
                image_folder=image_folder,
                label_key=label_key,
                confidence_key=confidence_key,
            )

            key = f"{ann_dict['filename']}{ann_dict['size']}"
            output[key] = ann_dict

        return output

    def save_via_json(
        self,
        path: PathLike,
        *,
        image_folder: Path,
        label_key: str = "label_id",
        confidence_key: str = "confidence",
        verbose: bool = False,
    ):
        path = Path(path).expanduser().resolve()

        if path.suffix == "":
            path = path.with_suffix(".json")

        assert path.suffix == ".json", f"Path '{path}' suffix should be '.json'."

        output = self.to_via_json(
            image_folder=image_folder,
            label_key=label_key,
            confidence_key=confidence_key,
            verbose=verbose,
        )

        with open_atomic(path, "w") as f:
            json.dump(output, fp=f)

    @staticmethod
    def parse_names_file(path: PathLike) -> "dict[str, str]":
        """Parse .names file.

        Parameters:
        - path: the path to the .names file.

        Returns:
        - A dictionary mapping label number with label names."""
        # TODO: Add error handling
        path = Path(path).expanduser().resolve()

        return {str(i): v for i, v in enumerate(path.read_text().splitlines())}

    @staticmethod
    def parse_mid_file(path: PathLike) -> "dict[str, str]":
        path = Path(path).expanduser().resolve()

        with path.open() as f:
            reader = csv.DictReader(f, fieldnames=("LabelName", "DisplayName"))
            return {l["LabelName"]: l["DisplayName"] for l in reader}

    def show_stats(self, *, verbose: bool = False):
        """
        Print in the console a synthetic view of the dataset annotations (distribution of
        bounding boxes and images by label).
        """
        from rich import print as rprint
        from rich.table import Table

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
