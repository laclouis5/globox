from globox import Annotation, AnnotationSet, BoundingBox, BoxFormat
from globox.file_utils import glob
from .constants import *
import pytest
from PIL import Image
from math import isclose


def test_annotationset():
    a = Annotation(image_id="a")
    b = Annotation(image_id="b")
    c = Annotation(image_id="c")

    annotations = AnnotationSet(annotations=(a, b, c))

    assert annotations.image_ids == {"a", "b", "c"}
    assert len(annotations) == 3
    # assert annotations._id_to_label is None
    # assert annotations._id_to_imageid is None

    assert annotations["b"].image_id == "b"
    assert a in annotations
    assert annotations["b"] is b

    with pytest.raises(KeyError):
        _ = annotations["d"]

    assert annotations.get("d") is None

    with pytest.raises(AssertionError):
        annotations.add(a)

    annotations.add(Annotation(image_id="a", image_size=(100, 100)), override=True)
    assert annotations["a"].image_size is not None


def test_annotation_set_2():
    files = list(glob(pascal_path, ".xml"))
    set1 = AnnotationSet(Annotation.from_xml(f) for f in files[:50])
    set2 = AnnotationSet(Annotation.from_xml(f) for f in files[50:])
    set3 = set1 | set2
    annotation = Annotation.from_xml(files[0])

    assert set3.image_ids == (set(set1.image_ids).union(set(set2.image_ids)))

    with pytest.raises(AssertionError):
        set3 |= set1

    with pytest.raises(AssertionError):
        set3.add(annotation)


def test_openimage_conversion(tmp_path: Path):
    box = BoundingBox.create(label="dining,table", coords=(0, 0, 10, 10))
    annotation = Annotation(image_id="", boxes=[box])
    annotationset = AnnotationSet(annotations=[annotation])

    # "," in label
    with pytest.raises(ValueError):
        _ = annotationset.save_openimage(tmp_path / "cvat.csv")

    box.label = "dining_table"

    # No image_size specified
    with pytest.raises(ValueError):
        _ = annotationset.save_openimage(tmp_path / "cvat.csv")

    annotation.image_size = (640, 480)
    _ = annotationset.save_openimage(tmp_path / "cvat.csv")


def test_save_txt_conf_first(tmp_path: Path):
    annotation = Annotation(
        image_id="image_id",
        boxes=[
            BoundingBox(
                label="label", xmin=10, ymin=20, xmax=30, ymax=40, confidence=0.25
            ),
        ],
    )

    annotationset = AnnotationSet(annotations=[annotation])
    annotationset.save_txt(tmp_path)

    content = (tmp_path / "image_id.txt").read_text()

    assert content == "label 0.25 10 20 30 40"


def test_save_txt_conf_last(tmp_path: Path):
    annotation = Annotation(
        image_id="image_id",
        boxes=[
            BoundingBox(
                label="label", xmin=10, ymin=20, xmax=30, ymax=40, confidence=0.25
            ),
        ],
    )

    annotationset = AnnotationSet(annotations=[annotation])
    annotationset.save_txt(tmp_path, conf_last=True)

    content = (tmp_path / "image_id.txt").read_text()

    assert content == "label 10 20 30 40 0.25"


def test_save_yolo_conf_first(tmp_path: Path):
    annotation = Annotation(
        image_id="image_id",
        image_size=(1_000, 1_000),
        boxes=[
            BoundingBox.create(
                label="label",
                coords=(125, 250, 500, 1_000),
                confidence=0.25,
                box_format=BoxFormat.XYWH,
            ),
        ],
    )

    annotationset = AnnotationSet(annotations=[annotation])
    annotationset.save_yolo(tmp_path)

    content = (tmp_path / "image_id.txt").read_text()

    assert content == "label 0.25 0.125 0.25 0.5 1.0"


def test_save_yolo_conf_last(tmp_path: Path):
    annotation = Annotation(
        image_id="image_id",
        image_size=(1_000, 1_000),
        boxes=[
            BoundingBox.create(
                label="label",
                coords=(125, 250, 500, 1_000),
                confidence=0.25,
                box_format=BoxFormat.XYWH,
            ),
        ],
    )

    annotationset = AnnotationSet(annotations=[annotation])
    annotationset.save_yolo(tmp_path, conf_last=True)

    content = (tmp_path / "image_id.txt").read_text()

    assert content == "label 0.125 0.25 0.5 1.0 0.25"


def test_from_txt_conf_first(tmp_path: Path):
    file_path = tmp_path / "txt_1.txt"
    content = """label 0.25 10 20 30 40"""
    file_path.write_text(content)

    annotationset = AnnotationSet.from_txt(tmp_path)
    assert len(annotationset) == 1

    annotation = annotationset["txt_1.jpg"]
    assert len(annotation.boxes) == 1

    box = annotation.boxes[0]
    assert box.confidence == 0.25


def test_from_txt_conf_last(tmp_path: Path):
    file_path = tmp_path / "txt_2.txt"
    content = """label 10 20 30 40 0.25"""
    file_path.write_text(content)

    annotationset = AnnotationSet.from_txt(tmp_path, conf_last=True)
    assert len(annotationset) == 1

    annotation = annotationset["txt_2.jpg"]
    assert len(annotation.boxes) == 1

    box = annotation.boxes[0]
    assert box.confidence == 0.25


def test_from_coco_id_string():
    gts = AnnotationSet.from_coco(coco_str_id_path)

    assert len(gts) == 100  # images defined in coco_file
    assert gts["2007_001585.jpg"] is not None

    gts_box = gts["2007_001585.jpg"].boxes[0]

    coco_box = BoundingBox(
        label="bottle", xmin=58.0, ymin=158.0, xmax=58.0 + 14.0, ymax=158 + 33.0
    )

    assert gts_box == coco_box


def test_from_yolo_darknet(tmp_path: Path):
    path = tmp_path / "annotation.txt"
    path.write_text("label 0.25 0.25 0.25 0.5 0.5\nlabel_2 0.25 0.25 0.25 0.5 0.5")

    img_path = tmp_path / "annotation.jpg"
    img = Image.new(mode="1", size=(100, 100))
    img.save(img_path)

    annotations = AnnotationSet.from_yolo_darknet(tmp_path, image_folder=tmp_path)

    assert len(annotations) == 1

    annotation = annotations["annotation.jpg"]

    assert len(annotation.boxes) == 2
    assert annotation.image_id == "annotation.jpg"
    assert annotation.image_size == (100, 100)

    bbox = annotation.boxes[0]

    assert bbox.label == "label"
    assert bbox.confidence == 0.25

    (xmin, ymin, xmax, ymax) = bbox.ltrb

    assert isclose(xmin, 0.0)
    assert isclose(ymin, 0.0)
    assert isclose(xmax, 50.0)
    assert isclose(ymax, 50.0)

    assert isclose(bbox.confidence, 0.25)


def test_from_yolo_v5(tmp_path: Path):
    path = tmp_path / "annotation.txt"
    path.write_text("label 0.25 0.25 0.5 0.5 0.25\nlabel_2 0.25 0.25 0.5 0.5 0.25")

    img_path = tmp_path / "annotation.jpg"
    img = Image.new(mode="1", size=(100, 100))
    img.save(img_path)

    annotations = AnnotationSet.from_yolo_v5(tmp_path, image_folder=tmp_path)

    assert len(annotations) == 1

    annotation = annotations["annotation.jpg"]

    assert len(annotation.boxes) == 2
    assert annotation.image_id == "annotation.jpg"
    assert annotation.image_size == (100, 100)

    bbox = annotation.boxes[0]

    assert bbox.label == "label"
    assert bbox.confidence == 0.25

    (xmin, ymin, xmax, ymax) = bbox.ltrb

    assert isclose(xmin, 0.0)
    assert isclose(ymin, 0.0)
    assert isclose(xmax, 50.0)
    assert isclose(ymax, 50.0)

    assert isclose(bbox.confidence, 0.25)


def test_save_yolo_darknet(tmp_path: Path):
    bboxes = [
        BoundingBox(
            label="label_1", xmin=0.0, ymin=0.0, xmax=50.0, ymax=50.0, confidence=0.25
        ),
        BoundingBox(
            label="label_2", xmin=25.0, ymin=25.0, xmax=75.0, ymax=75.0, confidence=0.75
        ),
    ]

    annotation = Annotation(
        image_id="annotation.jpg", image_size=(100, 100), boxes=bboxes
    )
    annotations = AnnotationSet([annotation])
    annotations.save_yolo_darknet(tmp_path)

    path = tmp_path / "annotation.txt"

    content = path.read_text()

    assert content == "label_1 0.25 0.25 0.25 0.5 0.5\nlabel_2 0.75 0.5 0.5 0.5 0.5"


def test_save_yolo_v5(tmp_path: Path):
    bboxes = [
        BoundingBox(
            label="label_1", xmin=0.0, ymin=0.0, xmax=50.0, ymax=50.0, confidence=0.25
        ),
        BoundingBox(
            label="label_2", xmin=25.0, ymin=25.0, xmax=75.0, ymax=75.0, confidence=0.75
        ),
    ]

    annotation = Annotation(
        image_id="annotation.jpg", image_size=(100, 100), boxes=bboxes
    )
    annotations = AnnotationSet([annotation])
    annotations.save_yolo_v5(tmp_path)

    path = tmp_path / "annotation.txt"

    content = path.read_text()

    assert content == "label_1 0.25 0.25 0.5 0.5 0.25\nlabel_2 0.5 0.5 0.5 0.5 0.75"
