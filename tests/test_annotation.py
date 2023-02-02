from globox import BoundingBox, Annotation, BoxFormat
import pytest
from pathlib import Path
from math import isclose


def test_init():
    a = Annotation(image_id="a")
    assert len(a.boxes) == 0
    assert a.image_size is None

    b = Annotation(image_id="b")
    b1 = BoundingBox(label="b1", xmin=0, ymin=0, xmax=0, ymax=0)
    b.add(b1)
    assert len(b.boxes) == 1

    b2 = BoundingBox(label="b2", xmin=5, ymin=0, xmax=10, ymax=10)
    c = Annotation(image_id="c", image_size=(20, 10), boxes=[b1, b2])
    assert c.image_width == 20 and c.image_height == 10
    assert len(c.boxes) == 2


def test_image_size():
    with pytest.raises(AssertionError):
        _ = Annotation(image_id="_a", image_size=(1, 0))

    with pytest.raises(AssertionError):
        _ = Annotation(image_id="_a", image_size=(-1, 10))

    with pytest.raises(AssertionError):
        _ = Annotation(image_id="_a", image_size=(10, 10.2))  # type: ignore


def test_map_labels():
    b1 = BoundingBox(label="b1", xmin=0, ymin=0, xmax=0, ymax=0)
    b2 = BoundingBox(label="b2", xmin=5, ymin=0, xmax=10, ymax=10)
    annotation = Annotation(image_id="image", boxes=[b1, b2])
    annotation.map_labels({"b1": "B1", "b2": "B2"})
    b1, b2 = annotation.boxes

    assert b1.label == "B1"
    assert b2.label == "B2"

    with pytest.raises(KeyError):
        annotation.map_labels({"B1": "b1"})  # B2 missing


def test_from_txt_conf_first(tmp_path: Path):
    file_path = tmp_path / "txt_1.txt"
    content = """label 0.25 10 20 30 40"""
    file_path.write_text(content)

    annotation = Annotation.from_txt(file_path, image_id="txt_1.jpg")

    assert len(annotation.boxes) == 1
    assert annotation.boxes[0].confidence == 0.25


def test_from_txt_conf_last(tmp_path: Path):
    file_path = tmp_path / "txt_2.txt"
    content = """label 10 20 30 40 0.25"""
    file_path.write_text(content)

    annotation = Annotation.from_txt(file_path, image_id="txt_2.jpg", conf_last=True)

    assert len(annotation.boxes) == 1
    assert annotation.boxes[0].confidence == 0.25


def test_from_yolo_conf_first(tmp_path: Path):
    file_path = tmp_path / "yolo_1.txt"
    content = """label 0.25 0.1 0.2 0.3 0.4"""
    file_path.write_text(content)

    annotation = Annotation.from_yolo_darknet(
        file_path, image_size=(640, 480), image_id="yolo_1.jpg"
    )

    assert len(annotation.boxes) == 1
    assert annotation.boxes[0].confidence == 0.25


def test_from_yolo_conf_last(tmp_path: Path):
    file_path = tmp_path / "yolo_2.txt"
    content = """label 0.1 0.2 0.3 0.4 0.25"""
    file_path.write_text(content)

    annotation = Annotation.from_yolo_v5(
        file_path, image_size=(640, 480), image_id="yolo_2.jpg"
    )

    assert len(annotation.boxes) == 1
    assert annotation.boxes[0].confidence == 0.25


def test_save_txt_conf_first(tmp_path: Path):
    file_path = tmp_path / "txt_first.txt"

    annotation = Annotation(
        image_id="image_id",
        boxes=[
            BoundingBox(
                label="label", xmin=10, ymin=20, xmax=30, ymax=40, confidence=0.25
            ),
        ],
    )

    annotation.save_txt(file_path)
    content = file_path.read_text()

    assert content == "label 0.25 10 20 30 40"


def test_save_txt_conf_last(tmp_path: Path):
    file_path = tmp_path / "txt_first.txt"

    annotation = Annotation(
        image_id="image_id",
        boxes=[
            BoundingBox(
                label="label", xmin=10, ymin=20, xmax=30, ymax=40, confidence=0.25
            ),
        ],
    )

    annotation.save_txt(file_path, conf_last=True)
    content = file_path.read_text()

    assert content == "label 10 20 30 40 0.25"


def test_save_yolo_conf_first(tmp_path: Path):
    file_path = tmp_path / "txt_first.txt"

    annotation = Annotation(
        image_id="image_id",
        image_size=(1_000, 1_000),
        boxes=[
            BoundingBox.create(
                label="label",
                coords=(125, 250, 500, 1_000),
                confidence=0.25,
                box_format=BoxFormat.XYWH,
            )
        ],
    )

    annotation.save_yolo(file_path)
    content = file_path.read_text()

    assert content == "label 0.25 0.125 0.25 0.5 1.0"


def test_save_yolo_conf_last(tmp_path: Path):
    file_path = tmp_path / "txt_last.txt"

    annotation = Annotation(
        image_id="image_id",
        image_size=(1_000, 1_000),
        boxes=[
            BoundingBox.create(
                label="label",
                coords=(125, 250, 500, 1_000),
                confidence=0.25,
                box_format=BoxFormat.XYWH,
            )
        ],
    )

    annotation.save_yolo(file_path, conf_last=True)
    content = file_path.read_text()

    assert content == "label 0.125 0.25 0.5 1.0 0.25"


def test_from_yolo_darknet(tmp_path: Path):
    path = tmp_path / "annotation.txt"
    path.write_text("label 0.25 0.25 0.25 0.5 0.5")

    annotation = Annotation.from_yolo_darknet(path, image_size=(100, 100))

    assert len(annotation.boxes) == 1
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
    path.write_text("label 0.25 0.25 0.5 0.5 0.25")

    annotation = Annotation.from_yolo_v5(path, image_size=(100, 100))

    assert len(annotation.boxes) == 1
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


def test_to_yolo_darknet():
    bbox = BoundingBox(
        label="label", xmin=0.0, ymin=0.0, xmax=50.0, ymax=50.0, confidence=0.25
    )
    annotation = Annotation(
        image_id="annotation.jpg", image_size=(100, 100), boxes=[bbox]
    )

    content = annotation.to_yolo_darknet()

    assert content == "label 0.25 0.25 0.25 0.5 0.5"


def test_to_yolo_v5():
    bbox = BoundingBox(
        label="label", xmin=0.0, ymin=0.0, xmax=50.0, ymax=50.0, confidence=0.25
    )
    annotation = Annotation(
        image_id="annotation.jpg", image_size=(100, 100), boxes=[bbox]
    )

    content = annotation.to_yolo_v5()

    assert content == "label 0.25 0.25 0.5 0.5 0.25"


def test_save_yolo_darknet(tmp_path: Path):
    bbox = BoundingBox(
        label="label", xmin=0.0, ymin=0.0, xmax=50.0, ymax=50.0, confidence=0.25
    )
    annotation = Annotation(
        image_id="annotation.jpg", image_size=(100, 100), boxes=[bbox]
    )

    path = tmp_path / "annotation.txt"
    annotation.save_yolo_darknet(path)

    content = path.read_text()

    assert content == "label 0.25 0.25 0.25 0.5 0.5"


def test_save_yolo_v5(tmp_path: Path):
    bbox = BoundingBox(
        label="label", xmin=0.0, ymin=0.0, xmax=50.0, ymax=50.0, confidence=0.25
    )
    annotation = Annotation(
        image_id="annotation.jpg", image_size=(100, 100), boxes=[bbox]
    )

    path = tmp_path / "annotation.txt"
    annotation.save_yolo_v5(path)

    content = path.read_text()

    assert content == "label 0.25 0.25 0.5 0.5 0.25"
