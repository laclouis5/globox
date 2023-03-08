from globox import BoxFormat, BoundingBox
from math import isclose
import pytest


def test_init():
    # xmax < xmin
    with pytest.raises(AssertionError):
        box = BoundingBox(label="", xmin=10, ymin=10, xmax=5, ymax=10)

    # ymax < ymin
    with pytest.raises(AssertionError):
        box = BoundingBox(label="", xmin=10, ymin=10, xmax=15, ymax=9)

    # Negative confidence
    with pytest.raises(AssertionError):
        box = BoundingBox(label="", xmin=0, ymin=0, xmax=0, ymax=0, confidence=-0.1)

    # Confidence > 1.0
    with pytest.raises(AssertionError):
        box = BoundingBox(label="", xmin=0, ymin=0, xmax=0, ymax=0, confidence=1.1)

    box = BoundingBox(label="", xmin=-1, ymin=0, xmax=1, ymax=2)
    assert box.xmin == -1
    assert box.ymin == 0
    assert box.xmax == 1
    assert box.ymax == 2
    assert box.xmid == 0.0
    assert box.ymid == 1.0
    assert box.width == 2.0
    assert box.height == 2.0
    assert box.area == 4.0
    assert box.is_ground_truth

    assert box.ltrb == (-1, 0, 1, 2)
    assert box.ltwh == (-1, 0, 2, 2)
    assert box.xywh == (0, 1, 2, 2)


def test_confidence():
    box = BoundingBox(label="", xmin=0, ymin=0, xmax=0, ymax=0, confidence=0.25)
    assert box.confidence == 0.25
    assert box.is_detection

    box = BoundingBox(label="", xmin=0, ymin=0, xmax=0, ymax=0, confidence=0.0)
    assert box.confidence == 0.0

    box = BoundingBox(label="", xmin=0, ymin=0, xmax=0, ymax=0, confidence=1.0)
    assert box.confidence == 1.0

    box = BoundingBox(label="", xmin=0, ymin=0, xmax=0, ymax=0)
    assert box.confidence is None
    assert box.is_ground_truth


def test_iou():
    b1 = BoundingBox(label="", xmin=0, ymin=0, xmax=10, ymax=10)
    b2 = BoundingBox(label="", xmin=5, ymin=0, xmax=10, ymax=10)
    assert b1.iou(b2) == 0.5

    b1 = BoundingBox(label="", xmin=0, ymin=0, xmax=10, ymax=10)
    b2 = BoundingBox(label="", xmin=10, ymin=10, xmax=15, ymax=15)
    assert b1.iou(b2) == 0.0

    b1 = BoundingBox(label="", xmin=0, ymin=0, xmax=10, ymax=10)
    b2 = BoundingBox(label="", xmin=20, ymin=20, xmax=30, ymax=30)
    assert b1.iou(b2) == 0.0

    b1 = BoundingBox(label="", xmin=0, ymin=0, xmax=0, ymax=0)
    b2 = BoundingBox(label="", xmin=5, ymin=5, xmax=15, ymax=15)
    assert b1.iou(b2) == 0.0

    b1 = BoundingBox(label="", xmin=0, ymin=0, xmax=0, ymax=0)
    b2 = BoundingBox(label="", xmin=0, ymin=0, xmax=0, ymax=0)
    assert b1.iou(b2) == 1.0

    b1 = BoundingBox(label="", xmin=0, ymin=0, xmax=0, ymax=0)
    b2 = BoundingBox(label="", xmin=1, ymin=1, xmax=1, ymax=1)
    assert b1.iou(b2) == 0.0


def test_create():
    with pytest.raises(AssertionError):
        box = BoundingBox.create(label="", coords=(0, 0, 0, 0), relative=True)

    box = BoundingBox.create(
        label="", coords=(0, 0, 0.75, 1.0), relative=True, image_size=(100, 200)
    )
    assert isclose(box.xmin, 0)
    assert isclose(box.ymin, 0)
    assert isclose(box.xmax, 75)
    assert isclose(box.ymax, 200)
    assert isclose(box.xmid, 37.5)
    assert isclose(box.ymid, 100)
    assert isclose(box.width, 75)
    assert isclose(box.height, 200)

    box = BoundingBox.create(
        label="",
        coords=(0.5, 0.5, 0.5, 1.0),
        box_format=BoxFormat.XYWH,
        relative=True,
        image_size=(100, 200),
    )
    assert isclose(box.xmin, 25)
    assert isclose(box.ymin, 0)
    assert isclose(box.xmax, 75)
    assert isclose(box.ymax, 200)
    assert isclose(box.xmid, 50)
    assert isclose(box.ymid, 100)
    assert isclose(box.width, 50)
    assert isclose(box.height, 200)

    box = BoundingBox.create(
        label="",
        coords=(0, 0, 0.5, 1.0),
        box_format=BoxFormat.LTWH,
        relative=True,
        image_size=(100, 200),
    )
    assert isclose(box.xmin, 0)
    assert isclose(box.ymin, 0)
    assert isclose(box.xmax, 50)
    assert isclose(box.ymax, 200)
    assert isclose(box.xmid, 25)
    assert isclose(box.ymid, 100)
    assert isclose(box.width, 50)
    assert isclose(box.height, 200)


def test_txt_conversion():
    box = BoundingBox.create(label="dining table", coords=(0, 0, 10, 10))
    with pytest.raises(AssertionError):
        _ = box.to_txt()
    box.to_txt(label_to_id={"dining table": "dining_table"})

    box = BoundingBox.create(label="dining_table", coords=(0, 0, 10, 10))
    with pytest.raises(AssertionError):
        _ = box.to_txt(separator="\n")


def test_to_txt_conf_last():
    box = BoundingBox.create(label="label", coords=(0, 0, 10, 10), confidence=0.5)

    line = box.to_txt(conf_last=True)
    assert line == "label 0 0 10 10 0.5"

    line = box.to_txt()
    assert line == "label 0.5 0 0 10 10"


def test_to_yolo_conf_last():
    box = BoundingBox.create(label="label", coords=(0, 0, 10, 10), confidence=0.25)

    line = box.to_yolo(image_size=(10, 10), conf_last=True)
    assert line == "label 0.5 0.5 1.0 1.0 0.25"

    line = box.to_yolo(image_size=(10, 10))
    assert line == "label 0.25 0.5 0.5 1.0 1.0"


def test_from_txt_conf_last():
    line = "label 10 20 30 40 0.25"
    box = BoundingBox.from_txt(line, conf_last=True)
    assert box.confidence == 0.25

    line = "label 0.25 10 20 30 40"
    box = BoundingBox.from_txt(line)
    assert box.confidence == 0.25


def test_from_yolo_v5():
    line = "label 0.25 0.25 0.5 0.5 0.25"
    bbox = BoundingBox.from_yolo_v5(line, image_size=(100, 100))

    assert bbox.label == "label"
    assert bbox.confidence == 0.25

    (xmin, ymin, xmax, ymax) = bbox.ltrb

    assert isclose(xmin, 0.0)
    assert isclose(ymin, 0.0)
    assert isclose(xmax, 50.0)
    assert isclose(ymax, 50.0)

    assert isclose(bbox.confidence, 0.25)


def test_from_yolo_v7():
    line = "label 0.25 0.25 0.5 0.5 0.25"
    bbox = BoundingBox.from_yolo_v7(line, image_size=(100, 100))

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
    string = bbox.to_yolo_darknet(image_size=(100, 100))

    assert string == "label 0.25 0.25 0.25 0.5 0.5"


def test_to_yolo_v5():
    bbox = BoundingBox(
        label="label", xmin=0.0, ymin=0.0, xmax=50.0, ymax=50.0, confidence=0.25
    )
    string = bbox.to_yolo_v5(image_size=(100, 100))

    assert string == "label 0.25 0.25 0.5 0.5 0.25"


def test_to_yolo_v7():
    bbox = BoundingBox(
        label="label", xmin=0.0, ymin=0.0, xmax=50.0, ymax=50.0, confidence=0.25
    )
    string = bbox.to_yolo_v7(image_size=(100, 100))

    assert string == "label 0.25 0.25 0.5 0.5 0.25"


def test_eq():
    box = BoundingBox(
        label="image_0.jpg", xmin=1.0, ymin=2.0, xmax=4.0, ymax=8.0, confidence=0.5
    )

    assert box == box

    # Same
    b1 = BoundingBox(
        label="image_0.jpg", xmin=1.0, ymin=2.0, xmax=4.0, ymax=8.0, confidence=0.5
    )

    assert box == b1

    # Different label
    b2 = BoundingBox(
        label="image_1.jpg", xmin=1.0, ymin=2.0, xmax=4.0, ymax=8.0, confidence=0.5
    )

    assert box != b2

    # Different coords
    b3 = BoundingBox(
        label="image_0.jpg", xmin=0.0, ymin=2.0, xmax=4.0, ymax=8.0, confidence=0.5
    )

    assert box != b3

    # Different confidence
    b4 = BoundingBox(
        label="image_0.jpg", xmin=1.0, ymin=2.0, xmax=4.0, ymax=8.0, confidence=0.25
    )

    assert box != b4

    # No confidence
    b5 = BoundingBox(
        label="image_0.jpg", xmin=1.0, ymin=2.0, xmax=4.0, ymax=8.0, confidence=None
    )

    assert box != b5

    # Different object
    with pytest.raises(NotImplementedError):
        _ = box == "Different object"
