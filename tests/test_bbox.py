from globox import BoxFormat, BoundingBox
from math import isclose
import pytest


def test_init():
    with pytest.raises(AssertionError):
        box = BoundingBox(label="", xmin=10, ymin=10, xmax=5, ymax=10)

    with pytest.raises(AssertionError):
        box = BoundingBox(label="", xmin=10, ymin=10, xmax=15, ymax=9)

    with pytest.raises(AssertionError):
        box = BoundingBox(label="", xmin=0, ymin=0, xmax=0, ymax=0, confidence=-0.1)

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
    box = BoundingBox(label="", xmin=0, ymin=0, xmax=0, ymax=0, confidence=0.666)
    assert box._confidence == 0.666
    assert box.is_detection

    box = BoundingBox(label="", xmin=0, ymin=0, xmax=0, ymax=0, confidence=0.0)
    assert box._confidence == 0.0

    box = BoundingBox(label="", xmin=0, ymin=0, xmax=0, ymax=0, confidence=1.0)
    assert box._confidence == 1.0

    box = BoundingBox(label="", xmin=0, ymin=0, xmax=0, ymax=0)
    assert box._confidence is None
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
    
    box = BoundingBox.create(label="", coords=(0, 0, 0.75, 1.0), relative=True, image_size=(100, 200))
    assert isclose(box.xmin, 0)
    assert isclose(box.ymin, 0)
    assert isclose(box.xmax, 75)
    assert isclose(box.ymax, 200)
    assert isclose(box.xmid, 37.5)
    assert isclose(box.ymid, 100)
    assert isclose(box.width, 75)
    assert isclose(box.height, 200)

    box = BoundingBox.create(label="", coords=(0.5, 0.5, 0.5, 1.0), box_format=BoxFormat.XYWH, relative=True, image_size=(100, 200))
    assert isclose(box.xmin, 25)
    assert isclose(box.ymin, 0)
    assert isclose(box.xmax, 75)
    assert isclose(box.ymax, 200)
    assert isclose(box.xmid, 50)
    assert isclose(box.ymid, 100)
    assert isclose(box.width, 50)
    assert isclose(box.height, 200)

    box = BoundingBox.create(label="", coords=(0, 0, 0.5, 1.0), box_format=BoxFormat.LTWH, relative=True, image_size=(100, 200))
    assert isclose(box.xmin, 0)
    assert isclose(box.ymin, 0)
    assert isclose(box.xmax, 50)
    assert isclose(box.ymax, 200)
    assert isclose(box.xmid, 25)
    assert isclose(box.ymid, 100)
    assert isclose(box.width, 50)
    assert isclose(box.height, 200)
