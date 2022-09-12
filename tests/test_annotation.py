from globox import BoundingBox, Annotation
import pytest


def test_init():
    a = Annotation(image_id="a")
    assert len(a.boxes) == 0
    assert a.image_size is None
    
    b = Annotation(image_id="b")
    b1 = BoundingBox(label="b1", xmin=0, ymin=0, xmax=0, ymax=0)
    b.add(b1)
    assert len(b.boxes) == 1

    b2 = BoundingBox(label="b2", xmin=5, ymin=0, xmax=10, ymax=10)
    c = Annotation(image_id="c", image_size=(20.0, 10.0), boxes=[b1, b2])
    assert c.image_width == 20 and c.image_height == 10
    assert len(c.boxes) == 2


def test_image_size():
    with pytest.raises(AssertionError):
        _ = Annotation(image_id="_a", image_size=(1, 0))

    with pytest.raises(AssertionError):
        _ = Annotation(image_id="_a", image_size=(-1, 10))

    with pytest.raises(AssertionError):
        _ = Annotation(image_id="_a", image_size=(10.1, 10))


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