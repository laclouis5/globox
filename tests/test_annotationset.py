from globox import Annotation, AnnotationSet
from globox.file_utils import glob
from .constants import *
import pytest


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