from ObjectDetectionEval import *
from pathlib import Path

from PIL import Image


def tests():
    try:
        b = BoundingBox(label="", xmin=10, ymin=10, xmax=5, ymax=10)
        raise ValueError
    except AssertionError:
        pass

    try:
        b = BoundingBox(label="", xmin=10, ymin=10, xmax=15, ymax=9)
        raise ValueError
    except AssertionError:
        pass

    try:
        b = BoundingBox(label="", xmin=0, ymin=0, xmax=0, ymax=0, confidence=-0.1)
        raise ValueError
    except AssertionError:
        pass

    try:
        b = BoundingBox(label="", xmin=0, ymin=0, xmax=0, ymax=0, confidence=1.1)
        raise ValueError
    except AssertionError:
        pass

    b = BoundingBox(label="", xmin=0, ymin=0, xmax=0, ymax=0, confidence=0.666)
    assert b.confidence == 0.666

    b = BoundingBox(label="", xmin=0, ymin=0, xmax=0, ymax=0, confidence=0.0)
    assert b.confidence == 0.0

    b = BoundingBox(label="", xmin=0, ymin=0, xmax=0, ymax=0, confidence=1.0)
    assert b.confidence == 1.0

    b = BoundingBox(label="", xmin=-1, ymin=0, xmax=1, ymax=2)
    assert b.xmin == -1
    assert b.ymin == 0
    assert b.xmax == 1
    assert b.ymax == 2
    assert b.xmid == 0.0
    assert b.ymid == 1.0
    assert b.width == 2.0
    assert b.height == 2.0
    assert b.area == 4.0

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

    a = Annotation(image_id="a", image_size=(20, 10))
    assert len(a.boxes) == 0
    
    b = Annotation(image_id="b", image_size=(20, 10))
    b.boxes.append(b1)
    assert len(a.boxes) == 0 and len(b.boxes) == 1

    c = Annotation(image_id="c", image_size=(20.0, 10.0), boxes = [b1, b2])
    assert c.image_width == 20 and c.image_height == 10
    assert len(c.boxes) == 2

    try:
        a = Annotation(image_id="_a", image_size=(1, 0))
        raise ValueError
    except AssertionError:
        pass

    try:
        a = Annotation(image_id="_a", image_size=(-1, 10))
        raise ValueError
    except AssertionError:
        pass

    try:
        a = Annotation(image_id="_a", image_size=(10.1, 10))
        raise ValueError
    except AssertionError:
        pass

    annotations = AnnotationSet((a, b, c))
    for id in annotations.image_ids:
        print(annotations[id].image_id)

def tests_2():
    data_path = Path("data/")
    gts_path = data_path / "gts/"
    dets_path = data_path / "dets/"
    image_folder = data_path / "images"
    visu_folder = data_path / "visualizations"
    names_file = gts_path / "yolo_format/obj.names"

    id_to_label = AnnotationSet.parse_names_file(names_file)
    labels = set(id_to_label.values())

    # Gts
    coco1_path = gts_path / "coco_format_v1/instances_default.json"
    coco2_path = gts_path / "coco_format_v2/instances_v2.json"
    yolo_path = gts_path / "yolo_format/obj_train_data"
    cvat_path = gts_path / "cvat_format/annotations.xml"
    labelme_path = gts_path / "labelme_format/"
    openimage_path = gts_path / "openimages_format/all_bounding_boxes.csv"
    pascal_path = gts_path / "pascalvoc_format/"

    # Dets
    coco_dets_path = dets_path / "coco_format/coco_dets.json"
    abs_ltrb = dets_path / "abs_ltrb/"
    abs_ltwh = dets_path / "abs_ltwh/"
    rel_ltwh = dets_path / "rel_ltwh/"

    # AnnotationSets
    coco1_set = AnnotationSet.from_coco(coco1_path)
    coco2_set = AnnotationSet.from_coco(coco2_path)
    yolo_set = AnnotationSet.from_yolo(yolo_path, image_folder, names_file)
    cvat_set = AnnotationSet.from_cvat(cvat_path)
    labelme_set = AnnotationSet.from_labelme(labelme_path)
    openimage_set = AnnotationSet.from_openimage(openimage_path, image_folder)
    pascal_set = AnnotationSet.from_xml(pascal_path)
    
    abs_ltrb_set = AnnotationSet.from_txt(abs_ltrb, image_folder, box_format=BoxFormat.LTRB, relative=False, id_to_label=id_to_label)
    abs_ltwh_set = AnnotationSet.from_txt(abs_ltwh, image_folder, box_format=BoxFormat.LTWH, relative=False, id_to_label=id_to_label)
    rel_ltwh_set = AnnotationSet.from_txt(rel_ltwh, image_folder, box_format=BoxFormat.LTWH, relative=True, id_to_label=id_to_label)
    coco_det_set = AnnotationSet.from_coco(coco_dets_path)

    gts_sets = [coco1_set, coco2_set, yolo_set, cvat_set, labelme_set, openimage_set, pascal_set]
    dets_sets = [abs_ltrb_set, abs_ltwh_set, rel_ltwh_set, coco_det_set]
    all_sets = dets_sets + gts_sets

    for i, s in enumerate(all_sets):
        assert s._labels() == labels
        for annotation in s:
            assert isinstance(annotation.image_id, str)
            assert isinstance(annotation.boxes, list)
            for box in annotation.boxes:
                assert isinstance(box.label, str)
                assert all(isinstance(c, float) for c in box.ltrb)
                assert any(c > 1 for c in box.ltrb), f"dataset {i}, {box.ltrb}"

    for s in dets_sets:
        for b in s.all_boxes:
            assert isinstance(b.confidence, float) and 0 <= b.confidence <= 1

    for i, s in enumerate(gts_sets):
        for b in s.all_boxes:
            assert b.confidence is None, f"dataset: {i}, Conf: {type(b.confidence)}"
            for c, s in zip(box.ltrb, annotation.image_size*2):
                assert c < s, f"dataset {i}, {c}, {annotation.image_size}, {annotation.image_id}"


    # for dataset in gts_sets:
    #     for annotation in dataset:
    #         image = visu_folder / annotation.image_id
    #         img = Image.open(image)
    #         annotation.draw(img)
    #         img.save(image)


if __name__ == "__main__":
    # tests()
    tests_2()