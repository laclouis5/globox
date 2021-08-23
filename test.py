from ObjectDetectionEval import *

from pathlib import Path
from time import perf_counter
from timeit import timeit
from math import isclose

# from PIL import Image


data_path = Path("data/")
gts_path = data_path / "gts/"
dets_path = data_path / "dets/"
image_folder = data_path / "images"
visu_folder = data_path / "visualizations"
names_file = gts_path / "yolo_format/obj.names"

# Gts
coco1_path = gts_path / "coco_format_v1/instances_default.json"
coco2_path = gts_path / "coco_format_v2/instances_v2.json"
yolo_path = gts_path / "yolo_format/obj_train_data"
cvat_path = gts_path / "cvat_format/annotations.xml"
imagenet_path = gts_path / "imagenet_format/Annotations/"
labelme_path = gts_path / "labelme_format/"
openimage_path = gts_path / "openimages_format/all_bounding_boxes.csv"
pascal_path = gts_path / "pascalvoc_format/"

# Dets
coco_dets_path = dets_path / "coco_format/coco_dets.json"
abs_ltrb = dets_path / "abs_ltrb/"
abs_ltwh = dets_path / "abs_ltwh/"
rel_ltwh = dets_path / "rel_ltwh/"

id_to_label = AnnotationSet.parse_names_file(names_file)
labels = set(id_to_label.values())
label_to_id = {v: k for k, v in id_to_label.items()}

def tests_bounding_box():
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
        assert id == annotations[id].image_id

def test_annotationset():
    files = list(glob(pascal_path, ".xml")) 
    set1 = AnnotationSet(Annotation.from_xml(f) for f in files[:50])
    set2 = AnnotationSet(Annotation.from_xml(f) for f in files[50:])  
    set3 = set1 + set2 
    annotation = Annotation.from_xml(files[0])
    
    assert set3.image_ids == (set(set1.image_ids).union(set(set2.image_ids)))

    try:
        set3 += set1
        raise ValueError
    except AssertionError:
        pass

    try:
        set3.add(annotation)
        raise ValueError
    except AssertionError:
        pass

def tests_parsing():
    start = perf_counter()

    coco1_set = AnnotationSet.from_coco(coco1_path)
    coco2_set = AnnotationSet.from_coco(coco2_path)
    yolo_set = AnnotationSet.from_yolo(yolo_path, image_folder).map_labels(id_to_label)
    cvat_set = AnnotationSet.from_cvat(cvat_path)
    imagenet_set = AnnotationSet.from_xml(folder=imagenet_path)
    labelme_set = AnnotationSet.from_labelme(labelme_path)
    openimage_set = AnnotationSet.from_openimage(openimage_path, image_folder)
    pascal_set = AnnotationSet.from_xml(pascal_path)
    
    abs_ltrb_set = AnnotationSet.from_txt(abs_ltrb, image_folder).map_labels(id_to_label)
    abs_ltwh_set = AnnotationSet.from_txt(abs_ltwh, image_folder, box_format=BoxFormat.LTWH).map_labels(id_to_label)
    rel_ltwh_set = AnnotationSet.from_txt(rel_ltwh, image_folder, box_format=BoxFormat.LTWH, relative=True).map_labels(id_to_label)
    coco_det_set = AnnotationSet.from_coco(coco_dets_path)

    elapsed = perf_counter() - start
    print(f"Parsing took {elapsed*1_000:.2f}ms")

    dets_sets = [abs_ltrb_set, abs_ltwh_set, rel_ltwh_set, coco_det_set]
    gts_sets = [coco1_set, coco2_set, yolo_set, cvat_set, imagenet_set, labelme_set, openimage_set, pascal_set]
    all_sets = dets_sets + gts_sets

    assert all_equal(s.image_ids for s in gts_sets)
    assert all_equal(s.image_ids for s in dets_sets)
    assert all_equal(len(s) for s in dets_sets)
    assert all_equal(len(s) for s in gts_sets)

    for image_id in all_sets[0].image_ids:
        assert all_equal(len(d[image_id].boxes) for d in dets_sets)
        assert all_equal(d[image_id].image_size for d in dets_sets)
        assert all_equal(len(d[image_id].boxes) for d in gts_sets)
        assert all_equal(d[image_id].image_size for d in gts_sets)

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

    # for s in all_sets:
    #     s.show_stats()

    # for dataset in gts_sets:
    #     for annotation in dataset:
    #         image = visu_folder / annotation.image_id
    #         img = Image.open(image)
    #         annotation.draw(img)
    #         img.save(image)

def test_conversion():
    save_dir = Path("/tmp/").expanduser()
    txt_dir = save_dir / "txt/"
    yolo_dir = save_dir / "yolo/"
    xml_dir = save_dir / "xml/"
    cvat_path = save_dir / "cvat.xml"
    coco_path = save_dir / "coco.json"
    labelme_dir = save_dir / "labelme"
    openimage_path = save_dir / "openimage.csv"

    boxes = AnnotationSet.from_coco(file_path=coco2_path)

    start = perf_counter()

    boxes.save_txt(txt_dir)
    boxes.save_yolo(yolo_dir, label_to_id=label_to_id)
    boxes.save_xml(xml_dir)
    boxes.save_cvat(cvat_path)
    boxes.save_coco(coco_path)
    boxes.save_labelme(labelme_dir)
    boxes.save_openimage(openimage_path)

    elapsed = perf_counter() - start
    print(f"Conversion took {elapsed*1_000:.2f}ms")

    dets_sets = [
        AnnotationSet.from_txt(txt_dir, image_folder),
        AnnotationSet.from_yolo(yolo_dir, image_folder).map_labels(id_to_label),
        AnnotationSet.from_xml(xml_dir),
        AnnotationSet.from_cvat(cvat_path),
        AnnotationSet.from_coco(coco_path),
        AnnotationSet.from_labelme(labelme_dir),
        AnnotationSet.from_openimage(openimage_path, image_folder),
    ]

    all_sets = dets_sets

    assert all_equal(s.image_ids for s in dets_sets)    
    assert all_equal(len(s) for s in dets_sets)

    for image_id in all_sets[0].image_ids:
        assert all_equal(len(d[image_id].boxes) for d in dets_sets)
        assert all_equal(d[image_id].image_size for d in dets_sets)

    for i, s in enumerate(all_sets):
        assert s._labels() == labels, f"{i}"
        for annotation in s:
            assert isinstance(annotation.image_id, str)
            assert isinstance(annotation.boxes, list)
            for box in annotation.boxes:
                assert isinstance(box.label, str)
                assert all(isinstance(c, float) for c in box.ltrb)
                assert any(c > 1 for c in box.ltrb), f"dataset {i}, {box.ltrb}"

def test_speed():
    iterations = 1
    base = Path("/Users/louislac/Downloads/val/")
    images = base / "images"

    coco = base / "coco.json"
    cvat = base / "cvat.xml"
    oi = base / "openimage.csv"
    labelme = base / "labelme"
    xml = base / "xml"
    yolo = base / "yolo"
    txt = base / "txt"

    gts = AnnotationSet.from_coco(coco)
    labels = gts._labels()
    label_to_id = {str(l): i for i, l in enumerate(labels)}
    
    coco_s = timeit(lambda: gts.save_coco(coco), number=iterations) / iterations
    cvat_s = timeit(lambda: gts.save_cvat(cvat), number=iterations) / iterations
    oi_s = timeit(lambda: gts.save_openimage(oi), number=iterations) / iterations
    labelme_s = timeit(lambda: gts.save_labelme(labelme), number=iterations) / iterations
    xml_s = timeit(lambda: gts.save_xml(xml), number=iterations) / iterations
    yolo_s = timeit(lambda: gts.save_yolo(yolo, label_to_id), number=iterations) / iterations
    txt_s = timeit(lambda: gts.save_txt(txt, label_to_id), number=iterations) / iterations

    coco_p = timeit(lambda: AnnotationSet.from_coco(coco), number=iterations) / iterations
    cvat_p = timeit(lambda: AnnotationSet.from_cvat(cvat), number=iterations) / iterations
    oi_p = timeit(lambda: AnnotationSet.from_openimage(oi, images), number=iterations) / iterations
    labelme_p = timeit(lambda: AnnotationSet.from_labelme(labelme), number=iterations) / iterations
    xml_p = timeit(lambda: AnnotationSet.from_xml(xml), number=iterations) / iterations
    yolo_p = timeit(lambda: AnnotationSet.from_yolo(yolo, images), number=iterations) / iterations
    txt_p = timeit(lambda: AnnotationSet.from_txt(txt, images), number=iterations) / iterations

    stats_t = timeit(lambda: gts.show_stats(), number=iterations) / iterations

    print(coco_p, cvat_p, oi_p, labelme_p, xml_p, yolo_p, txt_p)
    print(coco_s, cvat_s, oi_s, labelme_s, xml_s, yolo_s, txt_s)
    print(stats_t)
    
def test_evaluation():
    coco_test_path = Path("data/test_coco_eval")

    coco_gt = AnnotationSet.from_coco(
        coco_test_path / "gts/sampled_gts_bbox_area.json")
    coco_det = AnnotationSet.from_coco(
        coco_test_path / "dets/sampled_bbox_results.json")

    evaluator = COCOEvaluator(coco_gt, coco_det)
    evaluator.show_summary()

    assert isclose(evaluator.ap(), 0.503647, rel_tol=1e-4)
    assert isclose(evaluator.ap_50(), 0.696973, rel_tol=1e-4)
    assert isclose(evaluator.ap_75(), 0.571667, rel_tol=1e-4)

    assert isclose(evaluator.ap_small(), 0.593252, rel_tol=1e-4)
    assert isclose(evaluator.ap_medium(), 0.557991, rel_tol=1e-4)
    assert isclose(evaluator.ap_large(), 0.489363, rel_tol=1e-4)

    assert isclose(evaluator.ar_1(), 0.386813, rel_tol=1e-4)
    assert isclose(evaluator.ar_10(), 0.593680, rel_tol=1e-4)
    assert isclose(evaluator.ar_100(), 0.595353, rel_tol=1e-4)

    assert isclose(evaluator.ar_small(), 0.654764, rel_tol=1e-4)
    assert isclose(evaluator.ar_medium(), 0.603130, rel_tol=1e-4)
    assert isclose(evaluator.ar_large(), 0.553744, rel_tol=1e-4)

if __name__ == "__main__":
    # tests_bounding_box()
    # test_annotationset()
    # tests_parsing()
    # test_conversion()
    # test_speed()
    test_evaluation()