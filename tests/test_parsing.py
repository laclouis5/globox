from globox import BoxFormat, AnnotationSet
from globox.utils import all_equal
from .constants import *


def tests_parsing():
    coco1_set = AnnotationSet.from_coco(coco1_path)
    coco2_set = AnnotationSet.from_coco(coco2_path)
    coco3_set = AnnotationSet.from_coco(coco_str_id_path)
    coco_gts_set = AnnotationSet.from_coco(coco_gts_path)
    yolo_set = AnnotationSet.from_yolo_darknet(
        yolo_path, image_folder=image_folder
    ).map_labels(id_to_label)
    cvat_set = AnnotationSet.from_cvat(cvat_path)
    imagenet_set = AnnotationSet.from_imagenet(imagenet_path)
    labelme_set = AnnotationSet.from_labelme(labelme_path)
    openimage_set = AnnotationSet.from_openimage(
        openimage_path, image_folder=image_folder
    )
    pascal_set = AnnotationSet.from_pascal_voc(pascal_path)
    via_json_set = AnnotationSet.from_via_json(via_json_path, image_folder=image_folder)

    abs_ltrb_set = AnnotationSet.from_txt(
        abs_ltrb, image_folder=image_folder
    ).map_labels(id_to_label)
    abs_ltwh_set = AnnotationSet.from_txt(
        abs_ltwh, image_folder=image_folder, box_format=BoxFormat.LTWH
    ).map_labels(id_to_label)
    rel_ltwh_set = AnnotationSet.from_txt(
        rel_ltwh, image_folder=image_folder, box_format=BoxFormat.LTWH, relative=True
    ).map_labels(id_to_label)
    _ = coco_gts_set.from_results(coco_results_path)

    dets_sets = [abs_ltrb_set, abs_ltwh_set, rel_ltwh_set]
    gts_sets = [
        coco1_set,
        coco2_set,
        coco3_set,
        yolo_set,
        cvat_set,
        imagenet_set,
        labelme_set,
        openimage_set,
        pascal_set,
        via_json_set,
    ]
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

    for s in dets_sets:
        for b in s.all_boxes:
            assert isinstance(b._confidence, float) and 0 <= b._confidence <= 1

    for i, s in enumerate(gts_sets):
        for b in s.all_boxes:
            assert b._confidence is None, f"dataset: {i}, Conf: {type(b._confidence)}"
