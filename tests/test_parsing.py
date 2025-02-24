from globox import AnnotationSet, BoxFormat
from globox.utils import all_equal

from . import constants as C


def tests_parsing():
    coco1_set = AnnotationSet.from_coco(C.coco1_path)
    coco2_set = AnnotationSet.from_coco(C.coco2_path)
    coco3_set = AnnotationSet.from_coco(C.coco_str_id_path)
    coco_gts_set = AnnotationSet.from_coco(C.coco_gts_path)
    yolo_set = AnnotationSet.from_yolo_darknet(
        C.yolo_path, image_folder=C.image_folder
    ).map_labels(C.id_to_label)
    yolo_seg_set = AnnotationSet.from_yolo_seg(
        folder=C.yolo_seg_path, image_folder=C.image_folder
    ).map_labels(C.id_to_label)
    cvat_set = AnnotationSet.from_cvat(C.cvat_path)
    imagenet_set = AnnotationSet.from_imagenet(C.imagenet_path)
    labelme_set = AnnotationSet.from_labelme(C.labelme_path)
    labelme_poly_set = AnnotationSet.from_labelme(
        C.labelme_poly_path, include_poly=True
    )
    openimage_set = AnnotationSet.from_openimage(
        C.openimage_path, image_folder=C.image_folder
    )
    pascal_set = AnnotationSet.from_pascal_voc(C.pascal_path)
    via_json_set = AnnotationSet.from_via_json(
        C.via_json_path, image_folder=C.image_folder
    )

    abs_ltrb_set = AnnotationSet.from_txt(
        C.abs_ltrb, image_folder=C.image_folder
    ).map_labels(C.id_to_label)
    abs_ltwh_set = AnnotationSet.from_txt(
        C.abs_ltwh, image_folder=C.image_folder, box_format=BoxFormat.LTWH
    ).map_labels(C.id_to_label)
    rel_ltwh_set = AnnotationSet.from_txt(
        C.rel_ltwh,
        image_folder=C.image_folder,
        box_format=BoxFormat.LTWH,
        relative=True,
    ).map_labels(C.id_to_label)
    _ = coco_gts_set.from_results(C.coco_results_path)

    dets_sets = [abs_ltrb_set, abs_ltwh_set, rel_ltwh_set]
    gts_sets = [
        coco1_set,
        coco2_set,
        coco3_set,
        yolo_set,
        yolo_seg_set,
        cvat_set,
        imagenet_set,
        labelme_set,
        labelme_poly_set,
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
        assert s._labels() == C.labels
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
