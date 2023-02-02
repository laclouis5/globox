from globox import AnnotationSet
from globox.utils import all_equal
from pathlib import Path

from .constants import coco2_path, label_to_id, id_to_label, image_folder, labels


def test_conversion(tmp_path: Path):
    txt_dir = tmp_path / "txt/"
    yolo_dir = tmp_path / "yolo/"
    xml_dir = tmp_path / "xml/"
    cvat_path = tmp_path / "cvat.xml"
    coco_path = tmp_path / "coco.json"
    labelme_dir = tmp_path / "labelme"
    openimage_path = tmp_path / "openimage.csv"
    via_json_path = tmp_path / "via_json.json"

    boxes = AnnotationSet.from_coco(file_path=coco2_path)

    boxes.save_txt(txt_dir)
    boxes.save_yolo_darknet(yolo_dir, label_to_id=label_to_id)
    boxes.save_xml(xml_dir)
    boxes.save_cvat(cvat_path)
    boxes.save_coco(coco_path)
    boxes.save_labelme(labelme_dir)
    boxes.save_openimage(openimage_path)
    boxes.save_via_json(via_json_path, image_folder=image_folder)

    dets_sets = [
        AnnotationSet.from_txt(txt_dir, image_folder=image_folder),
        AnnotationSet.from_yolo_darknet(yolo_dir, image_folder=image_folder).map_labels(
            id_to_label
        ),
        AnnotationSet.from_xml(xml_dir),
        AnnotationSet.from_cvat(cvat_path),
        AnnotationSet.from_coco(coco_path),
        AnnotationSet.from_labelme(labelme_dir),
        AnnotationSet.from_openimage(openimage_path, image_folder=image_folder),
        AnnotationSet.from_via_json(via_json_path, image_folder=image_folder),
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
