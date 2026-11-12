from pathlib import Path

from globox import Annotation, AnnotationSet, BoundingBox
from globox.utils import all_equal

from .constants import coco2_path, id_to_label, image_folder, label_to_id, labels


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


def test_conversion_non_ascii_labels(tmp_path: Path):
    # Files are UTF-8, whatever the platform's default encoding is.
    labels = ["caf\u00e9", "\u732b"]
    boxes = [
        BoundingBox(label=label, xmin=10.0, ymin=20.0, xmax=110.0, ymax=80.0)
        for label in labels
    ]
    annotations = AnnotationSet(
        [Annotation(image_id="image.jpg", image_size=(200, 100), boxes=boxes)]
    )

    txt_dir = tmp_path / "txt/"
    xml_dir = tmp_path / "xml/"
    cvat_path = tmp_path / "cvat.xml"
    coco_path = tmp_path / "coco.json"
    labelme_dir = tmp_path / "labelme"
    openimage_path = tmp_path / "openimage.csv"

    annotations.save_txt(txt_dir)
    annotations.save_xml(xml_dir)
    annotations.save_cvat(cvat_path)
    annotations.save_coco(coco_path, auto_ids=True)
    annotations.save_labelme(labelme_dir)
    annotations.save_openimage(openimage_path)

    assert labels[1] in openimage_path.read_text(encoding="utf-8")

    parsed_sets = [
        AnnotationSet.from_txt(txt_dir),
        AnnotationSet.from_xml(xml_dir),
        AnnotationSet.from_cvat(cvat_path),
        AnnotationSet.from_coco(coco_path),
        AnnotationSet.from_labelme(labelme_dir),
    ]

    for parsed in parsed_sets:
        assert sorted(box.label for box in parsed.all_boxes) == sorted(labels)
