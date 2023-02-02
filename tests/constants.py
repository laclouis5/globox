from globox import AnnotationSet
from pathlib import Path


data_path = (Path(__file__).parent / "globox_test_data/").resolve()
gts_path = data_path / "gts/"
dets_path = data_path / "dets/"
image_folder = data_path / "images/"
names_file = gts_path / "yolo_format/obj.names"

coco_str_id_path = gts_path / "coco_format_v3/instances_v3.json"
coco1_path = gts_path / "coco_format_v1/instances_default.json"
coco2_path = gts_path / "coco_format_v2/instances_v2.json"
coco_gts_path = data_path / "coco_eval/ground_truths.json"
yolo_path = gts_path / "yolo_format/obj_train_data"
cvat_path = gts_path / "cvat_format/annotations.xml"
imagenet_path = gts_path / "imagenet_format/Annotations/"
labelme_path = gts_path / "labelme_format/"
openimage_path = gts_path / "openimages_format/all_bounding_boxes.csv"
pascal_path = gts_path / "pascalvoc_format/"
via_json_path = gts_path / "via/annotations.json"

coco_results_path = data_path / "coco_eval/results.json"
abs_ltrb = dets_path / "abs_ltrb/"
abs_ltwh = dets_path / "abs_ltwh/"
rel_ltwh = dets_path / "rel_ltwh/"

id_to_label = AnnotationSet.parse_names_file(names_file)
labels = set(id_to_label.values())
label_to_id = {v: k for k, v in id_to_label.items()}
