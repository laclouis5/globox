from ObjectDetectionEval import *
from pathlib import Path


def main() -> None:
    gts_path = Path("/Volumes/DEEPWATER/backups/data/database_8.2_norm/val/")
    gts_path_2 = Path("/Volumes/DEEPWATER/backups/data/database_8.2_norm/train/")

    dets_path = Path("/Volumes/DEEPWATER/backups/results/yolov4-tiny_10/predictions")
    names_path = Path("/Volumes/DEEPWATER/backups/results/yolov4-tiny_10/obj.names")

    nb_to_label = AnnotationSet.parse_names_file(names_path)

    gts = AnnotationSet.from_yolo(gts_path).map_labels(nb_to_label) \
        + AnnotationSet.from_yolo(gts_path_2).map_labels(nb_to_label)
    # dets = AnnotationSet.from_yolo(dets_path, gts_path)

    gts.show_stats()

    # evaluator = COCOEvaluator(gts, dets, labels=["maize", "bean", "stem_maize", "stem_bean"])
    # evaluator.show_summary()

if __name__ == "__main__":
    main()