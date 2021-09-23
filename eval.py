from ObjectDetectionEval import *
from pathlib import Path


def main() -> None:
    gts_path = Path("~/Documents/darknet/data/database_12.0_norm/val/").expanduser()
    dets_path = Path("~/Documents/darknet/results/yolov4-tiny_15/predictions/").expanduser()

    names = {str(i): n for i, n in enumerate(["maize", "bean", "leek", "stem_maize", "stem_bean", "stem_leek"])}

    gts = AnnotationSet.from_yolo(gts_path).map_labels(names)
    dets = AnnotationSet.from_yolo(dets_path, gts_path)

    evaluation = COCOEvaluator(gts, dets)
    evaluation.show_summary()


if __name__ == "__main__":
    main()