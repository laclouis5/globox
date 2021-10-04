from ObjectDetectionEval import *
from pathlib import Path


def main() -> None:
    darknet_dir = Path("/home/deepwater/Documents/darknet/")
    gts_path = Path("~/Documents/darknet/data/val/").expanduser()
    names = {str(i): n for i, n in enumerate(["maize", "bean", "leek", "stem_maize", "stem_bean", "stem_leek"])}
    gts = AnnotationSet.from_yolo(gts_path).map_labels(names)
    resolutions = [i*32 for i in range(10, 31)]

    for resolution in resolutions:
        dets_path = Path(f"~/Documents/darknet/results/yolov4-tiny_res_{resolution}/predictions/").expanduser()
        dets = AnnotationSet.from_yolo(dets_path, gts_path)

        evaluator = COCOEvaluator(gts, dets)
        evaluator.show_summary()
        evaluator.save_csv(darknet_dir / f"results/yolov4-tiny_res_{resolution}/evaluation_res_{resolution}.csv")

if __name__ == "__main__":
    main()