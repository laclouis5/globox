from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pathlib import Path


def main():
    gt_path = Path("data/coco/ground_truths.json").resolve()
    det_path = Path("data/coco/results.json").resolve()

    gts = COCO(str(gt_path))
    dets = gts.loadRes(str(det_path))

    evaluator = COCOeval(gts, dets, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()


if __name__ == "__main__":
    main()