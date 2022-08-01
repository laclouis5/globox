from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def main():
    gt_path = "ground_truths.json"
    det_path = "results.json"

    gts = COCO(gt_path)
    dets = gts.loadRes(det_path)

    evaluator = COCOeval(gts, dets, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()


if __name__ == "__main__":
    main()