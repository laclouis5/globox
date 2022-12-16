# Test if inveting gts & dets raises an error
# Test things with labels
# Etc...
# Test EvaluationItem and similar for invariants
# Test AP computation

from globox import *
from .constants import *
import pytest


def test_evaluation_item():
    ...


def test_evaluation_no_confidence():
    coco_gt = AnnotationSet.from_coco(coco_gts_path)

    evaluator = COCOEvaluator(
        ground_truths=coco_gt, 
        predictions=coco_gt,
    ).evaluate(
        iou_threshold=0.5,
        max_detections=100,
        size_range=COCOEvaluator.ALL_RANGE
    )
