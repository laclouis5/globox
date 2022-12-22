from globox import COCOEvaluator
from .constants import *
from math import isclose
import pytest


@pytest.fixture
def evaluator() -> COCOEvaluator:
    coco_gt = AnnotationSet.from_coco(coco_gts_path)
    coco_det = coco_gt.from_results(coco_results_path)

    return COCOEvaluator(
        ground_truths=coco_gt, 
        predictions=coco_det,
    )


def test_evaluation(evaluator: COCOEvaluator):
    evaluator.clear_cache()
    
    # Official figures returned by pycocotools (see pycocotools_results.py)
    assert isclose(evaluator.ap(), 0.503647, abs_tol=1e-6)
    assert isclose(evaluator.ap_50(), 0.696973, abs_tol=1e-6)
    assert isclose(evaluator.ap_75(), 0.571667, abs_tol=1e-6)

    assert isclose(evaluator.ap_small(), 0.593252, abs_tol=1e-6)
    assert isclose(evaluator.ap_medium(), 0.557991, abs_tol=1e-6)
    assert isclose(evaluator.ap_large(), 0.489363, abs_tol=1e-6)

    assert isclose(evaluator.ar_1(), 0.386813, abs_tol=1e-6)
    assert isclose(evaluator.ar_10(), 0.593680, abs_tol=1e-6)
    assert isclose(evaluator.ar_100(), 0.595353, abs_tol=1e-6)

    assert isclose(evaluator.ar_small(), 0.654764, abs_tol=1e-6)
    assert isclose(evaluator.ar_medium(), 0.603130, abs_tol=1e-6)
    assert isclose(evaluator.ar_large(), 0.553744, abs_tol=1e-6)

    assert evaluator.evaluate.cache_info().currsize == 60


def test_evaluation_no_confidence():
    coco_gt = AnnotationSet.from_coco(coco_gts_path)

    COCOEvaluator(
        ground_truths=coco_gt, 
        predictions=coco_gt,
    ).evaluate(
        iou_threshold=0.5,
        max_detections=100,
        size_range=COCOEvaluator.ALL_RANGE
    )


def test_evaluator_no_confidence_invariance_to_bboxes_order():
    from globox import AnnotationSet, Annotation, BoundingBox
    
    gts = AnnotationSet(annotations=[
        Annotation("img_1", image_size=(100, 100), boxes=[
            BoundingBox(label="cat", xmin=0, ymin=0, xmax=3, ymax=3),
            BoundingBox(label="cat", xmin=1, ymin=0, xmax=4, ymax=3),
        ])
    ])
    
    dets_1 = AnnotationSet(annotations=[
        Annotation("img_1", image_size=(100, 100), boxes=[
            BoundingBox(label="cat", xmin=-1, ymin=0, xmax=2, ymax=3),
            BoundingBox(label="cat", xmin=0, ymin=0, xmax=3, ymax=3),
        ])
    ])
    
    dets_2 = AnnotationSet(annotations=[
        Annotation("img_1", image_size=(100, 100), boxes=[
            BoundingBox(label="cat", xmin=0, ymin=0, xmax=3, ymax=3),
            BoundingBox(label="cat", xmin=-1, ymin=0, xmax=2, ymax=3),
        ])
    ])
    
    evaluator_1 = COCOEvaluator(ground_truths=gts, predictions=dets_1)
    ap_1 = evaluator_1.ap_50()
    
    evaluator_2 = COCOEvaluator(ground_truths=gts, predictions=dets_2)
    ap_2 = evaluator_2.ap_50()
    
    assert isclose(ap_1, ap_2)
