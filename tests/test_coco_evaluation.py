from globox import AnnotationSet, Annotation, BoundingBox, COCOEvaluator
from .constants import *
from math import isclose
import pytest


@pytest.fixture
def coco_evaluator() -> COCOEvaluator:
    coco_gt = AnnotationSet.from_coco(coco_gts_path)
    coco_det = coco_gt.from_results(coco_results_path)

    return COCOEvaluator(
        ground_truths=coco_gt,
        predictions=coco_det,
    )


def test_evaluation(coco_evaluator: COCOEvaluator):
    # Official figures returned by pycocotools (see pycocotools_results.py)
    assert isclose(coco_evaluator.ap(), 0.503647, abs_tol=1e-6)
    assert isclose(coco_evaluator.ap_50(), 0.696973, abs_tol=1e-6)
    assert isclose(coco_evaluator.ap_75(), 0.571667, abs_tol=1e-6)

    assert isclose(coco_evaluator.ap_small(), 0.593252, abs_tol=1e-6)
    assert isclose(coco_evaluator.ap_medium(), 0.557991, abs_tol=1e-6)
    assert isclose(coco_evaluator.ap_large(), 0.489363, abs_tol=1e-6)

    assert isclose(coco_evaluator.ar_1(), 0.386813, abs_tol=1e-6)
    assert isclose(coco_evaluator.ar_10(), 0.593680, abs_tol=1e-6)
    assert isclose(coco_evaluator.ar_100(), 0.595353, abs_tol=1e-6)

    assert isclose(coco_evaluator.ar_small(), 0.654764, abs_tol=1e-6)
    assert isclose(coco_evaluator.ar_medium(), 0.603130, abs_tol=1e-6)
    assert isclose(coco_evaluator.ar_large(), 0.553744, abs_tol=1e-6)

    assert coco_evaluator._cached_evaluate.cache_info().currsize == 60


def test_clear_cache(coco_evaluator: COCOEvaluator):
    assert coco_evaluator._cached_evaluate.cache_info().currsize == 0

    _ = coco_evaluator.ap_50()
    assert coco_evaluator._cached_evaluate.cache_info().currsize != 0

    coco_evaluator.clear_cache()
    assert coco_evaluator._cached_evaluate.cache_info().currsize == 0


def test_evaluate_defaults(coco_evaluator: COCOEvaluator):
    ev1 = coco_evaluator.evaluate(iou_threshold=0.5)
    ev2 = coco_evaluator.ap_50_evaluation()

    # Equality instead of `isclose` since the evaluation default
    # should exactly be `ap_50_evaluation` and should be retreived
    # from the `COCOEvaluator` cache.
    assert ev1.ap() == ev2.ap()
    assert ev1.ar() == ev2.ar()


def test_evaluation_invariance_to_bboxes_order():
    gts = AnnotationSet(
        annotations=[
            Annotation(
                "img_1",
                image_size=(100, 100),
                boxes=[
                    BoundingBox(label="cat", xmin=0, ymin=0, xmax=3, ymax=3),
                    BoundingBox(label="cat", xmin=1, ymin=0, xmax=4, ymax=3),
                ],
            )
        ]
    )

    dets_1 = AnnotationSet(
        annotations=[
            Annotation(
                "img_1",
                image_size=(100, 100),
                boxes=[
                    BoundingBox(
                        label="cat", xmin=-1, ymin=0, xmax=2, ymax=3, confidence=0.6
                    ),
                    BoundingBox(
                        label="cat", xmin=0, ymin=0, xmax=3, ymax=3, confidence=0.5
                    ),
                ],
            )
        ]
    )

    dets_2 = AnnotationSet(
        annotations=[
            Annotation(
                "img_1",
                image_size=(100, 100),
                boxes=[
                    BoundingBox(
                        label="cat", xmin=0, ymin=0, xmax=3, ymax=3, confidence=0.5
                    ),
                    BoundingBox(
                        label="cat", xmin=-1, ymin=0, xmax=2, ymax=3, confidence=0.6
                    ),
                ],
            )
        ]
    )

    evaluator_1 = COCOEvaluator(ground_truths=gts, predictions=dets_1)
    ap_1 = evaluator_1.ap_50()
    evaluator_2 = COCOEvaluator(ground_truths=gts, predictions=dets_2)
    ap_2 = evaluator_2.ap_50()

    assert isclose(ap_1, ap_2)


def test_predictions_missing_confidence():
    gts = AnnotationSet(
        annotations=[
            Annotation(
                "img_1",
                image_size=(100, 100),
                boxes=[
                    BoundingBox(label="cat", xmin=0, ymin=0, xmax=3, ymax=3),
                ],
            )
        ]
    )

    dets = AnnotationSet(
        annotations=[
            Annotation(
                "img_1",
                image_size=(100, 100),
                boxes=[
                    BoundingBox(label="cat", xmin=0, ymin=0, xmax=3, ymax=3),
                ],
            )
        ]
    )

    with pytest.raises(AssertionError):
        evaluator = COCOEvaluator(ground_truths=gts, predictions=dets)
        _ = evaluator.ap_50()


def test_evaluator_params(coco_evaluator: COCOEvaluator):
    with pytest.raises(AssertionError):
        coco_evaluator.evaluate(iou_threshold=1.1)

    with pytest.raises(AssertionError):
        coco_evaluator.evaluate(iou_threshold=-0.1)

    with pytest.raises(AssertionError):
        coco_evaluator.evaluate(
            iou_threshold=0.5,
            max_detections=-1,
        )

    with pytest.raises(AssertionError):
        coco_evaluator.evaluate(iou_threshold=0.5, size_range=(2.0, 1.0))

    with pytest.raises(AssertionError):
        coco_evaluator.evaluate(iou_threshold=0.5, size_range=(-1.0, 10.0))
