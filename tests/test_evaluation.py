# Test if inveting gts & dets raises an error
# Test things with labels
# Etc...
# Test EvaluationItem and similar for invariants
# Test AP computation

from globox import BoundingBox, COCOEvaluator


def test_evaluation_item(): ...


def test_detection_matching_ignored_ground_truth():
    # As in the COCO reference evaluation, a detection without a valid match to an
    # in-range ground truth is ignored if it matches an out-of-range ground truth.
    large = (96.0**2, float("inf"))
    gts = [
        BoundingBox(label="cat", xmin=0.0, ymin=0.0, xmax=100.0, ymax=100.0),
        BoundingBox(label="cat", xmin=200.0, ymin=0.0, xmax=290.0, ymax=90.0),
    ]
    det = BoundingBox(
        label="cat", xmin=195.0, ymin=0.0, xmax=295.0, ymax=100.0, confidence=0.9
    )

    evaluation = COCOEvaluator.evaluate_boxes(
        [det], gts, iou_threshold=0.5, max_detections=100, size_range=large
    )

    assert evaluation.npos == 1
    assert evaluation.ndet == 0
