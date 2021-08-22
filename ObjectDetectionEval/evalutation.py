from re import A
from .annotation import Annotation
from .annotationset import AnnotationSet
from .boundingbox import BoundingBox
from .utils import grouping, all_equal

from collections import defaultdict
from typing import Mapping, Optional
from copy import copy

import numpy as np
from rich.table import Table
from rich import print as pprint
from tqdm import tqdm


class EvaluationItem:

    def __init__(self, 
        tps: list[bool] = None,
        scores: list[float] = None,
        npos: int = 0
    ) -> None:
        self._tps = tps or []
        self._scores = scores or []
        self._npos = npos
        self._cache = {}

        assert self._npos >= 0
        assert len(self._tps) == len(self._scores)

    def __iadd__(self, other: "EvaluationItem") -> "EvaluationItem":
        self._tps += other._tps
        self._scores += other._scores
        self._npos += other._npos
        self.clear_cache()
        return self

    def __add__(self, other: "EvaluationItem") -> "EvaluationItem":
        copy_ = copy(self)
        copy_ += other
        return copy_
    
    @property
    def ndet(self) -> int:
        return len(self._tps)

    @property
    def npos(self) -> int:
        return self._npos

    def tp(self) -> int:
        tp = self._cache.get("tp", sum(self._tps))
        self._cache["tp"] = tp
        assert tp <= self.npos
        return tp

    def ap(self) -> Optional[float]:
        if (ap := self._cache.get("ap")) is not None:
            return ap
        ap = _compute_ap(self._scores, self._tps, self._npos)
        self._cache["ap"] = ap
        return ap

    def ar(self) -> Optional[float]:
        if (ar :=self._cache.get("ar")) is not None:
            return ar
        tp = self.tp()
        ar = tp / self.npos if (self._npos != 0 and tp is not None) else None
        self._cache["ar"] = ar
        return ar

    def clear_cache(self):
        self._cache.clear()


class Evaluation(defaultdict[str, EvaluationItem]):

    """Do not mutate this excepted with defined methods."""

    def __init__(self, items: Mapping[str, EvaluationItem] = None):
        if items is not None:
            super().__init__(lambda: EvaluationItem(), map=items)
        else:
            super().__init__(lambda: EvaluationItem())
        self._cache = {}

    def __iadd__(self, other: "Evaluation") -> "Evaluation":
        for key, value in other.items():
            self[key] += value
        self.clear_cache()
        return self

    def __add__(self, other: "Evaluation") -> "Evaluation":
        copy_ = copy(self)
        copy_ += other
        return copy_

    def ap(self) -> float:
        if (ap := self._cache.get("ap")) is not None:
            return ap
        ap = np.mean([ap for ev in self.values() if (ap := ev.ap()) is not None])
        self._cache["ap"] = ap
        return ap

    def ar(self) -> float:
        if (ar := self._cache.get("ar")) is not None:
            return ar
        ar = np.mean([ar for ev in self.values() if (ar := ev.ar()) is not None])
        self._cache["ar"] = ar
        return ar

    def clear_cache(self):
        self._cache.clear()
    

class COCOEvaluator:

    AP_THRESHOLDS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
    
    SMALL_RANGE = (0.0, 32.0**2)
    MEDIUM_RANGE = (32.0**2, 96.0**2)
    LARGE_RANGE = (96.0**2, float("inf"))

    def __init__(self, 
        predictions: AnnotationSet, 
        ground_truths: AnnotationSet
    ) -> None:
        self._predictions = predictions
        self._ground_truths = ground_truths
        self.evaluations: dict[(float, int, tuple[float, float]), Evaluation] = {}

    def clear_cache(self):
        self.evaluations = {}

    def evaluate(self,
        iou_threshold: float,
        max_detections: int = None, 
        size_range: tuple[int, int] = None
    ) -> Evaluation:
        key = (iou_threshold, max_detections, size_range)
        evaluation = self.evaluations.get(key)

        if evaluation is not None:
            return evaluation
        
        evaluation = self.evaluate_annotations(
            self._predictions, 
            self._ground_truths, 
            iou_threshold, 
            max_detections, 
            size_range)
        
        self.evaluations[key] = evaluation

        return evaluation

    def ap(self) -> float:
        ap = 0.0
        for iou_threshold in self.AP_THRESHOLDS:
            ap += self.evaluate(iou_threshold, 100).ap()
        return ap / len(self.AP_THRESHOLDS)

    def ap_50(self) -> float:
        return self.evaluate(0.5, 100).ap()

    def ap_75(self) -> float:
        return self.evaluate(0.75, 100).ap()

    def _ap_range(self, range_: tuple[float, float]) -> float:
        ap = 0.0
        for iou_threshold in self.AP_THRESHOLDS:
            ap += self.evaluate(iou_threshold, 100, range_).ap()
        return ap / len(self.AP_THRESHOLDS)

    def _ar_range(self, range_: tuple[float, float]) -> float:
        ar = 0.0
        for iou_threshold in self.AP_THRESHOLDS:
            ar += self.evaluate(iou_threshold, 100, range_).ar()
        return ar / len(self.AP_THRESHOLDS)

    def _ar_ndets(self, max_dets: int) -> float:
        ar = 0.0
        for iou_threshold in self.AP_THRESHOLDS:
            ar += self.evaluate(iou_threshold, max_dets).ar()
        return ar / len(self.AP_THRESHOLDS)

    def ap_small(self) -> float:
        return self._ap_range(self.SMALL_RANGE)

    def ap_medium(self) -> float:
        return self._ap_range(self.MEDIUM_RANGE)

    def ap_large(self) -> float:
        return self._ap_range(self.LARGE_RANGE)

    def ar_100(self) -> float:
        return self._ar_ndets(100)
    
    def ar_10(self) -> float:
        return self._ar_ndets(10)

    def ar_1(self) -> float:
        return self._ar_ndets(1)

    def ar_small(self) -> float:
        return self._ar_range(self.SMALL_RANGE)

    def ar_medium(self) -> float:
        return self._ar_range(self.MEDIUM_RANGE)

    def ar_large(self) -> float:
        return self._ar_range(self.LARGE_RANGE)

    @classmethod
    def evaluate_annotations(cls,
        predictions: AnnotationSet, 
        ground_truths: AnnotationSet,
        iou_threshold: float,
        max_detections: int = None,
        size_range: tuple[float, float] = None
    ) -> Evaluation:
        evaluation = Evaluation()
        image_ids = set(predictions.image_ids).union(ground_truths.image_ids)

        for image_id in image_ids:
            # Empty annotation is ugly
            pred = predictions.get(image_id, Annotation.empty())
            ref = ground_truths.get(image_id, Annotation.empty())

            evaluation += cls.evaluate_annotation(
                pred, ref, iou_threshold, max_detections, size_range)

        return evaluation

    @classmethod
    def evaluate_annotation(cls,
        prediction: Annotation, 
        ground_truth: Annotation,
        iou_threshold: float,
        max_detections: int = None,
        size_range: tuple[float, float] = None
    ) -> Evaluation:
        # TODO: Benchmark this redundant computation perf penalty
        # Those two can be hoisted up if slow
        preds = grouping(prediction.boxes, lambda box: box.label)
        refs = grouping(ground_truth.boxes, lambda box: box.label)
        
        labels = set(preds.keys()).union(refs.keys())
        evaluation = Evaluation()

        for label in labels:
            dets = preds.get(label, [])
            gts = refs.get(label, [])
            
            evaluation[label] += cls.evaluate_boxes(
                dets, gts, iou_threshold, max_detections, size_range)

        return evaluation

    @classmethod
    def evaluate_boxes(cls,
        predictions: list[BoundingBox],
        ground_truths: list[BoundingBox],
        iou_threshold: float,
        max_detections: int = None,
        size_range: tuple[float, float] = None
    ) -> EvaluationItem:
        # TODO: Optimize a little bit this
        # TODO: Benchmark asserts perf penalty
        assert 0.0 <= iou_threshold <= 1.0
        assert max_detections >= 0
        assert all(p.is_detection for p in predictions)
        assert all(g.is_ground_truth for g in ground_truths)
        assert all_equal(p.label for p in predictions)
        assert all_equal(g.label for g in ground_truths)

        size_range = size_range or (0.0, float("inf"))
        assert size_range[0] >= 0.0 and size_range[1] >= 0.0

        dets = sorted(predictions, key=lambda box: box.confidence, reverse=True)
        if max_detections is not None:
            dets = dets[:max_detections]

        gts = sorted(ground_truths, key=lambda box: not box.area_in(size_range))
        gt_ignore = [not g.area_in(size_range) for g in gts]  # Redundant `area_in`
        
        gt_matches = {}
        dt_matches = {}

        for idx_dt, det in enumerate(dets):
            best_iou = iou_threshold
            idx_best = -1

            for idx_gt, gt in enumerate(gts):
                if idx_gt in gt_matches:
                    continue
                if idx_best > -1 and not gt_ignore[idx_best] and gt_ignore[idx_gt]:
                    break 
                if (iou := det.iou(gt)) < best_iou:
                    continue

                best_iou = iou
                idx_best = idx_gt
            
            if idx_best == -1:
                continue

            # gt_matches.add(idx_best)
            dt_matches[idx_dt] = idx_best
            gt_matches[idx_best] = idx_dt
        
        # This is overly complicated
        dt_ignore = [gt_ignore[dt_matches[i]] if i in dt_matches else not d.area_in(size_range)
            for i, d in enumerate(dets)]

        scores = [d.confidence
            for i, d in enumerate(dets) if not dt_ignore[i]]
        matches = [i in dt_matches
            for i in range(len(dets)) if not dt_ignore[i]]
        npos = sum(1 for i in range(len(gts)) if not gt_ignore[i])

        return EvaluationItem(matches, scores, npos)

    def show_summary(self):
        table = Table(title="COCO Evaluation")
        table.add_column("Metric")
        table.add_column("Value", justify="right")

        with tqdm(desc="COCO Evaluation", total=12) as pbar:
            table.add_row("AP", f"{self.ap():.2%}")
            pbar.update()
            table.add_row("AP 50", f"{self.ap_50():.2%}")
            pbar.update()
            table.add_row("AP 75", f"{self.ap_75():.2%}")
            pbar.update()

            table.add_row("AP S", f"{self.ap_small():.2%}")
            pbar.update()
            table.add_row("AP M", f"{self.ap_medium():.2%}")
            pbar.update()
            table.add_row("AP L", f"{self.ap_large():.2%}")
            pbar.update()

            table.add_row("AR 1", f"{self.ar_1():.2%}")
            pbar.update()
            table.add_row("AR 10", f"{self.ar_10():.2%}")
            pbar.update()
            table.add_row("AR 100", f"{self.ar_100():.2%}")
            pbar.update()
            
            table.add_row("AR S", f"{self.ar_small():.2%}")
            pbar.update()
            table.add_row("AR M", f"{self.ar_medium():.2%}")
            pbar.update()
            table.add_row("AR L", f"{self.ar_large():.2%}")
            pbar.update()

        pprint(table)


def _compute_ap(scores: list[float], matched: list[bool], NP: int) -> float:
    """ This curve tracing method has some quirks that do not appear when only unique confidence thresholds
    are used (i.e. Scikit-learn's implementation), however, in order to be consistent, the COCO's method is reproduced. 
    
    Copyrights: https://github.com/rafaelpadilla/review_object_detection_metrics
    """
    if NP == 0:
        return None

    # by default evaluate on 101 recall levels
    recall_thresholds = np.linspace(0.0,
                                    1.00,
                                    int(np.round((1.00 - 0.0) / 0.01)) + 1,
                                    endpoint=True)

    # sort in descending score order
    scores = np.array(scores, dtype=float)
    matched = np.array(matched, dtype=bool)
    inds = np.argsort(-scores, kind="stable")

    scores = scores[inds]
    matched = matched[inds]

    tp = np.cumsum(matched)
    fp = np.cumsum(~matched)

    rc = tp / NP
    pr = tp / (tp + fp)

    # make precision monotonically decreasing
    i_pr = np.maximum.accumulate(pr[::-1])[::-1]
    rec_idx = np.searchsorted(rc, recall_thresholds, side="left")

    # get interpolated precision values at the evaluation thresholds
    i_pr = np.array([i_pr[r] if r < len(i_pr) else 0 for r in rec_idx])

    return np.mean(i_pr)