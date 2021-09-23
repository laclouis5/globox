from .annotation import Annotation
from .annotationset import AnnotationSet
from .boundingbox import BoundingBox
from .utils import grouping, all_equal, mean

from typing import DefaultDict, Dict, Mapping, Optional, Union, Iterable
from collections import defaultdict
from dataclasses import dataclass
import dataclasses
import numpy as np
from copy import copy
from math import isnan

from rich.table import Table
from rich import print as pprint
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

class PartialEvaluationItem:

    def __init__(self, 
        tps: "list[bool]" = None,
        scores: "list[float]" = None,
        npos: int = 0
    ) -> None:
        self._tps = tps or []
        self._scores = scores or []
        self._npos = npos
        self._cache = {}

        assert self._npos >= 0
        assert len(self._tps) == len(self._scores)

    def __iadd__(self, other: "PartialEvaluationItem") -> "PartialEvaluationItem":
        self._tps += other._tps
        self._scores += other._scores
        self._npos += other._npos
        self.clear_cache()
        return self

    def __add__(self, other: "PartialEvaluationItem") -> "PartialEvaluationItem":
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
        assert tp <= self._npos
        return tp

    def ap(self) -> float:
        if (ap := self._cache.get("ap")) is not None:
            return ap
        ap = _compute_ap(self._scores, self._tps, self._npos)
        self._cache["ap"] = ap
        return ap

    def ar(self) -> Optional[float]:
        if (ar :=self._cache.get("ar")) is not None:
            return ar
        tp = self.tp()
        ar = tp / self._npos if self._npos != 0 else float("nan")
        self._cache["ar"] = ar
        return ar

    def clear_cache(self):
        self._cache.clear()

    def evaluate(self) -> "EvaluationItem":
        return EvaluationItem(self.tp(), self.ndet, self._npos, self.ap(), self.ar())


class EvaluationItem:

    __slots__ = ("tp", "ndet", "npos", "ap", "ar")
    
    def __init__(self, tp: int, ndet: int, npos: int, ap: Union[float, None], ar: Union[float, None]) -> None:
        self.tp = tp
        self.ndet = ndet
        self.npos = npos
        self.ap = ap
        self.ar = ar


class PartialEvaluation(DefaultDict[str, PartialEvaluationItem]):

    """Do not mutate this excepted with defined methods."""

    def __init__(self, items: Mapping[str, PartialEvaluationItem] = None):
        super().__init__(lambda: PartialEvaluationItem())
        if items is not None:
            self.update(items)
        self._cache = {}

    def __iadd__(self, other: "PartialEvaluation") -> "PartialEvaluation":
        for key, value in other.items():
            self[key] += value
        self.clear_cache()
        return self

    def __add__(self, other: "PartialEvaluation") -> "PartialEvaluation":
        copy_ = copy(self)
        copy_ += other
        return copy_

    def ap(self) -> float:
        if (ap := self._cache.get("ap")) is not None:
            return ap
        ap = mean(ap for ev in self.values() if not isnan(ap := ev.ap()))
        self._cache["ap"] = ap
        return ap

    def ar(self) -> float:
        if (ar := self._cache.get("ar")) is not None:
            return ar
        ar = mean(ar for ev in self.values() if not isnan(ar := ev.ar()))
        self._cache["ar"] = ar
        return ar

    def clear_cache(self):
        self._cache.clear()
    
    def evaluate(self) -> "Evaluation":
        return Evaluation(self)


class Evaluation(DefaultDict[str, EvaluationItem]):
    
    def __init__(self, evaluation: PartialEvaluation) -> None:
        super().__init__(lambda: PartialEvaluation())
        self.update({label: ev.evaluate() for label, ev in evaluation.items()})
        
        self._ap = evaluation.ap()
        self._ar = evaluation.ar()

    def ap(self) -> float:
        return self._ap

    def ar(self) -> float:
        return self._ar


class MultiThresholdEvaluation(Dict[str, Dict[str, float]]):

    def __init__(self, evaluations: "list[Evaluation]") -> None:
        result = defaultdict(list)
        for evaluation in evaluations:
            for label, ev_item in evaluation.items():
                result[label].append(ev_item)

        super().__init__({
            label: {"ap": mean(ev.ap for ev in evs if not isnan(ev.ap)), "ar": mean(ev.ar for ev in evs if not isnan(ev.ar))} 
                for label, evs in result.items()})

    def ap(self) -> float:
        return mean(ev["ap"] for ev in self.values() if not isnan(ev["ap"]))
    
    def ar(self) -> float:
        return mean(ev["ar"] for ev in self.values() if not isnan(ev["ar"]))
    

@dataclass(unsafe_hash=True)
class EvaluationParams:
    iou_threshold: float
    max_detections: Optional[int]
    size_range: Optional["tuple[float, float]"]
    # recall_steps: RecallSteps

    def __post_init__(self):
        assert 0.0 <= self.iou_threshold <= 1.0

        if self.max_detections is not None:
            assert self.max_detections >= 0

        if self.size_range is None:
            self.size_range = (0.0, float("inf"))
            return

        low, high = self.size_range
        assert low >= 0 and high >= low

        # assert self.recall_steps in RecallSteps

    @classmethod
    def format(cls,
        iou_threshold: float, 
        max_detections: Optional[int], 
        size_range: Optional["tuple[float, float]"]
    ) -> "tuple[float, Optional[int], tuple[float, float]]":
        return dataclasses.astuple(cls(iou_threshold, max_detections, size_range))


class COCOEvaluator:

    AP_THRESHOLDS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
    
    SMALL_RANGE = (0.0, 32.0**2)
    MEDIUM_RANGE = (32.0**2, 96.0**2)
    LARGE_RANGE = (96.0**2, float("inf"))

    def __init__(self, 
        ground_truths: AnnotationSet,
        predictions: AnnotationSet ,
        labels: Iterable[str] = None,
    ) -> None:
        self._predictions = predictions
        self._ground_truths = ground_truths
        self.labels = labels
        
        self.evaluations: dict[EvaluationParams, Evaluation] = {}

    def clear_cache(self):
        self.evaluations = {}

    def evaluate(self,
        iou_threshold: float,
        max_detections: int = None, 
        size_range: "tuple[int, int]" = None
    ) -> Evaluation:
        key = EvaluationParams(iou_threshold, max_detections, size_range)
        if (evaluation := self.evaluations.get(key)) is not None:
            return evaluation
        
        evaluation = self.evaluate_annotations(
            self._predictions, 
            self._ground_truths,
            iou_threshold, 
            self.labels,
            max_detections, 
            size_range).evaluate()

        self.evaluations[key] = evaluation

        return evaluation

    def ap_evaluation(self) -> MultiThresholdEvaluation:
        evaluations = [self.evaluate(t, 100) for t in self.AP_THRESHOLDS]
        return MultiThresholdEvaluation(evaluations)

    def ap(self) -> float:
        return self.ap_evaluation().ap()

    def ap_50_evaluation(self) -> Evaluation:
        return self.evaluate(0.5, 100)

    def ap_50(self) -> float:
        return self.ap_50_evaluation().ap()

    def ap_75_evaluation(self) -> Evaluation:
        return self.evaluate(0.75, 100)

    def ap_75(self) -> float:
        return self.ap_75_evaluation().ap()

    def _range_evalation(self, range_: "tuple[float, float]") -> MultiThresholdEvaluation:
        evaluations = [self.evaluate(t, 100, range_) for t in self.AP_THRESHOLDS]
        return MultiThresholdEvaluation(evaluations)

    def _ndets_evaluation(self, max_dets: int) -> MultiThresholdEvaluation:
        evaluations = [self.evaluate(t, max_dets) for t in self.AP_THRESHOLDS]
        return MultiThresholdEvaluation(evaluations)

    def small_evaluation(self) -> MultiThresholdEvaluation:
        return self._range_evalation(self.SMALL_RANGE)

    def ap_small(self) -> float:
        return self.small_evaluation().ap()

    def medium_evaluation(self) -> MultiThresholdEvaluation:
        return self._range_evalation(self.MEDIUM_RANGE)

    def ap_medium(self) -> float:
        return self.medium_evaluation().ap()

    def large_evaluation(self) -> MultiThresholdEvaluation:
        return self._range_evalation(self.LARGE_RANGE)

    def ap_large(self) -> float:
        return self.large_evaluation().ap()

    def ar_100_evaluation(self) -> MultiThresholdEvaluation:
        return self._ndets_evaluation(100)

    def ar_100(self) -> float:
        return self.ar_100_evaluation().ar()
    
    def ar_10_evaluation(self) -> MultiThresholdEvaluation:
        return self._ndets_evaluation(10)

    def ar_10(self) -> float:
        return self.ar_10_evaluation().ar()

    def ar_1_evaluation(self) -> MultiThresholdEvaluation:
        return self._ndets_evaluation(1)

    def ar_1(self) -> float:
        return self.ar_1_evaluation().ar()

    def ar_small(self) -> float:
        return self.small_evaluation().ar()

    def ar_medium(self) -> float:
        return self.medium_evaluation().ar()

    def ar_large(self) -> float:
        return self.large_evaluation().ar()

    @classmethod
    def evaluate_annotations(cls,
        predictions: AnnotationSet, 
        ground_truths: AnnotationSet,
        iou_threshold: float,
        labels: Iterable[str] = None,
        max_detections: int = None,
        size_range: "tuple[float, float]" = None
    ) -> PartialEvaluation:
        image_ids = ground_truths.image_ids | predictions.image_ids
        evaluation = PartialEvaluation()

        for image_id in sorted(image_ids):  # Sorted to ensure reproductibility
            gt = ground_truths.get(image_id) or Annotation.empty_like(predictions[image_id])
            pred = predictions.get(image_id) or Annotation.empty_like(ground_truths[image_id])

            evaluation += cls.evaluate_annotation(
                pred, gt, iou_threshold, labels, max_detections, size_range)
        return evaluation

    @classmethod
    def evaluate_annotation(cls,
        prediction: Annotation, 
        ground_truth: Annotation,
        iou_threshold: float,
        labels: Iterable[str] = None,
        max_detections: int = None,
        size_range: "tuple[float, float]" = None
    ) -> PartialEvaluation:
        assert prediction.image_id == ground_truth.image_id

        preds = grouping(prediction.boxes, lambda box: box.label)
        refs = grouping(ground_truth.boxes, lambda box: box.label)
        labels = labels or set(preds.keys()).union(refs.keys())
        evaluation = PartialEvaluation()

        for label in labels:
            dets = preds.get(label, [])
            gts = refs.get(label, [])
            
            evaluation[label] += cls.evaluate_boxes(
                dets, gts, iou_threshold, max_detections, size_range)

        return evaluation

    @classmethod
    def evaluate_boxes(self,
        predictions: "list[BoundingBox]",
        ground_truths: "list[BoundingBox]",
        iou_threshold: float,
        max_detections: int = None,
        size_range: "tuple[float, float]" = None
    ) -> PartialEvaluationItem:
        assert all(p.is_detection for p in predictions)
        assert all(g.is_ground_truth for g in ground_truths)
        assert all_equal(p.label for p in predictions)
        assert all_equal(g.label for g in ground_truths)

        iou_threshold, max_detections, size_range = EvaluationParams.format(
            iou_threshold, max_detections, size_range)

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

        return PartialEvaluationItem(matches, scores, npos)

    def show_summary(self):
        table = Table(title="COCO Evaluation", show_footer=True)
        table.add_column("Label", footer="Total")

        # TODO: ProcessPoolExecutor + submit?
        # Requires to submit 'evaluate' jobs to the pool (not self.ap etc. because they mutate the underlying shared cache)
        metrics = {
            "AP 50:95": self.ap, "AP 50": self.ap_50, "AP 75": self.ap_75, 
            "AP S": self.ap_small, "AP M": self.ap_medium, "AP L": self.ap_large, 
            "AR 1": self.ar_1, "AR 10": self.ar_10, "AR 100": self.ar_100, 
            "AR S": self.ar_small, "AR M": self.ar_medium, "AR L": self.ar_large}

        for metric_name, metric in tqdm(metrics.items(), desc="Evaluation"):
            table.add_column(metric_name, justify="right", footer=f"{metric():.2%}")

        labels = self.labels or sorted(self.ap_evaluation().keys())

        for label in labels:
            ap = self.ap_evaluation()[label]["ap"]
            ap_50 = self.ap_50_evaluation()[label].ap
            ap_75 = self.ap_75_evaluation()[label].ap

            ap_s = self.small_evaluation()[label]["ap"]
            ap_m = self.medium_evaluation()[label]["ap"]
            ap_l = self.large_evaluation()[label]["ap"]

            ar_1 = self.ar_1_evaluation()[label]["ap"]
            ar_10 = self.ar_10_evaluation()[label]["ap"]
            ar_100 = self.ar_100_evaluation()[label]["ap"]

            ar_s = self.small_evaluation()[label]["ar"]
            ar_m = self.medium_evaluation()[label]["ar"]
            ar_l = self.large_evaluation()[label]["ar"]

            table.add_row(label, 
                f"{ap:.2%}", f"{ap_50:.2%}", f"{ap_75:.2%}", 
                f"{ap_s:.2%}", f"{ap_m:.2%}", f"{ap_l:.2%}", 
                f"{ar_1:.2%}", f"{ar_10:.2%}", f"{ar_100:.2%}", 
                f"{ar_s:.2%}", f"{ar_m:.2%}", f"{ar_l:.2%}")

        table.header_style = "bold"
        table.footer_style = "bold"
        table.row_styles = ["none", "dim"]
        
        for c in table.columns[1:4]: 
            c.style = "red"
            c.header_style = "red"
            c.footer_style = "red"
        for c in table.columns[4:7]: 
            c.style = "magenta"
            c.header_style = "magenta"
            c.footer_style = "magenta"
        for c in table.columns[7:10]: 
            c.style = "blue"
            c.header_style = "blue"
            c.footer_style = "blue"
        for c in table.columns[10:13]: 
            c.style = "green"
            c.header_style = "green"
            c.footer_style = "green"

        pprint(table)


def _compute_ap(scores: "list[float]", matched: "list[bool]", NP: int) -> float:
    """ This curve tracing method has some quirks that do not appear when only unique confidence thresholds
    are used (i.e. Scikit-learn's implementation), however, in order to be consistent, the COCO's method is reproduced. 
    
    Copyrights: https://github.com/rafaelpadilla/review_object_detection_metrics
    """
    if NP == 0:
        return float("nan")

    recall_steps = np.linspace(0.0, 1.0, 101, endpoint=True)

    # sort in descending score order
    scores = np.array(scores, dtype=float)
    matched = np.array(matched, dtype=bool)
    inds = np.argsort(-scores, kind="stable")

    scores = scores[inds]
    matched = matched[inds]

    tp = np.cumsum(matched)

    rc = tp / NP
    pr = tp / np.arange(1, len(matched)+1)
    # make precision monotonically decreasing
    i_pr = np.maximum.accumulate(pr[::-1])[::-1]
    rec_idx = np.searchsorted(rc, recall_steps, side="left")

    sum_ = i_pr[rec_idx[rec_idx < len(i_pr)]].sum()

    return sum_ / len(rec_idx)