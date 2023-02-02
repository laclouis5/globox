from .boundingbox import BoundingBox
from .annotation import Annotation
from .annotationset import AnnotationSet
from .utils import grouping, all_equal, mean
from .atomic import open_atomic
from .file_utils import PathLike

from typing import DefaultDict, Dict, Mapping, Optional, Sequence, Iterable, Any
from collections import defaultdict
import numpy as np
from copy import copy
from math import isnan
from enum import Enum, auto
from itertools import chain, product
from functools import lru_cache
from tqdm import tqdm
import csv


class RecallSteps(Enum):
    ELEVEN = auto()
    ALL = auto()


class PartialEvaluationItem:
    def __init__(
        self,
        tps: Optional["list[bool]"] = None,
        scores: Optional["list[float]"] = None,
        npos: int = 0,
    ) -> None:
        self._tps = tps or []
        self._scores = scores or []
        self._npos = npos
        self._cache: dict[str, Any] = {}

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
        ap = self._cache.get("ap")
        if ap is not None:
            return ap
        ap = COCOEvaluator._compute_ap(self._scores, self._tps, self._npos)
        self._cache["ap"] = ap
        return ap

    def ar(self) -> Optional[float]:
        ar = self._cache.get("ar")
        if ar is not None:
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
    """Evaluation of COCO metrics for one label."""

    __slots__ = ("tp", "ndet", "npos", "ap", "ar")

    def __init__(
        self, tp: int, ndet: int, npos: int, ap: Optional[float], ar: Optional[float]
    ) -> None:
        self.tp = tp
        self.ndet = ndet
        self.npos = npos
        self.ap = ap
        self.ar = ar


class PartialEvaluation(DefaultDict[str, PartialEvaluationItem]):
    """Do not mutate this excepted with defined methods."""

    def __init__(self, items: Optional[Mapping[str, PartialEvaluationItem]] = None):
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
        ap = self._cache.get("ap")
        if ap is not None:
            return ap
        ap = mean(a for a in (ev.ap() for ev in self.values()) if not isnan(a))
        self._cache["ap"] = ap
        return ap

    def ar(self) -> Optional[float]:
        ar = self._cache.get("ar")
        if ar is not None:
            return ar
        ar = mean(a for a in (ev.ar() for ev in self.values()) if not isnan(a))
        self._cache["ar"] = ar
        return ar

    def clear_cache(self):
        self._cache.clear()

    def evaluate(self) -> "Evaluation":
        return Evaluation(self)


class Evaluation(Dict[str, EvaluationItem]):
    """Evaluation of COCO metrics for multiple labels."""

    def __init__(self, evaluation: PartialEvaluation) -> None:
        super().__init__()
        self.update({label: ev.evaluate() for label, ev in evaluation.items()})

        self._ap = evaluation.ap()
        self._ar = evaluation.ar()

    def ap(self) -> float:
        return self._ap

    def ar(self) -> float:
        return self._ar


class MultiThresholdEvaluation(Dict[str, Dict[str, float]]):
    """Evaluation of COCO metrics for multiple labels and multiple
    IoU thresholds.
    """

    def __init__(self, evaluations: "list[Evaluation]") -> None:
        result: "defaultdict[str, list[EvaluationItem]]" = defaultdict(list)

        for evaluation in evaluations:
            for label, ev_item in evaluation.items():
                result[label].append(ev_item)

        super().__init__(
            {
                label: {
                    "ap": mean(ev.ap for ev in evs if not isnan(ev.ap)),
                    "ar": mean(ev.ar for ev in evs if not isnan(ev.ar)),
                }
                for label, evs in result.items()
            }
        )

    def ap(self) -> float:
        return mean(ev["ap"] for ev in self.values() if not isnan(ev["ap"]))

    def ar(self) -> float:
        return mean(ev["ar"] for ev in self.values() if not isnan(ev["ar"]))


class COCOEvaluator:
    """Class for evaluating standard COCO metrics efficently.

    This class use an internal cache to store evaluation results,
    hence, repeated calls to '.evaluate(...)', '.ap50()' and other
    such methods are fast.

    There are lots of asserts in the evaluation hot path to ensure
    a valid evaluation, speed gains can be otained by disabling them
    (python3 --OO ...).
    """

    AP_THRESHOLDS = np.linspace(0.5, 0.95, 10)

    SMALL_RANGE = (0.0, 32.0**2)
    MEDIUM_RANGE = (32.0**2, 96.0**2)
    LARGE_RANGE = (96.0**2, float("inf"))
    ALL_RANGE = (0.0, float("inf"))

    RECALL_STEPS = np.linspace(0.0, 1.0, 101)

    CSV_HEADERS = (
        "label",
        "AP 50:95",
        "AP 50",
        "AP 75",
        "AP S",
        "AP M",
        "AP L",
        "AR 1",
        "AR 10",
        "AR 100",
        "AR S",
        "AR M",
        "AR L",
    )

    def __init__(
        self,
        *,
        ground_truths: AnnotationSet,
        predictions: AnnotationSet,
        labels: Optional[Iterable[str]] = None,
    ) -> None:
        self._predictions = predictions
        self._ground_truths = ground_truths

        if labels is None:
            self.labels = []
        else:
            self.labels = list(labels)

        self._cached_evaluate = lru_cache(maxsize=60 * 4)(self.__evaluate)

    def evaluate(
        self,
        *,
        iou_threshold: float,
        max_detections: int = 100,
        size_range: Optional["tuple[float, float]"] = None,
    ) -> Evaluation:
        """COCO evaluation with custom parameters. The result
        is cached so that repeated call as fast.

        Parameters:
        - iou_threshold: the bounding box iou threshold to
        consider a ground-truth to detection association valid.
        - max_detections: the maximum number of detections taken
        into account (sorted by descreasing confidence). Defaults
        to 100.
        - size_range: the range of size (bounding box area) to
        consider. Defaults to all sizes.

        Returns:
        - An evaluation holding the metrics.
        """

        return self._cached_evaluate(
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            size_range=size_range,
        )

    def __evaluate(
        self,
        *,
        iou_threshold: float,
        max_detections: int = 100,
        size_range: Optional["tuple[float, float]"] = None,
    ) -> Evaluation:
        if size_range is None:
            size_range = COCOEvaluator.ALL_RANGE

        self._assert_params(iou_threshold, max_detections, size_range)

        return self.evaluate_annotations(
            self._predictions,
            self._ground_truths,
            iou_threshold,
            max_detections,
            size_range,
            self.labels,
        ).evaluate()

    def ap(self) -> float:
        return self.ap_evaluation().ap()

    def ap_50(self) -> float:
        return self.ap_50_evaluation().ap()

    def ap_75(self) -> float:
        return self.ap_75_evaluation().ap()

    def ap_small(self) -> float:
        return self.small_evaluation().ap()

    def ap_medium(self) -> float:
        return self.medium_evaluation().ap()

    def ap_large(self) -> float:
        return self.large_evaluation().ap()

    def ar_1(self) -> float:
        return self.ndets_1_evaluation().ar()

    def ar_10(self) -> float:
        return self.ndets_10_evaluation().ar()

    def ar_100(self) -> float:
        return self.ndets_100_evaluation().ar()

    def ar_small(self) -> float:
        return self.small_evaluation().ar()

    def ar_medium(self) -> float:
        return self.medium_evaluation().ar()

    def ar_large(self) -> float:
        return self.large_evaluation().ar()

    def ap_evaluation(self) -> MultiThresholdEvaluation:
        evaluations = [
            self.evaluate(
                iou_threshold=t, max_detections=100, size_range=self.ALL_RANGE
            )
            for t in self.AP_THRESHOLDS
        ]

        return MultiThresholdEvaluation(evaluations)

    def ap_50_evaluation(self) -> Evaluation:
        return self.evaluate(
            iou_threshold=0.5, max_detections=100, size_range=self.ALL_RANGE
        )

    def ap_75_evaluation(self) -> Evaluation:
        return self.evaluate(
            iou_threshold=0.75, max_detections=100, size_range=self.ALL_RANGE
        )

    def small_evaluation(self) -> MultiThresholdEvaluation:
        return self._range_evalation(self.SMALL_RANGE)

    def medium_evaluation(self) -> MultiThresholdEvaluation:
        return self._range_evalation(self.MEDIUM_RANGE)

    def large_evaluation(self) -> MultiThresholdEvaluation:
        return self._range_evalation(self.LARGE_RANGE)

    def ndets_1_evaluation(self) -> MultiThresholdEvaluation:
        return self._ndets_evaluation(1)

    def ndets_10_evaluation(self) -> MultiThresholdEvaluation:
        return self._ndets_evaluation(10)

    def ndets_100_evaluation(self) -> MultiThresholdEvaluation:
        return self._ndets_evaluation(100)

    def _range_evalation(
        self, range_: "tuple[float, float]"
    ) -> MultiThresholdEvaluation:
        evaluations = [
            self.evaluate(iou_threshold=t, max_detections=100, size_range=range_)
            for t in self.AP_THRESHOLDS
        ]

        return MultiThresholdEvaluation(evaluations)

    def _ndets_evaluation(self, max_dets: int) -> MultiThresholdEvaluation:
        evaluations = [
            self.evaluate(
                iou_threshold=t, max_detections=max_dets, size_range=self.ALL_RANGE
            )
            for t in self.AP_THRESHOLDS
        ]

        return MultiThresholdEvaluation(evaluations)

    @classmethod
    def evaluate_annotations(
        cls,
        predictions: AnnotationSet,
        ground_truths: AnnotationSet,
        iou_threshold: float,
        max_detections: int,
        size_range: "tuple[float, float]",
        labels: Optional[Sequence[str]] = None,
    ) -> PartialEvaluation:
        image_ids = ground_truths.image_ids | predictions.image_ids
        evaluation = PartialEvaluation()

        for image_id in sorted(image_ids):  # Sorted to ensure reproductibility
            gt = ground_truths.get(image_id) or Annotation(image_id)
            pred = predictions.get(image_id) or Annotation(image_id)

            evaluation += cls.evaluate_annotation(
                pred, gt, iou_threshold, max_detections, size_range, labels
            )

        return evaluation

    @classmethod
    def evaluate_annotation(
        cls,
        prediction: Annotation,
        ground_truth: Annotation,
        iou_threshold: float,
        max_detections: int,
        size_range: "tuple[float, float]",
        labels: Optional[Iterable[str]] = None,
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
                dets, gts, iou_threshold, max_detections, size_range
            )

        return evaluation

    @classmethod
    def evaluate_boxes(
        cls,
        predictions: "list[BoundingBox]",
        ground_truths: "list[BoundingBox]",
        iou_threshold: float,
        max_detections: int,
        size_range: "tuple[float, float]",
    ) -> PartialEvaluationItem:
        assert all(p.is_detection for p in predictions)
        assert all(g.is_ground_truth for g in ground_truths)
        assert all_equal(p.label for p in predictions)
        assert all_equal(g.label for g in ground_truths)

        cls._assert_params(iou_threshold, max_detections, size_range)

        dets = sorted(predictions, key=lambda box: box._confidence, reverse=True)  # type: ignore
        dets = dets[:max_detections]

        gts = sorted(ground_truths, key=lambda box: not box.area_in(size_range))
        gt_ignore = [not g.area_in(size_range) for g in gts]  # Redundant `area_in`

        gt_matches = {}
        dt_matches = {}

        for idx_dt, det in enumerate(dets):
            best_iou = 0.0
            idx_best = -1

            for idx_gt, gt in enumerate(gts):
                if idx_gt in gt_matches:
                    continue
                if idx_best > -1 and not gt_ignore[idx_best] and gt_ignore[idx_gt]:
                    break

                iou = det.iou(gt)
                if iou < best_iou:
                    continue

                best_iou = iou
                idx_best = idx_gt

            if idx_best == -1 or best_iou < iou_threshold:
                continue

            dt_matches[idx_dt] = idx_best
            gt_matches[idx_best] = idx_dt

        # This is overly complicated
        dt_ignore = [
            gt_ignore[dt_matches[i]] if i in dt_matches else not d.area_in(size_range)
            for i, d in enumerate(dets)
        ]

        scores = [d._confidence for i, d in enumerate(dets) if not dt_ignore[i]]
        matches = [i in dt_matches for i in range(len(dets)) if not dt_ignore[i]]
        npos = sum(1 for i in range(len(gts)) if not gt_ignore[i])

        return PartialEvaluationItem(matches, scores, npos)

    @staticmethod
    def _assert_params(
        iou_threshold: float, max_detections: int, size_range: "tuple[float, float]"
    ):
        assert 0.0 <= iou_threshold <= 1.0
        assert max_detections >= 0

        low, high = size_range
        assert low >= 0.0 and high >= low

    def clear_cache(self):
        self._cached_evaluate.cache_clear()

    def _evaluate_all(self, *, verbose: bool = False):
        params = chain(
            product(
                self.AP_THRESHOLDS,
                (100,),
                (self.SMALL_RANGE, self.MEDIUM_RANGE, self.LARGE_RANGE, self.ALL_RANGE),
            ),
            product(self.AP_THRESHOLDS, (1, 10), (self.ALL_RANGE,)),
        )

        for t, d, r in tqdm(params, desc="Evaluation", total=60, disable=not verbose):
            self.evaluate(iou_threshold=t, max_detections=d, size_range=r)

    def show_summary(self, *, verbose: bool = False):
        """Compute and show the standard COCO metrics."""
        from rich.table import Table
        from rich import print as pprint

        self._evaluate_all(verbose=verbose)

        table = Table(title="COCO Evaluation", show_footer=True)
        table.add_column("Label", footer="Total")

        metrics = {
            "AP 50:95": self.ap(),
            "AP 50": self.ap_50(),
            "AP 75": self.ap_75(),
            "AP S": self.ap_small(),
            "AP M": self.ap_medium(),
            "AP L": self.ap_large(),
            "AR 1": self.ar_1(),
            "AR 10": self.ar_10(),
            "AR 100": self.ar_100(),
            "AR S": self.ar_small(),
            "AR M": self.ar_medium(),
            "AR L": self.ar_large(),
        }

        for metric_name, metric in metrics.items():
            table.add_column(metric_name, justify="right", footer=f"{metric:.2%}")

        labels = self.labels or sorted(self.ap_evaluation().keys())

        for label in labels:
            ap = self.ap_evaluation()[label]["ap"]
            ap_50 = self.ap_50_evaluation()[label].ap
            ap_75 = self.ap_75_evaluation()[label].ap

            ap_s = self.small_evaluation()[label]["ap"]
            ap_m = self.medium_evaluation()[label]["ap"]
            ap_l = self.large_evaluation()[label]["ap"]

            ar_1 = self.ndets_1_evaluation()[label]["ap"]
            ar_10 = self.ndets_10_evaluation()[label]["ap"]
            ar_100 = self.ndets_100_evaluation()[label]["ap"]

            ar_s = self.small_evaluation()[label]["ar"]
            ar_m = self.medium_evaluation()[label]["ar"]
            ar_l = self.large_evaluation()[label]["ar"]

            table.add_row(
                label,
                f"{ap:.2%}",
                f"{ap_50:.2%}",
                f"{ap_75:.2%}",
                f"{ap_s:.2%}",
                f"{ap_m:.2%}",
                f"{ap_l:.2%}",
                f"{ar_1:.2%}",
                f"{ar_10:.2%}",
                f"{ar_100:.2%}",
                f"{ar_s:.2%}",
                f"{ar_m:.2%}",
                f"{ar_l:.2%}",
            )

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

    def save_csv(self, path: PathLike, *, verbose: bool = False):
        self._evaluate_all(verbose=verbose)
        labels = self.labels or sorted(self.ap_evaluation().keys())

        with open_atomic(path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(COCOEvaluator.CSV_HEADERS)

            for label in labels:
                ap = self.ap_evaluation()[label]["ap"]
                ap_50 = self.ap_50_evaluation()[label].ap
                ap_75 = self.ap_75_evaluation()[label].ap

                ap_s = self.small_evaluation()[label]["ap"]
                ap_m = self.medium_evaluation()[label]["ap"]
                ap_l = self.large_evaluation()[label]["ap"]

                ar_1 = self.ndets_1_evaluation()[label]["ap"]
                ar_10 = self.ndets_10_evaluation()[label]["ap"]
                ar_100 = self.ndets_100_evaluation()[label]["ap"]

                ar_s = self.small_evaluation()[label]["ar"]
                ar_m = self.medium_evaluation()[label]["ar"]
                ar_l = self.large_evaluation()[label]["ar"]

                row = (
                    label,
                    ap,
                    ap_50,
                    ap_75,
                    ap_s,
                    ap_m,
                    ap_l,
                    ar_1,
                    ar_10,
                    ar_100,
                    ar_s,
                    ar_m,
                    ar_l,
                )

                writer.writerow(row)

    @classmethod
    def _compute_ap(
        cls, scores: "list[float]", matched: "list[bool]", NP: int
    ) -> float:
        """This curve tracing method has some quirks that do not appear when only unique confidence thresholds
        are used (i.e. Scikit-learn's implementation), however, in order to be consistent, the COCO's method is reproduced.

        Copyrights: https://github.com/rafaelpadilla/review_object_detection_metrics
        """
        if NP == 0:
            return float("nan")

        # sort in descending score order
        scores_ = np.array(scores, dtype=float)
        matched_ = np.array(matched, dtype=bool)
        inds = np.argsort(-scores_, kind="stable")

        scores_ = scores_[inds]
        matched_ = matched_[inds]

        tp = np.cumsum(matched_)

        rc = tp / NP
        pr = tp / np.arange(1, matched_.size + 1)
        # make precision monotonically decreasing
        i_pr = np.maximum.accumulate(pr[::-1])[::-1]
        rec_idx = np.searchsorted(rc, cls.RECALL_STEPS, side="left")

        sum_ = i_pr[rec_idx[rec_idx < i_pr.size]].sum()

        return sum_ / rec_idx.size
