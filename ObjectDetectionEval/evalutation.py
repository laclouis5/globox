from collections import defaultdict
from copy import copy
from dataclasses import dataclass
from typing import Mapping
from functools import reduce


@dataclass(frozen=True)
class EvaluationItem:

    tp: int = 0
    ndet: int = 0
    npos: int = 0

    def __post_init__(self):
        assert 0 <= self.tp
        assert self.tp <= self.ndet
        assert self.tp <= self.npos

    def __iadd__(self, other: "EvaluationItem") -> "EvaluationItem":
        self.tp += other.tp
        self.ndet += other.ndet
        self.npos += other.npos
        return self

    def __add__(self, other: "EvaluationItem") -> "EvaluationItem":
        copy_ = copy(self)
        copy_ += other
        return copy_

    @property
    def fp(self) -> int:
        return self.ndet - self.tp

    @property
    def fn(self) -> int:
        return self.npos - self.tp

    @property
    def precision(self) -> float:
        return self.tp / self.ndet if self.ndet != 0 else 1.0 if self.npos == 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / self.npos if self.npos != 0 else 1.0 if self.ndet == 0 else 0.0

    @property
    def f1_score(self) -> float:
        s = self.npos + self.ndet
        return 2.0 * self.tp / s if s != 0 else 1.0


class Evaluation(defaultdict):

    def __init__(self, items: Mapping[str, EvaluationItem] = None):
        super().__init__(lambda: EvaluationItem(), map=items or {})

    def total(self) -> EvaluationItem:
        return reduce(EvaluationItem.__iadd__, self.values(), EvaluationItem())

    def __iadd__(self, other: "Evaluation") -> "Evaluation":
        for key, value in other.items():
            self[key] += value
        return self

    def __add__(self, other: "Evaluation") -> "Evaluation":
        copy_ = copy(self)
        copy_ += other
        return copy_