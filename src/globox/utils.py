from collections import defaultdict
from itertools import groupby
from typing import Callable, Hashable, Iterable, TypeVar

U = TypeVar("U")
V = TypeVar("V", bound=Hashable)


def grouping(it: Iterable[U], by_key: Callable[[U], V]) -> "dict[V, list[U]]":
    result = defaultdict(list)
    for item in it:
        result[by_key(item)].append(item)
    return result


def all_equal(iterable: Iterable) -> bool:
    """https://stackoverflow.com/a/3844948/6324055"""
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def mean(it: Iterable[float]) -> float:
    sum_ = 0.0
    count = 0

    for value in it:
        sum_ += value
        count += 1

    if count == 0:
        return float("nan")

    return sum_ / count
