from collections import defaultdict
from itertools import groupby
from typing import Callable, Hashable, Iterable, TypeVar


U = TypeVar("U")
V = TypeVar("V", bound=Hashable)


def grouping(it: Iterable[U], by_key: Callable[[U], V]) -> dict[V, list[U]]:
    result = defaultdict(list)
    for item in it:
        result[by_key(item)].append(item)
    return result


def all_equal(iterable: Iterable) -> bool:
    """https://stackoverflow.com/a/3844948/6324055"""
    g = groupby(iterable)
    return next(g, True) and not next(g, False)