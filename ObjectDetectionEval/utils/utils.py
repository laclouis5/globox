from itertools import groupby
from typing import Iterable


def all_equal(iterable: Iterable) -> bool:
    """https://stackoverflow.com/a/3844948/6324055"""
    g = groupby(iterable)
    return next(g, True) and not next(g, False)