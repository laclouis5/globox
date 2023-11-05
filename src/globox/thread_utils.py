import atexit
from concurrent.futures import ThreadPoolExecutor
from operator import length_hint
from typing import Callable, Iterable, Optional, TypeVar

from tqdm import tqdm

SHARED_THREAD_POOL = ThreadPoolExecutor()


def at_exit():
    SHARED_THREAD_POOL.shutdown()


atexit.register(at_exit)


U = TypeVar("U")
V = TypeVar("V")


def thread_map(
    fn: Callable[[U], V],
    it: Iterable[U],
    desc: Optional[str] = None,
    total: Optional[int] = None,
    unit: str = "it",
    verbose: bool = False,
) -> "list[V]":
    disable = not verbose
    total = total or length_hint(it)
    results = []
    with tqdm(desc=desc, total=total, unit=unit, disable=disable) as pbar:
        futures = SHARED_THREAD_POOL.map(fn, it)
        for result in futures:
            results.append(result)
            pbar.update()
    return results
