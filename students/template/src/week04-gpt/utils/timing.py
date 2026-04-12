"""Timing helpers.

Includes a decorator so students can see a practical advanced Python pattern.
"""

from __future__ import annotations

from functools import wraps
from time import perf_counter
from typing import Any, Callable, TypeVar


T = TypeVar("T")


def timed(func: Callable[..., T]) -> Callable[..., tuple[T, float]]:
    """Return (result, elapsed_seconds) when calling the wrapped function."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> tuple[T, float]:
        t0 = perf_counter()
        result = func(*args, **kwargs)
        return result, perf_counter() - t0

    return wrapper
