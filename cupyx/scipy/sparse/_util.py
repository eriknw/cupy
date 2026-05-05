from __future__ import annotations

import cupy
from cupy._core import core


def isdense(x):
    return isinstance(x, core.ndarray)


def isintlike(x):
    try:
        return bool(int(x) == x)
    except (TypeError, ValueError):
        return False


def isscalarlike(x):
    return cupy.isscalar(x) or (isdense(x) and x.ndim == 0)


def isshape(x, nonneg=False):
    """Check that ``x`` is a 2-tuple of int-like values.

    Args:
        x: Object to check.
        nonneg (bool): When ``True``, additionally require both
            entries to be ``>= 0``.  Mirrors ``scipy.sparse._sputils.isshape``
            so callers can reject ``(10, -5)`` -style shapes that would
            otherwise build a phantom sparse object (silent corruption
            on slicing / assignment).
    """
    if not isinstance(x, tuple) or len(x) != 2:
        return False
    m, n = x
    if isinstance(n, tuple):
        return False
    if not (isintlike(m) and isintlike(n)):
        return False
    if nonneg and (int(m) < 0 or int(n) < 0):
        return False
    return True
