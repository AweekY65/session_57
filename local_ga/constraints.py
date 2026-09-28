"""约束工具。

约束以函数列表形式提供：``constraints(x) -> float``（或返回数组）。
约定返回值的含义：

* ``<= 0``：满足约束；
* ``>  0``：违反量大小。

本模块提供常用的构造器，也可以直接手写 lambda，例如
``constraints=[lambda x: x[0] + x[1] - 1.0]``（要求 x0 + x1 <= 1）。
"""

from __future__ import annotations

import math
from typing import Callable, List, Sequence

import numpy as np

Constraint = Callable[[np.ndarray], float]


def less_equal(value_fn: Callable[[np.ndarray], float], bound: float) -> Constraint:
    """生成 ``value_fn(x) <= bound`` 约束；返回 max(0, value - bound)。"""

    def _c(x: np.ndarray) -> float:
        return float(max(0.0, float(value_fn(x)) - bound))

    return _c


def greater_equal(value_fn: Callable[[np.ndarray], float], bound: float) -> Constraint:
    """生成 ``value_fn(x) >= bound`` 约束；返回 max(0, bound - value)。"""

    def _c(x: np.ndarray) -> float:
        return float(max(0.0, bound - float(value_fn(x))))

    return _c


def equal_to(value_fn: Callable[[np.ndarray], float], target: float, tol: float = 1e-6) -> Constraint:
    """生成 ``value_fn(x) == target``（容差 tol）约束；返回 max(0, |差| - tol)。"""

    def _c(x: np.ndarray) -> float:
        return float(max(0.0, abs(float(value_fn(x)) - target) - tol))

    return _c


def compute_violation(
    constraints: Sequence[Constraint],
    x: np.ndarray,
) -> tuple[float, bool]:
    """计算总约束违反量（各违反量平方和）。

    返回 ``(violation, ok)``：任一约束抛异常时 ok=False，violation=inf。
    """
    violations: List[float] = []
    for c in constraints:
        try:
            raw = c(x)
        except Exception:
            return math.inf, False
        arr = np.atleast_1d(np.asarray(raw, dtype=float)).ravel()
        for v in arr:
            v = float(v)
            if not np.isfinite(v):
                # NaN / inf 一律视为最严重违反
                violations.append(np.inf)
            elif v > 0.0:
                violations.append(v)
    if not violations:
        return 0.0, True
    if any(np.isinf(v) for v in violations):
        return np.inf, True
    return float(sum(v * v for v in violations)), True
