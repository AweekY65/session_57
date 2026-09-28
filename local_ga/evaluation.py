"""Objective evaluation and individual ranking.

Robustness contract
-------------------
* A user objective/constraint that raises an exception, returns NaN or returns
  infinity yields an *invalid* individual; it never crashes the search and it is
  always ranked below every valid individual.
* Constraints follow the convention ``g_i(x) <= 0`` (feasible). The total
  violation is ``sum(max(0, g_i))``.
* Ranking supports two explicit priority schemes:

  - ``feasibility`` (default, Deb-style rules, scale free):
      1. valid & feasible, ordered by objective score
      2. valid & infeasible, ordered by smaller total violation, then score
      3. invalid (NaN/exception), always last
  - ``penalty``: valid individuals are ordered by ``score - weight*violation``;
      feasible individuals still strictly outrank invalid ones.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from .variables import VariableSpec

ObjectiveFn = Callable[[np.ndarray], float]
ConstraintFn = Callable[[np.ndarray], Sequence[float]]


def evaluate_population(
    population: np.ndarray,
    objective: ObjectiveFn,
    maximize: bool,
    specs: Sequence[VariableSpec],
    constraints: Optional[ConstraintFn] = None,
    constraint_tol: float = 1e-12,
):
    """Evaluate every individual sequentially (keeps RNG-free order fixed).

    Returns
    -------
    raw_fitness : ndarray (n,)  objective values, NaN for invalid
    valid       : ndarray (n,) bool
    violation   : ndarray (n,)  total constraint violation (0 when invalid/feasible)
    score       : ndarray (n,)  direction-normalised objective (higher is better)
    feasible    : ndarray (n,) bool
    """
    n = population.shape[0]
    raw = np.full(n, np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    violation = np.zeros(n, dtype=np.float64)
    score = np.full(n, np.nan, dtype=np.float64)
    feasible = np.zeros(n, dtype=bool)

    for i, x in enumerate(population):
        try:
            f = float(objective(x))
        except Exception:
            continue
        if not math.isfinite(f):
            continue

        v = 0.0
        if constraints is not None:
            try:
                gs = constraints(x)
                vals = [float(g) for g in gs]
            except Exception:
                continue
            if any(not math.isfinite(g) for g in vals):
                continue
            v = float(sum(max(0.0, g) for g in vals))

        raw[i] = f
        valid[i] = True
        violation[i] = v
        score[i] = f if maximize else -f
        feasible[i] = v <= constraint_tol

    return raw, valid, violation, score, feasible


def rank_order(
    valid: np.ndarray,
    score: np.ndarray,
    violation: np.ndarray,
    feasible: np.ndarray,
    constraint_handling: str = "feasibility",
    penalty_weight: float = 1e6,
) -> np.ndarray:
    """Return indices ordered best -> worst under the chosen priority rules."""
    n = len(valid)
    keys = np.empty(n, dtype=[
        ("tier", "i4"),
        ("primary", "f8"),
        ("secondary", "f8"),
    ])
    for i in range(n):
        if not valid[i]:
            keys[i] = (0, 0.0, 0.0)
        elif constraint_handling == "penalty":
            # invalid excluded above; valid infeasible individuals are penalised
            # but the ordering among valid individuals is a single scalar rule
            keys[i] = (1, score[i] - penalty_weight * violation[i], 0.0)
        elif feasible[i]:
            keys[i] = (2, score[i], 0.0)
        else:
            # smaller violation first; score breaks ties
            keys[i] = (1, -violation[i], score[i])
    # ascending sort, then reverse: highest key is the best individual
    return np.argsort(keys, order=("tier", "primary", "secondary"))[::-1]
