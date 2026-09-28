"""Decision-variable definitions, random population sampling and constraint repair.

A chromosome is always a float64 numpy array internally. Integer variables are
rounded after every operator and clipped to inclusive [lower, upper] bounds, so
the optimizer's invariant holds: every individual that enters evaluation is
inside its box and every integer gene is integral.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Union

import numpy as np

# A variable can be configured with a VariableSpec or a plain tuple:
#   (lower, upper)            -> continuous
#   (lower, upper, integer)   -> continuous/integer flag
VariableLike = Union["VariableSpec", Sequence]


@dataclass(frozen=True)
class VariableSpec:
    """One decision variable with inclusive lower/upper bounds."""

    lower: float
    upper: float
    integer: bool = False

    def __post_init__(self) -> None:
        lower = float(self.lower)
        upper = float(self.upper)
        if not (np.isfinite(lower) and np.isfinite(upper)):
            raise ValueError(f"bounds must be finite, got [{lower}, {upper}]")
        if lower > upper:
            raise ValueError(f"lower ({lower}) must be <= upper ({upper})")
        if bool(self.integer):
            if not (float(self.lower).is_integer() and float(self.upper).is_integer()):
                raise ValueError(
                    f"integer variable requires integer bounds, got "
                    f"[{self.lower}, {self.upper}]"
                )
        # frozen dataclass -> set normalized values via object.__setattr__
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "integer", bool(self.integer))


def normalize_specs(variables: Iterable[VariableLike]) -> List[VariableSpec]:
    specs: List[VariableSpec] = []
    for v in variables:
        if isinstance(v, VariableSpec):
            specs.append(v)
        elif isinstance(v, (tuple, list)):
            if len(v) == 2:
                specs.append(VariableSpec(v[0], v[1]))
            elif len(v) == 3:
                specs.append(VariableSpec(v[0], v[1], bool(v[2])))
            else:
                raise ValueError(
                    "tuple variable must be (lower, upper) or (lower, upper, integer)"
                )
        else:
            raise TypeError(f"unsupported variable specification: {v!r}")
    if not specs:
        raise ValueError("at least one decision variable is required")
    return specs


def lower_upper(specs: Sequence[VariableSpec]):
    lower = np.array([s.lower for s in specs], dtype=np.float64)
    upper = np.array([s.upper for s in specs], dtype=np.float64)
    integer = np.array([s.integer for s in specs], dtype=bool)
    return lower, upper, integer


def random_population(
    specs: Sequence[VariableSpec], n: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample *n* feasible individuals uniformly (integers uniform on the grid)."""
    population = np.empty((n, len(specs)), dtype=np.float64)
    for j, spec in enumerate(specs):
        if spec.integer:
            population[:, j] = rng.integers(
                int(spec.lower), int(spec.upper) + 1, size=n
            )
        else:
            population[:, j] = rng.uniform(spec.lower, spec.upper, size=n)
    return population


def repair(x: np.ndarray, specs: Sequence[VariableSpec]) -> np.ndarray:
    """Clip to bounds and round integer genes. Always returns a feasible vector."""
    lower, upper, integer = lower_upper(specs)
    x = np.minimum(np.maximum(x, lower), upper)
    if integer.any():
        x[integer] = np.round(x[integer])
        # rounding cannot leave an integer box, but clip again defensively
        x = np.minimum(np.maximum(x, lower), upper)
    return x
