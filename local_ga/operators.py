"""Genetic operators: selection, crossover, mutation, elitism.

Every stochastic draw goes through the single ``np.random.Generator`` owned by
the optimizer, in a fixed per-generation order, which is what makes seeded runs
(and checkpoint resume) bit-for-bit reproducible.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from .variables import VariableSpec, lower_upper, repair

SELECTION_METHODS = ("tournament", "roulette")
CROSSOVER_METHODS = ("uniform", "onepoint", "sbx")
MUTATION_METHODS = ("gaussian", "uniform")


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def tournament_select(
    rng: np.random.Generator,
    ranks: np.ndarray,
    k: int = 3,
) -> int:
    """Pick one parent via k-way tournament; *ranks[i]* is rank position (0 best)."""
    contenders = rng.integers(0, len(ranks), size=int(k))
    best = contenders[0]
    best_rank = ranks[best]
    for c in contenders[1:]:
        if ranks[c] < best_rank:
            best = c
            best_rank = ranks[c]
    return int(best)


def roulette_select(
    rng: np.random.Generator,
    ranks: np.ndarray,
) -> int:
    """Rank-based roulette: rank 0 (best) gets probability N/Σrank, no fitness-scale issues."""
    n = len(ranks)
    # rank position r (0=best) -> weight n-r
    weights = (n - ranks).astype(np.float64)
    probs = weights / weights.sum()
    return int(rng.choice(n, p=probs))


def select_parents(
    method: str,
    rng: np.random.Generator,
    ranks: np.ndarray,
    tournament_size: int,
) -> Tuple[int, int]:
    """Select two (not necessarily distinct) parents in a fixed draw order."""
    if method == "tournament":
        p1 = tournament_select(rng, ranks, tournament_size)
        p2 = tournament_select(rng, ranks, tournament_size)
    elif method == "roulette":
        p1 = roulette_select(rng, ranks)
        p2 = roulette_select(rng, ranks)
    else:  # validated at construction; defensive only
        raise ValueError(f"unknown selection method: {method}")
    return p1, p2


# --------------------------------------------------------------------------- #
# Crossover  (outputs are repaired by the caller, after mutation)
# --------------------------------------------------------------------------- #
def crossover(
    method: str,
    p1: np.ndarray,
    p2: np.ndarray,
    specs: Sequence[VariableSpec],
    rng: np.random.Generator,
    crossover_rate: float,
    sbx_eta: float = 15.0,
) -> Tuple[np.ndarray, np.ndarray]:
    d = len(specs)
    if rng.random() >= crossover_rate:
        return p1.copy(), p2.copy()

    if method == "uniform":
        mask = rng.random(d) < 0.5
        c1 = np.where(mask, p1, p2)
        c2 = np.where(mask, p2, p1)
        return c1, c2

    if method == "onepoint":
        if d == 1:
            return p1.copy(), p2.copy()
        point = int(rng.integers(1, d))
        c1 = np.concatenate([p1[:point], p2[point:]])
        c2 = np.concatenate([p2[:point], p1[point:]])
        return c1, c2

    if method == "sbx":
        # Simulated binary crossover (Deb & Agrawal). Bounded by the variable
        # boxes; integer genes are re-rounded through repair afterwards.
        _, upper, integer = lower_upper(specs)
        lower = np.array([s.lower for s in specs])
        c1 = np.empty(d)
        c2 = np.empty(d)
        for j in range(d):
            if rng.random() >= 0.5 or abs(p1[j] - p2[j]) < 1e-14:
                c1[j], c2[j] = p1[j], p2[j]
                continue
            u = rng.random()
            if u <= 0.5:
                beta = (2.0 * u) ** (1.0 / (sbx_eta + 1.0))
            else:
                beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (sbx_eta + 1.0))
            c1[j] = 0.5 * ((p1[j] + p2[j]) - beta * abs(p2[j] - p1[j]))
            c2[j] = 0.5 * ((p1[j] + p2[j]) + beta * abs(p2[j] - p1[j]))
        return c1, c2

    raise ValueError(f"unknown crossover method: {method}")


# --------------------------------------------------------------------------- #
# Mutation
# --------------------------------------------------------------------------- #
def mutate(
    method: str,
    x: np.ndarray,
    specs: Sequence[VariableSpec],
    rng: np.random.Generator,
    mutation_rate: float,
    gene_rate: float,
    sigma: float,
) -> np.ndarray:
    """Per-individual mutation_rate gate, then per-gene application."""
    if rng.random() >= mutation_rate:
        return x
    d = len(specs)
    lower, upper, integer = lower_upper(specs)
    span = upper - lower
    x = x.copy()
    for j in range(d):
        if rng.random() >= gene_rate:
            continue
        if method == "gaussian":
            step = sigma * span[j] * rng.normal()
            if integer[j] and span[j] > 0:
                # integer geometric step: 70% local +/-1..3, occasional long jump
                mag = int(abs(rng.normal(0, max(1.0, sigma * span[j])))) + 1
                step = float(mag if rng.random() < 0.5 else -mag)
            x[j] += step
        elif method == "uniform":
            if integer[j]:
                x[j] = rng.integers(int(lower[j]), int(upper[j]) + 1)
            else:
                x[j] = rng.uniform(lower[j], upper[j])
        else:
            raise ValueError(f"unknown mutation method: {method}")
    return repair(x, specs)


# --------------------------------------------------------------------------- #
# Elitism
# --------------------------------------------------------------------------- #
def elite_indices(order: np.ndarray, n_elite: int) -> np.ndarray:
    return order[:n_elite]
