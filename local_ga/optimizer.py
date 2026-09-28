"""GeneticOptimizer: the local genetic-algorithm main loop.

Everything lives in process memory or in a local checkpoint file — no network,
no database, no external services. All stochastic decisions draw from one
``numpy.random.Generator`` whose state is advanced in a fixed order, so a seeded
run and a checkpoint-resumed run produce identical trajectories.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .evaluation import (
    ConstraintFn,
    ObjectiveFn,
    evaluate_population,
    rank_order,
)
from .operators import (
    CROSSOVER_METHODS,
    MUTATION_METHODS,
    SELECTION_METHODS,
    crossover,
    elite_indices,
    mutate,
    repair,
    select_parents,
)
from .variables import (
    VariableLike,
    VariableSpec,
    normalize_specs,
    random_population,
)

HistoryRecord = Dict[str, Any]
GenerationCallback = Callable[[HistoryRecord], None]


@dataclass
class OptimizeResult:
    best_x: np.ndarray
    best_fitness: float
    best_is_feasible: bool
    best_is_valid: bool
    best_violation: float
    history: List[HistoryRecord] = field(default_factory=list)
    population: Optional[np.ndarray] = None
    fitnesses: Optional[np.ndarray] = None
    maximize: bool = False
    specs: Sequence[VariableSpec] = field(default_factory=list)

    @property
    def best_x_list(self) -> List[Any]:
        """Best point as a Python list with integer genes as ``int``."""
        out: List[Any] = []
        for v, spec in zip(self.best_x, self.specs):
            out.append(int(round(v)) if spec.integer else float(v))
        return out


class GeneticOptimizer:
    def __init__(
        self,
        variables: Sequence[VariableLike],
        objective: ObjectiveFn,
        maximize: bool = False,
        pop_size: int = 40,
        max_generations: int = 100,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.2,
        gene_mutation_rate: Optional[float] = None,
        mutation_sigma: float = 0.1,
        elitism: int = 2,
        selection: str = "tournament",
        tournament_size: int = 3,
        crossover_method: str = "uniform",
        mutation_method: str = "gaussian",
        sbx_eta: float = 15.0,
        constraints: Optional[ConstraintFn] = None,
        constraint_handling: str = "feasibility",
        penalty_weight: float = 1e6,
        constraint_tol: float = 1e-12,
        seed: Optional[int] = 42,
        verbose: bool = True,
        on_generation: Optional[GenerationCallback] = None,
    ) -> None:
        self.specs = normalize_specs(variables)
        self.objective = objective
        self.constraints = constraints
        self.maximize = bool(maximize)

        if pop_size < 2:
            raise ValueError("pop_size must be >= 2")
        if max_generations < 0:
            raise ValueError("max_generations must be >= 0")
        if not 0.0 <= crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be in [0, 1]")
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be in [0, 1]")
        if mutation_sigma <= 0:
            raise ValueError("mutation_sigma must be > 0")
        if elitism < 0 or elitism >= pop_size:
            raise ValueError("elitism must be in [0, pop_size)")
        if selection not in SELECTION_METHODS:
            raise ValueError(f"selection must be one of {SELECTION_METHODS}")
        if tournament_size < 1:
            raise ValueError("tournament_size must be >= 1")
        if crossover_method not in CROSSOVER_METHODS:
            raise ValueError(f"crossover_method must be one of {CROSSOVER_METHODS}")
        if mutation_method not in MUTATION_METHODS:
            raise ValueError(f"mutation_method must be one of {MUTATION_METHODS}")
        if constraint_handling not in ("feasibility", "penalty"):
            raise ValueError("constraint_handling must be 'feasibility' or 'penalty'")
        if penalty_weight <= 0:
            raise ValueError("penalty_weight must be > 0")

        self.pop_size = int(pop_size)
        self.max_generations = int(max_generations)
        self.crossover_rate = float(crossover_rate)
        self.mutation_rate = float(mutation_rate)
        self.gene_mutation_rate = (
            1.0 / len(self.specs) if gene_mutation_rate is None
            else float(gene_mutation_rate)
        )
        if not 0.0 <= self.gene_mutation_rate <= 1.0:
            raise ValueError("gene_mutation_rate must be in [0, 1]")
        self.mutation_sigma = float(mutation_sigma)
        self.elitism = int(elitism)
        self.selection = selection
        self.tournament_size = int(tournament_size)
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.sbx_eta = float(sbx_eta)
        self.constraint_handling = constraint_handling
        self.penalty_weight = float(penalty_weight)
        self.constraint_tol = float(constraint_tol)
        self.seed = seed
        self.verbose = bool(verbose)
        self.on_generation = on_generation

        # Mutable evolution state
        self.rng = np.random.default_rng(seed)
        self.generation = 0
        self.evaluations = 0
        self.history: List[HistoryRecord] = []
        self.population = random_population(self.specs, self.pop_size, self.rng)
        (
            self.fitnesses,
            self.valid,
            self.violations,
            self.scores,
            self.feasible,
        ) = self._evaluate(self.population)
        self.order = self._rank()
        self._record_generation()

    # ------------------------------------------------------------------ #
    # Core mechanics
    # ------------------------------------------------------------------ #
    def _evaluate(self, population: np.ndarray):
        self.evaluations += population.shape[0]
        return evaluate_population(
            population,
            self.objective,
            self.maximize,
            self.specs,
            self.constraints,
            self.constraint_tol,
        )

    def _rank(self) -> np.ndarray:
        return rank_order(
            self.valid,
            self.scores,
            self.violations,
            self.feasible,
            self.constraint_handling,
            self.penalty_weight,
        )

    def _step(self) -> None:
        n = self.pop_size
        ranks = np.empty(n, dtype=np.int64)
        ranks[self.order] = np.arange(n)  # rank 0 = best
        d = len(self.specs)

        children = np.empty((n, d), dtype=np.float64)
        n_filled = 0

        # 1) Elitism: best individuals copied unchanged
        if self.elitism > 0:
            elite = self.population[elite_indices(self.order, self.elitism)]
            children[: self.elitism] = elite
            n_filled = self.elitism

        # 2) Selection -> crossover -> mutation -> repair, pair by pair
        while n_filled < n:
            p1_idx, p2_idx = select_parents(
                self.selection, self.rng, ranks, self.tournament_size
            )
            c1, c2 = crossover(
                self.crossover_method,
                self.population[p1_idx],
                self.population[p2_idx],
                self.specs,
                self.rng,
                self.crossover_rate,
                self.sbx_eta,
            )
            c1 = repair(c1, self.specs)
            c2 = repair(c2, self.specs)
            c1 = mutate(
                self.mutation_method, c1, self.specs, self.rng,
                self.mutation_rate, self.gene_mutation_rate,
                self.mutation_sigma,
            )
            c2 = mutate(
                self.mutation_method, c2, self.specs, self.rng,
                self.mutation_rate, self.gene_mutation_rate,
                self.mutation_sigma,
            )
            children[n_filled] = c1
            n_filled += 1
            if n_filled < n:
                children[n_filled] = c2
                n_filled += 1

        self.population = children
        (
            self.fitnesses,
            self.valid,
            self.violations,
            self.scores,
            self.feasible,
        ) = self._evaluate(children)
        self.order = self._rank()
        self.generation += 1
        self._record_generation()

    # ------------------------------------------------------------------ #
    # Logging / stats
    # ------------------------------------------------------------------ #
    def _record_generation(self) -> None:
        best_i = int(self.order[0])
        valid_f = self.fitnesses[self.valid]
        # worst = least preferred individual (rank order is best -> worst)
        worst_i = int(self.order[-1])
        record: HistoryRecord = {
            "generation": self.generation,
            "best": float(self.fitnesses[best_i]) if self.valid[best_i] else np.nan,
            "mean": float(np.mean(valid_f)) if valid_f.size else np.nan,
            "worst": float(self.fitnesses[worst_i]) if self.valid[worst_i] else np.nan,
            "best_violation": float(self.violations[best_i]),
            "feasible": int(np.count_nonzero(self.feasible)),
            "valid": int(np.count_nonzero(self.valid)),
            "best_x": self.population[best_i].copy(),
        }
        self.history.append(record)
        if self.verbose:
            self._print_record(record)
        if self.on_generation is not None:
            self.on_generation(record)

    def _print_record(self, rec: HistoryRecord) -> None:
        if self.generation == 0:
            print(
                f"{'gen':>4} {'best':>14} {'mean':>14} {'worst':>14} "
                f"{'feas':>5}/{self.pop_size:<3} {'viol':>10} "
                f"{'dir':>3}",
                flush=True,
            )
        direction = "max" if self.maximize else "min"
        best = "nan" if np.isnan(rec["best"]) else f"{rec['best']:.6g}"
        mean = "nan" if np.isnan(rec["mean"]) else f"{rec['mean']:.6g}"
        worst = "nan" if np.isnan(rec["worst"]) else f"{rec['worst']:.6g}"
        print(
            f"{rec['generation']:>4} {best:>14} {mean:>14} {worst:>14} "
            f"{rec['feasible']:>5}       {rec['best_violation']:>10.3e} "
            f"{direction:>3}",
            flush=True,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self, generations: Optional[int] = None) -> OptimizeResult:
        """Evolve for *generations* more steps.

        With no argument, run until ``max_generations``. Calling ``run()`` after
        that returns immediately; use ``run(n)`` to extend beyond it.
        """
        if generations is None:
            target = self.max_generations
            steps = max(0, target - self.generation)
        else:
            if generations < 0:
                raise ValueError("generations must be >= 0")
            steps = int(generations)
        for _ in range(steps):
            self._step()
        return self.result

    @property
    def result(self) -> OptimizeResult:
        best_i = int(self.order[0])
        return OptimizeResult(
            best_x=self.population[best_i].copy(),
            best_fitness=float(self.fitnesses[best_i]),
            best_is_feasible=bool(self.feasible[best_i]),
            best_is_valid=bool(self.valid[best_i]),
            best_violation=float(self.violations[best_i]),
            history=list(self.history),
            population=self.population.copy(),
            fitnesses=self.fitnesses.copy(),
            maximize=self.maximize,
            specs=self.specs,
        )

    # ------------------------------------------------------------------ #
    # Checkpoint
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, path: str) -> None:
        from .checkpoint import save_checkpoint

        save_checkpoint(self, path)

    @classmethod
    def load_checkpoint(cls, path: str) -> "GeneticOptimizer":
        from .checkpoint import load_checkpoint

        return load_checkpoint(path)

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        # callbacks are user-owned and may not be picklable; drop on save
        state["on_generation"] = None
        state["rng"] = self.rng.bit_generator.state
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        rng_state = state.pop("rng")
        rng = np.random.default_rng()
        rng.bit_generator.state = rng_state
        state["rng"] = rng
        state.setdefault("on_generation", None)
        self.__dict__.update(state)
