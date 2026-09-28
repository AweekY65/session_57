"""local_ga — a fully local, dependency-light genetic algorithm framework.

Only NumPy and the Python standard library are used. Populations, fitness
values, random state, checkpoints and results live in process memory or in
local files; there is no network, database or SaaS dependency anywhere.
"""

from .optimizer import GeneticOptimizer, OptimizeResult
from .variables import VariableSpec

__all__ = ["GeneticOptimizer", "OptimizeResult", "VariableSpec"]
__version__ = "1.0.0"
