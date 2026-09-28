"""核心数据结构：个体、每代统计、优化结果。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class Individual:
    """一个个体（一条候选解）。

    Attributes
    ----------
    genes:
        决策变量向量。整数变量也以 float64 存储（值为整数），便于 NumPy 运算。
    objective:
        用户目标函数的原始返回值；评估失败（NaN / inf / 抛异常）时为 NaN。
    violation:
        约束违反量（各约束违反量平方和，>=0）；0 表示完全可行。
        评估失败时为 inf。
    feasible:
        是否可行（violation 在容差范围内，且目标值有效）。
    valid:
        目标函数是否成功返回了有限数值。
    """

    genes: np.ndarray
    objective: float = math.nan
    violation: float = math.inf
    feasible: bool = False
    valid: bool = False

    def copy(self) -> "Individual":
        return Individual(
            genes=self.genes.copy(),
            objective=self.objective,
            violation=self.violation,
            feasible=self.feasible,
            valid=self.valid,
        )


@dataclass
class GenerationStats:
    """一代结束后的关键指标。"""

    generation: int
    best_objective: float
    mean_objective: float
    worst_objective: float
    best_violation: float
    n_feasible: int
    n_valid: int

    def as_row(self) -> str:
        return (
            f"gen={self.generation:4d}  best={self.best_objective:+.6g}  "
            f"mean={self.mean_objective:+.6g}  worst={self.worst_objective:+.6g}  "
            f"feasible={self.n_feasible}  valid={self.n_valid}  "
            f"min_viol={self.best_violation:.3g}"
        )


@dataclass
class Result:
    """优化结果。"""

    best_genes: np.ndarray
    best_objective: float
    best_feasible: bool
    best_violation: float
    generations_run: int
    maximize: bool
    history: List[GenerationStats] = field(default_factory=list)

    @property
    def x(self) -> np.ndarray:
        """最优解的变量向量。"""
        return self.best_genes

    @property
    def fun(self) -> float:
        """最优解对应的目标函数值。"""
        return self.best_objective

    def summary(self) -> str:
        lines = [
            "=" * 64,
            f"优化结束（{'最大化' if self.maximize else '最小化'}），"
            f"共运行 {self.generations_run} 代",
            f"最优目标值: {self.best_objective:.10g}",
            f"是否可行:   {self.best_feasible}（违反量 {self.best_violation:.3g}）",
            f"最优变量:   {np.array2string(self.best_genes, precision=6, separator=', ')}",
            "=" * 64,
        ]
        return "\n".join(lines)
