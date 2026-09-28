"""遗传算法配置。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

SelectionMethod = str  # "tournament" | "roulette"
CrossoverMethod = str  # "sbx" | "uniform" | "onepoint"
MutationMethod = str  # "gaussian" | "uniform"
ConstraintStrategy = str  # "feasible_first" | "penalty"


@dataclass
class GAConfig:
    """遗传算法的全部可调参数。

    进化流程
    --------
    1. 在变量上下界内随机初始化 ``pop_size`` 个个体（整数变量取整）；
    2. 评估目标函数与约束违反量；
    3. 精英保留：把当前种群中最好的 ``elitism`` 个个体原样送入下一代；
    4. 选择父代（锦标赛 / 轮盘赌，基于排序的适应度）；
    5. 按 ``crossover_prob`` 执行交叉（SBX / 均匀 / 单点）；
    6. 每个基因按 ``mutation_prob`` 执行变异（高斯 / 均匀重置），
       新值一律裁剪（整数变量再取整）到变量范围内；
    7. 重复 2–6 直到达到 ``n_generations``。

    关键参数
    --------
    maximize:
        True 为最大化，False 为最小化（默认最小化）。
    seed:
        固定随机种子；相同配置 + 相同种子产生完全一致的进化历史。
    constraint_strategy:
        * ``"feasible_first"``（默认，Deb 可行解优先规则）：
          任意可行解优于任意不可行解；不可行个体之间按违反量排序；
        * ``"penalty"``：fitness = objective + sign * penalty_weight *
          violation（最小化时 sign=+1，最大化时 sign=-1）。
    """

    pop_size: int = 50
    n_generations: int = 100
    crossover_prob: float = 0.9
    mutation_prob: float = 0.1
    eta: float = 15.0
    # 高斯变异初始步长；None 时连续变量取 5% 变量宽度、整数变量取 max(1, 10% 宽度)
    sigma: Optional[float] = None
    # 步长随代数线性收缩，末代步长 = sigma_end_fraction * 初始步长；
    # 1.0 表示不收缩。仅在 sigma=None（自动步长）时生效。
    sigma_end_fraction: float = 0.02
    mutation_method: MutationMethod = "gaussian"
    crossover_method: CrossoverMethod = "sbx"
    selection_method: SelectionMethod = "tournament"
    tournament_size: int = 3
    elitism: int = 1
    maximize: bool = False
    seed: Optional[int] = 42
    constraint_strategy: ConstraintStrategy = "feasible_first"
    penalty_weight: float = 1e6
    eq_tolerance: float = 1e-6
    checkpoint_every: int = 0
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.pop_size < 2:
            raise ValueError("pop_size 至少为 2")
        if self.n_generations < 1:
            raise ValueError("n_generations 至少为 1")
        if not 0.0 <= self.crossover_prob <= 1.0:
            raise ValueError("crossover_prob 必须在 [0, 1] 内")
        if not 0.0 <= self.mutation_prob <= 1.0:
            raise ValueError("mutation_prob 必须在 [0, 1] 内")
        if self.eta <= 0:
            raise ValueError("eta 必须为正数")
        if self.tournament_size < 2:
            raise ValueError("tournament_size 至少为 2")
        if self.elitism < 0 or self.elitism > self.pop_size:
            raise ValueError("elitism 必须在 [0, pop_size] 内")
        if self.selection_method not in ("tournament", "roulette"):
            raise ValueError(f"未知的选择方法: {self.selection_method}")
        if self.crossover_method not in ("sbx", "uniform", "onepoint"):
            raise ValueError(f"未知的交叉方法: {self.crossover_method}")
        if self.mutation_method not in ("gaussian", "uniform"):
            raise ValueError(f"未知的变异方法: {self.mutation_method}")
        if self.constraint_strategy not in ("feasible_first", "penalty"):
            raise ValueError(f"未知的约束策略: {self.constraint_strategy}")
        if self.penalty_weight <= 0:
            raise ValueError("penalty_weight 必须为正数")
        if self.eq_tolerance < 0:
            raise ValueError("eq_tolerance 不能为负")
        if not 0.0 < self.sigma_end_fraction <= 1.0:
            raise ValueError("sigma_end_fraction 必须在 (0, 1] 内")
