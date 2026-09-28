"""遗传算子：选择、交叉、变异。

所有产生新基因的算子在返回前都会调用 :func:`repair`，把每个基因裁剪
回变量上下界；整数变量再做就近取整。因此种群中永远不会出现违反
变量约束的个体。
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .model import Individual
from .variable import Variable


# ---------------------------------------------------------------------------
# 边界修复
# ---------------------------------------------------------------------------

def repair(genes: np.ndarray, variables: Sequence[Variable]) -> np.ndarray:
    """把基因裁剪到上下界内，整数变量取整。始终返回新数组。"""

    out = np.array(genes, dtype=float, copy=True)
    for i, var in enumerate(variables):
        lo, hi = float(var.lower), float(var.upper)
        g = float(out[i])
        if not np.isfinite(g):
            # 变异产生了 NaN/inf（例如 sigma 过大），回退到下界
            g = lo if np.isfinite(lo) else 0.0
        if np.isfinite(lo) and g < lo:
            g = lo
        if np.isfinite(hi) and g > hi:
            g = hi
        if var.integer:
            g = float(int(round(g)))
        out[i] = g
    return out


# ---------------------------------------------------------------------------
# 选择
# ---------------------------------------------------------------------------

def sort_population(
    pop: List[Individual],
    maximize: bool,
    strategy: str,
    penalty_weight: float,
) -> List[Individual]:
    """按“从好到差”返回排序后的个体列表（不修改原列表）。

    排序键约定：键越小，个体越好（见 :func:`_sort_key`）。
    """

    return sorted(pop, key=lambda ind: _sort_key(ind, maximize, strategy, penalty_weight))


def _sort_key(ind: Individual, maximize: bool, strategy: str, penalty_weight: float):
    """排序键：键越小个体越好（``sorted`` 升序后最好的个体在最前）。

    层级：
      1. 有效个体优于无效个体（NaN/异常/不可评估）；
      2. feasible_first 策略：可行优于不可行，不可行之间违反量越小越好；
      3. penalty 策略：用 目标 ± weight*violation 直接比较；
      4. 可行个体之间按目标方向比较。
    """

    if not ind.valid:
        # 无效个体排在最后；同类之间违反量小的略好
        return (2, ind.violation, 0.0)

    obj = ind.objective

    if strategy == "penalty":
        sign = -1.0 if maximize else 1.0
        penalized = obj + sign * penalty_weight * ind.violation
        # 最大化时 penalized 越大越好 → 取负使其“越小越好”
        key_obj = -penalized if maximize else penalized
        return (0, 0.0, key_obj)

    # feasible_first（Deb 规则）
    if ind.feasible:
        # 可行个体：最小化 obj 越小越好；最大化取 -obj
        return (0, 0.0, -obj if maximize else obj)
    # 不可行个体：违反量越小越好，全部排在可行个体之后
    return (1, ind.violation, 0.0)


def is_better(
    a: Individual,
    b: Individual,
    maximize: bool,
    strategy: str,
    penalty_weight: float,
) -> bool:
    """个体 a 是否严格优于个体 b。"""

    return _sort_key(a, maximize, strategy, penalty_weight) < _sort_key(
        b, maximize, strategy, penalty_weight
    )


def select_parents(
    pop: List[Individual],
    n_offspring: int,
    config,
    rng: np.random.Generator,
) -> List[Individual]:
    """选择产生 n_offspring 个后代所需的父代（每两个父代配成一对）。"""

    ordered = sort_population(pop, config.maximize, config.constraint_strategy, config.penalty_weight)
    n_parents = n_offspring  # 调用方按 (p1, p2) 成对取用

    if config.selection_method == "tournament":
        chosen: List[Individual] = []
        k = min(config.tournament_size, len(ordered))
        for _ in range(n_parents):
            idxs = rng.integers(0, len(ordered), size=k)
            best = None
            for i in idxs:
                cand = ordered[int(i)]
                if best is None or is_better(
                    cand, best, config.maximize, config.constraint_strategy, config.penalty_weight
                ):
                    best = cand
            chosen.append(best)
        return chosen

    # 轮盘赌：基于线性排名的概率（不依赖目标尺度，NaN 已被隔离在最差端）
    # ordered 从好到差，故排名权重从 N 递减到 1
    ranks = np.arange(len(ordered), 0, -1, dtype=float)
    probs = ranks / ranks.sum()
    idx = rng.choice(len(ordered), size=n_parents, replace=True, p=probs)
    return [ordered[int(i)] for i in idx]


# ---------------------------------------------------------------------------
# 交叉
# ---------------------------------------------------------------------------

def crossover(
    p1: np.ndarray,
    p2: np.ndarray,
    variables: Sequence[Variable],
    config,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """对两条父代染色体交叉，返回两个修复后的子代。"""

    method = config.crossover_method
    n = len(variables)

    if method == "sbx":
        c1, c2 = _sbx(p1, p2, variables, config.eta, rng)
    elif method == "uniform":
        mask = rng.random(n) < 0.5
        c1 = np.where(mask, p1, p2)
        c2 = np.where(mask, p2, p1)
    else:  # onepoint
        point = int(rng.integers(1, n)) if n > 1 else n
        c1 = np.concatenate([p1[:point], p2[point:]])
        c2 = np.concatenate([p2[:point], p1[point:]])

    return repair(c1, variables), repair(c2, variables)


def _sbx(
    p1: np.ndarray,
    p2: np.ndarray,
    variables: Sequence[Variable],
    eta: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """模拟二进制交叉（Simulated Binary Crossover, Deb & Agrawal）。"""

    n = len(variables)
    c1 = np.empty(n)
    c2 = np.empty(n)
    u = rng.random(n)

    for j, var in enumerate(variables):
        x1, x2 = float(p1[j]), float(p2[j])
        if x1 == x2:
            c1[j] = x1
            c2[j] = x2
            continue

        xlo, xhi = float(var.lower), float(var.upper)
        if x1 > x2:
            x1, x2 = x2, x1
        dx = x2 - x1

        # 计算 beta 时同时考虑两侧边界，保证子代落在 [lower, upper] 内
        beta1 = 1.0 + 2.0 * (x1 - xlo) / dx if np.isfinite(xlo) else 1.0
        beta2 = 1.0 + 2.0 * (xhi - x2) / dx if np.isfinite(xhi) else 1.0
        alpha1 = 2.0 - beta1 ** (-(eta + 1.0))
        alpha2 = 2.0 - beta2 ** (-(eta + 1.0))

        uj = float(u[j])
        if uj <= 1.0 / alpha1:
            bq1 = (uj * alpha1) ** (1.0 / (eta + 1.0))
        else:
            bq1 = (1.0 / (2.0 - uj * alpha1)) ** (1.0 / (eta + 1.0))
        if uj <= 1.0 / alpha2:
            bq2 = (uj * alpha2) ** (1.0 / (eta + 1.0))
        else:
            bq2 = (1.0 / (2.0 - uj * alpha2)) ** (1.0 / (eta + 1.0))

        # 一个子代在 p1 侧扩展，另一个在 p2 侧扩展
        c1[j] = 0.5 * ((x1 + x2) - bq1 * dx)
        c2[j] = 0.5 * ((x1 + x2) + bq2 * dx)

    return c1, c2


# ---------------------------------------------------------------------------
# 变异
# ---------------------------------------------------------------------------

def mutate(
    genes: np.ndarray,
    variables: Sequence[Variable],
    config,
    rng: np.random.Generator,
    sigma_scale: float = 1.0,
) -> np.ndarray:
    """按基因独立地以 mutation_prob 变异，并修复边界。

    sigma_scale:
        高斯变异步长的缩放系数（来自步长收缩计划），均匀变异不受影响。
    """

    child = np.array(genes, dtype=float, copy=True)
    n = len(variables)
    mask = rng.random(n) < config.mutation_prob

    for i, var in enumerate(variables):
        if not mask[i]:
            continue
        lo, hi = float(var.lower), float(var.upper)
        width = (hi - lo) if (np.isfinite(lo) and np.isfinite(hi)) else 1.0

        if var.integer:
            if config.mutation_method == "uniform" or width == 0.0:
                child[i] = float(rng.integers(var.int_lower, var.int_upper + 1))
            else:
                base = config.sigma if config.sigma is not None else max(1.0, 0.1 * width)
                child[i] = child[i] + rng.normal(0.0, base * sigma_scale)
        else:
            if config.mutation_method == "uniform" or not (np.isfinite(lo) and np.isfinite(hi)):
                # 连续变量无有限边界时无法均匀重置，退化为高斯扰动
                if np.isfinite(lo) and np.isfinite(hi):
                    child[i] = float(rng.uniform(lo, hi))
                else:
                    base = config.sigma if config.sigma is not None else 1.0
                    child[i] = child[i] + rng.normal(0.0, base * sigma_scale)
            else:
                # 默认初始步长为变量范围的 5%，再乘以收缩系数
                base = config.sigma if config.sigma is not None else 0.05 * width
                child[i] = child[i] + rng.normal(0.0, base * sigma_scale)

    return repair(child, variables)


def random_individual_genes(
    variables: Sequence[Variable],
    rng: np.random.Generator,
) -> np.ndarray:
    """在上下界内随机初始化一条染色体。"""

    genes = np.empty(len(variables))
    for i, var in enumerate(variables):
        lo, hi = float(var.lower), float(var.upper)
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValueError("初始化种群时所有变量必须使用有限的上下界")
        if var.integer:
            genes[i] = float(rng.integers(var.int_lower, var.int_upper + 1))
        else:
            genes[i] = float(rng.uniform(lo, hi))
    return genes
