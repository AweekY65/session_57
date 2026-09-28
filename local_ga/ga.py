"""遗传算法主引擎：评估、进化循环、检查点与结果保存。

所有状态只存在于内存或本地文件：

* 随机状态：``numpy.random.Generator`` 的 bit generator state；
* 检查点：本地 pickle 文件（原子写入），只含数据，不含函数；
* 优化结果：内存中的 :class:`~local_ga.model.Result`，可选写成本地 JSON。
"""

from __future__ import annotations

import json
import math
import os
import pickle
import tempfile
from typing import Callable, List, Optional, Sequence

import numpy as np

from .config import GAConfig
from .constraints import Constraint, compute_violation
from .model import GenerationStats, Individual, Result
from .operators import (
    crossover,
    is_better,
    mutate,
    random_individual_genes,
    repair,
    select_parents,
    sort_population,
)
from .variable import Variable

FitnessFn = Callable[[np.ndarray], float]

_CHECKPOINT_VERSION = 1


class GeneticAlgorithm:
    """遗传算法优化器。

    Parameters
    ----------
    objective:
        目标函数 ``f(x) -> float``。方向由 ``config.maximize`` 决定。
        抛出异常、返回 NaN/inf 的个体被标记为无效并永远排在有效个体之后。
    variables:
        每个决策变量的上下界与类型（连续/整数）。
    config:
        算法参数；``seed`` 固定后相同配置可复现完全一致的进化历史。
    constraints:
        约束函数列表，每个返回 <=0 表示满足，>0 表示违反量。
        不可行为个体在 ``feasible_first`` 策略下永远劣于可行个体。
    """

    def __init__(
        self,
        objective: FitnessFn,
        variables: Sequence[Variable],
        config: Optional[GAConfig] = None,
        constraints: Optional[Sequence[Constraint]] = None,
    ) -> None:
        if len(variables) == 0:
            raise ValueError("至少需要一个决策变量")
        self.objective = objective
        self.variables: List[Variable] = list(variables)
        self.config = config if config is not None else GAConfig()
        self.constraints: List[Constraint] = list(constraints or [])

        self.rng = np.random.default_rng(self.config.seed)
        self.population: List[Individual] = []
        self.generation: int = 0
        self.history: List[GenerationStats] = []

    # ------------------------------------------------------------------
    # 评估
    # ------------------------------------------------------------------

    def evaluate(self, ind: Individual) -> Individual:
        """评估单个个体：目标值 + 约束违反量，就地填充字段。"""

        x = ind.genes
        try:
            raw = self.objective(x)
            val = float(raw)
        except Exception:
            val = math.nan

        violation, _ = compute_violation(self.constraints, x)

        ind.objective = val
        ind.violation = violation
        ind.valid = bool(np.isfinite(val))
        ind.feasible = bool(
            ind.valid and violation <= self.config.eq_tolerance
        )
        return ind

    def _stats(self, generation: int) -> GenerationStats:
        valid = [i for i in self.population if i.valid]
        objs = np.array([i.objective for i in valid], dtype=float)
        if valid:
            best = self._best_of(valid)
            best_obj = float(best.objective)
            mean_obj = float(objs.mean())
            # “最差”按优化方向的反方向取
            if self.config.maximize:
                worst_obj = float(objs.min())
            else:
                worst_obj = float(objs.max())
            best_viol = float(best.violation)
        else:
            best_obj = mean_obj = worst_obj = math.nan
            best_viol = float(min(i.violation for i in self.population))
        n_feasible = sum(1 for i in self.population if i.feasible)
        return GenerationStats(
            generation=generation,
            best_objective=best_obj,
            mean_objective=mean_obj,
            worst_objective=worst_obj,
            best_violation=best_viol,
            n_feasible=n_feasible,
            n_valid=len(valid),
        )

    def _best_of(self, individuals: Sequence[Individual]) -> Individual:
        best = individuals[0]
        for ind in individuals[1:]:
            if is_better(
                ind,
                best,
                self.config.maximize,
                self.config.constraint_strategy,
                self.config.penalty_weight,
            ):
                best = ind
        return best

    def best(self) -> Individual:
        """当前种群中的最优个体。"""

        return self._best_of(self.population)

    # ------------------------------------------------------------------
    # 进化
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """随机初始化种群并完成评估（可复现）。"""

        self.population = []
        for _ in range(self.config.pop_size):
            genes = random_individual_genes(self.variables, self.rng)
            self.population.append(self.evaluate(Individual(genes=genes)))
        self.generation = 0
        self.history = [self._stats(0)]
        if self.config.verbose:
            print(self.history[-1].as_row(), flush=True)

    def _evolve_one(self) -> None:
        cfg = self.config
        ordered = sort_population(
            self.population, cfg.maximize, cfg.constraint_strategy, cfg.penalty_weight
        )
        # 精英：原样保留（含已评估字段，不再重复调用目标函数）
        new_pop: List[Individual] = [ind.copy() for ind in ordered[: cfg.elitism]]

        n_offspring = cfg.pop_size - len(new_pop)
        if n_offspring > 0:
            parents = select_parents(self.population, n_offspring, cfg, self.rng)
            # 高斯步长线性收缩：第 1 代 1.0，到第 n_generations 代降到
            # sigma_end_fraction。只依赖代数，因此检查点恢复后完全一致。
            t = self.generation + 1
            total = max(1, cfg.n_generations)
            frac = cfg.sigma_end_fraction
            if t >= total:
                sigma_scale = frac
            else:
                sigma_scale = 1.0 + (frac - 1.0) * (t - 1) / (total - 1)

            children: List[Individual] = []
            for j in range(n_offspring):
                a = parents[j]
                b = parents[(j + 1) % len(parents)]
                if self.rng.random() < cfg.crossover_prob:
                    c1, _ = crossover(a.genes, b.genes, self.variables, cfg, self.rng)
                    child_genes = c1
                else:
                    child_genes = a.genes.copy()
                child_genes = mutate(
                    child_genes, self.variables, cfg, self.rng, sigma_scale=sigma_scale
                )
                # 变异后强制再修复一次（双保险）
                child_genes = repair(child_genes, self.variables)
                children.append(self.evaluate(Individual(genes=child_genes)))
            new_pop.extend(children)

        self.population = new_pop
        self.generation += 1
        stats = self._stats(self.generation)
        self.history.append(stats)
        if cfg.verbose:
            print(stats.as_row(), flush=True)

    def run(
        self,
        generations: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        result_path: Optional[str] = None,
    ) -> Result:
        """运行进化。

        Parameters
        ----------
        generations:
            目标总代数（含已运行代数）。None 时使用 ``config.n_generations``。
        checkpoint_path:
            若给定且文件已存在，则从该检查点恢复（种群、代数、随机状态、
            历史），继续运行后的进化轨迹与一次性连续执行完全一致；
            运行过程中按 ``config.checkpoint_every`` 周期写入该文件。
        result_path:
            可选，把最终结果以 JSON 写入本地文件。
        """

        target = self.config.n_generations if generations is None else int(generations)
        if target < 0:
            raise ValueError("generations 不能为负")

        resumed = False
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)
            resumed = True

        if not resumed:
            self.initialize()
        elif self.config.verbose:
            print(
                f"--> 从检查点恢复：第 {self.generation} 代，继续进化到第 {target} 代",
                flush=True,
            )

        while self.generation < target:
            self._evolve_one()
            if (
                checkpoint_path is not None
                and self.config.checkpoint_every > 0
                and self.generation % self.config.checkpoint_every == 0
            ):
                self.save_checkpoint(checkpoint_path)

        if checkpoint_path is not None:
            # 结束时总是落盘一次，保证最终状态可恢复
            self.save_checkpoint(checkpoint_path)

        best = self.best()
        result = Result(
            best_genes=best.genes.copy(),
            best_objective=float(best.objective),
            best_feasible=bool(best.feasible),
            best_violation=float(best.violation),
            generations_run=self.generation,
            maximize=self.config.maximize,
            history=list(self.history),
        )
        if result_path is not None:
            self._save_result(result, result_path)
        if self.config.verbose:
            print(result.summary(), flush=True)
        return result

    # ------------------------------------------------------------------
    # 检查点（纯数据；目标函数/约束函数不参与序列化）
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """把种群、代数、随机状态与历史原子写入本地检查点文件。

        ``n_generations`` 属于“运行计划”而非进化状态，不写入检查点：
        恢复后可以继续跑到任意新的目标代数（这也保证步长收缩等依赖
        目标代数的计划与一次连续执行完全一致）。
        """

        state_config = {k: v for k, v in self.config.__dict__.items()
                        if k != "n_generations"}
        payload = {
            "version": _CHECKPOINT_VERSION,
            "config": state_config,
            "variables": [vars(v) for v in self.variables],
            "generation": self.generation,
            "rng_state": self.rng.bit_generator.state,
            "population": [
                {
                    "genes": ind.genes.copy(),
                    "objective": ind.objective,
                    "violation": ind.violation,
                    "feasible": ind.feasible,
                    "valid": ind.valid,
                }
                for ind in self.population
            ],
            "history": [vars(s) for s in self.history],
        }
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".ckpt-", dir=directory)
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    def load_checkpoint(self, path: str) -> None:
        """从本地检查点恢复种群、代数、随机状态与历史。"""

        with open(path, "rb") as f:
            payload = pickle.load(f)
        if payload.get("version") != _CHECKPOINT_VERSION:
            raise ValueError(f"不支持的检查点版本: {payload.get('version')}")

        # 以检查点内的配置为准（种子等参数与中断前一致），但保留本次的
        # 目标代数 n_generations（它是运行计划，不属于进化状态）
        merged = dict(payload["config"])
        merged["n_generations"] = self.config.n_generations
        self.config = GAConfig(**merged)
        self.variables = [Variable(**kw) for kw in payload["variables"]]
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = payload["rng_state"]
        self.generation = payload["generation"]
        self.population = [
            Individual(
                genes=np.array(p["genes"], dtype=float),
                objective=p["objective"],
                violation=p["violation"],
                feasible=p["feasible"],
                valid=p["valid"],
            )
            for p in payload["population"]
        ]
        self.history = [GenerationStats(**kw) for kw in payload["history"]]

    # ------------------------------------------------------------------
    # 结果导出
    # ------------------------------------------------------------------

    @staticmethod
    def _save_result(result: Result, path: str) -> None:
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        data = {
            "best_genes": result.best_genes.tolist(),
            "best_objective": _json_safe(result.best_objective),
            "best_feasible": result.best_feasible,
            "best_violation": _json_safe(result.best_violation),
            "generations_run": result.generations_run,
            "maximize": result.maximize,
            "history": [
                {
                    "generation": s.generation,
                    "best_objective": _json_safe(s.best_objective),
                    "mean_objective": _json_safe(s.mean_objective),
                    "worst_objective": _json_safe(s.worst_objective),
                    "best_violation": _json_safe(s.best_violation),
                    "n_feasible": s.n_feasible,
                    "n_valid": s.n_valid,
                }
                for s in result.history
            ],
        }
        fd, tmp = tempfile.mkstemp(prefix=".result-", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise


def _json_safe(v: float):
    if math.isnan(v) or math.isinf(v):
        return None
    return float(v)
