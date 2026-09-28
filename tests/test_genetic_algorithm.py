"""local_ga 自动化测试。

直接运行（终端会打印每代关键指标）：

    pytest tests/ -s -v

所有测试均为纯本地计算，不访问网络或任何外部服务。
"""

import math

import numpy as np
import pytest

from local_ga import (
    GAConfig,
    GeneticAlgorithm,
    Variable,
    equal_to,
    greater_equal,
)
from local_ga.model import Individual
from local_ga.operators import is_better, repair, sort_population


# ---------------------------------------------------------------------------
# 基准测试函数
# ---------------------------------------------------------------------------

def sphere(x: np.ndarray) -> float:
    return float(np.sum(x * x))


def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


# ---------------------------------------------------------------------------
# 1. 已知函数：Sphere / Rosenbrock
# ---------------------------------------------------------------------------

def test_sphere_minimize(capsys):
    print("\n[test_sphere_minimize] min sum(x^2), 5 维, [-5.12, 5.12]")
    variables = [Variable(-5.12, 5.12) for _ in range(5)]
    ga = GeneticAlgorithm(
        sphere,
        variables,
        GAConfig(
            pop_size=100,
            n_generations=250,
            seed=2026,
            mutation_prob=0.2,
            eta=20,
            elitism=2,
            verbose=True,
        ),
    )
    result = ga.run()
    assert result.best_objective < 1e-3
    assert result.best_feasible
    assert len(result.history) == 251  # 初始代 + 250 代
    # 每代指标都被记录且目标值有效
    assert all(np.isfinite(s.best_objective) for s in result.history)
    # 单调不增（最小化时，每代最佳应当不退化）
    bests = [s.best_objective for s in result.history]
    assert all(bests[i + 1] <= bests[i] + 1e-12 for i in range(len(bests) - 1))


def test_rosenbrock_2d(capsys):
    print("\n[test_rosenbrock_2d] 2 维 Rosenbrock，理论最优点 (1,1) -> 0")
    variables = [Variable(-2.0, 2.0), Variable(-2.0, 2.0)]
    ga = GeneticAlgorithm(
        rosenbrock,
        variables,
        GAConfig(
            pop_size=120,
            n_generations=250,
            seed=36,
            mutation_prob=0.4,
            eta=20,
            elitism=2,
            verbose=True,
        ),
    )
    result = ga.run()
    assert result.best_objective < 1e-3
    np.testing.assert_allclose(result.best_genes, [1.0, 1.0], atol=0.02)


# ---------------------------------------------------------------------------
# 2. 整数染色体与边界
# ---------------------------------------------------------------------------

def test_integer_chromosome():
    print("\n[test_integer_chromosome] 整数变量最小化 sum(x_i^2)")
    variables = [
        Variable(-10, 10, integer=True),
        Variable(-20, 20, integer=True, name="x2"),
    ]
    ga = GeneticAlgorithm(
        sphere,
        variables,
        GAConfig(pop_size=40, n_generations=40, seed=1, verbose=True),
    )
    ga.initialize()
    for _ in range(40):
        ga._evolve_one()
        for ind in ga.population:
            assert np.all(ind.genes == np.round(ind.genes))
            assert -10 <= ind.genes[0] <= 10
            assert -20 <= ind.genes[1] <= 20
    best = ga.best()
    assert abs(best.objective) < 1e-9
    assert np.all(best.genes == 0.0)


def test_bounds_respected_every_generation():
    print("\n[test_bounds_respected_every_generation] 逐代检查连续变量边界")
    variables = [
        Variable(-2.0, 3.0),
        Variable(0.0, 0.5),
        Variable(-7.5, -7.1),
    ]
    ga = GeneticAlgorithm(
        sphere,
        variables,
        GAConfig(
            pop_size=30,
            n_generations=30,
            seed=99,
            mutation_prob=0.5,
            mutation_method="uniform",
        ),
    )
    ga.initialize()
    lows = np.array([v.lower for v in variables])
    highs = np.array([v.upper for v in variables])
    for _ in range(30):
        ga._evolve_one()
        for ind in ga.population:
            assert np.all(ind.genes >= lows)
            assert np.all(ind.genes <= highs)


def test_zero_width_integer_variable():
    print("\n[test_zero_width_integer_variable] 固定变量 x0=3 永远不变")
    variables = [Variable(3, 3, integer=True), Variable(-5, 5, integer=True)]
    ga = GeneticAlgorithm(
        sphere,
        variables,
        GAConfig(pop_size=20, n_generations=20, seed=3, mutation_prob=0.8),
    )
    ga.initialize()
    for _ in range(20):
        ga._evolve_one()
        assert all(ind.genes[0] == 3.0 for ind in ga.population)


def test_repair_function():
    variables = [Variable(0.0, 1.0), Variable(-2, 2, integer=True)]
    # 先裁剪到边界，再就近取整
    fixed = repair(np.array([-5.0, 2.7]), variables)
    assert fixed[0] == 0.0
    assert fixed[1] == 2.0 and fixed[1] == float(int(fixed[1]))
    fixed = repair(np.array([5.0, -2.7]), variables)
    assert fixed[0] == 1.0
    assert fixed[1] == -2.0
    # 先取整再越界时仍然被裁剪
    fixed = repair(np.array([0.5, 1.8]), variables)
    assert fixed[1] == 2.0
    # NaN 回退到下界
    fixed = repair(np.array([np.nan, 0.2]), variables)
    assert fixed[0] == 0.0


def test_invalid_variable_definition():
    with pytest.raises(ValueError):
        Variable(2.0, 1.0)
    with pytest.raises(ValueError):
        Variable(0.5, 2.0, integer=True)


# ---------------------------------------------------------------------------
# 3. 最大化
# ---------------------------------------------------------------------------

def test_maximize():
    print("\n[test_maximize] max -(x0-3)^2 - (x1+2)^2，理论最优 (3,-2) -> 0")

    def neg_quadratic(x):
        return -((x[0] - 3.0) ** 2 + (x[1] + 2.0) ** 2)

    ga = GeneticAlgorithm(
        neg_quadratic,
        [Variable(-5.0, 5.0), Variable(-5.0, 5.0)],
        GAConfig(pop_size=50, n_generations=60, seed=11, maximize=True),
    )
    result = ga.run()
    assert result.best_objective > -1e-3
    np.testing.assert_allclose(result.best_genes, [3.0, -2.0], atol=0.05)
    bests = [s.best_objective for s in result.history]
    # 最大化时每代最佳应单调不减
    assert all(bests[i + 1] >= bests[i] - 1e-12 for i in range(len(bests) - 1))


# ---------------------------------------------------------------------------
# 4. 约束优化
# ---------------------------------------------------------------------------

def test_constrained_feasible_first():
    print("\n[test_constrained_feasible_first] min x^2+y^2 s.t. x+y>=3，最优 4.5")
    ga = GeneticAlgorithm(
        sphere,
        [Variable(-5.0, 5.0), Variable(-5.0, 5.0)],
        GAConfig(pop_size=60, n_generations=100, seed=7),
        constraints=[greater_equal(lambda x: x[0] + x[1], 3.0)],
    )
    result = ga.run()
    assert result.best_feasible
    assert abs(result.best_objective - 4.5) < 0.05


def test_constrained_penalty_strategy():
    print("\n[test_constrained_penalty_strategy] 同样的问题，penalty 策略")
    ga = GeneticAlgorithm(
        sphere,
        [Variable(-5.0, 5.0), Variable(-5.0, 5.0)],
        GAConfig(
            pop_size=60,
            n_generations=100,
            seed=7,
            constraint_strategy="penalty",
            penalty_weight=1e4,
        ),
        constraints=[greater_equal(lambda x: x[0] + x[1], 3.0)],
    )
    result = ga.run()
    assert result.best_feasible
    assert abs(result.best_objective - 4.5) < 0.1


def test_equality_constraint():
    print("\n[test_equality_constraint] min 0.01*y^2 s.t. x == 2.0（x 不进目标函数）")
    ga = GeneticAlgorithm(
        lambda x: float(0.01 * x[1] ** 2),
        [Variable(-5.0, 5.0), Variable(-5.0, 5.0)],
        GAConfig(pop_size=80, n_generations=150, seed=5, sigma=0.2, verbose=True),
        constraints=[equal_to(lambda x: x[0], 2.0, tol=0.05)],
    )
    result = ga.run()
    assert result.best_feasible
    assert abs(result.best_genes[0] - 2.0) < 0.06


def test_feasible_individual_always_ranks_first():
    feasible = Individual(genes=np.zeros(2), objective=99.0, violation=0.0,
                          feasible=True, valid=True)
    infeasible = Individual(genes=np.zeros(2), objective=0.0, violation=1.0,
                            feasible=False, valid=True)
    invalid = Individual(genes=np.zeros(2), objective=math.nan, violation=math.inf,
                         feasible=False, valid=False)

    # 最小化：目标差但可行 > 目标好但不可行
    assert is_better(feasible, infeasible, False, "feasible_first", 1e6)
    assert not is_better(infeasible, feasible, False, "feasible_first", 1e6)
    # 不可行中违反量小者更好
    less_bad = Individual(genes=np.zeros(2), objective=0.0, violation=0.1,
                          feasible=False, valid=True)
    assert is_better(less_bad, infeasible, False, "feasible_first", 1e6)
    # 任何有效个体优于无效个体
    assert is_better(infeasible, invalid, False, "feasible_first", 1e6)

    # penalty 策略：重罚下可行解仍占优
    assert is_better(feasible, infeasible, False, "penalty", 1e6)

    ordered = sort_population([invalid, feasible, infeasible], False,
                              "feasible_first", 1e6)
    assert ordered[0] is feasible  # 列表开头为最优
    assert ordered[-1] is invalid  # 无效个体排在最后


# ---------------------------------------------------------------------------
# 5. NaN / 异常 fitness
# ---------------------------------------------------------------------------

def test_nan_fitness_region():
    print("\n[test_nan_fitness] 一半空间返回 NaN，仍应收敛到原点")

    def half_nan(x):
        if x[0] < 0.0:
            return math.nan
        return float(np.sum(x * x))

    ga = GeneticAlgorithm(
        half_nan,
        [Variable(-5.0, 5.0), Variable(-5.0, 5.0)],
        GAConfig(pop_size=60, n_generations=80, seed=21),
    )
    result = ga.run()
    assert result.best_feasible
    assert result.best_genes[0] >= 0.0
    assert result.best_objective < 0.05
    # 初始代至少有部分无效个体被正确统计
    assert result.history[0].n_valid < ga.config.pop_size


def test_exception_fitness_region():
    print("\n[test_exception_fitness] 一半空间抛异常，仍应收敛到原点")

    def half_raise(x):
        if x[0] > 0.0:
            raise RuntimeError("boom: 模拟目标函数失败")
        return float(np.sum(x * x))

    ga = GeneticAlgorithm(
        half_raise,
        [Variable(-5.0, 5.0), Variable(-5.0, 5.0)],
        GAConfig(pop_size=60, n_generations=80, seed=22),
    )
    result = ga.run()
    assert result.best_feasible
    assert result.best_genes[0] <= 0.0
    assert result.best_objective < 0.05
    assert result.history[0].n_valid < ga.config.pop_size


def test_constraint_exception_is_infeasible():
    print("\n[test_constraint_exception] 约束抛异常的个体按不可行处理")

    def bad_constraint(x):
        raise RuntimeError("constraint failure")

    ga = GeneticAlgorithm(
        sphere,
        [Variable(-1.0, 1.0)],
        GAConfig(pop_size=20, n_generations=10, seed=1),
        constraints=[bad_constraint],
    )
    ga.initialize()
    assert all(not ind.feasible for ind in ga.population)
    assert all(np.isinf(ind.violation) for ind in ga.population)


# ---------------------------------------------------------------------------
# 6. 随机确定性
# ---------------------------------------------------------------------------

def _run_full(seed, generations=30, dims=3, **overrides):
    cfg = GAConfig(pop_size=40, n_generations=generations, seed=seed,
                   verbose=False, **overrides)
    ga = GeneticAlgorithm(sphere, [Variable(-5.0, 5.0) for _ in range(dims)], cfg)
    return ga.run()


def test_same_seed_identical_history():
    print("\n[test_same_seed_identical_history] 相同种子进化历史逐位一致")
    r1 = _run_full(123)
    r2 = _run_full(123)
    for s1, s2 in zip(r1.history, r2.history):
        assert s1 == s2
    np.testing.assert_array_equal(r1.best_genes, r2.best_genes)


def test_different_seeds_differ():
    print("\n[test_different_seeds_differ] 不同种子产生不同初始种群")
    cfg1 = GAConfig(pop_size=40, n_generations=5, seed=1, verbose=False)
    cfg2 = GAConfig(pop_size=40, n_generations=5, seed=2, verbose=False)
    g1 = GeneticAlgorithm(sphere, [Variable(-5.0, 5.0)], cfg1)
    g2 = GeneticAlgorithm(sphere, [Variable(-5.0, 5.0)], cfg2)
    g1.initialize()
    g2.initialize()
    assert not np.allclose(g1.population[0].genes, g2.population[0].genes)


@pytest.mark.parametrize("selection", ["tournament", "roulette"])
@pytest.mark.parametrize("crossover_", ["sbx", "uniform", "onepoint"])
@pytest.mark.parametrize("mutation", ["gaussian", "uniform"])
def test_all_operator_combinations_deterministic(selection, crossover_, mutation):
    print(f"\n[operators] selection={selection} crossover={crossover_} "
          f"mutation={mutation} 可复现性")
    kw = dict(selection_method=selection, crossover_method=crossover_,
              mutation_method=mutation)
    r1 = _run_full(555, generations=15, **kw)
    r2 = _run_full(555, generations=15, **kw)
    np.testing.assert_array_equal(r1.best_genes, r2.best_genes)
    for s1, s2 in zip(r1.history, r2.history):
        assert s1 == s2
    assert np.isfinite(r1.best_objective)


def test_each_crossover_method_improves():
    print("\n[test_each_crossover_method_improves] 三种交叉都能在球函数上改进")
    for method in ("sbx", "uniform", "onepoint"):
        cfg = GAConfig(pop_size=50, n_generations=50, seed=777, verbose=False,
                       crossover_method=method)
        ga = GeneticAlgorithm(sphere, [Variable(-5.0, 5.0) for _ in range(2)], cfg)
        r = ga.run()
        assert r.best_objective < 0.1, method


# ---------------------------------------------------------------------------
# 7. Checkpoint 恢复
# ---------------------------------------------------------------------------

def test_checkpoint_resume_matches_continuous(tmp_path):
    print("\n[test_checkpoint_resume_matches_continuous] 断点续跑 == 连续执行")
    ckpt_split = str(tmp_path / "split.ckpt")

    variables = [Variable(-5.0, 5.0) for _ in range(3)]

    # 连续执行 40 代
    cont_cfg = GAConfig(pop_size=40, n_generations=40, seed=314, verbose=False)
    cont = GeneticAlgorithm(sphere, variables, cont_cfg)
    r_cont = cont.run()

    # 分段执行：计划始终是 40 代，先跑到第 15 代（15 是 checkpoint_every 倍数）
    cfg_a = GAConfig(pop_size=40, n_generations=40, seed=314, verbose=False,
                     checkpoint_every=5)
    ga_a = GeneticAlgorithm(sphere, variables, cfg_a)
    ga_a.run(generations=15, checkpoint_path=ckpt_split)

    # 全新实例从检查点恢复，继续到 40 代
    cfg_b = GAConfig(pop_size=40, n_generations=40, seed=314, verbose=False)
    ga_b = GeneticAlgorithm(sphere, variables, cfg_b)
    r_split = ga_b.run(generations=40, checkpoint_path=ckpt_split)

    assert r_split.generations_run == 40
    assert len(r_split.history) == len(r_cont.history)
    for s_a, s_b in zip(r_cont.history, r_split.history):
        assert s_a == s_b
    np.testing.assert_array_equal(r_cont.best_genes, r_split.best_genes)

    # 恢复后种群基因也逐位一致
    for ind_c, ind_s in zip(cont.population, ga_b.population):
        np.testing.assert_array_equal(ind_c.genes, ind_s.genes)


def test_checkpoint_resume_improvement_continues(tmp_path):
    print("\n[test_checkpoint_resume_improvement_continues] 恢复后继续优化不报错且有效")
    ckpt = str(tmp_path / "c.ckpt")
    variables = [Variable(-5.0, 5.0) for _ in range(2)]
    # 整个运行计划为 60 代，先执行 20 代并落盘
    cfg = GAConfig(pop_size=40, n_generations=60, seed=8, verbose=False,
                   checkpoint_every=10)
    ga = GeneticAlgorithm(sphere, variables, cfg)
    r20 = ga.run(generations=20, checkpoint_path=ckpt)

    cfg2 = GAConfig(pop_size=40, n_generations=60, seed=8, verbose=False)
    ga2 = GeneticAlgorithm(sphere, variables, cfg2)
    r60 = ga2.run(generations=60, checkpoint_path=ckpt)
    assert r60.best_objective <= r20.best_objective + 1e-12
    assert r60.best_objective < 1e-2


# ---------------------------------------------------------------------------
# 8. 结果导出
# ---------------------------------------------------------------------------

def test_result_json_export(tmp_path):
    print("\n[test_result_json_export] 结果写入本地 JSON")
    import json

    path = str(tmp_path / "result.json")
    ga = GeneticAlgorithm(
        sphere,
        [Variable(-2.0, 2.0)],
        GAConfig(pop_size=20, n_generations=10, seed=4, verbose=False),
    )
    ga.run(result_path=path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["generations_run"] == 10
    assert len(data["history"]) == 11
    assert "best_genes" in data


# ---------------------------------------------------------------------------
# 9. 参数校验
# ---------------------------------------------------------------------------

def test_config_validation():
    with pytest.raises(ValueError):
        GAConfig(pop_size=1)
    with pytest.raises(ValueError):
        GAConfig(crossover_prob=1.5)
    with pytest.raises(ValueError):
        GAConfig(selection_method="unknown")
    with pytest.raises(ValueError):
        GAConfig(elitism=999)
