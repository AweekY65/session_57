"""示例：在 Sphere / Rosenbrock / 带约束问题上运行本地遗传算法。

运行：python examples/run_benchmarks.py
每代关键指标会打印到终端，结果写入本地 results/ 目录。
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local_ga import GAConfig, GeneticAlgorithm, Variable, greater_equal  # noqa: E402


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x * x))


def rosenbrock(x: np.ndarray) -> float:
    return float(
        np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)
    )


def main() -> None:
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(out_dir, exist_ok=True)

    # --- 1. Sphere（5 维，最小化，理论最优点 0）---
    print("\n### Sphere 5D: min sum(x_i^2), x_i in [-5.12, 5.12]")
    vars_sphere = [Variable(-5.12, 5.12, name=f"x{i}") for i in range(5)]
    ga = GeneticAlgorithm(
        sphere,
        vars_sphere,
        GAConfig(
            pop_size=100,
            n_generations=250,
            seed=2026,
            checkpoint_every=50,
            mutation_prob=0.2,
            eta=20,
            elitism=2,
        ),
    )
    r = ga.run(
        checkpoint_path=os.path.join(out_dir, "sphere.ckpt"),
        result_path=os.path.join(out_dir, "sphere_result.json"),
    )
    assert r.best_objective < 1e-3

    # --- 2. Rosenbrock（2 维，最小化，理论最优点 (1,1)，最优值 0）---
    print("\n### Rosenbrock 2D: min f, x_i in [-2, 2]")
    vars_rb = [Variable(-2.0, 2.0, name="x"), Variable(-2.0, 2.0, name="y")]
    ga = GeneticAlgorithm(
        rosenbrock,
        vars_rb,
        GAConfig(
            pop_size=120,
            n_generations=250,
            seed=36,
            mutation_prob=0.4,
            eta=20,
            elitism=2,
        ),
    )
    r = ga.run(result_path=os.path.join(out_dir, "rosenbrock_result.json"))
    assert r.best_objective < 1e-2

    # --- 3. 约束问题：min x^2 + y^2,  s.t. x + y >= 3 ---
    # 理论最优 (1.5, 1.5)，最优值 4.5
    print("\n### 约束优化: min x^2 + y^2,  s.t. x + y >= 3")
    ga = GeneticAlgorithm(
        lambda x: x[0] ** 2 + x[1] ** 2,
        [Variable(-5.0, 5.0), Variable(-5.0, 5.0)],
        GAConfig(pop_size=60, n_generations=100, seed=7),
        constraints=[greater_equal(lambda x: x[0] + x[1], 3.0)],
    )
    r = ga.run(result_path=os.path.join(out_dir, "constrained_result.json"))
    assert r.best_feasible
    assert abs(r.best_objective - 4.5) < 0.05


if __name__ == "__main__":
    main()
