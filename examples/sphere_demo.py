"""Minimal end-to-end demo for local_ga.

Runs Sphere minimization twice:
  1. a continuous 40-generation run with a checkpoint in the middle,
  2. proving that resume-from-checkpoint matches one continuous run.

Run:  python3 examples/sphere_demo.py
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local_ga import GeneticOptimizer  # noqa: E402


def sphere(x):
    return float(np.sum(np.asarray(x) ** 2))


def main():
    gens = 40
    split = 20

    print("# Continuous run")
    continuous = GeneticOptimizer(
        variables=[(-5.12, 5.12)] * 3,
        objective=sphere,
        maximize=False,
        pop_size=40,
        max_generations=gens,
        elitism=2,
        mutation_sigma=0.1,
        seed=42,
        verbose=True,
    )
    c_res = continuous.run()

    print("\n# Split run: 20 generations -> checkpoint -> 20 generations")
    split_opt = GeneticOptimizer(
        variables=[(-5.12, 5.12)] * 3,
        objective=sphere,
        maximize=False,
        pop_size=40,
        max_generations=gens,
        elitism=2,
        mutation_sigma=0.1,
        seed=42,
        verbose=True,
    )
    split_opt.run(split)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sphere.ckpt")
        split_opt.save_checkpoint(path)
        print(f"\n[checkpoint written to local file: {path}, "
              f"generation={split_opt.generation}]")
        resumed = GeneticOptimizer.load_checkpoint(path)
        resumed.run(gens - split)

    identical = np.array_equal(
        resumed.result.population, c_res.population
    ) and np.array_equal(resumed.result.best_x, c_res.best_x)

    print("\n# Summary")
    print(f"continuous best : x={np.round(c_res.best_x, 6)}, f={c_res.best_fitness:.3e}")
    print(f"resumed best    : x={np.round(resumed.result.best_x, 6)}, "
          f"f={resumed.result.best_fitness:.3e}")
    print(f"identical trajectories: {identical}")
    assert identical, "checkpoint resume must reproduce the continuous run"
    print("OK — local checkpoint resume is exactly equivalent to continuous execution.")


if __name__ == "__main__":
    main()
