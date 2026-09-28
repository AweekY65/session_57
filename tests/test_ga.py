"""Automated tests for the local_ga framework.

Runs under both ``python tests/test_ga.py`` and ``pytest``. Every scenario
evolves a real GA and prints per-generation key metrics (best / mean / worst
fitness and feasible-individual count) to the terminal.
"""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local_ga import GeneticOptimizer, VariableSpec  # noqa: E402


# --------------------------------------------------------------------------- #
# Known benchmark functions
# --------------------------------------------------------------------------- #
def sphere(x):
    return float(np.sum(np.asarray(x) ** 2))


def rosenbrock(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def banner(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72, flush=True)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
class TestGeneticAlgorithm(unittest.TestCase):
    # -- 1. Sphere --------------------------------------------------------- #
    def test_sphere_minimize(self):
        banner("[1] Sphere minimization (3d, bounds [-5.12, 5.12])")
        opt = GeneticOptimizer(
            variables=[(-5.12, 5.12)] * 3,
            objective=sphere,
            maximize=False,
            pop_size=50,
            max_generations=80,
            mutation_sigma=0.1,
            elitism=3,
            seed=7,
            verbose=True,
        )
        res = opt.run()
        self.assertTrue(np.isfinite(res.best_fitness))
        self.assertLess(res.best_fitness, 0.05)
        self.assertLessEqual(res.best_fitness, res.history[0]["best"])
        print(f"  -> best_x = {np.round(res.best_x, 4)}, best f = {res.best_fitness:.3e}")

    # -- 2. Rosenbrock ----------------------------------------------------- #
    def test_rosenbrock_minimize(self):
        banner("[2] Rosenbrock minimization (2d, optimum f* = 0 at (1, 1))")
        opt = GeneticOptimizer(
            variables=[(-2.0, 2.0)] * 2,
            objective=rosenbrock,
            maximize=False,
            pop_size=80,
            max_generations=150,
            crossover_method="sbx",
            mutation_sigma=0.05,
            elitism=4,
            seed=11,
            verbose=True,
        )
        res = opt.run()
        self.assertTrue(np.isfinite(res.best_fitness))
        self.assertLess(res.best_fitness, 1.0)
        self.assertLess(res.best_fitness, res.history[0]["best"])
        print(f"  -> best_x = {np.round(res.best_x, 4)}, best f = {res.best_fitness:.3e}")

    # -- 3. Maximization drives solution to upper bounds ------------------- #
    def test_maximize_hits_upper_bounds(self):
        banner("[3a] Maximization sum(x), continuous -> should hit upper bounds")
        bounds = [(0.0, 1.0), (-2.0, 3.0), (1.0, 5.0)]
        opt = GeneticOptimizer(
            variables=bounds,
            objective=lambda x: float(np.sum(x)),
            maximize=True,
            pop_size=40,
            max_generations=60,
            mutation_sigma=0.1,
            elitism=2,
            seed=3,
            verbose=True,
        )
        res = opt.run()
        uppers = np.array([b[1] for b in bounds])
        self.assertAlmostEqual(res.best_fitness, float(uppers.sum()), places=4)
        np.testing.assert_allclose(res.best_x, uppers, atol=2e-3)
        print(f"  -> best_x = {np.round(res.best_x, 4)} (upper = {uppers})")

        banner("[3b] Maximization sum(x), integer -> should hit integer uppers")
        ibounds = [(0, 5, True), (-3, 4, True)]
        opt2 = GeneticOptimizer(
            variables=ibounds,
            objective=lambda x: float(np.sum(x)),
            maximize=True,
            pop_size=40,
            max_generations=40,
            elitism=2,
            seed=3,
            verbose=True,
        )
        res2 = opt2.run()
        self.assertEqual(res2.best_x_list, [5, 4])
        print(f"  -> best_x = {res2.best_x_list} (upper = [5, 4])")

    # -- 4. Bounds + integrality invariant across the whole history -------- #
    def test_bounds_invariant(self):
        banner("[4] Bounds / integrality invariant (mixed int+float population)")
        specs = [
            VariableSpec(-3.0, 6.0),
            VariableSpec(-2, 7, integer=True),
            VariableSpec(0.0, 1.0, integer=False),
            VariableSpec(-4, -1, integer=True),
        ]
        lower = np.array([s.lower for s in specs])
        upper = np.array([s.upper for s in specs])
        int_mask = np.array([s.integer for s in specs])

        seen = []

        opt = GeneticOptimizer(
            variables=specs,
            objective=sphere,
            pop_size=60,
            max_generations=50,
            mutation_method="gaussian",
            seed=99,
            verbose=True,
        )

        # instrument: capture population at every generation by monkeypatching
        captured = []
        orig_step = opt._step

        def cap_step():
            orig_step()
            captured.append(opt.population.copy())

        opt._step = cap_step
        opt.run()

        for gen, pop in enumerate(captured):
            self.assertTrue(
                np.all(pop >= lower - 1e-12) and np.all(pop <= upper + 1e-12),
                msg=f"generation {gen} violated bounds",
            )
            int_vals = pop[:, int_mask]
            np.testing.assert_array_equal(int_vals, np.round(int_vals))
        self.assertEqual(len(captured), 50)
        print("  -> all 50 generations: every gene inside bounds, integer genes integral")

    # -- 5. Constrained optimization, feasibility priority ----------------- #
    def test_constrained(self):
        banner("[5] Constrained: min (x-3)^2 + (y-3)^2 s.t. x+y <= 4, x,y >= 0")
        # unconstrained optimum (3,3) f=0 infeasible; constrained optimum on
        # x+y=4 at (2,2) with f = 2
        def constraints(x):
            return [
                float(x[0] + x[1] - 4.0),  # x + y <= 4
                float(-x[0]),               # x >= 0
                float(-x[1]),               # y >= 0
            ]

        opt = GeneticOptimizer(
            variables=[(0.0, 6.0), (0.0, 6.0)],
            objective=lambda x: float((x[0] - 3.0) ** 2 + (x[1] - 3.0) ** 2),
            maximize=False,
            constraints=constraints,
            constraint_handling="feasibility",
            pop_size=60,
            max_generations=80,
            mutation_sigma=0.08,
            elitism=2,
            seed=21,
            verbose=True,
        )
        res = opt.run()
        self.assertTrue(res.best_is_feasible)
        self.assertAlmostEqual(res.best_fitness, 2.0, places=1)
        np.testing.assert_allclose(res.best_x, [2.0, 2.0], atol=0.15)
        # feasible individuals must outrank every infeasible one
        self.assertTrue((res.history[-1]["feasible"]) > 0)
        print(f"  -> best_x = {np.round(res.best_x, 4)}, f = {res.best_fitness:.4f} "
              f"(optimum ≈ (2,2), f* = 2)")

    # -- 6. Penalty mode --------------------------------------------------- #
    def test_penalty_mode(self):
        banner("[6] Same constrained problem with constraint_handling='penalty'")
        def constraints(x):
            return [float(x[0] + x[1] - 4.0), float(-x[0]), float(-x[1])]

        opt = GeneticOptimizer(
            variables=[(0.0, 6.0), (0.0, 6.0)],
            objective=lambda x: float((x[0] - 3.0) ** 2 + (x[1] - 3.0) ** 2),
            maximize=False,
            constraints=constraints,
            constraint_handling="penalty",
            penalty_weight=1e3,
            pop_size=60,
            max_generations=120,
            seed=21,
            verbose=True,
        )
        res = opt.run()
        self.assertTrue(res.best_is_feasible)
        self.assertLess(abs(res.best_fitness - 2.0), 0.1)
        print(f"  -> best_x = {np.round(res.best_x, 4)}, f = {res.best_fitness:.4f}")

    # -- 7. Determinism ---------------------------------------------------- #
    def test_determinism(self):
        banner("[7] Determinism: identical seeds -> identical history")
        def build(seed):
            return GeneticOptimizer(
                variables=[(-5.0, 5.0)] * 3,
                objective=sphere,
                pop_size=40,
                max_generations=40,
                crossover_method="uniform",
                mutation_method="gaussian",
                selection="tournament",
                seed=seed,
                verbose=False,
            )

        r1 = build(123).run()
        r2 = build(123).run()
        for h1, h2 in zip(r1.history, r2.history):
            self.assertEqual(h1["generation"], h2["generation"])
            np.testing.assert_array_equal(h1["best_x"], h2["best_x"])
            self.assertEqual(h1["best"], h2["best"])
        np.testing.assert_array_equal(r1.population, r2.population)
        np.testing.assert_array_equal(r1.best_x, r2.best_x)

        # different seed should produce a different trajectory (with
        # overwhelming probability; compare initial populations)
        r3 = build(999).run()
        self.assertFalse(np.allclose(r1.population, r3.population))

        # print one run's per-generation metrics so this scenario also logs
        opt = build(123)
        opt.verbose = True
        opt.run()
        print("  -> two independent seed=123 runs match element-wise; "
              "seed=999 differs")

    # -- 8. NaN / exception fitness --------------------------------------- #
    def test_invalid_fitness(self):
        banner("[8] Robustness: objective raises / returns NaN in part of space")
        def noisy_objective(x):
            if x[0] > 2.0:
                raise FloatingPointError("simulated user objective failure")
            if x[1] < -2.0:
                return float("nan")
            if abs(x[0]) > 4.5:
                return float("inf")
            return sphere(x)

        opt = GeneticOptimizer(
            variables=[(-5.0, 5.0), (-5.0, 5.0)],
            objective=noisy_objective,
            maximize=False,
            pop_size=60,
            max_generations=60,
            mutation_sigma=0.1,
            elitism=2,
            seed=55,
            verbose=True,
        )
        res = opt.run()
        self.assertTrue(res.best_is_valid)
        self.assertLessEqual(res.best_x[0], 2.0 + 1e-9)
        self.assertGreaterEqual(res.best_x[1], -2.0 - 1e-9)
        self.assertLess(res.best_fitness, 0.5)
        # invalid individuals exist at some point but never win
        for rec in res.history:
            self.assertTrue(np.isfinite(rec["best"]))
        print(f"  -> best_x = {np.round(res.best_x, 4)} in valid region, "
              f"f = {res.best_fitness:.3e}")

    # -- 9. Checkpoint resume == continuous run --------------------------- #
    def test_checkpoint_resume(self):
        banner("[9] Checkpoint: 30+30 split generations must equal 60 continuous")
        def build(seed):
            return GeneticOptimizer(
                variables=[(-4.0, 4.0)] * 3,
                objective=sphere,
                maximize=False,
                pop_size=50,
                max_generations=60,
                crossover_method="onepoint",
                mutation_method="gaussian",
                mutation_sigma=0.1,
                elitism=2,
                seed=seed,
                verbose=False,
            )

        continuous = build(2024)
        continuous.run()

        split = build(2024)
        split.run(30)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = os.path.join(tmp, "ga.ckpt")
            split.save_checkpoint(ckpt)
            self.assertTrue(os.path.exists(ckpt))
            resumed = GeneticOptimizer.load_checkpoint(ckpt)
            self.assertEqual(resumed.generation, 30)
            resumed.verbose = True  # show second-half per-generation metrics
            resumed.run(30)

        self.assertEqual(resumed.generation, continuous.generation)
        self.assertEqual(len(resumed.history), len(continuous.history))
        for hc, hs in zip(continuous.history, resumed.history):
            np.testing.assert_array_equal(hc["best_x"], hs["best_x"])
            self.assertEqual(hc["best"], hs["best"])
        np.testing.assert_array_equal(resumed.population, continuous.population)
        np.testing.assert_array_equal(resumed.fitnesses, continuous.fitnesses)
        np.testing.assert_array_equal(
            resumed.result.best_x, continuous.result.best_x
        )
        print("  -> split run's populations, fitnesses and best_x match the "
              "continuous run exactly")

    # -- 9b. Checkpoint with integer chromosomes -------------------------- #
    def test_checkpoint_integer(self):
        banner("[9b] Checkpoint resume on integer Sphere")
        def build():
            return GeneticOptimizer(
                variables=[(-10, 10, True)] * 3,
                objective=sphere,
                pop_size=40,
                max_generations=40,
                mutation_method="uniform",
                elitism=2,
                seed=8,
                verbose=False,
            )

        a = build(); a.run()
        b = build(); b.run(20)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "int.ckpt")
            b.save_checkpoint(path)
            b2 = GeneticOptimizer.load_checkpoint(path)
            b2.run(20)
        np.testing.assert_array_equal(b2.population, a.population)
        self.assertEqual(b2.result.best_x_list, a.result.best_x_list)
        for row in b2.population:
            np.testing.assert_array_equal(row, np.round(row))
        print(f"  -> integer best_x = {b2.result.best_x_list}, "
              f"f = {b2.result.best_fitness}")

    # -- 10. Pure integer chromosome -------------------------------------- #
    def test_integer_chromosome(self):
        banner("[10] Integer Sphere (4d, [-8, 8])")
        opt = GeneticOptimizer(
            variables=[VariableSpec(-8, 8, integer=True)] * 4,
            objective=sphere,
            maximize=False,
            pop_size=50,
            max_generations=80,
            mutation_method="gaussian",
            mutation_sigma=0.15,
            elitism=2,
            seed=321,
            verbose=True,
        )
        res = opt.run()
        for v, spec in zip(res.best_x, opt.specs):
            self.assertTrue(float(v).is_integer())
            self.assertTrue(spec.lower <= v <= spec.upper)
        self.assertLessEqual(res.best_fitness, 1.0)  # at least one zero point found region
        self.assertEqual(res.best_x_list, [int(round(v)) for v in res.best_x])
        print(f"  -> best_x = {res.best_x_list}, f = {res.best_fitness}")

    # -- 11. Roulette selection still converges & reproducible ------------ #
    def test_roulette_and_uniform_mutation(self):
        banner("[11] Roulette selection + uniform mutation on Sphere")
        opt = GeneticOptimizer(
            variables=[(-5.12, 5.12)] * 2,
            objective=sphere,
            selection="roulette",
            mutation_method="uniform",
            pop_size=40,
            max_generations=60,
            mutation_rate=0.3,
            elitism=2,
            seed=77,
            verbose=True,
        )
        res = opt.run()
        self.assertLess(res.best_fitness, 0.5)
        print(f"  -> best f = {res.best_fitness:.3e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
