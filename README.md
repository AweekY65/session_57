# local_ga — 纯本地遗传算法优化框架

一个**完全本地运行**的遗传算法（Genetic Algorithm）框架。所有种群、适应度、随机数状态、检查点和优化结果都只存在于进程内存或你指定的本地文件中：

- 仅依赖 **NumPy** 与 Python 标准库（`pickle`、`tempfile` 等）；
- 核心 GA 逻辑（选择 / 交叉 / 变异 / 精英保留 / 约束处理）全部自行实现；
- **没有任何**网络请求、SaaS、数据库或云服务调用。

## 目录结构

```
local_ga/
├── variables.py     # 变量定义、初始采样、边界修复（repair）
├── evaluation.py    # 目标评估、NaN/异常处理、约束违约度、个体排序
├── operators.py     # selection / crossover / mutation / elitism
├── optimizer.py     # GeneticOptimizer 主循环与 OptimizeResult
└── checkpoint.py    # 本地 pickle 检查点（原子写入）
tests/test_ga.py     # 自动化测试（终端打印每代关键指标）
examples/sphere_demo.py
```

## 快速开始

```python
import numpy as np
from local_ga import GeneticOptimizer, VariableSpec

def sphere(x):                       # 最小化目标
    return float(np.sum(x ** 2))

opt = GeneticOptimizer(
    variables=[VariableSpec(-5.12, 5.12)] * 3,  # 也可以写 (-5.12, 5.12)
    objective=sphere,
    maximize=False,                  # True = 最大化
    pop_size=40,
    max_generations=100,
    seed=42,                         # 固定种子 => 完全可复现
    verbose=True,                    # 终端打印每代 best/mean/worst/feasible
)
result = opt.run()
print(result.best_x_list, result.best_fitness)
```

整数变量（混合染色体也支持）：

```python
GeneticOptimizer(
    variables=[
        VariableSpec(0.0, 1.0, integer=False),  # 连续
        VariableSpec(-5, 5, integer=True),      # 整数（界必须为整数）
    ],
    objective=my_fn,
    ...
)
# 简写元组：(lower, upper) 或 (lower, upper, integer)
```

## 算法参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `variables` | 必填 | `VariableSpec(lower, upper, integer=False)` 列表，或 `(lo, hi[, int])` 元组；界为闭区间 |
| `objective` | 必填 | 目标函数 `f(x) -> float`，`x` 为 numpy 数组 |
| `maximize` | `False` | `False` 最小化 / `True` 最大化，内部统一转为“越大越优” |
| `pop_size` | `40` | 种群规模，必须 ≥ 2 |
| `max_generations` | `100` | `run()` 默认运行到的代数（第 0 代为初始种群） |
| `selection` | `"tournament"` | 选择策略，见下 |
| `tournament_size` | `3` | 锦标赛大小 k |
| `crossover_method` | `"uniform"` | 交叉策略：`uniform` / `onepoint` / `sbx` |
| `crossover_rate` | `0.9` | 每对亲本执行交叉的概率 |
| `sbx_eta` | `15.0` | SBX 分布指数，越大子代越接近亲本 |
| `mutation_method` | `"gaussian"` | 变异策略：`gaussian` / `uniform` |
| `mutation_rate` | `0.2` | 个体发生变异的概率（先过个体门，再按基因概率） |
| `gene_mutation_rate` | `1/d` | 单个基因被变异的概率，默认每代平均变异一个基因 |
| `mutation_sigma` | `0.1` | 高斯变异步长 = `sigma × (ub − lb)`；整数变量取整为几何步长 |
| `elitism` | `2` | 每代原样保留的最优个体数，范围 `[0, pop_size)` |
| `constraints` | `None` | 约束函数 `g(x) -> [g_1, …]`，约定 **g_i(x) ≤ 0 可行** |
| `constraint_handling` | `"feasibility"` | `feasibility`（默认）或 `penalty` |
| `penalty_weight` | `1e6` | penalty 模式权重，适应度按 `score − weight × 违约度` 排序 |
| `constraint_tol` | `1e-12` | 判定可行的违约度容差 |
| `seed` | `42` | `numpy.random.default_rng` 种子；`None` 为随机种子 |
| `verbose` | `True` | 打印每代关键指标 |
| `on_generation` | `None` | 每代回调 `fn(record)` |

每代打印的列为：`gen / best / mean / worst / feas（可行个体数）/ viol（最优个体违约度）/ dir(max|min)`。
`mean` 只统计有效（未抛异常、非 NaN/inf）个体。

## 选择策略

- **锦标赛选择 `tournament`**（默认）：每次有放回随机抽 k 个个体，排序键最高者获胜；`tournament_size` 越大选择压力越强。
- **名次轮盘 `roulette`**：按名次赋权（最优权重 N、最差权重 1）再轮盘赌。基于名次而非原始适应度，因此对负目标值、尺度差异、最大化/最小化都稳健。

## 交叉策略

- **均匀交叉 `uniform`**（默认）：每个基因以 0.5 概率来自任一亲本。
- **单点交叉 `onepoint`**：随机一个切分点交换基因片段（一维染色体退化为克隆）。
- **SBX `sbx`**：模拟二进制交叉（Deb–Agrawal），连续变量的经典实值算子，由 `sbx_eta` 控制扩展程度。

## 变异策略

- **高斯变异 `gaussian`**（默认）：`x_j += sigma·(ub_j−lb_j)·N(0,1)`；整数基因使用取整的几何随机步长（多数为局部 ±1~3，偶有长跳）。
- **均匀变异 `uniform`**：该基因在自身界内均匀重采样（整数即在整数格点上重采样）。

## 边界保证（子代始终可行）

任何算子产出的新个体都统一经过 `repair`：先 `clip` 到 `[lower, upper]`，再对整数基因四舍五入取整。因此主循环中**每个进入评估的个体都保证在界内，且整数基因严格为整数**（测试逐代校验该不变量）。

## NaN、异常与不可行个体的处理

- 目标函数**抛异常**、返回 **NaN**、`+/-inf`：个体标记为无效，不会中断搜索；
- 约束函数抛异常或返回非有限值：同样标记为无效；
- **无效个体永远排在所有有效个体之后**，因此不可能成为最优解。

## 约束处理与优先规则

约束统一写成 `g_i(x) <= 0`。总违约度为 `violation = Σ max(0, g_i)`。

1. **`feasibility`（默认，Deb 可行优先规则，无量纲尺度问题）**，排序优先级从高到低：
   1. 有效且可行 —— 按目标分数排序；
   2. 有效但不可行 —— 先比违约度（小者优先），违约度相同再比目标分数；
   3. 无效（NaN/异常）—— 永远垫底。
2. **`penalty`**：有效个体按 `score − penalty_weight × violation` 单一标量排序；可行个体仍严格优先于无效个体。请按目标函数量级选择足够大的 `penalty_weight`。

两种模式下可行解都有明确、确定的优先规则。

## 可复现性（固定随机种子）

全部随机性来自优化器持有的**唯一一个** `numpy.random.Generator`，且每代严格按固定顺序取随机数（初始采样 → 选择 → 交叉 → 变异）。相同配置 + 相同种子会生成**逐元素一致**的进化历史（种群、适应度、最优解），测试 `test_determinism` 对此做精确断言。

## Checkpoint（检查点续跑）

```python
opt = GeneticOptimizer(..., seed=42, verbose=False)
opt.run(30)
opt.save_checkpoint("./run.ckpt")          # 本地 pickle 文件，原子替换写入

# …新进程/稍后…
opt2 = GeneticOptimizer.load_checkpoint("./run.ckpt")
opt2.run(30)                               # 与连续跑 60 代完全一致
```

检查点保存：全部配置、种群、适应度/有效性/违约度、当前代数、历史、评估计数以及 **RNG bit-generator 状态**。由于 RNG 状态随检查点保存，续跑的随机数抽取顺序与连续执行完全相同，测试 `test_checkpoint_resume`（连续场景）与 `test_checkpoint_integer`（整数场景）对种群和适应度做逐元素相等断言。

> 注意：pickle 仅序列化数据；自定义 `objective` / `constraints` 函数需在加载环境中仍可导入（请使用模块级函数，而不是 lambda/闭包）。每代回调 `on_generation` 不随检查点保存，加载后可重新设置。

## 运行测试

```bash
# 方式一：直接执行（unittest，终端会打印每代关键指标）
python3 tests/test_ga.py

# 方式二：pytest（加 -s 才能看到逐代打印）
python3 -m pytest tests/ -v -s

# 端到端示例（含 checkpoint 续跑一致性验证）
python3 examples/sphere_demo.py
```

测试覆盖 12 个场景：

| # | 场景 |
|---|---|
| 1 | Sphere 最小化收敛到 0 |
| 2 | Rosenbrock（SBX）收敛到已知最优 (1,1)，f*≈0 |
| 3 | 最大化连续/整数问题，最优解命中上界 |
| 4 | 全部进化代的边界不变量与整数不变量 |
| 5 | 约束优化（feasibility 规则）找到边界可行最优 (2,2) |
| 6 | 约束优化（penalty 模式） |
| 7 | 随机确定性：同种子逐元素一致，不同种子轨迹不同 |
| 8 | 目标抛异常 / NaN / inf：不崩溃、无效个体垫底、正常收敛 |
| 9 | Checkpoint 分段续跑与连续执行逐元素一致 |
| 9b | 整数染色体的 checkpoint 续跑一致性 |
| 10 | 纯整数染色体的合法性与收敛 |
| 11 | roulette 选择 + uniform 变异组合 |

## 环境要求

- Python ≥ 3.9
- NumPy ≥ 1.24（`pip install -r requirements.txt`）
