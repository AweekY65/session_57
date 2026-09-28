# local-ga：纯本地遗传算法优化框架

一个**只在本地运行**的遗传算法（Genetic Algorithm, GA）框架：所有种群、
适应度、随机状态、检查点与优化结果都保存在内存或本地文件中，**不依赖任何
优化 SaaS、数据库、云计算平台或外部服务**。核心 GA 逻辑全部自行实现，
唯一的数值计算依赖是 NumPy。

## 特性一览

- **连续 / 整数染色体**：每个变量可独立配置上下界与类型；初始化、交叉、
  变异产生的新个体一律裁剪回边界，整数变量就近取整，种群中永远不会出现
  违反变量约束的个体。
- **选择 / 交叉 / 变异 / 精英保留**：
  - 选择：锦标赛选择（默认）、基于线性排名的轮盘赌；
  - 交叉：模拟二进制交叉 SBX（默认，边界感知）、均匀交叉、单点交叉；
  - 变异：高斯变异（默认，步长随代数线性收缩）、均匀重置变异；
  - 精英保留：每代最好的 `elitism` 个个体原样进入下一代。
- **最大化 / 最小化**：`maximize=True/False` 自动处理比较方向。
- **鲁棒的适应度处理**：目标函数返回 `NaN`/`±inf` 或抛异常的个体标记为
  “无效”，在选择中永远排在有效个体之后，不会污染进化过程。
- **约束优化**：
  - `feasible_first`（默认，Deb 可行解优先规则）：任意可行解优于任意
    不可行解；不可行个体之间按约束违反量排序；
  - `penalty`：目标值加上 `penalty_weight × 违反量`（最大化时为减）。
- **完全可复现**：固定 `seed` 后，相同配置生成**逐位一致**的进化历史。
- **检查点（Checkpoint）**：原子写入本地文件，保存种群、代数、随机数
  状态和历史；恢复后续跑的进化轨迹与一次性连续执行完全一致。
- **每代关键指标**：终端实时打印当代最优/均值/最差目标值、可行个体数、
  有效个体数、最小约束违反量；最终结果可导出为本地 JSON。

## 安装

```bash
pip install -r requirements.txt   # 仅 numpy（pytest 用于测试）
```

无网络环境下，只要环境里已有 NumPy 即可直接使用，无需安装任何东西。

## 快速开始

```python
import numpy as np
from local_ga import GAConfig, GeneticAlgorithm, Variable, greater_equal

# 最小化 x0^2 + x1^2，约束 x0 + x1 >= 3（理论最优 (1.5, 1.5)，最优值 4.5）
def objective(x):
    return float(x[0] ** 2 + x[1] ** 2)

ga = GeneticAlgorithm(
    objective,
    variables=[Variable(-5.0, 5.0, name="x0"), Variable(-5.0, 5.0, name="x1")],
    config=GAConfig(
        pop_size=60,          # 种群大小
        n_generations=100,    # 进化代数
        crossover_prob=0.9,   # 交叉概率
        mutation_prob=0.1,    # 每个基因的变异概率
        elitism=1,            # 精英保留数量
        maximize=False,       # 最小化
        seed=42,              # 固定随机种子
    ),
    constraints=[greater_equal(lambda x: x[0] + x[1], 3.0)],
)
result = ga.run()
print(result.x, result.fun)      # [1.42 ... 1.57 ...] ≈ 4.5
```

终端输出形如：

```
gen=   0  best=+9.39162  mean=+16.7208  worst=+26.4513  feasible=10  valid=60  min_viol=0.0282
gen=   1  best=+7.69117  mean=+10.1063  worst=+18.4185  feasible=13  valid=60  min_viol=7.7e-05
...
gen= 100  best=+4.50816  mean=+4.50666  worst=+4.58222  feasible=50  valid=60  min_viol=9.99e-07
```

### 整数变量

```python
variables = [
    Variable(0, 100, integer=True, name="num_workers"),
    Variable(1, 16,  integer=True, name="batch_size"),
    Variable(0.0, 1.0, name="threshold"),   # 连续变量可混用
]
```

整数变量初始化时均匀抽取整数，高斯变异后就近取整，且在每一步都被裁剪到
`[lower, upper]` 的整数集合内。

### 最大化

只需 `GAConfig(maximize=True)`，选择方向、统计指标、精英保留会全部反转。

### Checkpoint：中断后续跑

```python
# 第一次运行（计划共 100 代，先跑 40 代，每 10 代写一次检查点）
cfg = GAConfig(pop_size=60, n_generations=100, seed=42, checkpoint_every=10)
ga = GeneticAlgorithm(objective, variables, cfg)
ga.run(generations=40, checkpoint_path="run/ga.ckpt")

# 之后（甚至新进程）用同一份“运行计划”从检查点恢复到第 100 代：
ga2 = GeneticAlgorithm(objective, variables,
                       GAConfig(pop_size=60, n_generations=100, seed=42))
result = ga2.run(generations=100, checkpoint_path="run/ga.ckpt")
```

检查点是原子写入（临时文件 + `os.replace`），内容为 pickle，包含：

| 字段 | 内容 |
|---|---|
| `population` | 全部个体的基因、目标值、违反量、可行/有效标记 |
| `generation` | 已完成的代数 |
| `rng_state` | `numpy.random.Generator` 的完整 bit-generator 状态 |
| `history` | 每代统计指标 |
| `config` / `variables` | 算法参数与变量定义（除“目标代数”运行计划外） |

> **注意（确定性约定）**：高斯步长收缩依赖“目标总代数”。要让分段续跑与
> 连续执行**完全一致**，两次 `run` 必须使用相同的 `n_generations` 总计划
> （中间用 `generations=` 参数指定本次跑到第几代）。`n_generations` 是
> 运行计划，不属于进化状态，因此不写入检查点。不想用收缩时设
> `sigma_end_fraction=1.0`，此时各代步长恒定，恢复更自由。

### 结果导出

```python
result = ga.run(result_path="run/result.json")
```

JSON 内含最优解、目标值、可行性与逐代指标（`NaN`/`inf` 导出为 `null`）。

## 算法参数说明（`GAConfig`）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `pop_size` | 50 | 种群大小 |
| `n_generations` | 100 | 目标总代数（第 0 代为初始种群，共记录 `n_generations+1` 行） |
| `crossover_prob` | 0.9 | 每对父代发生交叉的概率，否则直接复制父代 1 |
| `mutation_prob` | 0.1 | **每个基因**独立发生变异的概率 |
| `eta` | 15.0 | SBX 分布指数：越大子代越靠近父代 |
| `sigma` | `None` | 高斯变异初始步长；`None` 时连续变量取 5% 变量宽度，整数取 `max(1, 10% 宽度)` |
| `sigma_end_fraction` | 0.02 | 末代步长相对初始步长的比例（线性收缩）；`1.0` 表示不收缩 |
| `mutation_method` | `"gaussian"` | `"gaussian"` 或 `"uniform"`（在边界内均匀重置） |
| `crossover_method` | `"sbx"` | `"sbx"` / `"uniform"` / `"onepoint"` |
| `selection_method` | `"tournament"` | `"tournament"` / `"roulette"` |
| `tournament_size` | 3 | 锦标赛大小 |
| `elitism` | 1 | 每代原样保留的最优个体数 |
| `maximize` | `False` | `True` 为最大化，`False` 为最小化 |
| `seed` | 42 | 随机种子；相同种子 + 相同配置 ⇒ 完全一致的进化历史 |
| `constraint_strategy` | `"feasible_first"` | `"feasible_first"`（Deb 规则）或 `"penalty"` |
| `penalty_weight` | 1e6 | penalty 策略下的惩罚系数 |
| `eq_tolerance` | 1e-6 | 违反量 ≤ 该值即视为可行 |
| `checkpoint_every` | 0 | 每 N 代写一次检查点（0 表示仅结束时写） |
| `verbose` | `True` | 是否在终端打印每代指标 |

## 选择与变异策略

### 选择

- **锦标赛（默认）**：每次随机抽 `tournament_size` 个个体，按统一的比较
  规则选最好的一个。选择压稳定、对目标尺度不敏感。
- **轮盘赌**：按线性排名（最好权重 N，最差权重 1）分配概率。用排名而非
  原始目标值，避免目标尺度过大时早熟，也天然兼容 NaN 个体（它们排在
  排名末位）。

### 交叉

- **SBX（默认）**：模拟二进制交叉，按多项式分布在两个父代周围生成子代，
  并在计算展开因子时同时考虑左右两侧变量边界，子代天然落在
  `[lower, upper]` 内。
- **均匀交叉**：逐基因掷硬币选择来自哪个父代。
- **单点交叉**：随机选一个切点交换尾部。

### 变异

- **高斯变异（默认）**：基因加上 `N(0, sigma_t)`，`sigma_t` 随代数从
  `sigma` 线性收缩到 `sigma_end_fraction·sigma`——前期保持探索能力，
  后期精细收敛。变异后裁剪回边界。
- **均匀重置变异**：直接在变量边界内均匀抽取一个新值（整数变量取整数），
  适合离散问题或需要强探索的场景。

所有新个体在入种群前还会再做一次 `repair`（裁剪 + 取整）作为双保险。

## 约束处理

约束以函数列表传入：`c(x) -> float`（也支持返回数组），**返回值 ≤ 0 表示
满足，> 0 表示违反量大小**。构造器：

```python
from local_ga import less_equal, greater_equal, equal_to

constraints=[
    less_equal(lambda x: x[0] + x[1], 10.0),      # x0 + x1 <= 10
    greater_equal(lambda x: x[1], 0.0),            # x1 >= 0
    equal_to(lambda x: x[2], 1.0, tol=1e-3),      # x2 == 1（容差）
]
```

总违反量为各约束违反量的**平方和**；约束函数自身抛异常的个体按最严重
不可行处理。两种策略下的明确优先规则：

- `feasible_first`：**有效可行 > 有效不可行（违反量小者优先）> 无效**，
  同层内再按目标方向比较；
- `penalty`：最小化时比较 `objective + weight·violation`，最大化时比较
  `objective − weight·violation`；无效个体依然排在所有有效个体之后。

`fitness` 返回 NaN/inf 或抛异常 ⇒ 个体标记为无效，不参与有效统计，
永远不会被选为精英。

## 可复现性

所有随机性都来自**唯一一个**以 `seed` 初始化的
`numpy.random.Generator`，选择、交叉、变异、初始化全部通过它取随机数；
评估是无副作用的纯函数调用，精英直接复制、不重新评估。因此：

```python
r1 = GeneticAlgorithm(f, vars, GAConfig(seed=123, verbose=False)).run()
r2 = GeneticAlgorithm(f, vars, GAConfig(seed=123, verbose=False)).run()
assert np.array_equal(r1.best_genes, r2.best_genes)   # True
assert r1.history == r2.history                        # True
```

## 测试

测试覆盖：Sphere（5 维）/Rosenbrock（2 维）基准函数、整数染色体、
边界约束逐代校验、最大化、约束优化（不等式 + 等式，两种约束策略）、
相同种子逐位确定性、不同算子组合确定性、NaN/异常 fitness、约束异常、
checkpoint 恢复与连续执行一致性、结果 JSON 导出、参数校验。

```bash
# 终端执行；-s 用于显示每代关键指标
python3 -m pytest tests/ -s -v

# 也可以只跑部分用例
python3 -m pytest tests/test_genetic_algorithm.py::test_sphere_minimize -s
```

基准示例（含检查点与 JSON 结果，产物在 `results/`）：

```bash
python3 examples/run_benchmarks.py
```

## 项目结构

```
.
├── local_ga/
│   ├── __init__.py        # 对外 API
│   ├── config.py          # GAConfig 全部算法参数
│   ├── variable.py        # Variable：上下界 + 连续/整数类型
│   ├── model.py           # Individual / GenerationStats / Result
│   ├── constraints.py     # <=, >=, == 约束构造器与违反量计算
│   ├── operators.py       # 选择 / 交叉 / 变异 / 边界修复（自行实现）
│   └── ga.py              # 进化主循环、评估、checkpoint、结果导出
├── tests/
│   └── test_genetic_algorithm.py
├── examples/
│   └── run_benchmarks.py
├── requirements.txt
├── pytest.ini
└── README.md
```
