"""local_ga —— 纯本地遗传算法优化框架。

仅依赖 NumPy，所有种群、适应度、随机状态、检查点与优化结果都保存在
内存或本地文件中，不依赖任何 SaaS / 数据库 / 云服务。
"""

from .variable import Variable
from .config import GAConfig
from .model import Individual, GenerationStats, Result
from .constraints import less_equal, greater_equal, equal_to
from .ga import GeneticAlgorithm

__all__ = [
    "Variable",
    "GAConfig",
    "Individual",
    "GenerationStats",
    "Result",
    "less_equal",
    "greater_equal",
    "equal_to",
    "GeneticAlgorithm",
]

__version__ = "1.0.0"
