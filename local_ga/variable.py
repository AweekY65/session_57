"""变量（染色体基因）定义。"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Variable:
    """单个决策变量的描述。

    Parameters
    ----------
    lower, upper:
        变量的上下界（闭区间）。
    integer:
        为 True 时该变量只取整数；初始化、交叉、变异产生的新值都会被
        四舍五入并裁剪到合法整数范围内。
    name:
        可选的变量名，仅用于展示。
    """

    lower: float
    upper: float
    integer: bool = False
    name: str | None = None

    def __post_init__(self) -> None:
        lo, hi = float(self.lower), float(self.upper)
        if math.isnan(lo) or math.isnan(hi):
            raise ValueError("变量上下界不能是 NaN")
        if lo > hi:
            raise ValueError(f"变量下界 {lo} 不能大于上界 {hi}")
        if self.integer:
            if not (math.isfinite(lo) and math.isfinite(hi)):
                raise ValueError("整数变量必须使用有限的上下界")
            ilo, ihi = int(round(lo)), int(round(hi))
            if not math.isclose(float(ilo), lo) or not math.isclose(float(ihi), hi):
                raise ValueError(
                    f"整数变量的上下界必须是整数，收到 [{lo}, {hi}]"
                )

    @property
    def int_lower(self) -> int:
        return int(round(self.lower))

    @property
    def int_upper(self) -> int:
        return int(round(self.upper))
