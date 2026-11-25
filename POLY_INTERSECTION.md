# 整数多项式像集求交

`poly_intersection.py` 独立于编译器入口，默认只用 Python 标准库。它与 Lan 共用精确整数多项式的分段算法。生成元取全体整数，和 Lan 源码里默认取自然数的 `poly` 不同。系数按常数项到最高次项排列，默认闭值域为 `[-2**63, 2**64-1]`。

```python
from poly_intersection import PolynomialIntersection, solve_many

# x² 和 2y²+1 的公共值，限定在 [10**16, 2**64-1]
p = PolynomialIntersection((0, 0, 1), (1, 0, 2), lower=10**16)
print(p.contains(17_380_816_062_160_329))
print(p.witness(17_380_816_062_160_329))  # 返回 value、x、y
print(p.find())                         # 精确判空，找到交点时返回见证
print(p.is_empty())                     # 布尔结果

jobs = [p, PolynomialIntersection((0, 2), (1, 2))]
for result in solve_many(jobs, max_steps=2000):
    print(result.status, result.method, result.witness)
```

`find(max_steps=..., timeout=...)` 用于限预算的查询，返回 `nonempty`、`empty` 或 `unknown`，并记录使用的方法和搜索步数。超限时 `is_empty` 抛出 `SearchLimitExceeded`。步数预算计候选探测，时间预算在算法检查点生效。若只需要交点而不关心顺序，可以使用返回的任意见证。

`solve_set(max_steps=..., timeout=..., max_values=10_000, backend=...)` 返回公共**值**的完整表示。同一个值由多个整数根产生时只算一个。结果类型如下。

- `empty` 表示已经证明没有公共值。
- `progression` 用 `first, step, count` 表示有限等差数列。
- `polynomial` 用 `images` 表示有限个有界多项式像集的并集。每个生成元都取全体整数。除了同像集，还识别能通过系数恒等式验证的整系数组合包含。例如 `x²` 与 `y⁴` 的交集保留为四次方像集。
- `explicit` 的 `values` 是排序去重的完整结果。Pell 路径会遍历所有适用轨道，其他难例使用有界筛选。
- `unknown` 表示没有得到完整解集。达到搜索或显式输出预算时，库不会返回截断结果。此时仍可用原问题的 `contains(v)` 做精确单点判定。

有限值域内的公共值总是有限，但非线性并不保证公共值少。预算适用于显式枚举，紧凑表示不消耗 `max_values`。

一次式走同余和区间裁剪。二次式单点归属用精确整数判别式，二次求交有乘积分解、有限椭圆搜索和有界 Pell 轨道。其余情况先比较值域、做固定因子和模筛，再精确扫描候选较少的一侧。可选 `backend="sympy"` 会在合适的难题上调用 SymPy 的广义 Pell 代表元或因数分解，仍逐个核对原多项式和值域。指定 `timeout` 时只用内置路径，避免不可中断的符号求解。

Python 3.10+。在 Windows 用 `workers>1` 时，请在脚本的 `if __name__ == "__main__":` 内启动批处理。

```powershell
python -B -m unittest -v test_poly_intersection
python -B benchmark_poly_intersection.py --count 20000
python -B benchmark_poly_intersection.py --stress --count 1000 --backend sympy
```

有限值域保证无预算搜索有穷尽路径，实际耗时仍取决于系数、次数和窗口大小。即便是三次式，平移也能把有效自变量推到 64 位之外，库会从实际系数计算窗口。混合基准和高下界随机二次压力基准分别统计，吞吐量不能外推到任意输入。

本机 Windows 单进程测试中，20,000 道混合题用时约 4.1 秒，全部判定。1,000 道高下界随机二次题启用 SymPy 后用时约 37.4 秒，225 道非空，775 道空。相同的 2,000 步预算下，纯内置版本用时约 0.63 秒，留下 453 道未决题。
