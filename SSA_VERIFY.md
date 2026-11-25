# SSA 分支合流验证原型

`ssa_verify.py` 是独立的验证原型，不调用或改动 Lan 编译器。输入是结构化的无循环整数与布尔程序。整数按数学整数计算，乘法仅支持常量因子。若要检查定宽溢出、数组下标或转换安全，需要在对应位置显式提交证明目标。

```python
from ssa_verify import Program

p = Program()
b = p.boolean("b")
flow = p.flow
flow.declare("x", 0)
flow.declare("y", 0)
flow.branch(b, lambda s: s.assign("y", 1), lambda s: s.assign("x", 1))
goal = flow.check((flow.get("x") + flow.get("y")).eq(1), "sum")
print(p.verify(goal))
```

分支环境只记录本地写入。合流时只处理被写入的外层变量，生成共享条件的 `ite`，相同定义会共享同一 DAG 节点。证明查询只遍历目标和该处可达条件所引用的节点。未用到的输入界限不会进入查询，因为每个输入的界限独立且非空。当前保留完整的可达条件，没有进一步删除其中的合取项。

区间快路径先尝试证明。需要求解时，Z3 检查当前可达条件与目标的否定是否可同时成立。`proved` 表示已经证明，`refuted` 带有该查询所用输入的反例，`unknown` 表示求解器缺失、超限或未给结论。`assume` 会限制此后可达的输入，调用者负责保证这些假设来自契约或已知事实。已有证明目标不会受后续假设影响。

```powershell
python -m pip install z3-solver==4.15.4.0
python -B -m unittest -v test_ssa_verify
python -B benchmark_ssa_verify.py --counts 1000 10000
```

本机 Windows 上，1,000 个独立分支构建 4,005 个 DAG 节点，约 0.08 秒，Python 构建阶段峰值约 1.8 MiB。10,000 个分支构建 40,005 个节点，约 0.86 秒，峰值约 16.1 MiB。两次最终证明都只用到 8 个节点。这是独立分支的基准，内存数据只覆盖 Python 构建，不含 Z3。

目前没有接入 Lan 的 AST、差分约束、循环、调用或内存模型。每次 SMT 验证使用独立求解器实例。
