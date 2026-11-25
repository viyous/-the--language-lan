# 尺寸变化终止性检查

`size_change.py` 独立于编译器。把循环回边和递归调用表示成控制位置之间的转换。每条转换的边表示旧度量到新度量的关系，`STRICT` 意为新值严格小于旧值，`WEAK` 意为新值不大于旧值。度量 ID 只标识位置，所有可串接的度量必须在同一个良基序上比较，例如非负整数。

```python
from size_change import Change, Transition, check_termination

result = check_termination(
    {"loop": ("remaining",)},
    [Transition("loop", "loop", (("remaining", "remaining", Change.STRICT),))],
)
print(result.status)  # proved
```

调用方必须提供所有可能的转换。即使推不出一条大小关系，也要保留空边转换。不同调用点要分别给出，不能把它们的保证合并成一张图。检查器对递归强连通分量求复合闭包，要求每张幂等自环图都含有严格下降的自边。`not_proved` 附带一段输入转换的索引和复合图，表示抽象信息不足以证明终止。`unknown` 表示闭包超出图数或复合次数预算。

`infer_integer_transition` 可选接入 `ssa_verify.Program`。调用方传入转换守卫和更新前后的表达式。适配器先证明度量非负，再证明下降或保持。无界整数 `x-1` 单独不够，带有 `x>0` 的守卫就可以证明相应边。调用者仍负责保证表达式和守卫忠实表示实际转移。

```powershell
python -B -m unittest -v test_size_change
```
