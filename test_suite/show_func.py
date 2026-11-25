"""从某个用例的 .s 里单独抽出一个函数的汇编。

用法：
    python test_suite/show_func.py 31_matrix --list
    python test_suite/show_func.py 31_matrix mat_mul
    python test_suite/show_func.py 31_matrix mat_mul > mul.s

函数名可以写 Lan 里的名字（`mat_mul`），也可以写汇编里的符号（`lan_fn_mat_mul`）。

函数列表取自 COFF 的符号元数据（`.def <name>` 后跟 `.type 32`），
这样 `main` 里被提升到 `.rdata` 的局部常量数组（`lan_fn_main.lv_87_prog` 之类）
不会被误当成函数。
"""
import re
import sys
from pathlib import Path

SUITE = Path(__file__).resolve().parent

LABEL = re.compile(r"^([A-Za-z_$.][\w$.]*):")
DEF = re.compile(r"^\s*\.def\s+([^;]+);")
TYPE = re.compile(r"^\s*\.type\s+(\d+);")
# 函数体结束时紧跟的元数据 / 分节指令，抽单个函数时切掉
CUT = re.compile(r"^\s*(#\s*--\s*End function|\.def\b|\.endef\b|\.section\b|\.pdata\b|\.rdata\b)")

FUNC_TYPE = 32


def function_names(lines):
    """从 COFF 元数据里挑出真正的函数符号（.def X; .scl ..; .type 32; .endef）。"""
    names, pending = [], None
    for ln in lines:
        m = DEF.match(ln)
        if m:
            pending = m.group(1).strip()
            continue
        m = TYPE.match(ln)
        if m and pending:
            if int(m.group(1)) == FUNC_TYPE:
                names.append(pending)
            pending = None
    return names


def functions(lines):
    """返回 [(符号名, 起始行, 结束行)]，1-based 闭区间。"""
    order = function_names(lines)
    starts = {}
    for i, ln in enumerate(lines):
        m = LABEL.match(ln)
        if m and m.group(1) in order and m.group(1) not in starts:
            starts[m.group(1)] = i
    marks = sorted((starts[n], n) for n in order if n in starts)
    out = []
    for k, (i, name) in enumerate(marks):
        end = (marks[k + 1][0] - 1) if k + 1 < len(marks) else len(lines) - 1
        out.append((name, i, end))
    return out


def trim(lines, start, end):
    """去掉尾部紧跟的符号元数据块和分节指令。"""
    body = lines[start:end + 1]
    for i, ln in enumerate(body):
        if i and CUT.match(ln):
            return body[:i]
    return body


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    case = sys.argv[1]
    asm = SUITE / case / f"{case}.s"
    if not asm.exists():
        print(f"找不到 {asm}，先跑 python test_suite/verify.py {case}")
        return 1
    lines = asm.read_text(encoding="utf-8", errors="replace").split("\n")
    funcs = functions(lines)

    if len(sys.argv) < 3 or sys.argv[2] in ("--list", "-l"):
        print(f"{case}: {len(funcs)} 个函数（{asm.name} 共 {len(lines)} 行）")
        for name, a, b in funcs:
            print(f"  {name:32s} L{a + 1}-{b + 1}  ({b - a + 1} 行)")
        return 0

    want = sys.argv[2]
    for name, a, b in funcs:
        if name == want or name == f"lan_fn_{want}" or name.lstrip("_") == want:
            print(f"# {case}: {name}  ({asm.name}:{a + 1}-{b + 1})")
            print("\n".join(trim(lines, a, b)))
            return 0
    print(f"没有 {want!r}；用 --list 看看有哪些")
    return 1


if __name__ == "__main__":
    sys.exit(main())
