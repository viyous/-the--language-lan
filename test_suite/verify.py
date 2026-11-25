"""测试套件校验脚本：Lan 编译 + C 编译器 -O2 + 运行，逐个用例报告。

保留的产物（供分析与优化使用）：
    <case>.c       Lan 生成的 C11
    <case>.table   跨模块接口
    <case>.theorem 已证范围定理
    <case>.proof   SSA 查询及诊断轨迹
    <case>.s       zig cc -std=gnu11 -O2 -S 生成的汇编（真正的产物）

可执行文件只是用来跑断言的，编译到临时目录，跑完就删，不落在用例目录里。

用法：
    python test_suite/verify.py                 # 全部用例
    python test_suite/verify.py 19_chains       # 单个用例
    python test_suite/verify.py 31_matrix 32_search
"""
import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

LAN = Path(__file__).resolve().parent.parent
SUITE = LAN / "test_suite"
CFLAGS = ["-std=gnu11", "-O2"]

# 汇编产物必须干净，否则全是噪声、没法看：
#   -g0                 关掉 CodeView 调试信息（.cv_func_id / .cv_loc / #DEBUG_VALUE）
#   -fno-unwind-tables  关掉 Windows x64 的 SEH 展开指令（.seh_proc / .seh_pushreg ...）
# 不加这两条时 31_matrix.s 是 13531 行，加上之后 1575 行，其中真正的指令占绝大多数。
ASMFLAGS = CFLAGS + ["-g0", "-fno-unwind-tables"]
COFF_DEBUG_SECTION = re.compile(r"^\s*\.section\s+\.debug\$\w+")
ASM_SECTION = re.compile(r"^\s*\.(?:section|text|data|bss|rdata)\b")


def strip_coff_debug_sections(path):
    """Remove CodeView metadata, including the compiler's implicit object path."""
    lines = path.read_bytes().splitlines(keepends=True)
    kept = []
    debug = False
    for line in lines:
        directive = line.decode("ascii", errors="replace")
        if ASM_SECTION.match(directive):
            debug = bool(COFF_DEBUG_SECTION.match(directive))
        if not debug:
            kept.append(line)
    if len(kept) != len(lines):
        path.write_bytes(b"".join(kept))


def clean_stale(name):
    """删掉旧版本留下的 .exe / .pdb，保留编译产物。"""
    d = SUITE / name
    for pat in ("*.exe", "*.pdb"):
        for f in d.glob(pat):
            f.unlink()


def run_case(name, build, cc):
    d = SUITE / name
    src = d / f"{name}.lang"
    if not src.exists():
        return name, "MISSING", ""
    clean_stale(name)

    out = d / f"{name}.c"
    lib = ["--library"] if name == "shared" else []
    p = subprocess.run([sys.executable, "-B", str(LAN / "lanc.py"), str(src), "-o", str(out), *lib],
                       capture_output=True, text=True, cwd=str(LAN))
    if p.returncode != 0:
        return name, "LANFAIL", (p.stdout + p.stderr).strip().splitlines()[:3]

    extra = [c for c in sorted(d.glob("*.c")) if c != out]

    # 产物：-O2 汇编。COFF 下 -S 不能一次输出多个源，逐个生成。
    for source in [out, *extra]:
        p = subprocess.run([*cc, *ASMFLAGS, f"-ffile-prefix-map={d}=.",
                            "-S", source.name, "-o", source.with_suffix(".s").name],
                           capture_output=True, text=True, cwd=d)
        if p.returncode != 0:
            return name, "ASMFAIL", (p.stdout + p.stderr).strip().splitlines()[:5]
        strip_coff_debug_sections(source.with_suffix(".s"))

    if name == "shared":
        return name, "OK", ""

    # 跑断言用的可执行文件放临时目录，不污染用例目录
    exe = build / (f"{name}.exe" if sys.platform == "win32" else name)
    p = subprocess.run([*cc, *CFLAGS, out.name, *(source.name for source in extra), "-o", str(exe)],
                       capture_output=True, text=True, cwd=d)
    if p.returncode != 0:
        return name, "CCFAIL", (p.stdout + p.stderr).strip().splitlines()[:5]

    r = subprocess.run([str(exe)], capture_output=True, text=True)
    if r.returncode != 0:
        return name, f"RUN{r.returncode}", ""
    return name, "OK", ""


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cc", help="C compiler executable; defaults to clang, gcc or zig on PATH")
    parser.add_argument("--cc-arg", action="append", default=[], help="argument before the default C flags")
    parser.add_argument("cases", nargs="*", help="case names; defaults to all cases")
    args = parser.parse_args()
    compiler = shutil.which(args.cc) if args.cc else next(
        (path for name in ("clang", "gcc", "zig") if (path := shutil.which(name))), None)
    if not compiler:
        parser.error("C compiler not found; install clang, gcc or zig, or pass --cc")
    cc = [str(Path(compiler).absolute()), *args.cc_arg]
    if not args.cc_arg and Path(compiler).stem.lower() == "zig":
        cc.append("cc")
    available = {p.name for p in SUITE.iterdir() if p.is_dir() and (p / f"{p.name}.lang").is_file()}
    if unknown := set(args.cases) - available:
        parser.error("unknown test case: " + ", ".join(sorted(unknown)))
    names = args.cases or sorted(available)
    bad = 0
    build = Path(tempfile.mkdtemp(prefix="lan-build-"))
    try:
        for n in names:
            name, status, detail = run_case(n, build, cc)
            if status != "OK":
                bad += 1
                print(f"[{status}] {name}")
                if detail:
                    for line in (detail if isinstance(detail, list) else [detail]):
                        print(f"        {line}")
            else:
                print(f"[OK] {name}")
    finally:
        shutil.rmtree(build, ignore_errors=True)
    print(f"\n{len(names) - bad}/{len(names)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
