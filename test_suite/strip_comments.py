"""去掉 test_suite 下所有 .lang 文件里的注释。

保留 `/** @oracle ... */` 这类**循环契约块** —— 它们是语法的一部分，不是注释。
"""
import sys
from pathlib import Path

SUITE = Path(__file__).resolve().parent


def strip(src: str) -> str:
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        # 字符串字面量
        if c == '"':
            j = i + 1
            while j < n:
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == '"':
                    j += 1
                    break
                j += 1
            out.append(src[i:j])
            i = j
            continue
        # 字符字面量
        if c == "'":
            j = i + 1
            while j < n:
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == "'":
                    j += 1
                    break
                j += 1
            out.append(src[i:j])
            i = j
            continue
        # 行注释
        if src.startswith("//", i):
            j = src.find("\n", i)
            if j < 0:
                break
            i = j          # 保留换行
            continue
        # 块注释
        if src.startswith("/*", i):
            j = src.find("*/", i + 2)
            body = src[i:j + 2] if j >= 0 else src[i:]
            # `/** @oracle ... */` 是契约指令，保留
            if body.startswith("/**") and "@" in body:
                out.append(body)
            i = (j + 2) if j >= 0 else n
            continue
        out.append(c)
        i += 1
    return "".join(out)


def tidy(text: str) -> str:
    lines = [ln.rstrip() for ln in text.split("\n")]
    res = []
    for ln in lines:
        if ln == "" and res and res[-1] == "":
            continue
        res.append(ln)
    while res and res[0] == "":
        res.pop(0)
    while res and res[-1] == "":
        res.pop()
    return "\n".join(res) + "\n"


def main():
    total = 0
    for f in sorted(SUITE.rglob("*.lang")):
        src = f.read_text(encoding="utf-8")
        new = tidy(strip(src))
        if new != src:
            f.write_text(new, encoding="utf-8", newline="\n")
            total += 1
            print(f"{f.parent.name:20s} {len(src):6d} -> {len(new):6d} bytes")
    print(f"\n{total} 个文件已处理")


if __name__ == "__main__":
    sys.exit(main())
