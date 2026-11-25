# Lan 语言使用手册与完整参考（面向开发者 & AI Agent）

**Lan** 是一门面向系统编程、嵌入式（含 32 位 / ARM7TDMI / 裸机 freestanding）与高可靠内核的静态验证编译型语言，后端输出**标准、可移植、零运行时依赖的 C 代码**（无 `__int128`、无 GNU 语句表达式 `({ ... })`、无 `__label__` / `__auto_type` / `__typeof__`，裸机模式可零宿主头文件编译）。

Lan 的核心思想只有一句话：**类型就是值的集合**。所有数组越界、除零、整数溢出、空指针解引用、悬垂借用、堆内存泄漏，都在编译期按路径与值域静态证明；证明通过后，后端把边界知识变成优化假设（或 32 位窄化算术），生成与手写 C 几乎同速的机器码。

---

## 目录

- [0. 速查卡片（Agent / 老手必读避坑清单）](#0-速查卡片agent--老手必读避坑清单)
- [1. 快速上手与 CLI 命令](#1-快速上手与-cli-命令)
- [2. 词法规则、字面量、注释与保留字](#2-词法规则字面量注释与保留字)
- [3. 类型系统：标量、区间、集合与多项式](#3-类型系统标量区间集合与多项式)
- [4. 类型转换：`as` 与全部转换策略](#4-类型转换as-与全部转换策略)
- [5. 模块级常量与局部变量：`define`](#5-模块级常量与局部变量define)
- [6. 表达式、运算符优先级与 32 位窄化规则](#6-表达式运算符优先级与-32-位窄化规则)
- [7. 控制流：`if` / `while` / `interation` / `continue` / `return` / `assert`](#7-控制流if--while--interation--continue--return--assert)
- [8. 函数、依赖契约、代数属性与递归](#8-函数依赖契约代数属性与递归)
- [9. 定长数组、切片、字符串与 `.count(...)`](#9-定长数组切片字符串与-count)
- [10. 记录、键值记录（`type enum`）、枚举与位域](#10-记录键值记录type-enum枚举与位域)
- [11. 借用、别名、可空指针、堆所有权与链表/图](#11-借用别名可空指针堆所有权与链表图)
- [12. 错误联合与 `?` 早退传播](#12-错误联合与--早退传播)
- [13. 接口与回调：`interface`、`callable`、`extern callable`](#13-接口与回调interfacecallableextern-callable)
- [14. 类型对象合并（`·`）、限定符与编译期类型函数](#14-类型对象合并限定符与编译期类型函数)
- [15. 定理、公理与 Oracle 注释](#15-定理公理与-oracle-注释)
- [16. 模块系统、分模块编译与 C / 裸机互操作](#16-模块系统分模块编译与-c--裸机互操作)
- [17. 完整语法速览（EBNF）与常见报错速修表](#17-完整语法速览ebnf与常见报错速修表)

---

## 0. 速查卡片（Agent / 老手必读避坑清单）

在开始写 Lan 代码前，请务必牢记下面 16 条与 C / Rust 不同的硬规则（**90% 的编译报错都来自这里**）：

1. **没有 `break`、没有 `for`、没有 `switch`、没有三元运算符 `?:`**。
   - 提前退出循环请把条件写进 `while (cond)`，或者在函数内用 `return`。
   - 固定次数循环用 `interation(i; 0; uint8[0, 10]; 10; 1) { ... }`（注意拼写就是 **`interation`**，且类型上界必须包含退出步的值 `10`，循环体内自动收窄到 `[0, 9]`）。
2. **`if` / `while` 的条件必须有小括号，分支体必须有大括号**：必须写 `if (x > 0) { ... }`，不允许省略 `()` 或 `{}`。
3. **没有隐式真值转换（标量整数不能直接放进 `if (n)`）**：必须写 `if (n != 0) { ... }`。仅 `boolean`、可空别名/指针/函数指针（如 `if (p != null)` 或 `if (p)`）可直接作条件。
4. **声明一律用 `define`，不用 `let` / `const` / `var` / `fn`**：
   - 模块顶层常量：`define MAX_HP: uint16 = 999;` 或 `define N = 16;`（自动为只读 `static const`）。
   - 局部变量：`define x: uint8 = 0;` 或 `define x = 0;`。
   - 函数：`define add(a: uint8, b: uint8) -> uint16 { return a + b; }`。
5. **类型合并符是中点 `·`（Unicode `U+00B7`，不是 `.`）**：
   - 正确：`uint8[..] · {immutable: true, len: [1, 16]}`
   - 常用属性键名只有 9 个：`len`、`elem`、`immutable`、`pass`、`storage`、`from`、`offset`、`field`、`repr`（以及构建纯类型对象用的 `kind`、`bits`）。不要编造 `mut`、`readonly`、`access`、`const` 等不存在的键。
6. **数组/记录按参数传递时，默认就是按借用（指针）传递**，不要在类型前写 `&` 或 `*`：
   - 记录参数 `p: Pair` 在 C 里就是 `Pair *`。
   - 标量参数默认按值传递；需要输出参数（C 的 `uint32_t *`）时写 `x: uint32 · {pass: alias}`。
   - 需要按值拷贝传递记录或定长数组时写 `p: Pair · {pass: copy}`。
7. **类型转换：`as`（或 `as: T`）只允许静态证明不越界的安全转换**；允许回绕/饱和/截断/报错必须显式写策略：
   - 安全转换：`x as uint32` 或 `x as: uint32` 或 `x as: safe uint32`
   - 模运算回绕：`expr as: wrap uint32`（GBA/32 位 LCG 如 `(seed * 1103515245 + 12345) as: wrap uint32` 自动走纯 32 位模运算）
   - 饱和夹紧：`expr as: sat uint8[0, 100]`
   - 运行时断言陷入：`expr as: trap uint8[0, 100]`
   - 浮点转整数：`f as: trunc int32`、`f as: trunc_sat int32`、`f as: floor_trap int32` 等。
8. **切片语法用逗号 `,`，不是冒号 `:` 或 `..`**：
   - 切片类型：`uint8[..]`
   - 切片表达式：`a[l, r)`（半开）、`a[l, r]`（闭区间）、`a[, r)`、`a[l, )`、`a[, ]`（全范围）。
9. **多行数组字面量每行元素之间必须有逗号**（允许最后一个元素后带尾逗号）：
   ```lan
   define LUT: uint8[4] = [
       10, 20,
       30, 40,
   ];
   ```
10. **整数运算必须能静态证明不越界/不除零**：
    - `a / b` 或 `a % b` 前，必须保证 `b` 的值域不含 `0`（例如类型写成 `uint32[1, 1000]`，或先 `if (b == 0) { return 0; }`）。
    - 数组索引 `a[i]` 前，必须保证 `i` 落在 `[0, a.len)` 内。
11. **数组元素批量拷贝必须写 `dst[l, r) = src[a, b) as: copy;`**；普通赋值 `s = a[l, r);` 只是重绑切片描述符。
12. **堆对象（`storage: heap`）必须在每条退出路径上显式 `free(x);` 或作为拥有权返回值移交出去**，否则报 `heap leak`。
13. **可空借用与可空 C 回调使用前必须先判空**：
    - 可空记录/标量借用必须带 `pass: alias`，写 `Box · {pass: alias} | null`（不能写裸的 `Box | null`）；
    - 可空 C 回调写 `RejectFn | null`；`if (p != null) { ... }` 分支内自动收窄为非空。
14. **函数指针分两种**：
    - Lan 内部封闭回调集：`callable(f1, f2)`，值用 `callback(f1)` 构造（不能跨模块导出，不能接 C 函数指针）。
    - **C 函数指针回调**：`extern callable(x: uint32) -> boolean writes {}`（支持 `| null`，可导出，可直接接 C 传来的函数指针或 `NULL`，也可传 `callback(lan_fn)`）。
15. **C 互操作双方向**：
    - Lan 调 C：写 `extern contract foo(x: uint32) -> uint32 writes { external };`
    - C 调 Lan：编译时加 `--library --export-prefix <前缀>`，导出函数同时生成无指纹稳定 C 符号 `<前缀><导出名>`。
16. **独立可执行程序入口**：`define main() -> int32 { return 0; }`（无参数，返回可装入 `int32` 的整数，不能是 `boolean`）。

---

## 1. 快速上手与 CLI 命令

### 1.1 第一个程序

保存为 `hello.lang`：

```lan
define MAX_HP: uint16 = 999;

define clamp_hp(hp: uint16) -> uint16[0, MAX_HP] {
    if (hp > MAX_HP) {
        return MAX_HP;
    }
    return hp;
}

define main() -> int32 {
    define v: uint16[0, MAX_HP] = clamp_hp(1200);
    if (v != 999) {
        return 1;
    }
    return 0;
}
```

编译并运行：

```bash
python lanc.py hello.lang -o hello.c
cc -O2 hello.c -o hello
./hello
```

### 1.2 编译器 CLI 完整参数（`lanc.py`）

```bash
python lanc.py <source.lang> [options]
```

| 参数 | 说明 |
| --- | --- |
| `<source.lang>` | 主入口源码文件路径 |
| `--check` | 仅做静态类型/值域/所有权验证，不写任何输出文件（与 `-o` 互斥） |
| `-o <out.c>` | 输出 `<out>.c`，并在同目录生成 `<out>.table`（模块接口）、`<out>.theorem`（已证定理 JSONL）、`<out>.proof`（验证轨迹 JSONL）；省略 `-o` 时默认输出 `<source>.c` |
| `--library` | 库模式编译：隐藏内部实现函数为 `static`，不生成 C 的 `int main(void)`，只对外暴露 `export` 符号 |
| `--export-prefix <PREFIX>` | 为每个 `export` 函数额外生成**不带指纹**的稳定 C 符号 `<PREFIX><导出名>`，供手写 C 直接调用（带契约指纹的 `lan_api_*` 符号仍保留）。`<PREFIX>` 可为空串 `""`；不得与 `main`、`lan_fn_*`、`lan_api_*`、`lh_*` 等冲突 |
| `--module-root <dir>` | 指定多目录分模块项目时的统一基准根目录，使 `.table` 中的模块路径可移植 |
| `--debug-bounds` | 对部分未能静态证明的数组/切片边界退化为运行时断言检查（仅用于排错对比，不能替代算术溢出/所有权证明） |
| `--max-steps <N>` / `--solver-steps <N>` | 指定多项式方程求交与 SSA/SMT 推理的最大步数预算（默认 `2048`；遇到可解的高次/丢番图方程因默认步数不足未收敛时可调大，如 `--max-steps 100000`，多模块项目会自动传递到所有导入模块） |
| `--insert-oracle` | 当遇到需要循环不变量或释放提示的验证缺口时，自动把建议的 `@oracle` 注释插回源码 |

### 1.3 32 位与裸机（Freestanding / GBA）编译

`lanc.py` 生成的 C 代码天然支持 64 位、32 位与裸机嵌入式：

```bash
# 生成库代码，导出不带指纹的 gba_ 前缀符号
python lanc.py battle_rng.lang -o battle_rng.c --library --export-prefix gba_

# GBA ARM7TDMI 裸机编译（无需标准库头文件）
arm-none-eabi-gcc -mcpu=arm7tdmi -ffreestanding -O2 -c battle_rng.c -o battle_rng.o
# 或使用 zig cc：
zig cc -target arm-freestanding-eabi -mcpu=arm7tdmi -ffreestanding -O2 -c battle_rng.c -o battle_rng.o
```

生成的 C 文件头（`C_HEADER`）自动处理：
- 在 `#if !defined(__STDC_HOSTED__) || __STDC_HOSTED__` 下包含 `<string.h>`、`<stdlib.h>`、`<math.h>`、`<float.h>`；裸机模式仅依赖 `<stdint.h>` 和 `<stddef.h>`，并内置 `memcpy`/`memmove`/`memset`/`memcmp` 原型。
- 为 C11 之前的编译器提供 `_Static_assert` 降级宏；为 GCC < 4（如 `agbcc`）提供 `__builtin_unreachable()`、`__builtin_trap()` 和 `restrict` 降级宏。

---

## 2. 词法规则、字面量、注释与保留字

### 2.1 注释

- 单行注释：`// ...`
- 块注释：`/* ... */`（不可嵌套）
- **Oracle 注释**：以 `/* @oracle ... */` 或 `/** @oracle ... */` 开头的注释**会被词法器保留并参与编译期验证**（见第 15 节）。普通文档注释不要以 `@oracle` 开头。

### 2.2 标识符与保留字

- **标识符支持完整 Unicode（含中文）**：变量名、函数名、参数名、类型名、记录字段名、枚举名与成员名、模块级常量名、`export`/`import` 导出导入名均支持中文与中英混排（如 `define 最大血量: uint16 = 999;`、`type 角色 = { 血量: uint16 };`、`define 计算伤害(目标: 角色) -> uint16 { ... }`）。
- **C 后端自动转义为纯 ASCII 标识符**：非 ASCII 标识符在生成的 C 代码中会通过无碰撞单射编码 `lanu_<utf8十六进制>` 自动转为纯 ASCII C89 标识符，确保在 `agbcc`、ARM 裸机、GCC、Clang、MSVC 下零编码问题；纯 ASCII 标识符原样保留。
- **特殊说明**：
  - `·`（U+00B7）与中文标点（`，。；：` 等）不属于标识符字符，因此 `角色·{immutable: true}` 即使不加空格也能正确切分。
  - `extern contract` 声明的外部 C 函数名仍须是合法 ASCII C 标识符（对应实际链接的 C 符号）。

Lan 关键字（不可用作标识符）：

```text
type      define    contract  if        else      while     interation
break     continue  return    true      false     void      boolean
in        as        not       and       or        xor       assert
enum      extern    import    export    null      interface poly
bit       callable  callback
```

> 注意：`break` 仍是保留字，但使用时会直接报错 `"break was removed; express the exit in the loop condition"`。此外，导出给 C 的 `extern contract` 符号名不能与 C 关键字冲突（如 `int`、`register`、`default` 等）。

### 2.3 字面量

| 类别 | 写法示例 | 说明 |
| --- | --- | --- |
| 十进制整数 | `0`、`42`、`1103515245` | 精确任意精度编译期整数 |
| 十六进制整数 | `0xFF`、`0x1A2B` | 区分大小写前缀 `0x`/`0X` |
| 位字面量 / 二进制 | `0b1010`、`0b00001111` | **带精确位宽**（`0b1010` 宽 4 位，`0b00001111` 宽 8 位），可用于 `bit[N]` 初始化与移位，也可直接当普通整数使用（不超过 64 位） |
| 浮点数 | `0.5`、`.25`、`1e6`、`3.14e-2` | 默认按精确有理数/浮点单点追踪 |
| 布尔值 | `true`、`false` | 值域 `{1}` / `{0}`，底层为 `uint8_t` |
| 字符字面量 | `'A'`、`'\n'`、`'\x7F'`、`'\u0041'`、`'\U0001F600'` | 单一 Unicode 标量值 |
| 字符串字面量 | `"hello\n"` | UTF-8 字节串，支持 `\\` `\"` `\'` `\/` `\n` `\r` `\t` `\b` `\f` `\0` `\xHH` `\uHHHH` `\UHHHHHHHH` |
| 空值 | `null` | 用于可空借用、链表链接、`extern callable \| null` |

---

## 3. 类型系统：标量、区间、集合与多项式

### 3.1 内置基础类型

| 分类 | 类型名 | 值域 / 底层 C 类型 |
| --- | --- | --- |
| 无符号整数 | `uint8` / `uint16` / `uint32` / `uint64` | `[0, 2^N - 1]` → `uint8_t` .. `uint64_t` |
| 有符号整数 | `int8` / `int16` / `int32` / `int64` | `[-2^(N-1), 2^(N-1) - 1]` → `int8_t` .. `int64_t` |
| 布尔 | `boolean` | `{0, 1}` → `uint8_t` |
| 浮点 | `float32` / `float64` | IEEE-754 单/双精度 → `float` / `double` |
| 字符编码 | `ascii` / `utf8` / `utf16` / `unicode` | `uint8_t`（`[0,127]` / `[0,255]`）、`uint16_t`、`uint32_t`（排除代理项） |
| 空类型 | `void` / `null` | `void` / 空指针字面类型 |

### 3.2 区间、枚举集合与集合运算

在 Lan 中，任何整数或浮点类型都可以直接精化为一个**值域集合**：

```lan
type Percent   = uint8[0, 100];           // 闭区间 [0, 100]
type Index10   = uint8[0, 10);            // 半开区间 [0, 9]
type NonZero   = uint32[1, 4294967295];   // 非零 uint32
type SmallOdd  = uint8 & {1, 3, 5, 7, 9}; // 离散值集合
type NonZero8  = uint8 \ {0};             // 集合差：uint8 去掉 0
type ValidCode = uint8[1, 9] | uint8[20, 29]; // 集合并
type Positive  = int32 & > 0;             // 单边不等式界：< <= > >=
type UnitFloat = float32[0.0, 1.0];       // 浮点区间（自动排除 NaN 与 ±inf）
```

集合运算符优先级（从高到低）：
1. `·`（类型对象合并）
2. `&`（交集）、`\`（差集）
3. `|`（并集 / 联合）

### 3.3 多项式集合 `poly(...)`

用整数多项式定义非连续值域（如平方数、等差数列、偶数），支持静态证明与运行时 `in` 成员测试：

```lan
type Even8  = uint8 & poly(2 * n);
type Square = uint32 & poly(n * n);
type Mod3   = uint16 & poly(3 * n + 1);

define is_square(x: uint32) -> boolean {
    return x in Square;
}
```

### 3.4 编译期单点类型与物理类型封装

- 仅含单个值的类型别名（如 `type N = {16};` 或 `type N = uint8[16, 16];`）可作为编译期常量使用（更推荐直接用模块级 `define N = 16;`）。
- 纯数学集合（没有指定物理位宽，如 `type Small = [0, 100];`）在落盘到变量或参数时会自动选择最小可容纳的物理整数类型（如 `uint8`），也可显式写 `uint16 & Small` 指定位宽。

---

## 4. 类型转换：`as` 与全部转换策略

语法：
- 安全转换：`expr as Type` 或 `expr as: Type` 或 `expr as: safe Type`
- 策略转换：`expr as: <strategy> Type`
- 拷贝/C 字符串特化：`expr as: copy`、`expr as: c_string`

### 4.1 全部转换策略总表

| 写法 | 适用类型 | 语义与编译期检查 |
| --- | --- | --- |
| `x as T` / `x as: T` / `x as: safe T` | 整数、浮点、布尔、枚举、字符 | **必须静态证明** `x` 的所有可能值都在 `T` 的值域内；零运行时开销 |
| `x as: wrap T` | 整数 → 整数 | 按目标物理位宽模 $2^W$ 回绕（如 `uint32` → `uint16` 取低 16 位）；结果必须落入 `T` 的精化值域内 |
| `x as: sat T` | 整数 → 整数 | 饱和夹紧到 `T` 的连续区间 `[lo, hi]` |
| `x as: trap T` | 整数 → 整数 / 浮点 → 浮点 | 运行时检查是否在 `T` 内，越界触发 `__builtin_trap()`；通过后在当前路径上收窄 `x` 的值域 |
| `x as: view T` | 同宽整数/浮点/打包位数组/定长数组/记录 | **按位重解释（bitcast / `memcpy`）**，不改变底层比特（如 `float32` ↔ `uint32`、`uint8[4]` ↔ `uint32`） |
| `x as: trunc T` | 浮点 → 整数 | 向零取整；需静态证明取整后不越界且非 NaN |
| `x as: floor T` / `ceil T` / `round T` | 浮点 → 整数 | 向下 / 向上 / 四舍五入取整；需静态证明在目标值域内 |
| `x as: trunc_sat T` / `sat trunc T` | 浮点 → 整数 | 取整后饱和夹紧到 `T`（`floor`/`ceil`/`round` 均支持与 `sat`/`trap` 任意顺序组合，如 `floor_trap`、`trap ceil`） |
| `s as: copy` | 切片 / 数组 / 记录赋值右值 | 显式逐元素内存拷贝（见第 9.3 节） |
| `s as: c_string` | 以 `\0` 结尾的字节切片 | 零拷贝视图转为 C 字符串（`const char *`） |

### 4.2 关键转换示例

```lan
// 1. 拓宽与安全窄化（两者皆可写 as 或 as:）
define widen(a: uint16) -> uint32 {
    return a as uint32;          // 或 a as: uint32，甚至直接 return a;
}

// 2. 模 2^32 回绕（LCG 随机数、哈希，在 32 位 ARM 上走纯 32 位 mul/add）
define lcg_next(seed: uint32) -> uint32 {
    return (seed * 1103515245 + 12345) as: wrap uint32;
}

// 3. 饱和运算（血量扣减不低于 0）
define sub_hp(hp: uint16, dmg: uint16) -> uint16 {
    return (hp - dmg) as: sat uint16;
}

// 4. 布尔与整数互转
define bool_to_u8(flag: boolean) -> uint8[0, 1] {
    return flag as uint8;
}
define u8_to_bool(v: uint8) -> boolean {
    return v as boolean;         // 非零为 true，0 为 false
}

// 5. 枚举与底层整数互转（枚举不能直接做算术，需显式 as）
enum Dir: uint8 { Up = 0, Down = 1, Left = 2, Right = 3 }
define dir_code(d: Dir) -> uint8[0, 3] {
    return d as uint8;
}
```

---

## 5. 模块级常量与局部变量：`define`

### 5.1 模块级常量（顶层 `define`）

模块顶层的 `define` 用于定义**全局只读常量**（编译到 C 的 `static const`，在嵌入式/GBA 中直接落 ROM 只读段，不占 RAM）：

```lan
// 标量常量（可显式指定类型，也可省略类型自动推断）
define MAX_HP: uint16 = 999;
define LCG_MULT: uint32 = 1103515245;
define LCG_INC = 12345;
define DOUBLE_HP: uint16 = MAX_HP * 2;
define DEBUG_MODE = false;
define GRAVITY: float32 = 9.8;

// 数组、重复初始化、结构体、查找表、字符串常量
define N = 4;
type Move = {power: uint8[0, 200], acc: uint8[0, 100]};

define LUT: uint8[N] = [10, 20, 30, 40];
define ZEROS: uint8[16] = [0; 16];
define ZERO_DEFAULT: uint8[8];   // 省略初始值时全零初始化
define ORIGIN: Move = {power: 40, acc: 100};
define MOVES: Move[2] = [
    {power: 40, acc: 100},
    {power: 90, acc: 85},
];
define TITLE = "GBA";
```

模块级常量的特性：
- **编译期与运行期双可用**：标量常量、`LUT.len`、`MOVES[0].power`、`ORIGIN.acc` 可直接用在类型精化区间（`uint16[0, MAX_HP]`）、数组长度（`uint8[N]`）、重复初始化计数（`[0; N]`）以及另一个模块常量的初始化表达式中。
- **精确内容追踪**：函数里读取 `LUT[i]` 时，编译器静态知道结果落在 `[10, 40]`；读取 `MOVES[i].power` 知道结果落在 `[40, 90]`。
- **强制只读与静态生命周期**：不可被赋值修改；可直接返回其只读切片 `return LUT[, ];`。
- **支持 `export` / `import`**（见第 16 节）。

### 5.2 局部变量（块内 `define`）

```lan
define example(n: uint8[0, 10]) -> uint16 {
    define acc: uint16 = 0;                      // 显式类型，可变
    define twice = n + n;                        // 自动推断类型（此处为 uint8[0, 20]）
    define frozen: uint8 · {immutable: true} = n; // 不可变局部变量
    define buf: uint8[4] = [1, 2, 3, 4];         // 栈上定长数组
    define zero_buf: uint8[4];                   // 栈上全零定长数组
    define rom_lut: uint8[4] · {storage: rodata} = [1, 2, 3, 4]; // 函数内只读静态表
    acc += twice;
    return acc + frozen + buf[0] + rom_lut[0];
}
```

> **作用域规则**：函数参数与函数顶层局部变量共享同一作用域（局部变量不能与参数同名）；内层块 `{ ... }` 可嵌套声明局部变量。

---

## 6. 表达式、运算符优先级与 32 位窄化规则

### 6.1 完整运算符优先级（1 最低 → 14 最高）

| 优先级 | 运算符 | 说明 |
| --- | --- | --- |
| 1 | `\|\|` / `or` | 短路逻辑或（返回 `boolean`） |
| 2 | `^^` / `xor` | 逻辑异或（返回 `boolean`） |
| 3 | `&&` / `and` | 短路逻辑与（返回 `boolean`） |
| 4 | `\|` | 按位或 |
| 5 | `^` | 按位异或 |
| 6 | `&` | 按位与 |
| 7 | `==` `!=` `===` `!==` `===@k` `!==@k` `in` | 值相等、结构相等、k 阶结构相等、集合成员测试 |
| 8 | `<` `>` `<=` `>=` | 大小比较 |
| 9 | `<<` `>>` | 移位（移位量必须静态证明在 `[0, 128)`） |
| 10 | `+` `-` | 加、减 |
| 11 | `*` `/` `%` | 乘、向零截断除、取余（除数必须静态证明不含 `0`） |
| 12 | `as` / `as: 策略` | 类型转换 |
| 13 | `-` `+` `!` `not` `~` | 一元前缀运算符 |
| 14 | `f(...)`、`a[i]`、`a[l, r)`、`b[f: ...]`、`.field`、`.len`、`.count(...)`、`?` | 后缀调用、索引、切片、位域、成员、长度、计数、错误传播 |

复合赋值语句支持：`=` `+=` `-=` `*=` `/=` `%=` `&=` `|=` `^=` `<<=` `>>=`。

### 6.2 `in` 成员判断与 `===` 结构比较

- **`x in TypeSet`**：返回 `boolean`，并在 `if (x in T)` 分支内自动把 `x` 的静态值域收窄为 `x & T`（`else` 分支收窄为 `x \ T`）。
- **`a === b` / `a !== b`**：对记录、数组或切片做**逐元素结构内容比较**；`===@k` 指定嵌套比较深度阶数 `k`。

### 6.3 32 位算术自动窄化（无 64 位软浮点/软除法开销）

虽然 Lan 在编译期按精确大整数推理值域，但生成 C 代码时会自动窄化：
- 只要 `+` `-` `*` `/` `%` `&` `|` `^` `<<` `>>` 及比较运算的操作数与**精确数学结果**均静态证明落在 `uint32_t`（`[0, 2^32 - 1]`）或 `int32_t`（`[-2^31, 2^31 - 1]`）内，后端直接生成 32 位 `uint32_t` / `int32_t` 运算。
- 放在 `as: wrap uint32`（或 `uint16`/`uint8`/`int32` 等 $\le 32$ 位回绕转换）内的 `+`/`-`/`*`/`&`/`|`/`^`/`<<` 表达式树，后端自动走纯 `uint32_t` 模运算。
- 因此在 GBA / ARM7TDMI 等 32 位平台，所有 32 位运算（包括 LCG 乘法与随机取模）均**不会引入 `__muldi3` / `__moddi3` / `__divdi3` / `__aeabi_ldivmod`**。只有真正超出 32 位范围的运算才会提升到 `int64_t` / `uint64_t`。

---

## 7. 控制流：`if` / `while` / `interation` / `continue` / `return` / `assert`

### 7.1 `if` / `else if` / `else`（路径敏感值域收窄）

```lan
define classify(x: uint8) -> uint8[0, 2] {
    if (x == 0) {
        return 0;               // 此处 x 已知为 {0}
    } else if (x < 100) {
        return 1;               // 此处 x 已知为 [1, 99]
    } else {
        return 2;               // 此处 x 已知为 [100, 255]
    }
}
```

如果某条分支被静态证明为**不可达**（值域交为空），编译器会自动剪枝。所有可达路径必须返回符合签名契约的值（非 `void` 函数不能遗漏 `return`）。

### 7.2 计数循环：`interation`

用于固定步长、编译期或运行期已知次数的计数循环（不需要写 `@oracle` 即可自动证明终止与边界）：

```lan
// 语法：interation(循环变量; 初值; 循环变量类型; 迭代次数; 步长) { 循环体 }
define sum_array(a: uint8[10]) -> uint16[0, 2550] {
    define total: uint16 = 0;
    interation(i; 0; uint8[0, 10]; 10; 1) {
        total += a[i];
    }
    return total;
}
```

- `i` 的声明类型必须覆盖退出循环时的最终步进值（如 10 次步长 1 需写 `uint8[0, 10]`），而在循环体内编译器自动将 `i` 精确收窄为 `[0, 9]` 且只读。
- `count` 可以是常量，也可以是变量或切片长度 `s.len`。
- 支持 `continue;` 跳过本次迭代剩余语句（不支持 `break`）。

### 7.3 条件循环：`while` 与循环不变量

普通简单 `while` 循环编译器会自动做不动点迭代与终止性检查：

```lan
define find_zero(s: uint8[..]) -> uint32 {
    define i: uint32 = 0;
    while (i < s.len) {
        if (s[i] == 0) {
            return i;
        }
        i += 1;
    }
    return s.len as uint32;
}
```

当复杂循环的收敛域或终止度量无法自动推出时，在 `while` 正上方紧贴写 `/** @oracle ... */`（详见第 15.2 节）。

### 7.4 编译期断言：`assert`

Lan 的 `assert` 是**零运行时开销的静态证明检查**（不生成运行时代码，验证失败直接编译报错）：

```lan
// 按守卫条件分情况检查静态类型/不可达性
define check_assert(x: uint8[0, 10]) -> uint8[0, 10] {
    assert x: {
        [x < 5]  => uint8[0, 4],
        [x >= 5] => uint8[5, 10],
        [_]      => unreachable,
    };
    return x;
}
```

在链表函数中还支持静态可达性断言：`assert chain(head: first -> tail: last);`。

---

## 8. 函数、依赖契约、代数属性与递归

### 8.1 函数定义与独立契约

```lan
// 常规内联签名
define clamp(x: uint16, lo: uint16, hi: uint16) -> uint16 {
    if (lo > hi) { return lo; }
    if (x < lo) { return lo; }
    if (x > hi) { return hi; }
    return x;
}

// 独立契约类型 + 绑定实现
type BinaryOp = contract(a: uint8, b: uint8) -> uint16;
define add_u8: BinaryOp (a, b) {
    return a + b;
}

// 局部辅助函数可省略返回类型（由编译器推断）
define double_it(x: uint8) {
    return x + x;
}
```

### 8.2 依赖参数边界与依赖返回边界

后出现的参数类型和返回值类型可以直接引用前面参数的值或 `.len`：

```lan
// 参数 i 必须严格小于切片 s 的长度；调用方不满足时在调用处直接编译报错
define get_at(s: uint8[..], i: uint32[0, s.len)) -> uint8 {
    return s[i];
}

// 依赖返回上界：返回值静态保证落在 [0, hi]
define clamp_hi(x: uint16, hi: uint16) -> uint16[0, hi] {
    if (x > hi) { return hi; }
    return x;
}
```

### 8.3 代数属性验证（`property`）

函数签名后可附带编译器静态证明的代数性质（支持恒等、幂等、交换律、对角律）：

```lan
define min_u8(a: uint8, b: uint8) -> uint8 property: forall(a, b) => min_u8(a, b) == min_u8(b, a) {
    if (a < b) { return a; }
    return b;
}

define clamp10(x: uint8) -> uint8[0, 10] property: forall(x) => clamp10(clamp10(x)) == clamp10(x) {
    if (x > 10) { return 10; }
    return x;
}
```

### 8.4 递归与终止性

Lan 支持直接递归与互递归，编译器使用**大小变化终止分析（Size-Change Termination）**自动验证递归必然终止；若无法证明某个参数严格递减（有下界），会报终止性错误。

---

## 9. 定长数组、切片、字符串与 `.count(...)`

### 9.1 定长数组 `T[N]`

```lan
define array_demo(i: uint8[0, 3]) -> uint8 {
    define a: uint8[4] = [10, 20, 30, 40];
    define b: uint8[4] = [7; 4];            // 重复初始化，长度为编译期常量
    define mat: uint8[4][3];                // 3 行 × 4 列（注意维度从外到内：mat[row][col]）
    mat[0][i] = a[i];
    return mat[0][i] + b[0];                // a.len 为编译期常量 4
}
```

> **多维数组维度顺序**：`uint8[4][3]` 表示“3 个 `uint8[4]`”，即第 1 维长度 3，第 2 维长度 4，索引为 `mat[0..2][0..3]`。

### 9.2 切片类型 `T[..]` 与切片窗口表达式

切片在 C 底层是 `typedef struct { const void *data; size_t len; } lan_slice;`：

```lan
// 切片类型可加长度约束、元素约束与只读约束
type NonEmptyBytes = uint8[..] · {len: [1, 256], immutable: true};

define slice_ops(buf: uint8[16]) -> uint16 {
    define full = buf[, ];        // 全部元素 [0, 16)
    define head = buf[, 4);       // 前 4 个 [0, 4)
    define tail = buf[4, );       // 后 12 个 [4, 16)
    define mid  = buf[2, 5];      // 闭区间 [2, 5]（长度 4）
    define sub  = full[1, 3);     // 对切片再切片
    return head[0] + tail[0] + mid[0] + sub[0];
}
```

### 9.3 切片内容拷贝与返回借用切片

- **内存拷贝**：`dst[0, 4) = src[0, 4) as: copy;`（要求左右长度可证明相等，且目标可写；编译器自动处理重叠/非重叠并选择 `memcpy`/`memmove`）。
- **返回借用切片**：必须用 `from` 指明借用自第几个参数（从 `0` 开始计数），并用 `len` 与 `offset` 满足 `offset + len <= 来源最大长度`：
  ```lan
  define tail_slice(
      s: uint8[..] · {len: [2, 100]}
  ) -> uint8[..] · {len: [1, 99], from: 0, offset: 1} {
      return s[1, );
  }
  ```
- **返回静态只读切片**：来自模块级常量数组或 `rodata` 的切片可直接返回 `uint8[..] · {immutable: true}`，无需 `from`。

### 9.4 字符串与 `c_string`

```lan
define string_demo() -> uint8 {
    define s: ascii[..] · {immutable: true} = "hello";
    // s.len == 5，s[0] == 'h'
    define cstr: c_string = "hello" as: c_string; // 仅包含以 '\0' 结尾且无内嵌 '\0' 的字节串
    return s[0] as uint8;
}
```

- `ascii[..]`、`utf8[..]`、`c_string` 均支持 `.len`、下标读取与 `===` 比较。
- 传给 `extern contract` C 函数时，`c_string` 自动映射为 `const char *`，普通只读/可变切片映射为 `const T *` / `T *`。

### 9.5 自定义字符串编码类型（`type string ... = Map as slice | tag`）

用于嵌入式/游戏引擎（如 GBA 宝可梦双字节字库、Shift-JIS、自定义控制符）将源码中的中文/Unicode 字符串字面量在编译期直接编码为目标字节序列：

```lan
// 编译期 JSON 风格映射表（纯编译期对象，不生成 C 全局变量）
define GBA_MAP = {
    " ": 0x00,
    "A": 0xBB,
    "妙": [0x09, 0x86],
    "蛙": [0x0D, 0x44],
    "{PLAYER}": [0xFD, 0x01], // 多字符控制符，采用贪心最长匹配
    "\n": 0xFE,
    "@terminator": 0xFF,      // 隐式尾随终止符（可设为 null 表示无终止符）
    "@pad": 0xFF,             // 定长缓冲区剩余填充字节（省略时默认取单字节终止符或 0x00）
};

// 两种模式：
// 1) as slice（默认，可省略）：GbaText 本身即只读编码切片，GbaText[N] 为 N 字节定长缓冲
type string GbaText = GBA_MAP as slice;
// 2) as tag：GbaChar 为编码标签，GbaChar[..] 为只读编码切片，GbaChar[N] 为 N 字节定长缓冲
type string GbaChar = GBA_MAP as tag;

type Species = { id: uint16, name: GbaChar[8] } · {repr: c};
extern contract gba_print(msg: GbaText) -> void writes {external}; // C 侧自动映射为 const uint8_t *

define encode_demo() -> uint8 {
    define s: GbaText = "{PLAYER}妙蛙\nA";   // s.len == 8（不含尾随 0xFF），底层数组以 0xFF 结尾
    define lit = "妙蛙" as GbaText;           // 也可显式用 as 编码字面量
    define raw = lit as: view uint8[..] · {immutable: true}; // 零拷贝转普通只读字节切片
    define sp: Species = { id: 1, name: "妙蛙" };            // 编码+终止符后以 @pad 填满 8 字节，超长编译报错
    gba_print(s);
    gba_print("妙蛙");                        // 按参数期望编码自动编码字面量
    return (raw[0] + sp.name[4]) as: sat uint8;
}
```

编码类型安全规则：
- **严格编码隔离**：`string`、`c_string` 与不同自定义编码类型之间禁止隐式互传，需查看原始字节时使用 `as: view uint8[..] · {immutable: true}` 或 `as: view GbaText`。
- **编译期完备检查**：字面量中出现未映射字符，或定长缓冲 `Name[N]` 编码后（含 `@terminator`）超出 `N` 字节，均在编译期直接报错。
- **删除/覆盖条目**：合并两个映射表 `A · B` 时后写覆盖前写；将某键映射为 `null`（如 `BASE · { "A": null }`）可删除该映射条目。

### 9.6 集合元素计数 `.count(...)`

可直接统计数组/切片中落入某值域的元素个数（也可按结构体字段列统计），并参与静态范围推理：

```lan
type Slot = {state: uint8[0, 3], id: uint8};

define count_active(slots: Slot[8]) -> uint8[0, 8] {
    return slots.count({2}, state) as uint8[0, 8];
}

define count_zeros(a: uint8[16]) -> uint8[0, 16] {
    return a.count({0}) as uint8[0, 16];
}
```

---

## 10. 记录、键值记录（`type enum`）、枚举与位域

### 10.1 记录（结构体）与 C 内存布局

```lan
type Point = {x: int16, y: int16};
type CHeader = {magic: uint16, flags: uint8, count: uint8} · {repr: c};

define make_point(x: int16, y: int16) -> Point {
    return {x: x, y: y};
}

define move_x(p: Point, dx: int16) -> void {
    // 记录参数默认按借用（指针）传入，可直接原地修改
    p.x = (p.x + dx) as: sat int16;
}
```

- 默认记录会按字段对齐从大到小重排以减少填充；需要与 C `struct` 字段顺序一一对应时，必须加 `· {repr: c}`。
- 省略的记录初始化字段自动零初始化（如 `define p: Point = {x: 3};` 会把 `y` 初始化为 `0`）。
- 记录支持整体拷贝赋值：`dst = src as: copy;` 或直接初始化 `define b: Point = a;`。

### 10.2 枚举（`enum`）

```lan
enum Status: uint8 {
    Ok = 0,
    Busy = 1,
    Fail = 255,
}

define is_ok(s: Status) -> boolean {
    return s == Status.Ok;
}
```

- 省略 `: BaseType` 时默认底层为 `uint8`；未写 `= 值` 时从前一项加 1（首项为 0）。
- 枚举是强类型名义类型，不能隐式与整数或其他枚举混算或比较；需要取底层整数值时写 `s as uint8`。

### 10.3 键值记录（`type enum`）与动态字段索引

当一个记录的所有字段共享同一标量布局，且希望用枚举键动态索引字段（零越界查表）时使用 `type enum`：

```lan
type enum Stats {
    hp: uint16,
    atk: uint16,
    def: uint16,
}

define get_stat(s: Stats, k: Stats.key) -> uint16 {
    return s[k];                 // k 的类型是自动生成的 Stats.key 枚举
}

define boost_atk(s: Stats) -> void {
    s[Stats.key.atk] = (s[Stats.key.atk] + 10) as: sat uint16;
}
```

### 10.4 打包位数组（`bit[N]`）与位域视图

`bit[N]`（`N` 必须为 8 的正倍数）按字节紧凑存储单个比特，支持位字面量、位切片、以及自定义端序的位域映射视图：

```lan
define bitfield_demo(raw: uint32) -> uint16 {
    define storage: bit[32] = raw as: view bit[32];
    // 声明位域视图：字段名: 目标类型@bit(位宽, 位偏移, endian: le|be|native)
    define view = storage[
        lo: uint16@bit(16, 0, endian: le),
        hi: uint16@bit(16, 16, endian: le),
    ];
    return (view.lo + view.hi) as: wrap uint16;
}
```

- `storage[0]` 读/写第 0 位（`uint8[0, 1]`）；`storage[l, r)` 支持按位窗口切片与位拷贝。
- 支持 `endian: le`（小端，默认）、`endian: be`（大端）、`endian: native`（主机端序），也可写成 `@bit(16, 0) {endian: le}`。

---

## 11. 借用、别名、可空指针、堆所有权与链表/图

### 11.1 传递方式：`alias`（借用）与 `copy`（拷贝）

| 参数类型写法 | 语义 | 生成的 C 参数类型 |
| --- | --- | --- |
| `x: uint32` | 标量按值传 | `uint32_t` |
| `x: uint32 · {pass: alias}` | 可变标量借用（输出参数） | `uint32_t *` |
| `x: uint32 · {pass: alias, immutable: true}` | 只读标量借用 | `const uint32_t *` |
| `p: Point` | 可变记录借用（默认） | `Point *` |
| `p: Point · {immutable: true}` | 只读记录借用 | `const Point *` |
| `p: Point · {pass: copy}` | 记录按值拷贝 | `Point` |
| `a: uint8[16]` | 可变数组借用（默认） | `uint8_t *` |
| `a: uint8[16] · {pass: copy}` | 定长数组按值拷贝 | `lan_array_copy_...` |

> **别名反混叠规则**：如果函数同时接收多个可变借用参数，调用方必须能被静态证明它们指向互不重叠的内存对象，否则编译报错。

### 11.2 可空借用（`T | null`）与返回借用别名（`from`）

```lan
type Box = {value: uint16};

// 可空记录/标量借用必须显式带 {pass: alias} 再与 | null 组合
define read_or_zero(b: Box · {pass: alias} | null) -> uint16 {
    if (b == null) {
        return 0;
    }
    return b.value;              // 此处 b 已静态证明非空
}

// 返回借用自第 0 个参数字段的别名
define borrow_value(b: Box) -> uint16 · {pass: alias, from: 0, field: value} {
    return b.value;
}
```

### 11.3 堆对象（`storage: heap`）、所有权转移与 `free`

Lan 没有 GC，通过编译期生命周期状态（`Alive` / `Tombstone`）杜绝悬垂指针、重复释放与内存泄漏：

```lan
type Node = {value: uint32};

define alloc_and_use() -> uint32 {
    define n: Node · {storage: heap} = {value: 42};
    define v: uint32 = n.value;
    free(n);                     // 此后 n 变为 Tombstone，再访问或再 free 均编译报错
    return v;
}

// 所有权返回给调用方（调用方接手释放责任）
define make_node(v: uint32) -> Node · {storage: heap} {
    define n: Node · {storage: heap} = {value: v};
    return n;
}

// 动态长度堆数组
define heap_array_demo(n: uint8[1, 16]) -> uint8 {
    define arr: uint8[n] · {storage: heap};
    arr[0] = 7;
    define out: uint8 = arr[0];
    free(arr);
    return out;
}
```

### 11.4 链表与图记录（自引用记录）

记录字段可声明指向同类型（或其它记录）的可空链接 `NextNode | null`：

```lan
type Item = {value: uint8, next: Item | null};
```

- 编译器会自动根据函数体内的链接读写推断链/图借用形状（无需写已废弃的 `chain`/`graph`/`prune`/`attach` 限定符）。
- 断开或释放节点时，若仍有局部别名可能悬垂，编译器会直接报错（配合 `--insert-oracle` 可自动提示悬垂释放说明）。

---

## 12. 错误联合与 `?` 早退传播

Lan 用**类型并集 `Success | ErrorEnum`** 表示可失败返回，无异常机制，零堆分配：

### 12.1 标量不相交错误联合（零标签开销）

当成功值域与错误枚举值域**完全不相交**时，底层直接共用同一个标量整数，零额外结构体开销：

```lan
enum ParseErr: uint8 { Empty = 254, Overflow = 255 }
type Digit = uint8[0, 9] | ParseErr;

define parse_digit(ch: uint8) -> Digit {
    if (ch < 48 || ch > 57) {
        return ParseErr.Empty;
    }
    return (ch - 48) as uint8[0, 9];
}

define parse_pair(a: uint8, b: uint8) -> uint8[0, 18] | ParseErr {
    define x = parse_digit(a)?;  // 若为 ParseErr，自动早退返回；否则 x 精化为 uint8[0, 9]
    define y = parse_digit(b)?;
    return x + y;
}
```

### 12.2 复合带标签错误联合（记录 / 数组 / 堆对象 | ErrorEnum）

当成功臂是记录、数组或堆所有权对象时，后端自动生成带 `ok` 标志的 `lan_result_*` 结构体，`?` 同样生效（且早退路径会自动清理当前正在初始化的堆对象，防止泄漏）：

```lan
enum LookupErr: uint8 { Missing = 255 }
type Pair = {a: uint8[0, 100], b: uint8};
type Found = Pair | LookupErr;

define find_pair(ok: boolean) -> Found {
    if (!ok) { return LookupErr.Missing; }
    return {a: 3, b: 4};
}

define read_a(ok: boolean) -> uint8[0, 100] | LookupErr {
    define p: Pair = find_pair(ok)?;
    return p.a;
}
```

---

## 13. 接口与回调：`interface`、`callable`、`extern callable`

### 13.1 结构化接口（`interface`）

对具备相同字段名与类型的不同记录类型实现零堆分配多态（底层为 `{data指针, 字段偏移表指针}`）：

```lan
type HasHP = interface { hp: uint16 };
type Hero  = { name_id: uint8, hp: uint16 };
type Boss  = { hp: uint16, phase: uint8 };

define heal(unit: HasHP, amount: uint16) -> void {
    unit.hp = (unit.hp + amount) as: sat uint16;
}
```

### 13.2 模块内封闭回调集（`callable`）

适用于**全部候选函数在编译期已知**的 Lan 内部回调（编译器会合并全部候选函数的写集并做全静态副作用分析）：

```lan
define inc(x: uint8) -> uint8 { return (x + 1) as: sat uint8; }
define dec(x: uint8) -> uint8 { return (x - 1) as: sat uint8; }

type StepFn = callable(inc, dec);

define apply_step(cb: StepFn, x: uint8) -> uint8 {
    return cb(x);
}

define test() -> uint8 {
    return apply_step(callback(inc), 5);
}
```

> 限制：`callable(...)` 仅限模块内部使用，**不能跨模块 `export`，也不能接收外部 C 传来的函数指针**。

### 13.3 开放式 C 函数指针回调（`extern callable`）

用于接收**手写 C 传进来的任意函数指针**（例如游戏里的 `reject` 过滤回调、排序比较器、事件钩子），支持可空（`| null`）、显式写集声明、跨 `.table` 导入导出与 `--export-prefix`：

```lan
// 语法：extern callable(参数列表) -> 返回类型 [writes { ... }]
// 省略 writes 时默认等同 writes {}（纯读不改外部状态）
type RejectFn = extern callable(x: uint32) -> boolean writes {};

define random_except(
    lo: uint32[0, 1000],
    span: uint32[1, 1000],
    raw: uint32,
    reject: RejectFn | null
) -> uint32[0, 1999] {
    define pick: uint32[0, 1999] = lo + (raw % span);
    if (reject != null) {
        if (reject(pick)) {      // 未判空直接调用 reject(pick) 会编译报错！
            return lo;
        }
    }
    return pick;
}

// 也可在 Lan 内部传本地函数 callback(fn) 或 null 给 extern callable 参数
define is_even(x: uint32) -> boolean { return (x & 1) == 0; }
define demo(raw: uint32) -> uint32[0, 1999] {
    define a = random_except(10, 20, raw, callback(is_even));
    define b = random_except(10, 20, raw, null);
    return a;
}

export { random_except };
```

在 C 侧可直接声明并传普通函数指针或 `NULL`：

```c
uint32_t gba_random_except(uint32_t lo, uint32_t span, uint32_t raw, uint8_t (*reject)(uint32_t));
```

写集兼容性规则：传入的 `callback(fn)` 或 `extern callable` 变量的实际副作用集合必须是目标参数 `writes { ... }` 的子集（不允许把会写内存的函数传给声明为 `writes {}` 的回调参数）。

---

## 14. 类型对象合并（`·`）、限定符与编译期类型函数

### 14.1 类型对象合并符 `·` 与全部合法属性键

使用 `BaseType · {key: value, ...}` 给类型附加存储、借用、不可变、布局或来源属性（也可以直接从 `int`、`uint`、`array` 关键词构造基础类型对象，如 `uint · {bits: 16}`）：

| 属性键 | 允许取值 | 作用范围 |
| --- | --- | --- |
| `kind` | `int` \| `uint` \| `array` | 纯类型对象构造基础种类 |
| `bits` | `8` \| `16` \| `32` \| `64` | 整数位宽 |
| `len` | 编译期整数或区间（如 `4`、`[1, 32]`） | 定长数组长度或切片动态长度界 |
| `elem` | 类型或值域集合（如 `uint8`、`[0, 9]`） | 数组/切片元素类型或元素值域约束 |
| `immutable` | `true` \| `false` | 禁止经由该绑定/切片/借用修改内存 |
| `pass` | `alias` \| `copy` | 参数/返回值传递方式（借用指针 vs 独立拷贝） |
| `storage` | `heap` \| `stack` \| `rodata` | 存储区：堆所有权 / 栈 / 静态只读区 |
| `from` | 非负参数序号（从 `0` 起） | 返回借用切片/别名/接口时，绑定到第几个入参 |
| `offset` | 编译期整数或依赖偏移表达式 | 返回切片相对来源参数的起始偏移 |
| `field` | 字段路径（如 `value`、`a.b[0]`） | 返回别名精确指明借用自来源记录的子字段 |
| `repr` | `c` | 用在记录上强制保持声明字段序，与 C `struct` ABI 对齐 |

> 注意：`chain`、`graph`、`prune`、`attach` 为历史废弃限定符，现已由编译器自动推断，不要再写入类型对象。

### 14.2 编译期类型函数（`type Fn(...) { ... }`）

用于泛型与参数化类型构造（在编译期求值，不产生任何运行时代码）：

```lan
type Word(bits: uint8) {
    return uint · {bits: bits};
}

type Matrix(Elem: type, cols: uint8[1, 32], rows: uint8[1, 32]) {
    return Elem[cols][rows];
}

type ByteMat4x3 = Matrix(uint8, 4, 3);
type U16 = Word(16);
```

- 类型函数参数标注 `: type` 表示接收类型、映射表或列表，标注标量/定长数组类型表示接收对应编译期常量。
- 类型函数体内支持 `if`、`while`、`interation`、局部 `define`、`chr(code)` / `ord(ch)`、映射表合并 `m · { ... }`、计算键 `{ [code]: byte }`、下标写入 `m[key] = bytes` / `arr[i] = val` 与 `return`。
- 跨模块复用类型函数时，必须从 `.lang` 源码导入（不能仅从 `.table` 导入）。

### 14.3 编译期常量递增展开（`expand`）

当大型编译期常量（如几千条目的汉字字库映射表、查找表、参数配置）需要分散在多个文件或通过循环批量构造时，可在模块顶层使用 `expand 类型函数(目标常量[, 额外参数...]);`：

```lan
define GBA_MAP = { " ": 0x00, "@terminator": 0xFF };

// 用循环按 Unicode 码点批量填充 A-Z（支持 m[code] = byte、m[chr(code)] = byte 与 m · { [code]: byte }）
type AddAsciiUpper(m: type) {
    interation(i; 0; uint8; 26; 1) {
        m[0x41 + i] = 0xBB + i;
    }
    return m;
}
type AddDigits(m: type) {
    interation(i; 0; uint8; 10; 1) {
        m[chr(0x30 + i)] = 0xA1 + i;
    }
    return m;
}

expand AddAsciiUpper(GBA_MAP);
expand AddDigits(GBA_MAP);
type string GbaText = GBA_MAP as slice;

// expand 同样适用于标量、定长数组与只读切片常量
define MAX_HP: uint16 = 100;
define BEFORE_HP = MAX_HP * 2;        // 自顶向下求值：此时 MAX_HP == 100 -> BEFORE_HP == 200
type Scale(v: uint16, k: uint16) { return v * k; }
expand Scale(MAX_HP, 3);              // 更新 MAX_HP -> 300
define AFTER_HP = MAX_HP + 5;         // 此时 MAX_HP == 300 -> AFTER_HP == 305

define LUT = [10, 20];
type Append(a: type, v: uint16) { return a · [v]; }
expand Append(LUT, 30);               // 最终生成 3 元素只读数组 [10, 20, 30]

define verify_expand() -> uint16 {
    define msg: GbaText = "A0 ";
    // 运行期所有函数看到的均为全部编译期代码执行完毕后的最终值
    if (MAX_HP == 300 and BEFORE_HP == 200 and AFTER_HP == 305 and LUT[2] == 30) {
        return MAX_HP + msg[0];
    }
    return 0;
}
```

执行顺序与落盘语义：
- **自顶向下同步执行**：模块顶层的 `define`、`expand`、`type string`、`type` 在编译期阶段按源码顺序同步求值；后续语句看到此前已完成 `expand` 后的最新值。
- **最终值一次性落盘**：全部编译期代码执行完毕后，每个需要进入运行期的模块常量仅以**最终值**生成一次 `static const`，所有运行期函数看到的都是该最终值。
- **纯映射表零开销**：`JsonMap` 常量仅在编译期存活，不生成任何 C 全局符号；可通过 `export` / `import` 在 `.lang` 模块间分片组装大型映射表。

---

## 15. 定理、公理与 Oracle 注释

### 15.1 顶层集合定理与公理

```lan
type Even  = uint8 & poly(2 * n);
type Four  = uint8 & poly(4 * n);
type Odd   = uint8 & poly(2 * n + 1);

theorem subset(Four, Even);      // 编译器自动证明；证不出来则报错
theorem disjoint(Even, Odd);     // 证明交集为空
theorem equal(Even & Four, Four);// 证明相等，并自动加入后续交集推理规则
// 若某高阶多项式交集超出自动求解器能力，可显式声明公理（记录为 assumed）：
// axiom equal(A & B, Result);
```

也支持文档注释形式声明交集事实：`/* @oracle @theorem A & B => Result */`。

### 15.2 `while` 循环 `@oracle` 注解

当复杂 `while` 循环无法被自动不动点迭代收敛或自动证明终止时，在 `while` 正上方紧贴写 `/** @oracle ... */`：

```lan
define countdown(n: uint16) -> uint16[0, 0] {
    define cur: uint16 = n;
    /** @oracle
     * @invariant { state(cur) := uint16[0, 65535]; }
     * @proof { decreases: cur }
     * @ensures { state(cur) := uint16[0, 0]; }
     */
    while (cur > 0) {
        cur -= 1;
    }
    return cur;
}
```

支持指令：
- `@invariant { state(var) := Type; ... }`：循环不变量（编译器会检查初值满足且循环体保持）
- `@proof { decreases: expr }`：严格递减的非负终止度量
- `@transition => { state(var) := ...; }`（可加 `@trusted`）：复杂状态转移摘要
- `@ensures { [when (cond)] state(var) := Type; ... }`：退出循环后的后置条件

> 技巧：不确定某个循环缺什么不变量时，可运行 `python lanc.py foo.lang --check --insert-oracle`，编译器会自动把缺失的验证请求注释插到对应位置。

---

## 16. 模块系统、分模块编译与 C / 裸机互操作

### 16.1 模块导入导出（`import` / `export`）

**导出模块 `math_util.lang`**：

```lan
define MAX_VAL: uint16 = 1000;
type ClampVal = uint16[0, MAX_VAL];

define clamp(x: uint16) -> ClampVal {
    if (x > MAX_VAL) { return MAX_VAL; }
    return x;
}

export { MAX_VAL, ClampVal, clamp };
// 支持重命名导出：export { clamp as clamp_u16 };
```

**导入方 `app.lang`**：

```lan
// 方式 1：直接从 .lang 源码导入（编译到同一个 C 翻译单元，支持所有类型/函数/类型函数/全部模块常量）
import { MAX_VAL, ClampVal, clamp } from "./math_util.lang";

// 方式 2：从分模块构建产物 .table 导入（只需编译期接口，链接期链接 math_util.o）
// import { MAX_VAL, ClampVal, clamp } from "./math_util.table";
```

### 16.2 Lan 调用手写 C：`extern contract`

用于从 Lan 里调用外部 C 函数（如游戏引擎、驱动、libc、硬件寄存器封装）：

```lan
// 必须声明 writes { ... } 写集，告知编译器该 C 函数的副作用边界：
//   writes {}          — 纯函数，不写任何参数也不改外部状态
//   writes { buf }     — 可能修改参数 buf 的内容（支持字段路径如 s.hp）
//   writes { memory }  — 可能分配/释放堆内存
//   writes { external }— 可能读写全局变量 / MMIO / I/O
//   writes { all }     — 可能修改一切可写参数及外部状态

extern contract Random() -> uint16 writes { external };
extern contract CpuSet(src: uint8[..] · {immutable: true}, dst: uint8[..], mode: uint32) -> void writes { dst, external };

define roll_d6() -> uint16[1, 6] {
    return (Random() % 6) + 1;
}
```

### 16.3 手写 C 调用 Lan：`--library --export-prefix`

默认 `export { foo }` 生成带 ABI 与契约 SHA-256 指纹的符号 `lan_api_<abi>_<contract>_foo`，用于在多模块 Lan 之间防止签名漂移。

要让**手写 C 代码直接调用 Lan 函数**，编译时加上 `--library --export-prefix <前缀>`：

```bash
python lanc.py battle.lang -o battle.c --library --export-prefix battle_
```

此时 `export { iso_step }` 会同时生成：
```c
uint32_t lan_api_..._iso_step(uint32_t arg_0); // 保留指纹符号
uint32_t battle_iso_step(uint32_t arg_0);      // 稳定别名，供手写 C 直接声明调用
```

- 传 `--export-prefix ""`（空串）可直接按原导出名 `iso_step` 暴露。
- 注意：稳定别名去掉了契约指纹防呆校验，修改 Lan 函数签名后需同步更新 C 侧的 `extern` 声明。

### 16.4 Lan ↔ C ABI 类型映射表

| Lan 类型 | Lan 导出 / 内部 C 类型 | `extern contract` / `extern callable` C ABI 类型 |
| --- | --- | --- |
| `uint8`/`16`/`32`/`64` | `uint8_t`/`16`/`32`/`64_t` | 同左 |
| `int8`/`16`/`32`/`64` | `int8_t`/`16`/`32`/`64_t` | 同左 |
| `boolean` | `uint8_t`（值严格为 `0` 或 `1`） | `uint8_t` |
| `float32` / `float64` | `float` / `double` | 同左 |
| `T · {pass: alias}` | `T *`（加 `immutable: true` 为 `const T *`） | 同左 |
| `Record` 参数（默认借用） | `lan_record_* *`（布局加 `{repr: c}` 保序） | 同左 |
| `T[N]` 参数（默认借用） | `T *`（`bit[N]` 为 `uint8_t *`） | 同左 |
| `T[..]` 切片 | `lan_slice`（`{const void *data; size_t len;}`） | `T *` / `const T *`（自动拆指针传入） |
| `c_string` | `lan_slice` | `const char *` |
| `extern callable(...) -> R` | `typedef R (*lan_fnptr_...)(...)` | 标准 C 函数指针（与 `R (*)(...)` ABI 完全等价） |

---

## 17. 完整语法速览（EBNF）与常见报错速修表

### 17.1 顶层与语句核心 EBNF

```ebnf
Module       := TopItem* ;
TopItem      := ImportDecl | ExportDecl | EnumDecl | KeyedRecDecl
              | TypeAlias | TypeFuncDecl | ExternDecl | ConstDecl
              | FuncDecl | TheoremDecl | OracleComment ;

ImportDecl   := "import" "{" ModItem ("," ModItem)* ","? "}" "from" StringLit ";" ;
ExportDecl   := "export" "{" ModItem ("," ModItem)* ","? "}" ";" ;
ModItem      := Ident ("as" Ident)? ;

EnumDecl     := "enum" Ident (":" TypeExpr)? "{" EnumMem ("," EnumMem)* ","? "}" ";"? ;
KeyedRecDecl := "type" "enum" Ident "{" Param ("," Param)* ","? "}" ";"? ;
TypeAlias    := "type" Ident "=" TypeExpr ";" ;
TypeFuncDecl := "type" Ident "(" TypeParam ("," TypeParam)* ","? ")" Block ;

ExternDecl   := "extern" "contract" Ident "(" ParamList ")" "->" TypeExpr WriteSet ";" ;
WriteSet     := "writes" "{" (WritePath ("," WritePath)*)? "}" ;

ConstDecl    := "define" Ident "=" Expr ";"
              | "define" Ident ":" TypeExpr ("=" Expr)? ";" ;

FuncDecl     := "define" Ident "(" ImplParamList ")" ("->" TypeExpr)? PropClause? Block
              | "define" Ident ":" TypeExpr "(" IdentList ")" PropClause? Block ;

TheoremDecl  := ("theorem" | "axiom") ("equal" | "subset" | "disjoint")
                "(" TypeExpr "," TypeExpr ")" ";" ;

Block        := "{" Stmt* "}" ;
Stmt         := "define" Ident (":" TypeExpr)? ("=" Expr)? ";"
              | "if" "(" Expr ")" Block ("else" ("if" ... | Block))?
              | LoopOracle? "while" "(" Expr ")" Block
              | "interation" "(" Ident ";" Expr ";" TypeExpr ";" Expr ";" Expr ")" Block
              | "continue" ";"
              | "return" Expr? ";"
              | "assert" Expr ":" "{" AssertRow ("," AssertRow)* ","? "}" ";"
              | "assert" "chain" "(" "head" ":" Expr "->" "tail" ":" Expr ")" ";"
              | Expr AssignOp Expr ";"
              | Expr ";" ;

TypeExpr     := BaseType ("·" TypeExpr)* (("&" | "\\" | "|") TypeExpr)* ;
BaseType     := Ident ("[" Expr "," Expr ("]" | ")") | "[" ".." "]" | "[" Expr "]" | "(" TypeArgs ")")*
              | ("[" | "(") Expr "," Expr ("]" | ")")
              | ("<" | "<=" | ">" | ">=") Expr
              | "{" (Ident ":" TypeVal ("," ...)* | Expr ("," ...)*) "}"
              | "interface" "{" ParamList "}"
              | "contract" "(" ParamList ")" "->" TypeExpr
              | "callable" "(" IdentList ")"
              | "extern" "callable" "(" ParamList ")" "->" TypeExpr WriteSet?
              | "poly" "(" Expr ")" ;
```

### 17.2 常见编译报错速修表（Agent 调试指南）

| 报错关键词 | 原因 | 正确改法 |
| --- | --- | --- |
| `break was removed; express the exit in the loop condition` | 写了 `break;` | 删掉 `break`，改用 `while (running && ...)` 或在函数里直接 `return` |
| `expected '{', got ...`（在 `if`/`while` 处） | 条件没加外层括号或分支省略了 `{}` | 条件整体加 `(...)`，分支体必须用 `{ ... }` 包起来 |
| `expected ',' or ']'` | 多行数组字面量两行之间漏了逗号 | 在上一行最后一个元素末尾补 `,` |
| `unknown type object property '...'` | `· { ... }` 里写了不存在的属性名（如 `mut`/`readonly`/`access`） | 只读请写 `· {immutable: true}`；其余合法键见第 14.1 节总表 |
| `cannot prove value ... fits target ...` | 算术结果范围超过目标类型（如 `uint8 + uint8` 赋给 `uint8`） | 扩大目标类型（如 `uint16`），或显式写 `as: wrap uint8` / `as: sat uint8`，或先用 `if` 缩小范围 |
| `cannot prove index ... is within ...` | 索引上界或下界未被静态证明在 `[0, len)` 内 | 用依赖参数 `i: uint32[0, a.len)`、`interation`，或访问前加 `if (i < a.len) { ... }` |
| `cannot prove divisor ... is nonzero` | `/` 或 `%` 右操作数值域包含 `0` | 类型限定为 `uint32[1, ...]` / `uint32 \ {0}`，或运算前加 `if (d == 0) { return ...; }` |
| `cannot prove nullable ... is non-null` | 未判空就解引用可空指针或调用 `extern callable \| null` | 包在 `if (p != null) { ... }` 中，或先 `if (p == null) { return ...; }` |
| `heap leak: '...' leaves the function before free` | 某条分支退出前没释放 `storage: heap` 对象 | 在所有 `return` 及函数末尾路径上补 `free(x);` 或将所有权返回 |
| `a reachable path does not return a value` | 非 `void` 函数存在未覆盖的 `return` 路径 | 补齐 `else` 或函数末尾的 `return` |
| `expected a compile-time ...` / `rodata initializer requires a compile-time constant` | 数组长度、重复计数或常量初始化依赖了运行期非常量 | 改用模块级 `define N = ...;`、单点类型或当前路径已知为常量的表达式 |
| `unsupported conversion strategy '...'` | `as:` 后拼错了策略名 | 合法策略为 `safe`/`wrap`/`sat`/`trap`/`view`/`trunc`/`floor`/`ceil`/`round`（及其 `_sat`/`_trap` 组合）；安全转换直接写 `x as T` 或 `x as: T` |
| `write set exceeds extern callable parameter effects` | 传入的回调函数副作用超过 `extern callable` 声明的 `writes` | 补齐目标 `extern callable` 的 `writes { ... }`，或改为无副作用的纯回调 |
| `callable values are local to their Lan module` | 试图导出 `callable(...)` 或放进 `extern contract` | 跨模块/跨 C 边界的函数指针请改用 `extern callable(...) -> R writes { ... }` |
