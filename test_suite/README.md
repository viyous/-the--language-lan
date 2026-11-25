# test_suite — Lan 语言测试用例

每个目录一个独立用例，目录名与主文件名一致（`NN_name/NN_name.lang`）。
每个用例都是一个**能编译、能链接、能运行**的完整程序：`main` 返回 0 表示
全部断言通过，返回非 0 表示第几条检查失败（便于定位）。

## 运行

```bash
cd lan
python test_suite/verify.py                 # 跑全部
python test_suite/verify.py 11_records      # 只跑一个
python test_suite/verify.py 31_matrix 32_search
python test_suite/verify.py --cc clang 01_scalars
```

`verify.py` 默认从 PATH 依次查找 Clang、GCC、Zig。也可用 `--cc` 选择编译器。每个用例依次执行以下步骤。

1. `python lanc.py <case>.lang -o <case>.c`（`shared` 用例加 `--library`）
2. 逐个生成 `-O2` 汇编。使用相对文件名并移除 COFF 调试节，避免汇编携带本机路径。
3. 链接到临时目录并检查程序退出码。

同目录下的其它 `.c` 文件会自动参与链接与汇编，用于 `extern contract` 这类
需要 C 侧实现的用例（见 `21_extern/support.c`）。

状态码：`OK` / `LANFAIL`（Lan 编译失败）/ `ASMFAIL`（汇编失败）/ `CCFAIL`
（链接失败）/ `RUNn`（运行返回 n）。

## 产物（保留，用于分析与优化）

运行脚本后，对应产物会保留在用例目录，供分析和优化对比。Git 默认忽略这些生成文件。原生测试源码 `21_extern/support.c` 随源码发布。

| 文件 | 内容 |
| --- | --- |
| `<case>.c` | Lan 生成的 C11：`lh_contract_assume_N` 假设宏、标签 + goto 的分支/循环、`LAN_ORACLE_DEBUG` 断言 |
| `<case>.table` | 版本化模块接口 |
| `<case>.theorem` | 已证范围定理 |
| `<case>.proof` | SSA 查询数据及分析轨迹，包括`@function` 的入口参数值域、`assume entry`、每条路径的 `before`/`after` 状态、`loop head`/`loop body`/`loop back edge`/`loop exit`、`loop fixed point after N analysis round(s)`、`mark tombstone`、`free chain` 摘要 |
| `<case>.s` | `zig cc -std=gnu11 -O2 -S -g0 -fno-unwind-tables` 的汇编，用来核对证明是否真的喂给了优化器 |

可执行文件不保留：它只是用来跑断言的，编译在临时目录、跑完即删。

例如 `27_comprehensive.proof` 的诊断轨迹里，解释器主循环（`loop@43:5`）的收敛过程：

```
loop head   [loop@43:5] { ..., id#26 = [0, 16] }
loop body   [loop@43:5 and condition=true] { ..., id#26 = [0, 14] }
update entity id#26 [loop@43:5 and condition=true] := [2, 16]
loop fixed point after 9 analysis round(s)
loop back edge tail [...] { ..., id#26 = [2, 16] }
```

`id#26` 是 `pc`：入口 `[0, 16]`，循环体内被条件收窄到 `[0, 14]`（因为要
读两个字节），回边更新成 `[2, 16]`，九轮后到达不动点。

同一个循环里还能看到索引证明的行级记录：

```
verify index id#17 at 44:33: [0, 14] subset [0, 15] => proven
verify index id#17 at 45:37: [1, 15] subset [0, 15] => proven
```

工具链：`zig 0.17.1`（`zig cc` 包装 clang）+ `z3-solver`。MSVC 不能编译
Lan 输出（用到 `__int128`、GNU 语句表达式、`__builtin_assume`）。

## `-O2` 汇编观察

`.s` 用的是 `zig cc -std=gnu11 -O2 -S -g0 -fno-unwind-tables`。后两个开关是必须的，
否则整份汇编会被非代码内容淹没：

| 开关 | 去掉什么 | 效果（`31_matrix`） |
| --- | --- | --- |
| （不加） | — | 13531 行，其中 CodeView 调试信息（`.cv_func_id` / `.cv_loc` / `#DEBUG_VALUE`）和 SEH 展开指令（`.seh_proc` / `.seh_pushreg` / `.seh_savexmm` …）占九成 |
| `-g0` | CodeView 调试信息 | 1773 行 |
| `+ -fno-unwind-tables` | Windows x64 SEH 展开指令 | **1575 行** |

全库 37 个 `.s` 从 19.9 万行降到 3.2 万行，剩下的基本就是指令本身
（外加每个符号 5 行的 COFF `.def/.scl/.type/.endef` 元数据，那是给汇编器看的，
读代码时忽略即可）。不加这两条开关，分析任何单个函数都要在一堆 `.cv_loc` 里翻。

### 只看一个函数

`show_func.py` 直接从 `.s` 里按符号名抽出来，并切掉尾部的符号元数据：

```bash
python test_suite/show_func.py 31_matrix --list      # 列出所有函数和行号区间
python test_suite/show_func.py 31_matrix mat_mul     # 只打印 mat_mul
python test_suite/show_func.py 27_comprehensive pop
```

函数清单取自 COFF 的 `.def <name>` + `.type 32` 元数据，所以 `main` 里被提升到
`.rdata` 的局部常量数组（`lan_fn_main.lv_87_prog` 之类）不会被误当成函数。

### 几个典型结果

| 用例 | `.s` 行数 | AVX2 指令 | 说明 |
| --- | --- | --- | --- |
| `31_matrix` | 1575 | 110 | `mat_mul` 的 4×4 三层循环被**完全展开 + AVX2 向量化**（`vpmulld %ymm`） |
| `29_hashing` | 1876 | 124 | 哈希主循环向量化 |
| `30_bitops` | 1284 | 112 | SWAR popcount、位反转批量操作向量化 |
| `33_strings` | 2021 | 8 | 只有 RLE/大写转换的拷贝段向量化，匹配循环本身是分支密集 |
| `32_search` | 1932 | 0 | 二分/双指针全是数据相关分支，不可能向量化（符合预期） |
| `34_dp` / `35_dp2` | 1066 / 1737 | 10 / 23 | 滚动数组的初始化与拷贝段向量化，DP 递推本身有依赖链 |
| `27_comprehensive` | 536 | 0 | 字节码解释器，主循环是间接跳转，不可能向量化 |

`31_matrix` 里 `lan_fn_mat_mul` 的核心段（`python test_suite/show_func.py 31_matrix mat_mul`）：

```asm
lan_fn_mat_mul:
        vmovdqu (%rdx), %xmm0
        vmovdqu 16(%rdx), %xmm1
        ...
        vpmulld %ymm11, %ymm1, %ymm1
        vpmulld %ymm15, %ymm2, %ymm2
        vpaddd  %ymm2, %ymm1, %ymm2
```

`i < 4` / `k < 4` 这些**由类型和守卫证明出来的定长上界**，加上
`lh_contract_assume_N` 注入的值域假设，让 LLVM 敢把三层循环整个摊平、
再用 256 位整数乘法一次算完 —— 这正是"证明能换性能"的直接证据。

## 关于源码里的注释

`*.lang` 里的注释已经全部去掉了，包括文件头说明和分节注释；
`/** @oracle ... */` 这类**循环契约块保留**，因为它们是语法的一部分。
去注释用 `python test_suite/strip_comments.py`（幂等，可重复跑）。

设计意图、踩坑规则、缺陷复现步骤都集中在本文件里，不再写在源码注释里。

## 用例清单

| 用例 | 覆盖内容 |
| --- | --- |
| `shared` | 共享类型库：`Vec2/Vec3/Rgb/Span/Actor(32 字段)/Frame/Matrix4/ItemSlot/Inventory/Camera/Light/Scene/Element/SlotKind/Direction` 以及细化别名 |
| `01_scalars` | 各宽度整数、算术细化、有符号/无符号、位运算、`wrap`/`sat`/`trap` 三种溢出策略、布尔算术、常量折叠 |
| `02_intervals` | 闭/半开/开区间、单边区间、夹取、区间传递、依赖参数、循环推进区间 |
| `03_sets` | 离散集字面量、并/交/差、`in` 成员、集合算术 |
| `04_polynomials` | `poly(2n)` 等多项式像集、像集保持、不相交证明、大值成员判定 |
| `05_floats` | `float32/64` 运算与比较、取整/饱和转换、浮点记录、整数互转 |
| `06_conversions` | `as` / `as: wrap/sat/trap/trunc/view/copy`、有符号无符号重解释、整数与位数组互视、记录复制语义 |
| `07_bitfields` | 位数组、具名 `@bit` 字段、`endian: be`、位切片、二维位数组、`.len` |
| `08_bitwindows` | 8/16/32/64 位动态窗口、可变宽度、重叠与不相交复制、滑动循环、往返 |
| `09_arrays` | 列表/重复初始化、下标、`.len`、二维数组、记录数组、借用与复制参数 |
| `10_slices` | `{len: {...}}` 长度契约、定长转动态、子切片、循环、二维行切片 |
| `11_records` | 导入大结构体：零值/部分初始化、嵌套投影、记录数组、`Matrix4`、`Frame` 位标志 |
| `12_interfaces` | 鸭子接口、经接口读写、字段顺序不同、额外字段、多实现、带接口字段的堆记录按拥有型形参跨函数移交（只读字段、被调方释放、拷贝副本自带偏移表） |
| `13_enums` | 自增/显式取值、有符号基类型、枚举与整数互转、比较、导入枚举 |
| `14_control` | if/else 链、嵌套、短路、德摩根、while/continue、冒泡排序、线性查找、状态机 |
| `15_guards` | 守卫断言 `assert x: { [x == 0] => ..., [_] => ... }`、字段守卫、四路拆分 |
| `16_recursion` | 直接/相互/依赖结果递归、累加器、递归+循环、数组递归 |
| `17_properties` | `property:` 恒等/幂等/交换/对角 |
| `18_ownership` | 堆标量/数组/位数组/记录、定长堆转动态堆切片、消费型参数、递归返回堆值 |
| `19_chains` | `assert chain`、推导的只读链借用、遍历求和/计数/查找/取尾/最值、两种记录类型交替的链与树的释放 |
| `20_text` | `ascii_string`/`string`/`c_string`、长度契约、动态下标、字符分类、内容比较、子串返回、UTF-8 多字节 |
| `21_extern` | `extern contract`：标量/切片/定长数组形参、返回值域契约、C 侧 `support.c` |
| `22_modules` | 跨文件 `import`/`export`：细化类型与函数、多层依赖、转发层、别名导入 |
| `23_oracles` | 循环契约：`@invariant`、`@transition`、`@trusted`、`@ensures`、`@proof { decreases }` |
| `24_graphs` | 推导图借用、断边和挂接，覆盖分叉、共享和环 |
| `25_copies` | `{pass: copy}` 与默认借用的差别、`{storage: rodata}` 常量表、`as: copy`、副本独立性 |
| `26_ranks` | `===@k` / `!==@k` 定秩比较：`@0` 比地址、`@k` 比 k 层标量后比边界链接地址、切片比较 |
| `27_comprehensive` | 综合：枚举指令集 + rodata 程序段 + 堆栈 + 链式轨迹 + 记录出参的栈式字节码解释器 |
| `28_sorting` | 高性能算法：插入/选择/希尔/快速（递归 + Lomuto）/堆排序，含有序性、元素和不变、五算法一致性 |
| `29_hashing` | 高性能算法：FNV-1a、多项式哈希、开放寻址哈希表（含墓碑）、布隆过滤器（`bit[64]`） |
| `30_bitops` | 高性能算法：Kernighan / SWAR popcount、clz/ctz、位反转、格雷码、`next_pow2`、循环移位、位集合 |
| `31_matrix` | 高性能算法：4x4 矩阵乘/转置/迹、点积、矩阵乘向量、3x3 行列式、矩阵幂、行/列和、饱和加法 |
| `32_search` | 高性能算法：二分、下界/上界、指数搜索、旋转数组最小元、峰值、双指针、整数开方（对答案二分） |
| `33_strings` | 高性能算法：朴素匹配、KMP、Rabin-Karp、Boyer-Moore-Horspool、回文、atoi、trim、游程编码 |
| `34_dp` | 高性能算法：LCS（两行滚动）、编辑距离、LIS；演示 `as` 不保留细化与嵌套循环拆函数 |
| `35_dp2` | 高性能算法：0/1 背包（一维逆序）、零钱兑换、Kadane、网格路径计数（中间和夹取） |
| `36_type_functions` | 编译期类型函数：`type f(n: uint8){ ... return <整数集合>; }`，编译期循环/递归/布尔参数，使用处 `in {uint8, uint16} 32` 物理封口落成最小容器，算出的下标集合直接证明下标在界内 |
| `37_heap_alloc` | 动态分配：`uint8[n] · {storage: heap}` 元素数来自运行期参数（`calloc(n, ...)`）、拥有型数组形参移交、循环内逐轮分配并释放、运行期节点数建链后 `free` 一遍 |
| `38_dynamic_array` | 动态数组（JS 风格，元素类型固定）：`(缓冲区, count)` 表示变长数组，push/pop/at/set_at/insert/remove_at/index_of/reverse/copy_into，容量 16 的有界退化，以及 `grow()` 8→16 的重新分配搬移 |

## 写用例时反复踩到的规则

- **整数缩窄要证明值域兼容**：`uint16 → uint8` 可在已证无损时直接写，`as` 可用于显式写出同样的转换；
  真要截断用 `as: wrap`，要夹取用 `as: sat`。
- **循环累加需要证明每次写回安全**。对于计数器按固定步长前进、边界稳定的循环，编译器可以用次数和每轮增量推导累加范围。
- **守卫只收窄局部变量**：`if (m.sp < 16)` 不会收窄记录字段 `m.sp`，
  要先 `define sp: uint8 = m.sp;` 再判断。
- **记录不能整体赋值**：`s = observe(s, x)` 被拒绝，改用新名字。
- **记录返回/初始化要 `as: copy`**：`return r as: copy;`、`{ label: l as: copy }`。
- **多节点链不能整体转移所有权**：遍历函数只读借用，
  `free` 由调用方执行（`free` 会沿元图闭包一次释放）。
- **文本切片要写长度契约**：`ascii_string · {len: [0, 64]}`，否则 `.len` 的值域是
  `[0, 2147483647]`，转 `uint8` 会失败；返回文本切片要 `{from: 0}`。
- **`as: view` 要求同宽**：位切片宽度必须等于目标类型宽度。

## 写数值内核时的三条硬规则（28–33 总结）

1. **推不出累加范围时，循环头可能把累加器放宽到声明值域**。

   ```lan
   define s: int32[-20000000, 20000000] = 0;   // 声明域
   while (k < 4) { s = sat_wide(s + a * b); }  // 进循环时 s 已经是 [-2e7, 2e7]
   ```

   所以累加器的声明域必须能装下「自己 + 一步增量」，否则 `s += ...` 立刻报
   `cannot prove [...] fits [...]`。两种解法：
   - **每步饱和**：`s = sat(s + step)`（`31_matrix` 全部用这条）；
   - **条件收窄**：`while (k < 4 && s < 60000 && s > -60000)`，但会提前退出循环，
     只在"真值远小于上界"时安全。

2. **元素值域跟着类型走，所以操作数类型要选窄**。
   `int16[-4000, 4000]` 的元素两两相乘是 `[-1.6e7, 1.6e7]`，装得进 int32；
   如果操作数写成裸 `int32`，乘积是 `[-2^62, 2^62]`，什么都证不出来。
   `31_matrix` 用 `Mat = int16[-4000,4000][4][4]` + `Wide = int32[-2e7,2e7][4][4]`。

3. **守卫只收窄变量，不收窄表达式**。

   ```lan
   if (pos + jump < 64) { pos += jump; }   // ✗ pos + jump 仍按 [1, 64] 处理
   define nextp: uint8 = pos + jump;       // ✓ 先落到变量
   if (nextp < 64) { pos = nextp; }        //    守卫才收窄 nextp
   ```

   同理 `if (jump <= 63 - pos)` 也没用，必须落到变量上再判。

4. **`as` 不保留细化，会把值域放大到目标类型的全值域**。

   ```lan
   define cand: uint16 = (dp[c - w] as uint16) + 1;  // ✗ [0,65535]+1 溢出
   define cand: uint16 = dp[c - w] + 1;              // ✓ 元素本身就是 uint16[0,600]
   ```

   所以 DP 表的元素类型要直接写成 `uint16[0, 16]`，让值域一路带着走；
   或者把增量写进声明域（`uint16[0, 101]` 装 `dp[c-coin] + 1`），
   再用守卫收窄回目标域（`34_dp` / `35_dp2`）。

5. **`GAS_LIMIT = 100_000` 是按源文件算的，不是按函数**。
   `lanc.py` 的 `Compiler.__init__` 里 `self.gas = self.GAS_LIMIT` 只设一次、
   全程不重置，`spend()` 逐节点扣。所以函数多的文件即使每个函数都很简单，
   也会在后面某个普通循环上报 `analysis budget exhausted`。
   `34_dp` 一开始把 7 个 DP 算法写在一个文件里就撞上了这个墙，
   拆成 `34_dp`（LCS/编辑距离/LIS）+ `35_dp2`（背包/零钱/Kadane/路径）才过。

## 文本/下标相关补充

- **长度契约 `{len: [lo, hi]}` 只能保证 `lo` 个字节可访问**：`{len: [1, 16]}` 时
  只有下标 0 合法，`p[t]` 报 `cannot prove index [0, 15] is within {0}`。
  要按下标取字符，守卫必须直接写在 **`p.len` 字段**上（`while (t < pat.len)`），
  不能写在局部变量 `m = pat.len as uint8` 上。
- **切片 `s[lo, end)` 最好用 int32 游标**：证明器不会把 `uint8 n = s.len as uint8`
  和 `s.len` 联系起来，但认 `end = s.len` 这种直接赋值（见 `33_strings` 的 `trim`）。
- **外层循环里再套"数 run"的内层循环会耗尽分析预算**，改成单层循环 + 状态机
  （见 `33_strings` 的 `run_length_encode`）。

## 静态检查抓不到的错

编译器只保证**不越界、不溢出、不泄漏**，不保证**算法对**。`33_strings` 里
Rabin-Karp 首个窗口的循环漏写 `k < m`（写成 `k < s.len && k < 16`），把前 16 个
字符全哈希进去，下标全都合法、编译通过、断言才抓出来。所以每个用例都必须有
**独立算出的期望值**，不能拿"能跑通"当正确性证据。

## 已发现的编译器缺陷

1. **`widen_loop_head` 越界崩溃（已修复）**

   原触发条件：循环体里用循环变量初始化**堆记录**字段，同时循环条件或循环体里
   出现与**变量**的比较。

   ```lan
   define f(n: uint8[0, 8]) -> uint8 {
       define head: T · {storage: heap} = { pc: 0 };
       define tail: T · {pass: alias} = head;
       define i: uint8 = 0;
       while (i < n) {                              // 换成 i < 8 就好了
           define node: T · {storage: heap} = { pc: i };
           tail.next = node;
           tail = node;
           i += 1;
       }
       ...
   }
   ```

   原现象：`IndexError: list index out of range`，位置
   `widen_loop_head` 里 `self.symbols[uid - 1]`。原因是循环体保留的堆节点把
   局部绑定留在了下一轮 head 状态里，而对应符号表项已经回滚。现在每轮结束会把
   合流状态投影回循环入口 id，再作为下一轮 head，因此不再保留已回滚的 uid。
   `test_loop_body_may_initialize_a_heap_field_from_the_counter` 覆盖此回归。

2. **变长指令导致的路径爆炸**

   解释器里 `PUSH` 走 `pc += 2`、其余走 `pc += 1` 时，6 个分支各带一层嵌套
   `if` 会让循环证明预算直接耗尽（`analysis budget exhausted`）。
   改成定长两字节指令、`pc` 统一 `+2`，并把守卫折进分支条件
   （`else if (op == 2 && m.sp >= 2)`）即可收敛。

3. **分析预算按文件计（已修复）**

   原实现只在 `Compiler.__init__` 里设置一次 `self.gas = self.GAS_LIMIT`，
   `spend()` 每解析一个节点扣一次，全程不重置。所以大文件会"越写越紧"：
   前面几十个函数都过，后面一个三行的初始化循环突然报
   `analysis budget exhausted`。现在每个函数入口都会重置预算，预算仍然限制
   单个函数的分析规模，但不再让前面的函数挤占后面的函数。

4. **链遍历循环的预算**

   在同一个函数里既用字面量上界建链、又走一遍链遍历计数，会让
   `@oracle_request` 报预算耗尽。遍历循环放到独立函数、参数写
   `Node` 由编译器推导借用形状（见 `19_chains`）。
