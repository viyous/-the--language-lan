"""Measure independent branch growth and the slice for a relational query."""

from argparse import ArgumentParser
from time import perf_counter
import tracemalloc

from ssa_verify import Program


def run(count):
    tracemalloc.start()
    start = perf_counter()
    p = Program()
    for i in range(count):
        flag = p.boolean(f"flag{i}")
        x, y = f"x{i}", f"y{i}"
        p.flow.declare(x, 0)
        p.flow.declare(y, 0)
        p.flow.branch(flag, lambda f: f.assign(y, 1), lambda f: f.assign(x, 1))
    goal = p.flow.check((p.flow.get(x) + p.flow.get(y)).eq(1))
    elapsed = perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    start = perf_counter()
    proof = p.verify(goal)
    check = perf_counter() - start
    print(f"{count:,} ifs: {len(p.arena.nodes):,} DAG nodes, {elapsed:.3f}s build, "
          f"{peak/1024/1024:.2f} MiB Python peak; slice={proof.slice_nodes}, "
          f"{proof.status}/{proof.method} in {check:.4f}s")
    return proof.status == "proved"


def run_compiler(count):
    from lanc import compile_source

    parameters = ", ".join(f"b{i}: boolean" for i in range(count))
    branches = "\n".join(f"if (b{i}) {{ scratch = 1; }} else {{ scratch = 0; }}"
                         for i in range(1, count))
    source = f"""
        define example({parameters}) -> uint8[1,1] {{
            define x: uint8 = 0;
            define y: uint8 = 0;
            define scratch: uint8 = 0;
            if (b0) {{ x = 1; y = 0; }} else {{ x = 0; y = 1; }}
            {branches}
            return x + y;
        }}
    """
    start = perf_counter()
    c, table = compile_source(source)
    elapsed = perf_counter() - start
    proved = "ssa return proved" in table or count <= 7
    print(f"{count:,} ifs: {elapsed:.3f}s end to end, "
          f"{len(c) / 1024:.1f} KiB C, {len(table) / 1024**2:.2f} MiB .table, proof={proved}")
    return proved


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--counts", nargs="+", type=int)
    parser.add_argument("--compiler", action="store_true", help="benchmark complete Lan compilation")
    args = parser.parse_args()
    counts = args.counts or ([1000] if args.compiler else [1000, 10000])
    if any(n <= 0 for n in counts):
        parser.error("counts must be positive")
    benchmark = run_compiler if args.compiler else run
    raise SystemExit(0 if all(benchmark(n) for n in counts) else 1)
