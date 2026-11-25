"""Reproducible mixed workload and a separate bounded stress workload."""

from argparse import ArgumentParser
from collections import Counter
from random import Random
from time import perf_counter

from poly_intersection import PolynomialIntersection, solve_many


def mixed(count):
    for i in range(count):
        k = i // 6
        kind = i % 6
        if kind == 0:
            yield PolynomialIntersection((k, 6), (2*k, -14))
        elif kind == 1:
            yield PolynomialIntersection((2*k, 0, 101), (2*k+1, 0, 101))
        elif kind == 2:
            yield PolynomialIntersection((k, 0, 1), (k+1, 0, 2), 10**12+k)
        elif kind == 3:
            yield PolynomialIntersection((k, 0, 1), (k+1, 0, 1), k+2)
        elif kind == 4:
            yield PolynomialIntersection((k, 0, 0, 1), (k+1, 0, 1))
        else:
            yield PolynomialIntersection((-k, 0, 1), (100-k, 0, -1), 1-k, 99-k)


def stress(count):
    rng = Random(421)
    for _ in range(count):
        yield PolynomialIntersection(
            (rng.randint(-100, 100), rng.randint(-20, 20), rng.randrange(1, 80)),
            (rng.randint(-100, 100), rng.randint(-20, 20), rng.randrange(1, 80)),
            10**12,
        )


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=20_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--backend", choices=("builtin", "sympy"), default="builtin")
    parser.add_argument("--stress", action="store_true")
    args = parser.parse_args()
    jobs = stress(args.count) if args.stress else mixed(args.count)
    start = perf_counter()
    results = Counter()
    methods = Counter()
    for result in solve_many(jobs, workers=args.workers, max_steps=args.max_steps, backend=args.backend):
        results[result.status] += 1
        methods[result.method] += 1
    elapsed = perf_counter()-start
    print(f"{args.count:,} jobs in {elapsed:.3f}s; {sum(results.values())/elapsed:,.0f} jobs/s")
    print(f"statuses: {dict(results)}")
    print(f"methods: {dict(methods)}")


if __name__ == "__main__":
    main()
