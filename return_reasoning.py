"""Logical consequences of already-checked scalar return paths."""

from dataclasses import dataclass

from value_domain import Domain


@dataclass(frozen=True)
class ReturnCase:
    result: Domain
    arguments: tuple[Domain, ...]


@dataclass(frozen=True)
class ReturnObservation:
    parameters: tuple[int, ...]
    cases: tuple[ReturnCase, ...]


def return_implications(observation):
    """Yield (premises, conclusion) for executions that return."""
    cases = observation.cases
    if len(cases) < 2:
        return
    arguments = tuple(Domain.union(row.arguments[i] for row in cases)
                      for i in range(len(observation.parameters)))
    results = Domain.union(row.result for row in cases)
    result_name = ("result",)
    names = tuple(("parameter", index) for index in observation.parameters)

    seen = set()
    for row in cases:
        if row.result != results:
            premise = ((result_name, row.result),)
            if premise not in seen:
                seen.add(premise)
                overlapping = [case for case in cases if case.result & row.result]
                for i, name in enumerate(names):
                    restricted = Domain.union(case.arguments[i] for case in overlapping)
                    if restricted and restricted != arguments[i]:
                        yield premise, (name, restricted)

        narrowed = [(i, value) for i, value in enumerate(row.arguments) if value != arguments[i]]
        # Try individual tests and then their conjunction. Marginals that
        # overlap do not rule a path out unless one whole premise is disjoint.
        tests = [((i, value),) for i, value in narrowed]
        if len(narrowed) > 1:
            tests.append(tuple(narrowed))
        for test in tests:
            premise = tuple((names[i], value) for i, value in test)
            if premise in seen:
                continue
            seen.add(premise)
            restricted = Domain.union(case.result for case in cases
                                      if all(case.arguments[i] & value for i, value in test))
            if restricted and restricted != results:
                yield premise, (result_name, restricted)
            if len(seen) >= 32:
                return
