"""Simple benchmark problems for EA discovery."""

import random
from typing import List, Callable


def onemax(bitstring: List[int]) -> float:
    """OneMax: count the number of 1s. Optimal = all 1s."""
    return float(sum(bitstring))


def leading_ones(bitstring: List[int]) -> float:
    """LeadingOnes: count consecutive 1s from the start. Deceptive."""
    count = 0
    for bit in bitstring:
        if bit == 1:
            count += 1
        else:
            break
    return float(count)


def trap(bitstring: List[int], trap_size: int = 5) -> float:
    """Trap function: deceptive - rewards all 0s more than partial solutions."""
    total = 0.0
    n = len(bitstring)

    for i in range(0, n, trap_size):
        block = bitstring[i:i+trap_size]
        ones = sum(block)

        if ones == trap_size:
            # Global optimum: all 1s
            total += trap_size
        else:
            # Local optimum: all 0s
            total += (trap_size - 1 - ones)

    return total


BENCHMARKS = {
    "onemax": onemax,
    "leading_ones": leading_ones,
    "trap": trap,
}


def get_problem(problem_name: str, problem_size: int = 20) -> Callable:
    """Get a benchmark problem function."""
    if problem_name not in BENCHMARKS:
        raise ValueError(f"Unknown problem: {problem_name}")

    base_func = BENCHMARKS[problem_name]

    def problem_func(bitstring: List[int]) -> float:
        if len(bitstring) != problem_size:
            raise ValueError(f"Expected bitstring of length {problem_size}")
        return base_func(bitstring)

    return problem_func


def get_optimal_fitness(problem_name: str, problem_size: int = 20) -> float:
    """Return the known optimal fitness for a problem."""
    if problem_name == "onemax":
        return float(problem_size)
    elif problem_name == "leading_ones":
        return float(problem_size)
    elif problem_name == "trap":
        # Optimal is all 1s
        num_blocks = problem_size // 5
        return float(num_blocks * 5)
    else:
        raise ValueError(f"Unknown problem: {problem_name}")
