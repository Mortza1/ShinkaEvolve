"""Initial EA - intentionally trivial baseline."""

from typing import List
import random


# EVOLVE-BLOCK-START
def run_ea(problem_seed: int) -> dict:
    """
    Run an evolutionary algorithm on a bitstring optimization problem.

    Args:
        problem_seed: Random seed for problem instance

    Returns:
        dict with metrics: best_fitness, diversity, evaluations
    """
    import random

    # Get problem parameters (injected by evaluation framework)
    try:
        fitness_function = FITNESS_FUNCTION  # type: ignore
        problem_size = PROBLEM_SIZE  # type: ignore
        max_evals = MAX_EVALUATIONS  # type: ignore
    except NameError:
        # Fallback for testing - shouldn't happen in actual evolution
        problem_size = 20
        max_evals = 500
        def fitness_function(x):
            return float(sum(x))

    # Set seed for reproducibility
    random.seed(problem_seed)

    # Simple baseline EA: random search with population
    population_size = 10
    evaluations = 0

    # Initialize random population
    population = []
    fitnesses = []
    for _ in range(population_size):
        individual = [random.randint(0, 1) for _ in range(problem_size)]
        fitness = fitness_function(individual)
        population.append(individual)
        fitnesses.append(fitness)
        evaluations += 1

    # Simple evolution: just keep generating random solutions
    while evaluations < max_evals:
        # Generate new random individual
        new_individual = [random.randint(0, 1) for _ in range(problem_size)]
        new_fitness = fitness_function(new_individual)
        evaluations += 1

        # Replace worst if better
        worst_idx = fitnesses.index(min(fitnesses))
        if new_fitness > fitnesses[worst_idx]:
            population[worst_idx] = new_individual
            fitnesses[worst_idx] = new_fitness

    # Calculate diversity (Hamming distance between individuals)
    diversity_sum = 0.0
    count = 0
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            hamming = sum(a != b for a, b in zip(population[i], population[j]))
            diversity_sum += hamming / problem_size
            count += 1

    diversity = diversity_sum / count if count > 0 else 0.0

    return {
        "best_fitness": max(fitnesses),
        "diversity": diversity,
        "evaluations": evaluations,
    }
# EVOLVE-BLOCK-END


def run_experiment(random_seeds: List[int]) -> List[dict]:
    """Run EA on multiple seeds and return results."""
    results = []
    for seed in random_seeds:
        result = run_ea(seed)
        results.append(result)
        print(f"Seed {seed}: fitness={result['best_fitness']:.2f}, "
              f"diversity={result['diversity']:.2f}, evals={result['evaluations']}")
    return results
