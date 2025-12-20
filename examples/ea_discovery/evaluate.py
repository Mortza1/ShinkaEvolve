"""Evaluation function for EA discovery task."""

import sys
import json
import time
import importlib.util
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from benchmarks import get_problem, get_optimal_fitness

logger = logging.getLogger(__name__)

# Configuration
PROBLEM_SIZE = 20
MAX_EVALUATIONS = 500
NUM_TEST_SEEDS = 5
TIMEOUT_SECONDS = 30


def load_ea_program(program_path: str):
    """Load the EA program from a file."""
    spec = importlib.util.spec_from_file_location("ea_program", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load EA program from {program_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "run_ea"):
        raise AttributeError("EA program must define run_ea(problem_seed) function")

    return module.run_ea


def inject_fitness_function(problem_name: str):
    """
    Inject the fitness function into the EA's namespace.
    The EA can call FITNESS_FUNCTION(bitstring) to evaluate solutions.
    """
    problem_func = get_problem(problem_name, PROBLEM_SIZE)

    # Make it available globally so evolved EAs can access it
    import builtins
    builtins.FITNESS_FUNCTION = problem_func  # type: ignore
    builtins.PROBLEM_SIZE = PROBLEM_SIZE  # type: ignore
    builtins.MAX_EVALUATIONS = MAX_EVALUATIONS  # type: ignore


def evaluate_ea_on_problem(run_ea_func, problem_name: str, seed: int) -> Dict[str, float]:
    """Evaluate an EA on a single problem instance."""
    inject_fitness_function(problem_name)
    optimal = get_optimal_fitness(problem_name, PROBLEM_SIZE)

    try:
        # Run the EA
        result = run_ea_func(seed)

        # Extract metrics
        best_fitness = result.get("best_fitness", 0.0)
        diversity = result.get("diversity", 0.0)
        evaluations = result.get("evaluations", MAX_EVALUATIONS)

        # Normalize fitness (0-1 scale)
        normalized_fitness = min(best_fitness / optimal, 1.0) if optimal > 0 else 0.0

        # Normalize diversity (higher is better, cap at 1.0)
        normalized_diversity = min(diversity, 1.0)

        # Efficiency (fewer evals is better)
        efficiency = 1.0 - (evaluations / MAX_EVALUATIONS)

        return {
            "best_fitness": best_fitness,
            "normalized_fitness": normalized_fitness,
            "diversity": diversity,
            "normalized_diversity": normalized_diversity,
            "evaluations": evaluations,
            "efficiency": efficiency,
        }

    except Exception as e:
        logger.error(f"EA failed on {problem_name} seed {seed}: {e}")
        return {
            "best_fitness": 0.0,
            "normalized_fitness": 0.0,
            "diversity": 0.0,
            "normalized_diversity": 0.0,
            "evaluations": MAX_EVALUATIONS,
            "efficiency": 0.0,
        }


def evaluate_program(
    program_path: str,
    results_dir: str,
    num_samples: int = NUM_TEST_SEEDS,
    **kwargs
) -> Dict[str, Any]:
    """
    Main evaluation function for EA discovery.

    Runs the EA on multiple problems and seeds, aggregates metrics.
    """
    start_time = time.time()

    # Load the EA
    try:
        run_ea_func = load_ea_program(program_path)
    except Exception as e:
        logger.error(f"Failed to load EA program: {e}")

        # Save error metrics
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)

        metrics = {
            "runtime": time.time() - start_time,
            "public": {"combined_score": 0.0},
            "private": {},
            "combined_score": 0.0,
        }

        with open(results_path / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        with open(results_path / "correct.json", "w") as f:
            json.dump({"correct": False, "error": str(e)}, f, indent=2)

        return {
            "runtime": time.time() - start_time,
            "public": {"combined_score": 0.0},
            "private": {},
            "combined_score": 0.0,
        }

    # Problems to test
    problems = ["onemax", "leading_ones"]

    # Test seeds
    test_seeds = list(range(num_samples))

    # Collect results
    all_results = []

    for problem_name in problems:
        for seed in test_seeds:
            result = evaluate_ea_on_problem(run_ea_func, problem_name, seed)
            result["problem"] = problem_name
            result["seed"] = seed
            all_results.append(result)

    # Aggregate metrics
    fitness_scores = [r["normalized_fitness"] for r in all_results]
    diversity_scores = [r["normalized_diversity"] for r in all_results]
    efficiency_scores = [r["efficiency"] for r in all_results]

    mean_fitness = np.mean(fitness_scores)
    mean_diversity = np.mean(diversity_scores)
    mean_efficiency = np.mean(efficiency_scores)

    # Robustness = low variance across seeds
    robustness = 1.0 - min(np.std(fitness_scores), 1.0)

    # Combined score (geometric mean to force balance)
    combined_score = (mean_fitness * mean_diversity * robustness) ** (1/3)

    runtime = time.time() - start_time

    # Save detailed results
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    with open(results_path / "detailed_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    public_metrics = {
        "mean_fitness": mean_fitness,
        "mean_diversity": mean_diversity,
        "mean_efficiency": mean_efficiency,
        "robustness": robustness,
        "combined_score": combined_score,
    }

    metrics = {
        "runtime": runtime,
        "public": public_metrics,
        "private": {},
        "combined_score": combined_score,
    }

    # Save metrics.json (required by framework)
    with open(results_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save correct.json (required by framework)
    with open(results_path / "correct.json", "w") as f:
        json.dump({"correct": True, "error": ""}, f, indent=2)

    return metrics
