"""LLM judge prompts for EA discovery."""

import re
from typing import Dict, Any

# System prompt for the judge
SYSTEM_PROMPT = """You are an expert in evolutionary algorithms and metaheuristic optimization.

Your task is to evaluate whether an evolved EA represents a meaningful algorithmic innovation.

You will see:
1. The EA's source code
2. Performance metrics across multiple problem instances

Evaluate the EA on three dimensions:

1. **Algorithmic Sophistication** (0.0 to 1.0):
   - Does it use meaningful evolutionary mechanisms?
   - Look for: selection schemes, variation operators, population management
   - Penalize: trivial random search, hardcoded solutions

2. **Generalization** (0.0 to 1.0):
   - Does it perform consistently across different problems?
   - Look for: problem-agnostic mechanisms, adaptive behaviors
   - Penalize: overfitting to specific problem characteristics

3. **Novelty** (0.0 to 1.0):
   - Does it represent an interesting algorithmic pattern?
   - Look for: unusual selection, adaptive parameters, hybrid approaches
   - Penalize: standard textbook EAs with minor variations

**Critical**: Ignore cosmetic differences (variable names, comments, formatting).
Focus on ALGORITHMIC MECHANISMS.

Be strict. Only score > 0.5 in each dimension for truly impressive work.

Your final EA score is the product: sophistication × generalization × novelty
"""


def make_user_message(metrics: Dict[str, float]) -> str:
    """Create user message showing EA performance metrics."""
    return f"""
Here are the EA's performance metrics across test problems:

**Performance Summary:**
- Mean Fitness (normalized): {metrics.get('mean_fitness', 0.0):.3f}
- Mean Diversity: {metrics.get('mean_diversity', 0.0):.3f}
- Mean Efficiency: {metrics.get('mean_efficiency', 0.0):.3f}
- Robustness (1 - variance): {metrics.get('robustness', 0.0):.3f}
- Combined Score: {metrics.get('combined_score', 0.0):.3f}

**Evaluation:**
Please assess this EA and provide scores in this EXACT format:

algorithmic_sophistication: <score between 0.0 and 1.0>
generalization: <score between 0.0 and 1.0>
novelty: <score between 0.0 and 1.0>
ea_score: <product of the three scores>

Provide a brief explanation after the scores.
"""


def extract_scores(llm_response: str) -> Dict[str, float]:
    """Extract numeric scores from LLM response."""
    scores = {}

    patterns = {
        "algorithmic_sophistication": r"algorithmic[_\s]sophistication:\s*([0-9.]+)",
        "generalization": r"generalization:\s*([0-9.]+)",
        "novelty": r"novelty:\s*([0-9.]+)",
        "ea_score": r"ea[_\s]score:\s*([0-9.]+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, llm_response, re.IGNORECASE)
        if match:
            try:
                scores[key] = float(match.group(1))
            except ValueError:
                scores[key] = 0.0
        else:
            scores[key] = 0.0

    # Ensure ea_score is product of three dimensions
    if scores["ea_score"] == 0.0:
        scores["ea_score"] = (
            scores["algorithmic_sophistication"]
            * scores["generalization"]
            * scores["novelty"]
        )

    return scores


def make_lm_input_and_output_processors(number_of_samples: int = 5):
    """Create input/output processors for LLM judge."""

    def input_processor(outputs: list, code: str) -> tuple:
        """Process EA outputs and code into judge prompt."""
        # outputs is a list of metric dicts from evaluate.py
        # Aggregate them
        if not outputs:
            metrics = {
                "mean_fitness": 0.0,
                "mean_diversity": 0.0,
                "mean_efficiency": 0.0,
                "robustness": 0.0,
                "combined_score": 0.0,
            }
        else:
            # Take the last evaluation's public metrics
            metrics = outputs[0].get("public", {})

        user_msg = make_user_message(metrics)

        # Add code snippet
        user_msg += f"\n\n**EA Source Code:**\n```python\n{code}\n```"

        return SYSTEM_PROMPT, user_msg

    def output_processor(llm_response: str) -> dict:
        """Extract scores from LLM judge response."""
        scores = extract_scores(llm_response)

        return {
            "judge1_sophistication_score": scores["algorithmic_sophistication"],
            "judge1_generalization_score": scores["generalization"],
            "judge1_novelty_score": scores["novelty"],
            "judge1_ea_score": scores["ea_score"],
            "combined_score": scores["ea_score"],
        }

    return input_processor, output_processor
