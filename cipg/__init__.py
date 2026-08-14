"""
Cooperative Integer Programming Games (CIPG)

A framework for computing optimal coalition structures with stability
in cooperative knapsack games.

Main components:
- instance: CKGInstance definition and generation
- coalition_value: Compute V(S) for any coalition S
- solver_enumerate: Baseline cutting-plane with full enumeration
- solver_lazy: Gurobi lazy constraint callbacks (faster)
"""

__version__ = "1.0.0"

from .instance import CKGInstance, generate_instance
from .coalition_value import solve_coalition_value, CoalitionValueCache
from .ocs_model import build_OCS_model, extract_solution
from .solver_enumerate import solve_OCS, solve_OCSS_enumerate, OCSSResult
from .solver_lazy import solve_OCSS_lazy, LazyOCSSResult
from .heuristic_css import heuristic_css, CSSHeuristicResult
from .payoff_refinement import (
    shapley_value,
    nucleolus,
    core_violation,
    refine_coalition,
    refine_coalition_structure,
    CoalitionRefinement,
    RefinementResult,
)

__all__ = [
    # Instance
    "CKGInstance",
    "generate_instance",
    # Coalition values
    "solve_coalition_value",
    "CoalitionValueCache",
    # OCS model
    "build_OCS_model",
    "extract_solution",
    # Solvers
    "solve_OCS",
    "solve_OCSS_enumerate",
    "solve_OCSS_lazy",
    "OCSSResult",
    "LazyOCSSResult",
    # CSS-feasible heuristic
    "heuristic_css",
    "CSSHeuristicResult",
    # Payoff refinement
    "shapley_value",
    "nucleolus",
    "core_violation",
    "refine_coalition",
    "refine_coalition_structure",
    "CoalitionRefinement",
    "RefinementResult",
]
