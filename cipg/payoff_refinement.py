"""
Per-Coalition Solution Concepts and Payoff Refinement

Given a coalition structure (e.g., the OCSS returned by the solvers), this
module computes refined payoff allocations for each formed coalition c:

  - Shapley value of the restricted game (c, V_c), via exact enumeration
  - Core membership check for the Shapley value
  - Nucleolus of (c, V_c), via the standard sequential LP scheme
    (Maschler-Peleg-Shapley), solved with Gurobi

The characteristic values V(s) for all s subseteq c are computed with the
coalition-level IP (solve_coalition_value) and cached, requiring at most
2^|c| small IP solves per coalition.
"""

import math
import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from .instance import CKGInstance
from .coalition_value import solve_coalition_value, CoalitionValueCache

TOL = 1e-6


# ---------------------------------------------------------------------------
# Subset value enumeration
# ---------------------------------------------------------------------------

def compute_all_subset_values(
    inst: CKGInstance,
    coalition: Tuple[int, ...],
    alpha: float,
    cache: Optional[CoalitionValueCache] = None,
) -> Dict[FrozenSet[int], float]:
    """
    Compute V(s) for every nonempty subset s of `coalition` using the
    coalition-level IP (same alpha-restricted policy as the master problem).

    Returns a dict frozenset -> value, including the empty set (value 0).
    """
    members = sorted(coalition)
    values: Dict[FrozenSet[int], float] = {frozenset(): 0.0}
    for k in range(1, len(members) + 1):
        for combo in combinations(members, k):
            values[frozenset(combo)] = solve_coalition_value(
                inst, tuple(combo), alpha, cache
            )
    return values


# ---------------------------------------------------------------------------
# Shapley value
# ---------------------------------------------------------------------------

def shapley_value(
    values: Dict[FrozenSet[int], float],
    coalition: Tuple[int, ...],
) -> Dict[int, float]:
    """
    Exact Shapley value of the restricted game (coalition, V), where
    `values` contains V(s) for all s subseteq coalition.

    phi_i = sum_{s subseteq c\\{i}} |s|! (|c|-1-|s|)! / |c|! * [V(s+i) - V(s)]
    """
    members = sorted(coalition)
    k = len(members)
    fact = [math.factorial(m) for m in range(k + 1)]
    phi = {i: 0.0 for i in members}

    for i in members:
        others = [j for j in members if j != i]
        for m in range(0, k):
            w = fact[m] * fact[k - 1 - m] / fact[k]
            for combo in combinations(others, m):
                s = frozenset(combo)
                phi[i] += w * (values[s | {i}] - values[s])
    return phi


# ---------------------------------------------------------------------------
# Core membership
# ---------------------------------------------------------------------------

def core_violation(
    payoff: Dict[int, float],
    values: Dict[FrozenSet[int], float],
    coalition: Tuple[int, ...],
) -> Tuple[float, Optional[Tuple[int, ...]]]:
    """
    Maximum Core violation of `payoff` in the restricted game:

        max over proper nonempty s of  V(s) - payoff(s).

    A nonpositive value (within tolerance) means `payoff` is in the Core
    (efficiency is assumed to hold by construction).
    Returns (max_violation, worst_subset).
    """
    members = sorted(coalition)
    worst_v, worst_s = -math.inf, None
    for k in range(1, len(members)):
        for combo in combinations(members, k):
            viol = values[frozenset(combo)] - sum(payoff[i] for i in combo)
            if viol > worst_v:
                worst_v, worst_s = viol, combo
    if worst_s is None:  # singleton coalition: no proper nonempty subsets
        worst_v = 0.0
    return worst_v, worst_s


# ---------------------------------------------------------------------------
# Nucleolus (sequential LP scheme, Gurobi backend)
# ---------------------------------------------------------------------------

def nucleolus(
    values: Dict[FrozenSet[int], float],
    coalition: Tuple[int, ...],
    max_rounds: int = 100,
) -> Tuple[Dict[int, float], float]:
    """
    Nucleolus of the restricted game (coalition, V) via the standard
    sequential LP scheme:

      Round r: minimize the maximum excess e(s, v) = V(s) - v(s) over the
      remaining free subsets, subject to previously fixed excess levels and
      efficiency v(c) = V(c). Subsets whose excess is at the optimal level
      in EVERY optimal solution (verified by an auxiliary slack-test LP)
      are fixed, and the procedure recurses.

    Returns (payoff dict, least_core_eps) where least_core_eps is the
    optimal epsilon of the first round; least_core_eps <= 0 iff the Core
    is nonempty.
    """
    members = sorted(coalition)
    k = len(members)
    idx = {i: p for p, i in enumerate(members)}
    grand = frozenset(members)

    if k == 1:
        return {members[0]: values[grand]}, 0.0

    proper = [
        frozenset(c)
        for m in range(1, k)
        for c in combinations(members, m)
    ]

    def ind(s) -> np.ndarray:
        row = np.zeros(k)
        for i in s:
            row[idx[i]] = 1.0
        return row

    free: List[FrozenSet[int]] = list(proper)
    fixed: List[Tuple[FrozenSet[int], float]] = []
    least_core_eps: Optional[float] = None
    v_star: Optional[np.ndarray] = None

    for _ in range(max_rounds):
        # ---- main LP: min eps  s.t.  v(s) >= V(s) - eps  (s free),
        #      v(s) = V(s) - eps_r  (s fixed),  v(grand) = V(grand)
        m = gp.Model("nucleolus_main")
        m.Params.OutputFlag = 0
        v = m.addVars(k, lb=-GRB.INFINITY, name="v")
        eps = m.addVar(lb=-GRB.INFINITY, name="eps")

        for s in free:
            m.addConstr(
                gp.quicksum(v[idx[i]] for i in s) >= values[s] - eps
            )
        m.addConstr(gp.quicksum(v[p] for p in range(k)) == values[grand])
        for s, eps_r in fixed:
            m.addConstr(
                gp.quicksum(v[idx[i]] for i in s) == values[s] - eps_r
            )

        m.setObjective(eps, GRB.MINIMIZE)
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(
                f"LP failed in nucleolus computation (status {m.Status})"
            )
        v_star = np.array([v[p].X for p in range(k)])
        eps_star = float(eps.X)
        if least_core_eps is None:
            least_core_eps = eps_star

        # ---- identify subsets tight in every optimal solution
        candidates = [
            s for s in free
            if values[s] - float(ind(s) @ v_star) >= eps_star - 1e-7
        ]
        newly_fixed = []
        for t in candidates:
            # maximize v(t) subject to all constraints with eps = eps_star;
            # if v(t) cannot exceed V(t) - eps_star, then t is tight in
            # every optimum.
            m2 = gp.Model("nucleolus_slack")
            m2.Params.OutputFlag = 0
            w = m2.addVars(k, lb=-GRB.INFINITY, name="v")

            for s in free:
                m2.addConstr(
                    gp.quicksum(w[idx[i]] for i in s) >= values[s] - eps_star
                )
            m2.addConstr(
                gp.quicksum(w[p] for p in range(k)) == values[grand]
            )
            for s, eps_r in fixed:
                m2.addConstr(
                    gp.quicksum(w[idx[i]] for i in s) == values[s] - eps_r
                )

            m2.setObjective(
                gp.quicksum(w[idx[i]] for i in t), GRB.MAXIMIZE
            )
            m2.optimize()
            if m2.Status != GRB.OPTIMAL:
                raise RuntimeError(
                    f"Slack-test LP failed (status {m2.Status})"
                )
            if m2.ObjVal <= values[t] - eps_star + 1e-7:
                newly_fixed.append(t)

        if not newly_fixed:
            # Numerical guard: fix all candidates to guarantee progress.
            newly_fixed = candidates

        for t in newly_fixed:
            free.remove(t)
            fixed.append((t, float(eps_star)))

        # ---- termination: payoff uniquely determined by the equality system
        eq_rows = [ind(grand)] + [ind(s) for s, _ in fixed]
        if np.linalg.matrix_rank(np.array(eq_rows)) >= k or not free:
            break

    payoff = {i: float(v_star[idx[i]]) for i in members}
    return payoff, float(least_core_eps)


# ---------------------------------------------------------------------------
# Per-coalition refinement driver
# ---------------------------------------------------------------------------

@dataclass
class CoalitionRefinement:
    """Refined payoffs for one formed coalition."""
    coalition: Tuple[int, ...]
    size: int
    V_c: float
    shapley: Dict[int, float]
    shapley_in_core: bool
    shapley_max_violation: float
    nucleolus: Dict[int, float]
    least_core_eps: float
    n_subsets: int
    runtime_values: float
    runtime_shapley: float
    runtime_nucleolus: float


@dataclass
class RefinementResult:
    """Refinement of a full coalition structure."""
    coalitions: List[CoalitionRefinement] = field(default_factory=list)
    skipped: List[Tuple[int, ...]] = field(default_factory=list)

    @property
    def total_value(self) -> float:
        return sum(cr.V_c for cr in self.coalitions)


def refine_coalition(
    inst: CKGInstance,
    coalition: Tuple[int, ...],
    alpha: float,
    cache: Optional[CoalitionValueCache] = None,
) -> CoalitionRefinement:
    """Compute Shapley and nucleolus for one formed coalition."""
    members = tuple(sorted(coalition))

    t0 = time.time()
    values = compute_all_subset_values(inst, members, alpha, cache)
    t_values = time.time() - t0
    V_c = values[frozenset(members)]

    t0 = time.time()
    phi = shapley_value(values, members)
    t_shapley = time.time() - t0
    viol, _ = core_violation(phi, values, members)

    t0 = time.time()
    nuc, eps = nucleolus(values, members)
    t_nucleolus = time.time() - t0

    return CoalitionRefinement(
        coalition=members,
        size=len(members),
        V_c=V_c,
        shapley=phi,
        shapley_in_core=(viol <= TOL * max(1.0, abs(V_c))),
        shapley_max_violation=max(0.0, viol),
        nucleolus=nuc,
        least_core_eps=eps,
        n_subsets=2 ** len(members) - 1,
        runtime_values=t_values,
        runtime_shapley=t_shapley,
        runtime_nucleolus=t_nucleolus,
    )


def refine_coalition_structure(
    inst: CKGInstance,
    coalition_structure: List[Tuple[int, ...]],
    alpha: float,
    cache: Optional[CoalitionValueCache] = None,
    max_size: int = 14,
    verbose: bool = False,
) -> RefinementResult:
    """
    Refine payoffs for every coalition in a coalition structure (e.g., the
    OCSS partition). Coalitions larger than `max_size` are skipped (2^|c|
    subset IPs would be required) and recorded in `result.skipped`.
    """
    if cache is None:
        cache = CoalitionValueCache()
    result = RefinementResult()

    for c in coalition_structure:
        if len(c) > max_size:
            result.skipped.append(tuple(sorted(c)))
            if verbose:
                print(f"  skipped {tuple(sorted(c))} (size {len(c)} > {max_size})")
            continue
        cr = refine_coalition(inst, c, alpha, cache)
        result.coalitions.append(cr)
        if verbose:
            tag = "Shapley in Core" if cr.shapley_in_core else \
                  f"Shapley NOT in Core (viol {cr.shapley_max_violation:.2f})"
            print(
                f"  {cr.coalition}: V={cr.V_c:.0f}, {tag}, "
                f"least-core eps={cr.least_core_eps:.2f}, "
                f"({cr.runtime_values:.1f}s values, "
                f"{cr.runtime_nucleolus:.1f}s nucleolus)"
            )
    return result
