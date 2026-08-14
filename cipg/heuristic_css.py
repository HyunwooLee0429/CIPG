"""
CSS-Feasible Primal Heuristic

Two-phase heuristic: Phase 1 builds a pool of Core-feasible coalitions
bottom-up (merges accepted only if the merged Core is nonempty); Phase 2
solves a set partitioning IP over the pool. Every coalition in the
returned partition has a nonempty Core, so the output is CSS-feasible
by construction.
"""

import itertools
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

import gurobipy as gp
from gurobipy import GRB

from .instance import CKGInstance
from .coalition_value import solve_coalition_value, CoalitionValueCache


@dataclass
class CSSHeuristicResult:
    """Result from the CSS-feasible heuristic."""
    total_value: float
    coalitions: List[Tuple[int, ...]]
    payoffs: Dict[int, float]
    runtime: float
    method: str
    status: str                         # "completed" or "timeout"
    pool_size: int = 0                  # Number of Core-feasible coalitions in pool
    phase1_time: float = 0.0            # Time spent in Phase 1
    phase2_time: float = 0.0            # Time spent in Phase 2
    core_checks: int = 0               # Total Core feasibility checks performed
    core_feasible: int = 0             # Number that passed (nonempty Core)
    n_value_evals: int = 0             # Distinct coalition values V(s) computed


# ---------------------------------------------------------------------------
# Core feasibility check
# ---------------------------------------------------------------------------

def check_core_feasibility(
    inst: CKGInstance,
    coalition: Tuple[int, ...],
    alpha: float,
    cache: CoalitionValueCache,
    enum_threshold: int = 9,
) -> Tuple[bool, Optional[Dict[int, float]]]:
    """
    Check whether a coalition has a nonempty Core.

    Solves the Core LP:
        find  v_i >= 0
        s.t.  sum_{i in c} v_i = V(c)             [efficiency]
              sum_{i in s} v_i >= V(s)  for all proper nonempty s subset c  [stability]

    For |c| <= enum_threshold (default 9, matching the lazy solver):
        Full enumeration of all 2^|c| - 2 subcoalitions.

    For |c| > enum_threshold:
        Iterative cutting-plane with MIP-based separation. Solves the Core LP,
        then uses the separation oracle to find the most violated subcoalition.
        If violated, adds the constraint and re-solves. Repeats until no
        violation is found (exact) or the LP becomes infeasible (Core is empty).

    Returns:
        (is_feasible, payoffs)  where payoffs is a Core allocation if feasible,
        or (False, None) if the Core is empty.
    """
    if len(coalition) == 1:
        i = coalition[0]
        V_i = solve_coalition_value(inst, coalition, alpha, cache)
        return True, {i: V_i}

    V_c = solve_coalition_value(inst, coalition, alpha, cache)

    if len(coalition) <= enum_threshold:
        return _core_check_enumerate(inst, coalition, alpha, cache, V_c)
    else:
        return _core_check_separation(inst, coalition, alpha, cache, V_c)


def _core_check_enumerate(
    inst: CKGInstance,
    coalition: Tuple[int, ...],
    alpha: float,
    cache: CoalitionValueCache,
    V_c: float,
) -> Tuple[bool, Optional[Dict[int, float]]]:
    """Core check via full enumeration of all subcoalitions."""
    m = gp.Model("core_enum")
    m.Params.OutputFlag = 0
    m.Params.Threads = 1

    v = m.addVars(coalition, lb=0.0, name="v")
    m.addConstr(gp.quicksum(v[i] for i in coalition) == V_c, "efficiency")

    for r in range(1, len(coalition)):
        for s in itertools.combinations(coalition, r):
            V_s = solve_coalition_value(inst, tuple(s), alpha, cache)
            m.addConstr(
                gp.quicksum(v[i] for i in s) >= V_s,
                name=f"stab_{'_'.join(map(str, s))}"
            )

    m.setObjective(gp.quicksum(v[i] for i in coalition), GRB.MAXIMIZE)
    m.optimize()

    if m.Status == GRB.OPTIMAL:
        return True, {i: v[i].X for i in coalition}
    return False, None


def _core_check_separation(
    inst: CKGInstance,
    coalition: Tuple[int, ...],
    alpha: float,
    cache: CoalitionValueCache,
    V_c: float,
    max_rounds: int = 100,
    tol: float = 1e-6,
) -> Tuple[bool, Optional[Dict[int, float]]]:
    """
    Core check via iterative cutting-plane with MIP-based separation.

    Starts with singleton constraints only, then iteratively:
    1. Solve the Core LP to get candidate payoffs v.
    2. Solve the separation MIP: max_{s subset c} V(s) - sum_{i in s} v_i.
    3. If violation > tol, add the constraint and go to 1.
    4. If no violation, v is in the Core.
    """
    m = gp.Model("core_sep")
    m.Params.OutputFlag = 0
    m.Params.Threads = 1

    v = m.addVars(coalition, lb=0.0, name="v")
    m.addConstr(gp.quicksum(v[i] for i in coalition) == V_c, "efficiency")

    # Start with singleton constraints
    for i in coalition:
        V_i = solve_coalition_value(inst, (i,), alpha, cache)
        m.addConstr(v[i] >= V_i, name=f"stab_{i}")

    m.setObjective(gp.quicksum(v[i] for i in coalition), GRB.MAXIMIZE)

    for _ in range(max_rounds):
        m.optimize()

        if m.Status != GRB.OPTIMAL:
            # LP infeasible => Core is empty
            return False, None

        v_val = {i: v[i].X for i in coalition}

        # Separation: find most violated subcoalition
        blocking, V_s, violation = _separation_oracle(
            inst, coalition, alpha, v_val, cache
        )

        if blocking is None or violation <= tol:
            # No violation => v is in the Core
            return True, v_val

        # Add violated constraint
        s_key = '_'.join(map(str, blocking))
        m.addConstr(
            gp.quicksum(v[i] for i in blocking) >= V_s,
            name=f"sep_{s_key}"
        )

    # Exhausted max_rounds without convergence — conservatively say infeasible
    return False, None


def _separation_oracle(
    inst: CKGInstance,
    coalition: Tuple[int, ...],
    alpha: float,
    v_val: Dict[int, float],
    cache: CoalitionValueCache,
) -> Tuple[Optional[Tuple[int, ...]], float, float]:
    """
    MIP-based separation oracle for Core feasibility.

    Delegates to the shared exact oracle in solver_lazy
    (find_most_violated_subset), so that the heuristic's Core checks and
    the lazy solver callback run identical, exact separation: integrality
    on individual-item x, commonness indicators for the alpha-restricted
    bound, and exact V(s) recomputation. The certificate direction
    ('no violated subset' => Core allocation found) is then reliable,
    which is what makes the heuristic's SCS-feasibility guarantee sound.

    Returns:
        (blocking_coalition, V_s, violation) or (None, 0, 0) if no violation.
    """
    from .solver_lazy import find_most_violated_subset
    return find_most_violated_subset(inst, coalition, alpha, v_val, cache)


# ---------------------------------------------------------------------------
# Phase 1: Core-Verified Bottom-Up
# ---------------------------------------------------------------------------

def _phase1_core_verified_bottom_up(
    inst: CKGInstance,
    alpha: float,
    cache: CoalitionValueCache,
    deadline: Optional[float] = None,
) -> Tuple[List[Tuple[int, ...]], Dict[Tuple[int, ...], float],
           Dict[Tuple[int, ...], Dict[int, float]], int, int]:
    """
    Phase 1: Build coalitions bottom-up, only accepting Core-feasible merges.

    Returns:
        pool_coalitions:  list of all Core-feasible coalitions found
        pool_values:      {coalition: V(c)} for each pooled coalition
        pool_payoffs:     {coalition: {i: v_i}} for each pooled coalition
        core_checks:      number of Core checks performed
        core_feasible:    number of checks that passed
    """
    N = inst.players

    # Singletons are always Core-feasible
    pool_coalitions: List[Tuple[int, ...]] = []
    pool_values: Dict[Tuple[int, ...], float] = {}
    pool_payoffs: Dict[Tuple[int, ...], Dict[int, float]] = {}

    for i in N:
        c = (i,)
        V_c = solve_coalition_value(inst, c, alpha, cache)
        pool_coalitions.append(c)
        pool_values[c] = V_c
        pool_payoffs[c] = {i: V_c}

    core_checks = 0
    core_feasible = len(N)  # singletons

    # Current partition (start with singletons)
    current_partition: List[Tuple[int, ...]] = [c for c in pool_coalitions]

    improved = True
    while improved:
        if deadline is not None and time.time() >= deadline:
            break

        improved = False
        best_merge = None
        best_gain = 0
        best_merged_payoffs = None

        for i in range(len(current_partition)):
            for j in range(i + 1, len(current_partition)):
                if deadline is not None and time.time() >= deadline:
                    break

                c1 = current_partition[i]
                c2 = current_partition[j]
                merged = tuple(sorted(c1 + c2))

                # Skip if already in pool (already checked)
                if merged in pool_values:
                    # Already Core-feasible — consider as merge candidate
                    gain = pool_values[merged] - (pool_values[c1] + pool_values[c2])
                    if gain > best_gain:
                        best_gain = gain
                        best_merge = (i, j, merged)
                        best_merged_payoffs = pool_payoffs[merged]
                    continue

                # Compute V(merged) first — quick reject if not superadditive
                V_merged = solve_coalition_value(inst, merged, alpha, cache)
                V_c1 = pool_values[c1]
                V_c2 = pool_values[c2]

                if V_merged <= V_c1 + V_c2:
                    # Subadditive: merge not beneficial, skip Core check
                    continue

                # Core check
                core_checks += 1
                is_feasible, merged_payoffs = check_core_feasibility(
                    inst, merged, alpha, cache
                )

                if is_feasible:
                    core_feasible += 1
                    # Add to pool
                    pool_coalitions.append(merged)
                    pool_values[merged] = V_merged
                    pool_payoffs[merged] = merged_payoffs

                    gain = V_merged - (V_c1 + V_c2)
                    if gain > best_gain:
                        best_gain = gain
                        best_merge = (i, j, merged)
                        best_merged_payoffs = merged_payoffs

            if deadline is not None and time.time() >= deadline:
                break

        if best_merge is not None:
            i, j, merged = best_merge
            current_partition.pop(j)
            current_partition.pop(i)
            current_partition.append(merged)
            improved = True

    return pool_coalitions, pool_values, pool_payoffs, core_checks, core_feasible


# ---------------------------------------------------------------------------
# Phase 2: Set Partitioning IP
# ---------------------------------------------------------------------------

def _phase2_set_partitioning(
    inst: CKGInstance,
    pool_coalitions: List[Tuple[int, ...]],
    pool_values: Dict[Tuple[int, ...], float],
    pool_payoffs: Dict[Tuple[int, ...], Dict[int, float]],
    time_limit: Optional[float] = None,
) -> Tuple[List[Tuple[int, ...]], float]:
    """
    Phase 2: Select the best CSS-feasible partition from the pool.

    Solves:
        max  sum_{c in pool} V(c) * y_c
        s.t. sum_{c: i in c} y_c = 1   for all i in N
             y_c in {0, 1}

    Returns:
        (selected_coalitions, total_value)
    """
    N = inst.players

    m = gp.Model("set_partition")
    m.Params.OutputFlag = 0
    if time_limit is not None:
        m.Params.TimeLimit = time_limit

    # Decision variables: y[c] = 1 if coalition c is selected
    y = {}
    for idx, c in enumerate(pool_coalitions):
        y[idx] = m.addVar(vtype=GRB.BINARY, name=f"y_{idx}")

    # Objective: maximize total coalition value
    m.setObjective(
        gp.quicksum(pool_values[pool_coalitions[idx]] * y[idx]
                     for idx in range(len(pool_coalitions))),
        GRB.MAXIMIZE
    )

    # Partition constraints: each player in exactly one coalition
    for i in N:
        m.addConstr(
            gp.quicksum(
                y[idx] for idx, c in enumerate(pool_coalitions) if i in c
            ) == 1,
            name=f"cover_{i}"
        )

    m.optimize()

    if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and m.SolCount > 0:
        selected = [pool_coalitions[idx] for idx in range(len(pool_coalitions))
                     if y[idx].X > 0.5]
        total_value = sum(pool_values[c] for c in selected)
        return selected, total_value

    # Fallback: singletons (always feasible)
    singletons = [(i,) for i in N]
    total_value = sum(pool_values[(i,)] for i in N)
    return singletons, total_value


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def heuristic_css(
    inst: CKGInstance,
    alpha: float,
    time_limit: Optional[float] = None,
    cache: Optional[CoalitionValueCache] = None,
) -> CSSHeuristicResult:
    """
    CSS-feasible primal heuristic.

    Phase 1: Core-verified bottom-up — builds a pool of Core-feasible
             coalitions by starting from singletons and greedily merging,
             accepting only merges whose Core is nonempty.

    Phase 2: Set partitioning IP — selects the value-maximizing partition
             of N from the pool of Core-feasible coalitions.

    The output is guaranteed CSS-feasible.

    Args:
        inst:        CKG instance
        alpha:       Restriction factor for common items
        time_limit:  Total wall-clock time budget (seconds). None = no limit.
                     Phase 1 gets 2/3 of the budget, Phase 2 gets 1/3.

    Returns:
        CSSHeuristicResult with a CSS-feasible partition and payoffs.
    """
    t0 = time.time()
    cache = cache if cache is not None else CoalitionValueCache()
    timed_out = False

    # Phase 1 gets 2/3 of time_limit, Phase 2 gets 1/3.
    if time_limit is not None:
        phase1_deadline = t0 + time_limit * (2.0 / 3.0)
        phase2_limit = time_limit * (1.0 / 3.0)
    else:
        phase1_deadline = None
        phase2_limit = None

    # ---- Phase 1 ----
    t1 = time.time()
    pool_coalitions, pool_values, pool_payoffs, core_checks, core_feasible = \
        _phase1_core_verified_bottom_up(inst, alpha, cache, deadline=phase1_deadline)
    phase1_time = time.time() - t1

    if phase1_deadline is not None and time.time() >= phase1_deadline:
        timed_out = True

    # ---- Phase 2 ----
    t2 = time.time()
    # Adjust remaining time
    if time_limit is not None:
        remaining = time_limit - (time.time() - t0)
        phase2_limit = max(1.0, remaining)  # At least 1 second

    selected, total_value = _phase2_set_partitioning(
        inst, pool_coalitions, pool_values, pool_payoffs,
        time_limit=phase2_limit,
    )
    phase2_time = time.time() - t2

    # Collect payoffs for the selected partition
    payoffs = {}
    for c in selected:
        if c in pool_payoffs:
            payoffs.update(pool_payoffs[c])
        else:
            # Should not happen, but fallback
            V_c = pool_values.get(c, 0.0)
            for i in c:
                payoffs[i] = V_c / len(c)

    runtime = time.time() - t0

    return CSSHeuristicResult(
        total_value=total_value,
        coalitions=[tuple(sorted(c)) for c in selected],
        payoffs=payoffs,
        runtime=runtime,
        method="CSS_Heuristic",
        status="timeout" if timed_out else "completed",
        pool_size=len(pool_coalitions),
        phase1_time=phase1_time,
        phase2_time=phase2_time,
        core_checks=core_checks,
        core_feasible=core_feasible,
        n_value_evals=cache.misses,
    )
