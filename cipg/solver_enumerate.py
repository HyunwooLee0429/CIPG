"""
Enumeration-based OCSS Solver

Solves the Optimal Coalition Structure with Stability (OCSS) problem
using a cutting-plane algorithm with full enumeration of subcoalitions.

This is the baseline method - effective for small instances but
exponentially slow as problem size grows.
"""

import itertools
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

from .instance import CKGInstance
from .coalition_value import solve_coalition_value, CoalitionValueCache
from .ocs_model import build_OCS_model, extract_solution


@dataclass
class OCSSResult:
    """Result from OCSS solver."""
    obj: Optional[float]
    coalitions: Optional[List[Tuple[int, ...]]]
    payoffs: Optional[Dict[int, float]]
    runtime: float
    n_cuts: int
    n_iterations: int
    method: str
    status: str  # "optimal", "timeout", "infeasible"
    mip_gap: Optional[float] = None
    n_value_evals: int = 0     # Distinct coalition values V(s) computed


def solve_OCS(
    inst: CKGInstance,
    alpha: float,
    time_limit: Optional[float] = None,
    disaggregated: bool = True,
) -> OCSSResult:
    """Solve OCS without stability constraints."""
    m, v, z, x, y, Vg = build_OCS_model(inst, alpha, disaggregated=disaggregated)

    if time_limit is not None:
        m.Params.TimeLimit = time_limit

    t0 = time.time()
    m.optimize()
    runtime = time.time() - t0

    if m.Status == GRB.INFEASIBLE:
        return OCSSResult(None, None, None, runtime, 0, 1, "OCS", "infeasible")

    if m.SolCount == 0:
        return OCSSResult(None, None, None, runtime, 0, 1, "OCS", "timeout")

    coalitions, payoffs = extract_solution(inst, v, z)
    mip_gap = m.MIPGap if hasattr(m, 'MIPGap') and m.SolCount > 0 else None

    return OCSSResult(
        obj=m.ObjVal,
        coalitions=coalitions,
        payoffs=payoffs,
        runtime=runtime,
        n_cuts=0,
        n_iterations=1,
        method="OCS",
        status="optimal" if m.Status == GRB.OPTIMAL else "timeout",
        mip_gap=mip_gap,
    )


def separate_stability_enumerate(
    model: gp.Model,
    inst: CKGInstance,
    alpha: float,
    v: Dict,
    z: Dict,
    cache: CoalitionValueCache,
    tol: float = 1e-6,
    deadline: Optional[float] = None,
    use_lifted_cuts: bool = True,
    singleton_values: Optional[Dict[int, float]] = None,
) -> Tuple[int, bool]:
    """
    Separate stability cuts by enumerating all subcoalitions.

    For each formed coalition, checks all 2^|c| - 2 proper subcoalitions
    for stability violations.

    Args:
        deadline: Absolute time (time.time()) by which we must stop.
                  None means no limit.
        use_lifted_cuts: If True, use lifted stability cuts (with synergy
                         check). Falls back to basic cut when synergy < 0.
        singleton_values: Precomputed V({i}) for lifted cuts. Required
                          when use_lifted_cuts=True.

    Returns:
        (n_cuts, timed_out): number of cuts added and whether time limit was hit.
    """
    N = inst.players
    G = inst.players

    # Get current solution
    z_val = {(i, g): z[i, g].X for i in N for g in G}
    v_val = {i: v[i].X for i in N}

    n_cuts = 0
    timed_out = False

    for g in G:
        coalition = [i for i in N if z_val[(i, g)] > 0.5]
        if len(coalition) <= 1:
            continue

        # Check all proper nonempty subcoalitions
        for r in range(1, len(coalition)):
            for s in itertools.combinations(coalition, r):
                # Check time limit
                if deadline is not None and time.time() >= deadline:
                    return n_cuts, True

                s = tuple(s)

                V_s = solve_coalition_value(inst, s, alpha, cache)
                payoff_s = sum(v_val[i] for i in s)

                if V_s > payoff_s + tol:
                    lhs = gp.quicksum(v[i] for i in s)
                    cname = f"stab_{g}_" + "_".join(map(str, s))

                    if use_lifted_cuts and singleton_values is not None:
                        sum_singletons = sum(singleton_values[i] for i in s)
                        synergy = V_s - sum_singletons
                        if synergy >= 0:
                            z_sum = gp.quicksum(z[i, g] for i in s)
                            rhs = (synergy * (z_sum - len(s) + 1)
                                   + gp.quicksum(
                                       singleton_values[i] * z[i, g]
                                       for i in s
                                   ))
                            model.addConstr(lhs >= rhs, name=cname)
                        else:
                            M_s = V_s
                            rhs = V_s - M_s * (len(s) - gp.quicksum(z[i, g] for i in s))
                            model.addConstr(lhs >= rhs, name=cname)
                    else:
                        M_s = V_s
                        rhs = V_s - M_s * (len(s) - gp.quicksum(z[i, g] for i in s))
                        model.addConstr(lhs >= rhs, name=cname)
                    n_cuts += 1

    return n_cuts, timed_out


def solve_OCSS_enumerate(
    inst: CKGInstance,
    alpha: float,
    time_limit: Optional[float] = None,
    disaggregated: bool = True,
    use_lifted_cuts: bool = True,
) -> OCSSResult:
    """
    Solve OCSS using cutting-plane with full enumeration.

    This is the baseline method that checks all subcoalitions.
    Exponentially slow as coalition sizes grow.

    The time limit applies to the entire procedure, including both
    MIP solves and cut generation phases.

    Args:
        inst: The CKG instance
        alpha: Restriction factor for common items
        time_limit: Maximum total solve time in seconds
        disaggregated: If True (default), use the disaggregated formulation
        use_lifted_cuts: If True (default), use lifted stability cuts
    """
    m, v, z, x, y, Vg = build_OCS_model(inst, alpha, disaggregated=disaggregated)

    cache = CoalitionValueCache()

    # Precompute singleton values for lifted cuts
    singleton_values = None
    if use_lifted_cuts:
        singleton_values = {}
        for i in inst.players:
            singleton_values[i] = solve_coalition_value(
                inst, (i,), alpha, cache
            )

    total_cuts = 0
    n_iterations = 0
    t0 = time.time()
    deadline = (t0 + time_limit) if time_limit is not None else None

    while True:
        # Check overall time budget before solving
        if deadline is not None:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            m.Params.TimeLimit = remaining

        m.optimize()
        n_iterations += 1

        if m.Status == GRB.INFEASIBLE:
            return OCSSResult(
                None, None, None, time.time() - t0,
                total_cuts, n_iterations, "Enumerate", "infeasible"
            )

        if m.Status == GRB.TIME_LIMIT:
            break

        if m.SolCount == 0:
            break

        # Separate stability cuts (with time-awareness)
        cuts_added, cut_timed_out = separate_stability_enumerate(
            m, inst, alpha, v, z, cache, deadline=deadline,
            use_lifted_cuts=use_lifted_cuts,
            singleton_values=singleton_values,
        )
        total_cuts += cuts_added

        if cut_timed_out:
            # Ran out of time during cut generation
            break

        if cuts_added == 0:
            # No violations - solution is stable
            coalitions, payoffs = extract_solution(inst, v, z)
            return OCSSResult(
                obj=m.ObjVal,
                coalitions=coalitions,
                payoffs=payoffs,
                runtime=time.time() - t0,
                n_cuts=total_cuts,
                n_iterations=n_iterations,
                method="Enumerate",
                status="optimal",
                n_value_evals=cache.misses,
            )

    # Timeout - return best found if any
    runtime = time.time() - t0
    if m.SolCount > 0:
        coalitions, payoffs = extract_solution(inst, v, z)
        return OCSSResult(
            obj=m.ObjVal,
            coalitions=coalitions,
            payoffs=payoffs,
            runtime=runtime,
            n_cuts=total_cuts,
            n_iterations=n_iterations,
            method="Enumerate",
            status="timeout",
            n_value_evals=cache.misses,
        )

    return OCSSResult(
        None, None, None, runtime,
        total_cuts, n_iterations, "Enumerate", "timeout"
    )
