"""
Lazy Constraint OCSS Solver

Solves the Optimal Coalition Structure with Stability (OCSS) problem
using Gurobi's lazy constraint callbacks, so stability cuts are added
within a single branch-and-bound tree.
"""

import itertools
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

import gurobipy as gp
from gurobipy import GRB

from .instance import CKGInstance
from .coalition_value import solve_coalition_value, CoalitionValueCache
from .ocs_model import build_OCS_model, extract_solution


@dataclass 
class LazyOCSSResult:
    """Result from lazy constraint OCSS solver."""
    obj: Optional[float]
    coalitions: Optional[List[Tuple[int, ...]]]
    payoffs: Optional[Dict[int, float]]
    runtime: float
    n_cuts: int
    n_callbacks: int
    callback_time: float
    method: str
    status: str
    mip_gap: Optional[float] = None
    n_lifted_added: int = 0    # cbLazy calls using the lifted cut (8)
    n_basic_added: int = 0     # cbLazy calls using the basic cut (7)
    n_value_evals: int = 0     # Distinct coalition values V(s) computed


class StabilityLazyCallback:
    """
    Callback that adds stability constraints as lazy constraints.

    When an integer solution is found, checks for stability violations
    and adds lazy constraints for any blocking coalitions.

    Two separation modes for coalitions larger than enum_threshold:

    - "lazy":     Enumerate subsets in ordinal order, stop at the FIRST
                  violation, and add that single cut.
    - "lazy_MIP": Solve a MIP-based separation oracle to find the MOST
                  violated subcoalition, and add that single cut.

    For coalitions with |c| <= enum_threshold, BOTH modes enumerate ALL
    subsets and add ALL violated cuts.

    All subsets are checked for violations at every callback invocation
    (V(s) values are cached so re-checking is efficient).  The added_cuts
    set is used only to avoid adding duplicate cbLazy constraints —
    violation detection is never skipped.
    """

    def __init__(
        self,
        inst: CKGInstance,
        alpha: float,
        v: Dict,
        z: Dict,
        enum_threshold: int = 9,
        use_mip_separation: bool = True,
        use_lifted_cuts: bool = True,
        verbose: bool = False,
        cache: Optional[CoalitionValueCache] = None,
    ):
        self.inst = inst
        self.alpha = alpha
        self.v = v
        self.z = z
        self.enum_threshold = enum_threshold
        self.use_mip_separation = use_mip_separation
        self.use_lifted_cuts = use_lifted_cuts
        self.verbose = verbose

        self.cache = cache if cache is not None else CoalitionValueCache()
        self.n_callbacks = 0
        self.n_cuts = 0
        self.n_lifted_added = 0
        self.n_basic_added = 0
        self.callback_time = 0.0
        self.added_cuts: Set[Tuple] = set()  # Avoid duplicate cuts

        # Precompute singleton values for lifted cuts
        if self.use_lifted_cuts:
            self.singleton_values = {}
            for i in inst.players:
                self.singleton_values[i] = solve_coalition_value(
                    inst, (i,), alpha, self.cache
                )
    
    def __call__(self, model: gp.Model, where: int):
        """Gurobi callback function."""
        if where != GRB.Callback.MIPSOL:
            return

        t0 = time.time()
        self.n_callbacks += 1

        N = self.inst.players
        G = self.inst.players

        # Get solution values
        z_val = {(i, g): model.cbGetSolution(self.z[i, g]) for i in N for g in G}
        v_val = {i: model.cbGetSolution(self.v[i]) for i in N}

        # Find formed coalitions
        coalitions_found = []
        total_violations = 0
        new_cuts_this_cb = 0

        for g in G:
            coalition = tuple(i for i in N if z_val[(i, g)] > 0.5)
            if len(coalition) <= 1:
                continue

            coalitions_found.append((g, coalition))

            # Find ALL stability violations (no skipping)
            violations = self._find_violations(coalition, v_val, g=g)
            total_violations += len(violations)

            for s, V_s in violations:
                cut_key = (g, s)
                is_new = cut_key not in self.added_cuts
                if is_new:
                    self.added_cuts.add(cut_key)
                    new_cuts_this_cb += 1
                    self.n_cuts += 1

                # cbLazy must be called for every violation, even if the
                # constraint was added before: Gurobi only rejects the
                # current solution if some cbLazy call is made here.
                lhs = gp.quicksum(self.v[i] for i in s)

                if self.use_lifted_cuts:
                    sum_singletons = sum(self.singleton_values[i] for i in s)
                    synergy = V_s - sum_singletons

                    if synergy >= 0:
                        z_sum = gp.quicksum(self.z[i, g] for i in s)
                        rhs = (synergy * (z_sum - len(s) + 1)
                               + gp.quicksum(
                                   self.singleton_values[i] * self.z[i, g]
                                   for i in s
                               ))
                        model.cbLazy(lhs >= rhs)
                        if is_new:
                            self.n_lifted_added += 1
                    else:
                        M_s = V_s
                        rhs = V_s - M_s * (len(s) - gp.quicksum(self.z[i, g] for i in s))
                        model.cbLazy(lhs >= rhs)
                        if is_new:
                            self.n_basic_added += 1
                else:
                    M_s = V_s
                    rhs = V_s - M_s * (len(s) - gp.quicksum(self.z[i, g] for i in s))
                    model.cbLazy(lhs >= rhs)
                    if is_new:
                        self.n_basic_added += 1

        if self.verbose:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            coal_str = "; ".join(
                f"g{g}={list(c)}" for g, c in coalitions_found
            )
            payoff_str = ", ".join(f"v{i}={v_val[i]:.1f}" for i in N)
            print(
                f"  CB#{self.n_callbacks}: obj={obj:.1f} | {coal_str} | "
                f"violations={total_violations}, new_cuts={new_cuts_this_cb}, "
                f"total_cuts={self.n_cuts} | {payoff_str}"
            )

        self.callback_time += time.time() - t0
    
    def _find_violations(
        self,
        coalition: Tuple[int, ...],
        v_val: Dict[int, float],
        g: int = None,
        tol: float = 1e-6,
    ) -> List[Tuple[Tuple[int, ...], float]]:
        """
        Find violated stability constraints for a coalition.

        For |c| <= enum_threshold: enumerate ALL subsets, gather ALL violations.
        For |c| > enum_threshold:
            - lazy_MIP mode: use MIP-based separation for MOST violated subset.
            - otherwise:     enumerate in ordinal order, stop at FIRST violation.

        All subsets are always checked for violations (V(s) values are cached
        so re-checking is efficient).  The added_cuts set is only used in
        __call__ to avoid adding duplicate cbLazy constraints.
        """
        violations = []

        if len(coalition) <= self.enum_threshold:
            # Small coalition: enumerate all subsets, gather ALL violations
            for r in range(1, len(coalition)):
                for s in itertools.combinations(coalition, r):
                    s = tuple(s)
                    V_s = solve_coalition_value(self.inst, s, self.alpha, self.cache)
                    payoff_s = sum(v_val[i] for i in s)

                    if V_s > payoff_s + tol:
                        violations.append((s, V_s))
        else:
            if self.use_mip_separation:
                # lazy_MIP: MIP-based separation for most violated subset
                blocking, V_s, violation = self._find_most_violated_opt(coalition, v_val)
                if blocking is not None and violation > tol:
                    violations.append((blocking, V_s))
            else:
                # Enumerate in ordinal order, stop at FIRST violation
                for r in range(1, len(coalition)):
                    for s in itertools.combinations(coalition, r):
                        s = tuple(s)
                        V_s = solve_coalition_value(self.inst, s, self.alpha, self.cache)
                        payoff_s = sum(v_val[i] for i in s)

                        if V_s > payoff_s + tol:
                            violations.append((s, V_s))
                            return violations  # Stop at first violation

        return violations
    
    def _find_most_violated_opt(
        self,
        coalition: Tuple[int, ...],
        v_val: Dict[int, float],
    ) -> Tuple[Optional[Tuple[int, ...]], float, float]:
        """
        Find most violated constraint via optimization.

        Delegates to the shared exact separation oracle
        (find_most_violated_subset), which is also used by the SCS-feasible
        heuristic's Core checks so that both components run identical,
        exact separation.
        """
        return find_most_violated_subset(
            self.inst, coalition, self.alpha, v_val, self.cache
        )


def find_most_violated_subset(
    inst: CKGInstance,
    coalition: Tuple[int, ...],
    alpha: float,
    v_val: Dict[int, float],
    cache: Optional[CoalitionValueCache] = None,
    threads: int = 1,
) -> Tuple[Optional[Tuple[int, ...]], float, float]:
    """
    Exact MIP-based separation oracle (shared by the lazy solver callback
    and the heuristic's Core checks).

    Solves: max_{s proper nonempty subset of coalition} V(s) - sum_{i in s} v_i

    Exactness requires (i) integrality on individual-item x (single user in
    the parent coalition), mirroring the coalition IP; and (ii) a commonness
    indicator c_j per item so that the alpha-restricted bound applies only
    when 2+ SELECTED players can use item j (an item common in the parent
    coalition may be individual in the selected subset). Without (i)-(ii)
    the objective is not V(s) and the certificate 'no violated subset'
    is unreliable.

    Returns:
        (blocking, V_s, violation) with V_s recomputed exactly via
        solve_coalition_value, or (None, 0, 0) if no violation exists.
    """
    m = gp.Model("separation")
    m.Params.OutputFlag = 0
    m.Params.Threads = threads

    S = list(coalition)
    n = len(S)
    R = inst.items
    R_i = inst.R_i
    p = inst.profits
    a = inst.weights
    b = inst.capacity
    u = inst.u

    # w[i] = 1 if player i in blocking coalition
    w = m.addVars(S, vtype=GRB.BINARY, name="w")

    # Item variables
    x = {}
    for i in S:
        for j in R_i[i]:
            x[(i, j)] = m.addVar(lb=0.0, ub=u[(i, j)], name=f"x_{i}_{j}")

    # Precompute players_in_S for each item
    pj_map = {}
    R_ind, R_com = [], []
    for j in R:
        pj = [i for i in S if j in R_i[i]]
        if len(pj) == 0:
            continue
        pj_map[j] = pj
        if len(pj) == 1:
            R_ind.append(j)
        else:
            R_com.append(j)

    # (i) Integrality on individual items (mirrors the coalition IP (2)):
    # for j in R_ind, the single user's x must be integer; for j in R_com,
    # y is integer and x may stay continuous.
    for j in R_ind:
        i = pj_map[j][0]
        x[(i, j)].vtype = GRB.INTEGER

    # y[j] for common items
    y = {}
    for j in R_com:
        y[j] = m.addVar(lb=0.0, vtype=GRB.INTEGER, name=f"y_{j}")

    # Proper nonempty subset
    m.addConstr(gp.quicksum(w[i] for i in S) >= 1, "nonempty")
    m.addConstr(gp.quicksum(w[i] for i in S) <= n - 1, "proper")

    # Capacity
    cap_lhs = gp.LinExpr()
    for j in R_ind:
        i = pj_map[j][0]
        cap_lhs += a[j] * x[(i, j)]
    for j in R_com:
        cap_lhs += a[j] * y[j]

    m.addConstr(cap_lhs <= gp.quicksum(b[i] * w[i] for i in S), "capacity")

    # Linking for common items
    for j in R_com:
        m.addConstr(y[j] == gp.quicksum(x[(i, j)] for i in pj_map[j]), f"link_{j}")

    # x <= u * w
    for i in S:
        for j in R_i[i]:
            m.addConstr(x[(i, j)] <= u[(i, j)] * w[i], f"x_ub_{i}_{j}")

    # (ii) Coalition bounds (partially restricted) with commonness
    # indicators: c[j] = 1 iff 2+ SELECTED players can use item j; the
    # alpha restriction applies only when c[j] = 1, and the natural bound
    # otherwise — mirroring the master model's w indicators.
    c = {}
    for j in R_com:
        pj = pj_map[j]
        k_j = len(pj)
        c[j] = m.addVar(vtype=GRB.BINARY, name=f"c_{j}")

        w_sum_j = gp.quicksum(w[i] for i in pj)

        m.addConstr(w_sum_j >= 2 * c[j], f"c_lb_{j}")
        m.addConstr(w_sum_j - 1 <= (k_j - 1) * c[j], f"c_ub_{j}")

        M_j = sum(u[(i, j)] for i in pj)
        m.addConstr(
            y[j] <= alpha * gp.quicksum(u[(i, j)] * w[i] for i in pj)
                   + M_j * (1 - c[j]),
            f"coal_{j}"
        )
        m.addConstr(
            y[j] <= gp.quicksum(u[(i, j)] * w[i] for i in pj),
            f"coal_nat_{j}"
        )

    # Objective: V(s) - sum v_i
    profit_expr = gp.quicksum(
        p[(i, j)] * x[(i, j)] for i in S for j in R_i[i]
    )
    payoff_expr = gp.quicksum(v_val[i] * w[i] for i in S)

    m.setObjective(profit_expr - payoff_expr, GRB.MAXIMIZE)
    m.optimize()

    if m.Status == GRB.OPTIMAL and m.ObjVal > 1e-6:
        blocking = tuple(i for i in S if w[i].X > 0.5)
        # Recompute exact V(s) (numerical safety; also warms the cache)
        V_s = solve_coalition_value(inst, blocking, alpha, cache)
        violation = V_s - sum(v_val[i] for i in blocking)
        return blocking, V_s, violation

    return None, 0, 0


def solve_OCSS_lazy(
    inst: CKGInstance,
    alpha: float,
    time_limit: Optional[float] = None,
    enum_threshold: int = 9,
    use_mip_separation: bool = True,
    use_lifted_cuts: bool = True,
    disaggregated: bool = True,
    warm_start: Optional[Tuple[List[Tuple[int, ...]], Dict[int, float]]] = None,
    verbose: bool = False,
    cache: Optional[CoalitionValueCache] = None,
) -> LazyOCSSResult:
    """
    Solve OCSS using lazy constraint callbacks.

    Args:
        inst: The CKG instance
        alpha: Restriction factor for common items
        time_limit: Maximum solve time in seconds
        enum_threshold: Coalitions <= this size use enumeration, larger use optimization
        use_mip_separation: If True ("lazy_MIP"), use MIP-based separation for
            coalitions larger than enum_threshold. If False ("lazy"), enumerate
            subsets in ordinal order and stop at the first violation.
        use_lifted_cuts: If True (default), use lifted stability cuts that
            exploit singleton values V({i}) for tighter bounds. Falls back to
            basic cuts when synergy is negative (sub-additive case).
        disaggregated: If True (default), use the disaggregated formulation
            with per-group payoff variables for a tighter LP relaxation.
        warm_start: Optional (coalitions, payoffs) for MIP start

    Returns:
        LazyOCSSResult with solution and statistics
    """
    m, v, z, x, y, Vg = build_OCS_model(inst, alpha, disaggregated=disaggregated)

    if time_limit is not None:
        m.Params.TimeLimit = time_limit

    # Enable lazy constraints
    m.Params.LazyConstraints = 1

    # Apply warm start if provided
    if warm_start is not None:
        coalitions_ws, payoffs_ws = warm_start
        N = inst.players
        G = inst.players

        # Build coalition membership
        player_to_group = {}
        for g_idx, coal in enumerate(coalitions_ws):
            group_id = min(coal)  # Leader is smallest index
            for i in coal:
                player_to_group[i] = group_id

        # Set z start values
        for i in N:
            for g in G:
                if player_to_group.get(i) == g:
                    z[i, g].Start = 1.0
                else:
                    z[i, g].Start = 0.0

        # Set v start values
        for i in N:
            if i in payoffs_ws:
                v[i].Start = payoffs_ws[i]

    # Determine method label
    parts = []
    parts.append("Lazy_MIP" if use_mip_separation else "Lazy")
    if disaggregated:
        parts.append("disagg")
    if use_lifted_cuts:
        parts.append("lifted")
    method_label = "_".join(parts)

    # Create callback
    callback = StabilityLazyCallback(
        inst, alpha, v, z, enum_threshold,
        use_mip_separation=use_mip_separation,
        use_lifted_cuts=use_lifted_cuts,
        verbose=verbose,
        cache=cache,
    )

    t0 = time.time()
    m.optimize(callback)
    runtime = time.time() - t0

    status = "optimal" if m.Status == GRB.OPTIMAL else "timeout"
    if m.Status == GRB.INFEASIBLE:
        status = "infeasible"

    if m.SolCount == 0:
        return LazyOCSSResult(
            None, None, None, runtime,
            callback.n_cuts, callback.n_callbacks, callback.callback_time,
            method_label, status
        )

    coalitions, payoffs = extract_solution(inst, v, z)
    mip_gap = m.MIPGap if hasattr(m, 'MIPGap') and m.SolCount > 0 else None

    return LazyOCSSResult(
        obj=m.ObjVal,
        coalitions=coalitions,
        payoffs=payoffs,
        runtime=runtime,
        n_cuts=callback.n_cuts,
        n_callbacks=callback.n_callbacks,
        callback_time=callback.callback_time,
        method=method_label,
        status=status,
        mip_gap=mip_gap,
        n_lifted_added=callback.n_lifted_added,
        n_basic_added=callback.n_basic_added,
        n_value_evals=callback.cache.misses,
    )
