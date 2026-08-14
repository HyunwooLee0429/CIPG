"""
Optimal Coalition Structure (OCS) Model

Builds the mixed-integer program for finding the optimal coalition structure.
Supports both aggregated (McCormick) and disaggregated formulations.
"""

from typing import Dict, Tuple
import gurobipy as gp
from gurobipy import GRB

from .instance import CKGInstance


def build_OCS_model(
    inst: CKGInstance,
    alpha: float,
    disaggregated: bool = True,
) -> Tuple[gp.Model, Dict, Dict, Dict, Dict, Dict]:
    """
    Build the OCS MILP for finding optimal coalition structure.

    Uses PARTIALLY_RESTRICTED coalition bounds with factor alpha.

    Args:
        inst: The CKG instance
        alpha: Restriction factor for common items
        disaggregated: If True (default), use the disaggregated formulation
            with per-group payoff variables v_{ig} linked by v_i = sum_g v_{ig}.
            This yields a strictly tighter LP relaxation than the aggregated
            (McCormick) formulation by eliminating payoff double-counting.
            If False, use the standard McCormick linearization.

    Returns:
        (model, v, z, x, y, Vg) - the model and variable dictionaries
    """
    m = gp.Model("OCS")
    m.Params.OutputFlag = 0
    m.Params.Threads = 16   # matches the reported experimental setup

    N = inst.players
    G = inst.players  # Group indices (one per player max)
    R = inst.items
    R_i = inst.R_i
    I_j = inst.I_j
    p = inst.profits
    a = inst.weights
    b = inst.capacity
    u = inst.u

    # Variables

    # Payoffs
    v = m.addVars(N, lb=0.0, name="v")

    # Assignment: z[i,g] = 1 if player i in group g
    z = m.addVars(N, G, vtype=GRB.BINARY, name="z")

    # Item usage: x[i,g,j] for player i in group g using item j
    x = {}
    for i in N:
        for g in G:
            for j in R_i[i]:
                x[(i, g, j)] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{g}_{j}")

    # Coalition-level usage y[g,j]
    y = m.addVars(G, R, vtype=GRB.INTEGER, lb=0.0, name="y")

    # w[g,j] = 1 if item j is common in group g (2+ users)
    w = m.addVars(G, R, vtype=GRB.BINARY, name="w")

    # Coalition values
    Vg = m.addVars(G, lb=0.0, name="Vg")

    # Objective: maximize total payoff
    m.setObjective(gp.quicksum(v[i] for i in N), GRB.MAXIMIZE)

    # Constraints

    # Partition: each player in exactly one group
    for i in N:
        m.addConstr(gp.quicksum(z[i, g] for g in G) == 1, name=f"partition_{i}")

    # Capacity and item constraints for each group
    for g in G:
        # Capacity: weight of items <= pooled budget
        m.addConstr(
            gp.quicksum(a[j] * y[g, j] for j in R)
            <= gp.quicksum(b[i] * z[i, g] for i in N),
            name=f"cap_{g}"
        )

        for j in R:
            players_for_j = I_j[j]
            if not players_for_j:
                m.addConstr(y[g, j] == 0, name=f"y_zero_{g}_{j}")
                continue

            # Linking: y[g,j] = sum of x[i,g,j]
            m.addConstr(
                y[g, j] == gp.quicksum(x[(i, g, j)] for i in players_for_j),
                name=f"link_{g}_{j}"
            )

            # Upper bound on x
            for i in players_for_j:
                m.addConstr(
                    x[(i, g, j)] <= u[(i, j)] * z[i, g],
                    name=f"x_ub_{g}_{i}_{j}"
                )

            # Natural upper bound
            m.addConstr(
                y[g, j] <= gp.quicksum(u[(i, j)] * z[i, g] for i in players_for_j),
                name=f"coal_nat_{g}_{j}"
            )

            # Partially restricted bounds using w indicator
            k_j = len(players_for_j)

            # w[g,j] = 1 iff sum z[i,g] >= 2 (item is common)
            m.addConstr(
                gp.quicksum(z[i, g] for i in players_for_j) - 2 * w[g, j] >= 0,
                name=f"w_lb_{g}_{j}"
            )
            m.addConstr(
                gp.quicksum(z[i, g] for i in players_for_j) - 1 <= (k_j - 1) * w[g, j],
                name=f"w_ub_{g}_{j}"
            )

            # Partial restriction when w=1
            M_j = sum(u[(i, j)] for i in players_for_j)
            m.addConstr(
                y[g, j] <= alpha * gp.quicksum(u[(i, j)] * z[i, g] for i in players_for_j)
                         + M_j * (1 - w[g, j]),
                name=f"coal_part_{g}_{j}"
            )

    # ── Efficiency linearization ──
    if disaggregated:
        # Disaggregated formulation: per-group payoff variables v_ig
        # Strictly tighter LP relaxation than McCormick.

        # Global payoff bound M = sum over ALL players of p^i_j * u^i_j,
        # matching the paper's (6). A per-player bound sum_j p^i_j u^i_j is
        # NOT valid in general: a Core allocation may reward a player (e.g.,
        # a pure budget contributor) beyond its own profit potential, and a
        # per-player cap can then cut off every Core allocation of a formed
        # coalition. The LP tightening of the disaggregated formulation
        # comes from the linking constraint v_i = sum_g v_ig, not from the
        # magnitude of this bound.
        M_glob = sum(
            p.get((i, j), 0) * u.get((i, j), 0)
            for i in N for j in R
            if i in I_j.get(j, [])
        )
        U = {i: M_glob for i in N}

        # Per-group payoff variables
        v_ig = {}
        for i in N:
            for g in G:
                v_ig[(i, g)] = m.addVar(lb=0.0, name=f"vig_{i}_{g}")

        # Linking: v_i = sum_g v_{ig}
        for i in N:
            m.addConstr(
                v[i] == gp.quicksum(v_ig[(i, g)] for g in G),
                name=f"disagg_link_{i}"
            )

        # Variable upper bound: v_{ig} <= U_i * z_{ig}
        for i in N:
            for g in G:
                m.addConstr(
                    v_ig[(i, g)] <= U[i] * z[i, g],
                    name=f"disagg_ub_{i}_{g}"
                )

        # Coalition value definition
        for g in G:
            m.addConstr(
                Vg[g] == gp.quicksum(
                    p[(i, j)] * x[(i, g, j)]
                    for j in R
                    for i in I_j[j]
                    if (i, g, j) in x
                ),
                name=f"value_def_{g}"
            )

            # Efficiency: Vg = sum of per-group payoffs
            m.addConstr(
                Vg[g] == gp.quicksum(v_ig[(i, g)] for i in N),
                name=f"efficiency_{g}"
            )

    else:
        # Aggregated formulation: McCormick linearization (baseline)
        M_payoff = sum(
            p.get((i, j), 0) * u.get((i, j), 0)
            for i in N for j in R
            if i in I_j.get(j, [])
        )

        vz = {}
        for i in N:
            for g in G:
                vz[(i, g)] = m.addVar(lb=0.0, name=f"vz_{i}_{g}")

        # McCormick constraints for vz[i,g] = v[i] * z[i,g]
        for i in N:
            for g in G:
                # vz <= v
                m.addConstr(vz[(i, g)] <= v[i], name=f"vz_ub1_{i}_{g}")
                # vz <= M * z
                m.addConstr(vz[(i, g)] <= M_payoff * z[i, g], name=f"vz_ub2_{i}_{g}")
                # vz >= v - M * (1 - z)
                m.addConstr(vz[(i, g)] >= v[i] - M_payoff * (1 - z[i, g]), name=f"vz_lb_{i}_{g}")

        # Coalition value definition
        for g in G:
            m.addConstr(
                Vg[g] == gp.quicksum(
                    p[(i, j)] * x[(i, g, j)]
                    for j in R
                    for i in I_j[j]
                    if (i, g, j) in x
                ),
                name=f"value_def_{g}"
            )

            # Efficiency: Vg = sum of payoffs in group (linearized)
            m.addConstr(
                Vg[g] == gp.quicksum(vz[(i, g)] for i in N),
                name=f"efficiency_{g}"
            )

    # Symmetry breaking: group leader
    for g in G:
        for i in N:
            if i < g:
                m.addConstr(z[i, g] == 0, name=f"leader_sym_{g}_{i}")

        m.addConstr(
            gp.quicksum(z[i, g] for i in N) <= len(N) * z[g, g],
            name=f"leader_nonempty_{g}"
        )

    # Warm start: grand coalition (though may not be optimal)
    leader = N[0]
    for i in N:
        for g in G:
            z[i, g].Start = 1.0 if g == leader else 0.0

    return m, v, z, x, y, Vg


def extract_solution(inst: CKGInstance, v: Dict, z: Dict) -> Tuple:
    """Extract coalition structure and payoffs from solved model."""
    N = inst.players
    G = inst.players

    coalitions = []
    for g in G:
        members = [i for i in N if z[i, g].X > 0.5]
        if members:
            coalitions.append(tuple(sorted(members)))

    coalitions.sort(key=lambda c: (len(c), c))
    payoffs = {i: v[i].X for i in N}

    return coalitions, payoffs
