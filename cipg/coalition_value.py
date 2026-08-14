"""
Coalition Value Computation

Solves the coalition-level integer program to compute V(S) for any coalition S.
Includes caching for efficiency.
"""

from typing import Dict, Tuple, Optional
import gurobipy as gp
from gurobipy import GRB

from .instance import CKGInstance


class CoalitionValueCache:
    """Cache for coalition values to avoid recomputation."""
    
    def __init__(self):
        self._cache: Dict[Tuple[int, ...], float] = {}
        self.hits = 0
        self.misses = 0
    
    def get(self, coalition: Tuple[int, ...]) -> Optional[float]:
        key = tuple(sorted(coalition))
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None
    
    def put(self, coalition: Tuple[int, ...], value: float):
        key = tuple(sorted(coalition))
        self._cache[key] = value
    
    def clear(self):
        self._cache.clear()
        self.hits = 0
        self.misses = 0
    
    def stats(self) -> str:
        total = self.hits + self.misses
        hit_rate = self.hits / total * 100 if total > 0 else 0
        return f"Cache: {self.hits}/{total} hits ({hit_rate:.1f}%)"


def solve_coalition_value(
    inst: CKGInstance,
    coalition: Tuple[int, ...],
    alpha: float,
    cache: Optional[CoalitionValueCache] = None,
) -> float:
    """
    Solve the coalition-level knapsack IP for coalition S.
    
    Uses PARTIALLY_RESTRICTED bounds where common items are limited to
    alpha * (number of players who can use the item).
    
    Args:
        inst: The CKG instance
        coalition: Tuple of player indices in the coalition
        alpha: Restriction factor for common items (0.5 = half capacity)
        cache: Optional cache for coalition values
        
    Returns:
        V(S) - the optimal value for coalition S
    """
    # Check cache
    if cache is not None:
        cached = cache.get(coalition)
        if cached is not None:
            return cached
    
    m = gp.Model("coalition_value")
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    
    S = list(coalition)
    R = inst.items
    R_i = inst.R_i
    p = inst.profits
    a = inst.weights
    b = inst.capacity
    u = inst.u
    
    # Variables
    x = {}
    for i in S:
        for j in R_i[i]:
            x[(i, j)] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    
    # y[j] for common items (available to 2+ players in S)
    y = {}
    players_for_j_S = {}  # Cache I_j ∩ S
    R_ind_S = []  # Individual items
    R_com_S = []  # Common items
    
    for j in R:
        players_for_j = [i for i in S if j in R_i[i]]
        players_for_j_S[j] = players_for_j
        
        if len(players_for_j) == 0:
            continue
        elif len(players_for_j) == 1:
            R_ind_S.append(j)
        else:
            R_com_S.append(j)
    
    for j in R_com_S:
        y[j] = m.addVar(vtype=GRB.INTEGER, lb=0.0, name=f"y_{j}")
    
    # Objective: sum of profits
    m.setObjective(
        gp.quicksum(
            p[(i, j)] * x[(i, j)]
            for i in S
            for j in R_i[i]
        ),
        GRB.MAXIMIZE
    )
    
    # Capacity constraint
    cap_expr = gp.LinExpr()
    for j in R_ind_S:
        i_only = players_for_j_S[j][0]
        cap_expr += a[j] * x[(i_only, j)]
    for j in R_com_S:
        cap_expr += a[j] * y[j]
    
    m.addConstr(cap_expr <= sum(b[i] for i in S), name="capacity")
    
    # Linking and coalition bounds for common items
    for j in R_com_S:
        players_for_j = players_for_j_S[j]
        
        # Linking: y[j] = sum of x[i,j]
        m.addConstr(
            y[j] == gp.quicksum(x[(i, j)] for i in players_for_j),
            name=f"link_{j}"
        )
        
        # Partially restricted bound: y[j] <= alpha * |players who can use j|
        m.addConstr(
            y[j] <= alpha * gp.quicksum(u[(i, j)] for i in players_for_j),
            name=f"coal_bound_{j}"
        )
    
    m.optimize()
    
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Coalition IP not optimal, status {m.Status}")
    
    value = m.ObjVal
    
    # Store in cache
    if cache is not None:
        cache.put(coalition, value)
    
    return value
