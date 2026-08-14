"""
Cooperative Integer Programming Games - Instance Definition

This module defines the CKGInstance (Cooperative Knapsack Game) data structure
and instance generation utilities.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class CKGInstance:
    """
    A Cooperative Knapsack Game instance.
    
    Each player has items available to them with profits and weights.
    Players can form coalitions and pool their budgets.
    """
    n_players: int
    n_items: int
    q: float              # Item availability probability
    rho: float            # Budget ratio
    players: List[int]
    items: List[int]
    R_i: Dict[int, List[int]]    # i -> items available to player i
    I_j: Dict[int, List[int]]    # j -> players that can use item j
    profits: Dict[Tuple[int, int], int]   # (i,j) -> profit
    weights: Dict[int, int]       # j -> weight
    capacity: Dict[int, float]    # i -> budget
    u: Dict[Tuple[int, int], int] # (i,j) -> upper bound (0 or 1)


def generate_instance(
    n_players: int,
    n_items: int,
    q: float = 0.5,
    rho: float = 0.3,
    seed: Optional[int] = None,
    max_tries: int = 100,
) -> CKGInstance:
    """
    Generate a Cooperative Knapsack Game instance.
    
    Args:
        n_players: Number of players
        n_items: Number of items
        q: Probability that an item is available to a player
        rho: Budget ratio (fraction of total available item weight)
        seed: Random seed for reproducibility
        max_tries: Maximum generation attempts
        
    Returns:
        A valid CKGInstance where every player has at least one item
    """
    for attempt in range(max_tries):
        rng = random.Random(seed + attempt if seed is not None else None)
        
        players = list(range(n_players))
        items = list(range(n_items))
        
        # Item weights
        weights = {j: rng.randint(5, 10) for j in items}
        
        R_i = {i: [] for i in players}
        I_j = {j: [] for j in items}
        profits = {}
        u = {}
        
        # Availability and profits
        for i in players:
            for j in items:
                if rng.random() <= q:
                    R_i[i].append(j)
                    I_j[j].append(i)
                    profits[(i, j)] = rng.randint(10, 100)
                    u[(i, j)] = 1
                else:
                    u[(i, j)] = 0
        
        # Check every player has at least one item
        if any(len(R_i[i]) == 0 for i in players):
            continue
        
        # Capacities
        capacity = {}
        for i in players:
            total_w = sum(weights[j] for j in R_i[i])
            capacity[i] = rho * total_w
        
        return CKGInstance(
            n_players=n_players,
            n_items=n_items,
            q=q,
            rho=rho,
            players=players,
            items=items,
            R_i=R_i,
            I_j=I_j,
            profits=profits,
            weights=weights,
            capacity=capacity,
            u=u,
        )
    
    raise RuntimeError(f"Failed to generate valid instance after {max_tries} tries")
