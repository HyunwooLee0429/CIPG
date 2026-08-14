"""
Brute-Force Baseline

Exhaustive baseline in three stages:

  Stage V:   compute V(c) for ALL 2^n - 1 nonempty coalitions (IP each)
  Stage OCS: optimal partition by dynamic programming over subsets, O(3^n)
  Stage SOCS: Core check (LP with all 2^|c|-2 subset constraints) for every
              coalition, then DP restricted to Core-feasible blocks

Each stage has its own size cap; beyond a cap the script reports a time
projection extrapolated from the completed sizes.

Usage: python run_bruteforce_baseline.py [--values-max-n N] [--ocs-max-n N] [--socs-max-n N]

Output: results/bruteforce_baseline.csv
"""

import argparse
import os
import time

import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from cipg import generate_instance, CoalitionValueCache, solve_coalition_value

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
N_ITEMS, Q, RHO = 50, 0.5, 0.3
SEED_BASE = 42


def mask_to_tuple(mask, n):
    return tuple(i for i in range(n) if mask >> i & 1)


def stage_values(inst, n, alpha, cache):
    """V(c) for all nonempty subsets. Returns (values list, time)."""
    t0 = time.time()
    V = [0.0] * (1 << n)
    for mask in range(1, 1 << n):
        V[mask] = solve_coalition_value(inst, mask_to_tuple(mask, n), alpha, cache)
    return V, time.time() - t0


def stage_ocs_dp(V, n):
    """Best-partition DP over subsets, O(3^n)."""
    t0 = time.time()
    dp = [0.0] * (1 << n)
    for mask in range(1, 1 << n):
        low = mask & (-mask)
        best = 0.0
        # iterate submasks of mask that contain the lowest set bit
        sub = mask
        while sub:
            if sub & low:
                cand = V[sub] + dp[mask ^ sub]
                if cand > best:
                    best = cand
            sub = (sub - 1) & mask
        dp[mask] = best
    return dp[(1 << n) - 1], time.time() - t0


def core_feasible_lp(V, mask, n):
    """LP feasibility of the Core of coalition `mask` using all subset
    constraints (values V already available)."""
    members = mask_to_tuple(mask, n)
    if len(members) == 1:
        return True
    m = gp.Model()
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    v = m.addVars(members, lb=0.0)
    m.addConstr(gp.quicksum(v[i] for i in members) == V[mask])
    sub = (mask - 1) & mask
    while sub:
        m.addConstr(
            gp.quicksum(v[i] for i in mask_to_tuple(sub, n)) >= V[sub])
        sub = (sub - 1) & mask
    m.optimize()
    return m.Status == GRB.OPTIMAL


def stage_socs_dp(V, n):
    """Core-check every coalition, then best-partition DP over
    Core-feasible blocks."""
    t0 = time.time()
    ok = [False] * (1 << n)
    n_checks = 0
    for mask in range(1, 1 << n):
        ok[mask] = core_feasible_lp(V, mask, n)
        n_checks += 1
    t_core = time.time() - t0

    t1 = time.time()
    dp = [0.0] * (1 << n)
    for mask in range(1, 1 << n):
        low = mask & (-mask)
        best = None
        sub = mask
        while sub:
            if (sub & low) and ok[sub]:
                cand = V[sub] + dp[mask ^ sub]
                if best is None or cand > best:
                    best = cand
            sub = (sub - 1) & mask
        dp[mask] = best if best is not None else 0.0  # singletons ensure feasibility
    return dp[(1 << n) - 1], n_checks, t_core, time.time() - t1


def project(times_by_n, target_n):
    """Extrapolate runtime geometrically from the last two completed sizes."""
    ns = sorted(times_by_n)
    if len(ns) < 2:
        return None
    n1, n2 = ns[-2], ns[-1]
    t1, t2 = times_by_n[n1], times_by_n[n2]
    if t1 <= 0:
        return None
    ratio = (t2 / t1) ** (1.0 / (n2 - n1))
    return t2 * ratio ** (target_n - n2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--values-max-n", type=int, default=18)
    ap.add_argument("--ocs-max-n", type=int, default=16)
    ap.add_argument("--socs-max-n", type=int, default=14)
    ap.add_argument("--max-n", type=int, default=30,
                    help="Report projections up to this size")
    args = ap.parse_args()

    rows = []
    val_times, socs_times = {}, {}
    for n in range(2, args.max_n + 1, 2):
        seed = SEED_BASE + n
        for alpha in (0.5, 1.0):
            row = {"n": n, "seed": seed, "alpha": alpha,
                   "bf_n_values": (1 << n) - 1}
            if n <= args.values_max_n:
                inst = generate_instance(n, N_ITEMS, q=Q, rho=RHO, seed=seed)
                cache = CoalitionValueCache()
                V, tv = stage_values(inst, n, alpha, cache)
                row["bf_values_time"] = round(tv, 2)
                val_times[n] = tv
                if n <= args.ocs_max_n:
                    ocs, t_ocs = stage_ocs_dp(V, n)
                    row["bf_ocs_obj"] = round(ocs, 2)
                    row["bf_ocs_time"] = round(t_ocs, 2)
                if n <= args.socs_max_n:
                    socs, n_chk, t_core, t_dp = stage_socs_dp(V, n)
                    row["bf_socs_obj"] = round(socs, 2)
                    row["bf_core_checks"] = n_chk
                    row["bf_core_time"] = round(t_core, 2)
                    row["bf_socs_dp_time"] = round(t_dp, 2)
                    socs_times[n] = t_core + t_dp
            else:
                pv = project(val_times, n)
                row["bf_values_time_projected"] = round(pv, 1) if pv else ""
                ps = project(socs_times, n)
                row["bf_socs_time_projected"] = round(ps, 1) if ps else ""
            rows.append(row)
            print(row, flush=True)

    out = os.path.join(RESULTS_DIR, "bruteforce_baseline.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote {len(rows)} rows to {out}")

    # Sanity: compare against the main-grid objectives where available
    p1 = os.path.join(RESULTS_DIR, "exp1_results.csv")
    if os.path.exists(p1):
        exp = pd.read_csv(p1)
        bf = pd.DataFrame(rows)
        m_ocs = bf.dropna(subset=["bf_ocs_obj"]) if "bf_ocs_obj" in bf else None
        if m_ocs is not None and len(m_ocs):
            ocs_ref = exp[exp.method.isin(["OCS_McC", "OCS_DL"])] \
                .groupby(["n", "alpha"])["obj"].max().reset_index()
            cmp_ = m_ocs.merge(ocs_ref, on=["n", "alpha"])
            bad = cmp_[(cmp_["bf_ocs_obj"] - cmp_["obj"]).abs() > 1e-3]
            print(f"OCS match: {len(cmp_)-len(bad)}/{len(cmp_)}")
        if "bf_socs_obj" in bf:
            m_socs = bf.dropna(subset=["bf_socs_obj"])
            socs_ref = exp[exp.method == "lazy_MIP_ws_McC"][["n", "alpha", "obj"]]
            cmp_ = m_socs.merge(socs_ref, on=["n", "alpha"])
            bad = cmp_[(cmp_["bf_socs_obj"] - cmp_["obj"]).abs() > 1e-3]
            print(f"SOCS match: {len(cmp_)-len(bad)}/{len(cmp_)}")


if __name__ == "__main__":
    main()
