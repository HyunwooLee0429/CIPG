"""
Root LP Bound Comparison: McC vs. DL

Direct validation of the strict LP dominance result (Proposition 3):
for every experimental instance, build the OCS model under both the
aggregated (McCormick) and disaggregated formulations, relax all
integrality, and report the root LP bounds. The disaggregated bound
should be no larger (tighter or equal) on every instance.

Also verifies root inertness of the stability cuts: with --with-cuts,
all basic and lifted stability cuts for singleton subsets (plus any
subsets up to --cut-size) are added a priori to the LP relaxation,
and the optimal value is compared against the plain root LP.

Usage: python run_root_lp_bounds.py [--with-cuts]

Output: results/root_lp_bounds.csv
"""

import argparse
import os
import time
from itertools import combinations

import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from cipg import generate_instance, CoalitionValueCache, solve_coalition_value
from cipg.ocs_model import build_OCS_model

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
N_ITEMS, Q, RHO = 50, 0.5, 0.3


def root_lp_value(inst, alpha, disaggregated, add_cuts=False, cut_size=1,
                  cache=None):
    m, v, z, x, y, Vg = build_OCS_model(inst, alpha, disaggregated=disaggregated)
    if add_cuts:
        cache = cache or CoalitionValueCache()
        singles = {i: solve_coalition_value(inst, (i,), alpha, cache)
                   for i in inst.players}
        G = inst.players  # group indices
        for size in range(1, cut_size + 1):
            for s in combinations(inst.players, size):
                V_s = solve_coalition_value(inst, s, alpha, cache)
                syn = V_s - sum(singles[i] for i in s)
                for g in G:
                    lhs = gp.quicksum(v[i] for i in s)
                    zs = gp.quicksum(z[i, g] for i in s)
                    # basic cut (7), M_s = max(V_s, 0)
                    Ms = max(V_s, 0.0)
                    m.addConstr(lhs >= V_s - Ms * (len(s) - zs))
                    # lifted cut (8) when valid
                    if syn >= 0 and all(singles[i] >= 0 for i in s):
                        m.addConstr(
                            lhs >= syn * (zs - len(s) + 1)
                            + gp.quicksum(singles[i] * z[i, g] for i in s))
    m.update()  # flush pending vars/constrs before relaxing
    r = m.relax()
    r.Params.OutputFlag = 0
    r.optimize()
    return r.ObjVal if r.Status == GRB.OPTIMAL else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=30)
    ap.add_argument("--with-cuts", action="store_true",
                    help="Also compute root LP with a-priori stability cuts")
    ap.add_argument("--cut-size", type=int, default=2,
                    help="Add all cuts for subsets up to this size")
    args = ap.parse_args()

    rows = []
    for n in range(2, args.max_n + 1, 2):
        seed = 42 + n
        inst = generate_instance(n, N_ITEMS, q=Q, rho=RHO, seed=seed)
        for alpha in (0.5, 1.0):
            t0 = time.time()
            lp_mcc = root_lp_value(inst, alpha, disaggregated=False)
            lp_dl = root_lp_value(inst, alpha, disaggregated=True)
            row = {"n": n, "seed": seed, "alpha": alpha,
                   "rootLP_McC": lp_mcc, "rootLP_DL": lp_dl,
                   "tightening_%": None if not lp_mcc else
                   round((lp_mcc - lp_dl) / abs(lp_mcc) * 100, 4)}
            if args.with_cuts:
                cache = CoalitionValueCache()
                row["rootLP_DL_cuts"] = root_lp_value(
                    inst, alpha, disaggregated=True, add_cuts=True,
                    cut_size=args.cut_size, cache=cache)
                row["inert"] = (abs(row["rootLP_DL_cuts"] - lp_dl) < 1e-4
                                if row["rootLP_DL_cuts"] is not None else None)
            row["time_s"] = round(time.time() - t0, 1)
            rows.append(row)
            print(row)

    out = os.path.join(RESULTS_DIR, "root_lp_bounds.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote {len(rows)} rows to {out}")
    df = pd.DataFrame(rows)
    print(f"DL bound <= McC bound on {int((df['rootLP_DL'] <= df['rootLP_McC'] + 1e-6).sum())}"
          f"/{len(df)} instances")
    if args.with_cuts and "inert" in df:
        print(f"Root LP unchanged by a-priori cuts on {int(df['inert'].sum())}"
              f"/{df['inert'].notna().sum()} instances")


if __name__ == "__main__":
    main()
