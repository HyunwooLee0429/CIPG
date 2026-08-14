"""
Payoff Refinement Experiment

Post-processes the OCSS coalition structures from the recorded experiment
results: for every formed coalition, computes the Shapley value and the
nucleolus of the restricted game, and checks Core membership of the
Shapley value. Instances are regenerated from the (n, seed) recorded in
the results CSVs, and coalition structures are taken from the stored
`coalitions` column of the reference method (default: lazy_MIP_ws_McC).

Usage: python run_refinement.py [--max-n N] [--max-size K]

Output: results/refinement_results.csv, one row per formed coalition.
"""

import argparse
import ast
import json
import os
import time

import pandas as pd

from cipg import generate_instance, CoalitionValueCache
from cipg.payoff_refinement import refine_coalition_structure

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

N_ITEMS = 50
Q = 0.5
RHO = 0.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="lazy_MIP_ws_McC",
                    help="Reference method whose coalition structure is refined")
    ap.add_argument("--max-n", type=int, default=12,
                    help="Largest instance size n to process")
    ap.add_argument("--max-size", type=int, default=14,
                    help="Skip coalitions larger than this (2^|c| IP solves)")
    ap.add_argument("--out", default="refinement_results.csv")

    args = ap.parse_args()

    frames = []
    for fname in ("exp1_results.csv", "exp2_results.csv"):
        path = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["method"] == args.method) & (df["n"] <= args.max_n)]

    rows = []
    for _, r in df.sort_values(["n", "alpha"]).iterrows():
        n, seed, alpha = int(r["n"]), int(r["seed"]), float(r["alpha"])
        cs = [tuple(c) for c in ast.literal_eval(r["coalitions"])]
        print(f"n={n}, alpha={alpha} (seed {seed}): {len(cs)} coalitions")

        inst = generate_instance(n, N_ITEMS, q=Q, rho=RHO, seed=seed)
        cache = CoalitionValueCache()

        t0 = time.time()
        res = refine_coalition_structure(
            inst, cs, alpha, cache=cache,
            max_size=args.max_size, verbose=True,
        )
        elapsed = time.time() - t0

        # Sanity check: refined total value should match the reported obj.
        if not res.skipped:
            diff = abs(res.total_value - float(r["obj"]))
            if diff > 1e-4 * max(1.0, abs(float(r["obj"]))):
                print(f"  WARNING: refined total {res.total_value:.1f} != "
                      f"reported obj {float(r['obj']):.1f} "
                      f"(check seed/params consistency)")

        for cr in res.coalitions:
            rows.append({
                "n": n,
                "seed": seed,
                "alpha": alpha,
                "method": args.method,
                "coalition": str(cr.coalition),
                "size": cr.size,
                "V_c": cr.V_c,
                "shapley": json.dumps(
                    {str(i): round(v, 4) for i, v in sorted(cr.shapley.items())}),
                "shapley_in_core": cr.shapley_in_core,
                "shapley_max_violation": round(cr.shapley_max_violation, 6),
                "nucleolus": json.dumps(
                    {str(i): round(v, 4) for i, v in sorted(cr.nucleolus.items())}),
                "least_core_eps": round(cr.least_core_eps, 6),
                "n_subsets": cr.n_subsets,
                "runtime_values": round(cr.runtime_values, 3),
                "runtime_shapley": round(cr.runtime_shapley, 3),
                "runtime_nucleolus": round(cr.runtime_nucleolus, 3),
            })
        for c in res.skipped:
            rows.append({
                "n": n, "seed": seed, "alpha": alpha, "method": args.method,
                "coalition": str(c), "size": len(c), "V_c": None,
                "shapley": None, "shapley_in_core": None,
                "shapley_max_violation": None, "nucleolus": None,
                "least_core_eps": None, "n_subsets": 2 ** len(c) - 1,
                "runtime_values": None, "runtime_shapley": None,
                "runtime_nucleolus": None,
            })
        print(f"  done in {elapsed:.1f}s ({cache.stats()})\n")

    out_path = os.path.join(RESULTS_DIR, args.out)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {len(rows)} rows to {out_path}")

    # Summary
    done = [r for r in rows if r["V_c"] is not None]
    nontrivial = [r for r in done if r["size"] >= 2]
    in_core = sum(1 for r in nontrivial if r["shapley_in_core"])
    print(f"\nSummary: {len(done)} coalitions refined "
          f"({len(nontrivial)} of size >= 2); "
          f"Shapley in Core: {in_core}/{len(nontrivial)}")


if __name__ == "__main__":
    main()
