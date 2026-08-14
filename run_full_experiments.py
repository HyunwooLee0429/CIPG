"""
Main experimental grid (Experiments 1 and 2, all method configurations).

The lifted-vs-basic ablation is integrated: the lazy_MIP_McC runs provide
the lifted arm and the extra configuration lazy_MIP_basic_McC provides the
basic arm. Already-completed (n, alpha, method) rows found in the output
CSV are skipped, so the script can be stopped and restarted.

Usage: python run_full_experiments.py [--only exp1|exp2] [--no-exp2-basic]

Outputs: results/exp1_results.csv, results/exp2_results.csv
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime

from cipg import (
    generate_instance,
    solve_OCS,
    solve_OCSS_enumerate,
    solve_OCSS_lazy,
)
from cipg.heuristic_css import heuristic_css

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
N_ITEMS, Q, RHO = 50, 0.5, 0.3
SEED_BASE = 42
TAU = 9   # enumeration threshold
EXP1_NS = list(range(2, 21, 2))
EXP2_NS = list(range(22, 31, 2))
ALPHAS = [0.5, 1.0]
EXP1_TIME_LIMIT = 1800.0
EXP2_TIME_LIMIT = 3600.0

FIELDNAMES = [
    'n', 'seed', 'alpha', 'method', 'obj', 'runtime', 'n_cuts',
    'status', 'mip_gap',
    'n_callbacks', 'callback_time', 'n_iterations',
    'heur_runtime', 'heur_value', 'heur_status', 'heur_pool_size',
    'solver_runtime',
    'coalitions', 'n_coalitions', 'avg_coalition_size',
    'css_feasible',
    'n_lifted_added', 'n_basic_added', 'n_value_evals',
    'heur_phase1_time', 'heur_phase2_time',
    'heur_core_checks', 'heur_core_feasible',
    'heur_coalitions', 'heur_payoffs',
    'payoffs', 'timestamp',
]


# ---------------------------------------------------------------------------
# Method wrappers
# ---------------------------------------------------------------------------

def _payoffs_json(payoffs):
    if not payoffs:
        return ''
    return json.dumps({str(i): round(v, 4) for i, v in sorted(payoffs.items())})


def run_ocs(inst, alpha, time_limit, disaggregated):
    res = solve_OCS(inst, alpha, time_limit=time_limit,
                    disaggregated=disaggregated)
    return {
        'obj': res.obj, 'runtime': res.runtime, 'n_cuts': 0,
        'status': res.status, 'coalitions': res.coalitions,
        'mip_gap': res.mip_gap,
        'payoffs': _payoffs_json(res.payoffs),
    }


def run_enum(inst, alpha, time_limit, disaggregated):
    res = solve_OCSS_enumerate(
        inst, alpha, time_limit=time_limit,
        disaggregated=disaggregated, use_lifted_cuts=True,
    )
    return {
        'obj': res.obj, 'runtime': res.runtime, 'n_cuts': res.n_cuts,
        'n_iterations': res.n_iterations, 'status': res.status,
        'coalitions': res.coalitions,
        'n_value_evals': res.n_value_evals,
        'payoffs': _payoffs_json(res.payoffs),
    }


def run_lazy(inst, alpha, time_limit, use_mip_separation, disaggregated,
             use_lifted_cuts=True):
    res = solve_OCSS_lazy(
        inst, alpha, time_limit=time_limit,
        enum_threshold=TAU,
        use_mip_separation=use_mip_separation,
        disaggregated=disaggregated, use_lifted_cuts=use_lifted_cuts,
    )
    return {
        'obj': res.obj, 'runtime': res.runtime, 'n_cuts': res.n_cuts,
        'n_callbacks': res.n_callbacks, 'callback_time': res.callback_time,
        'status': res.status, 'coalitions': res.coalitions,
        'mip_gap': res.mip_gap,
        'n_lifted_added': res.n_lifted_added,
        'n_basic_added': res.n_basic_added,
        'n_value_evals': res.n_value_evals,
        'payoffs': _payoffs_json(res.payoffs),
    }


def run_lazy_mip_ws(inst, alpha, total_time_limit, disaggregated):
    # Heuristic and solver share one value cache, so n_value_evals counts
    # distinct values (cache.misses of the single shared cache).
    from cipg.coalition_value import CoalitionValueCache
    shared_cache = CoalitionValueCache()

    heur_budget = total_time_limit * (3.0 / 5.0)
    heur = heuristic_css(inst, alpha, time_limit=heur_budget,
                         cache=shared_cache)

    solver_budget = max(1.0, total_time_limit - heur.runtime)
    res = solve_OCSS_lazy(
        inst, alpha, time_limit=solver_budget,
        enum_threshold=TAU,
        use_mip_separation=True,
        warm_start=(heur.coalitions, heur.payoffs),
        disaggregated=disaggregated, use_lifted_cuts=True,
        cache=shared_cache,
    )
    return {
        'obj': res.obj, 'runtime': heur.runtime + res.runtime,
        'heur_runtime': heur.runtime, 'heur_value': heur.total_value,
        'heur_status': heur.status, 'heur_pool_size': heur.pool_size,
        'solver_runtime': res.runtime,
        'n_cuts': res.n_cuts, 'n_callbacks': res.n_callbacks,
        'callback_time': res.callback_time,
        'status': res.status, 'coalitions': res.coalitions,
        'mip_gap': res.mip_gap,
        'n_lifted_added': res.n_lifted_added,
        'n_basic_added': res.n_basic_added,
        # res.n_value_evals (shared cache misses) already includes the
        # heuristic's evaluations, so no summation.
        'n_value_evals': res.n_value_evals,
        'payoffs': _payoffs_json(res.payoffs),
        'heur_phase1_time': round(heur.phase1_time, 3),
        'heur_phase2_time': round(heur.phase2_time, 3),
        'heur_core_checks': heur.core_checks,
        'heur_core_feasible': heur.core_feasible,
        'heur_coalitions': str([tuple(c) for c in heur.coalitions]),
        'heur_payoffs': _payoffs_json(heur.payoffs),
    }


METHODS_EXP1 = [
    ('OCS_McC',            lambda i, a, t: run_ocs(i, a, t, disaggregated=False)),
    ('OCS_DL',             lambda i, a, t: run_ocs(i, a, t, disaggregated=True)),
    ('enum_McC',           lambda i, a, t: run_enum(i, a, t, disaggregated=False)),
    ('enum_DL',            lambda i, a, t: run_enum(i, a, t, disaggregated=True)),
    ('lazy_McC',           lambda i, a, t: run_lazy(i, a, t, False, False)),
    ('lazy_DL',            lambda i, a, t: run_lazy(i, a, t, False, True)),
    ('lazy_MIP_McC',       lambda i, a, t: run_lazy(i, a, t, True, False)),
    ('lazy_MIP_DL',        lambda i, a, t: run_lazy(i, a, t, True, True)),
    # Ablation basic arm (feeds the ablation table only; the lifted arm IS
    # lazy_MIP_McC above, shared with the SOCS methods table)
    ('lazy_MIP_basic_McC', lambda i, a, t: run_lazy(i, a, t, True, False,
                                                    use_lifted_cuts=False)),
    ('lazy_MIP_ws_McC',    lambda i, a, t: run_lazy_mip_ws(i, a, t, disaggregated=False)),
    ('lazy_MIP_ws_DL',     lambda i, a, t: run_lazy_mip_ws(i, a, t, disaggregated=True)),
]

METHODS_EXP2_BASE = [m for m in METHODS_EXP1 if not m[0].startswith('enum')]


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------

def coalition_stats(coalitions):
    if not coalitions:
        return None, None
    sizes = [len(c) for c in coalitions]
    return len(coalitions), sum(sizes) / len(sizes)


def determine_css_feasibility(method_name, status):
    """Stability methods produce SCS-feasible incumbents (the exact
    separation certificate makes this sound); OCS makes no such claim."""
    if method_name.startswith('OCS'):
        return ''
    return status in ('optimal', 'timeout')


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                done.add((int(row['n']), float(row['alpha']), row['method']))
    return done


def append_row(path, row, write_header):
    with open(path, 'a', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction='ignore')
        if write_header:
            wr.writeheader()
        wr.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def run_grid(tag, ns, methods, time_limit, out_name):
    path = os.path.join(RESULTS_DIR, out_name)
    done = load_done(path)
    need_header = not os.path.exists(path)
    print(f"\n#### {tag}: {len(ns)} sizes x {len(ALPHAS)} alphas x "
          f"{len(methods)} methods (skip {len(done)} done) ####")

    for n in ns:
        seed = SEED_BASE + n
        inst = generate_instance(n, N_ITEMS, q=Q, rho=RHO, seed=seed)
        for alpha in ALPHAS:
            for name, fn in methods:
                if (n, alpha, name) in done:
                    continue
                stamp = datetime.now().strftime('%m-%d %H:%M:%S')
                print(f"[{stamp}] {tag} n={n} a={alpha} {name} ...",
                      flush=True)
                r = fn(inst, alpha, time_limit)
                n_coal, avg_sz = coalition_stats(r.get('coalitions'))
                row = {
                    'n': n, 'seed': seed, 'alpha': alpha, 'method': name,
                    'obj': r.get('obj'), 'runtime': round(r.get('runtime', 0), 3),
                    'n_cuts': r.get('n_cuts', ''),
                    'status': r.get('status', ''),
                    'mip_gap': r.get('mip_gap', ''),
                    'n_callbacks': r.get('n_callbacks', ''),
                    'callback_time': r.get('callback_time', ''),
                    'n_iterations': r.get('n_iterations', ''),
                    'heur_runtime': r.get('heur_runtime', ''),
                    'heur_value': r.get('heur_value', ''),
                    'heur_status': r.get('heur_status', ''),
                    'heur_pool_size': r.get('heur_pool_size', ''),
                    'solver_runtime': r.get('solver_runtime', ''),
                    'coalitions': str(r.get('coalitions')),
                    'n_coalitions': n_coal if n_coal is not None else '',
                    'avg_coalition_size':
                        round(avg_sz, 2) if avg_sz is not None else '',
                    'css_feasible':
                        determine_css_feasibility(name, r.get('status', '')),
                    'n_lifted_added': r.get('n_lifted_added', ''),
                    'n_basic_added': r.get('n_basic_added', ''),
                    'n_value_evals': r.get('n_value_evals', ''),
                    'heur_phase1_time': r.get('heur_phase1_time', ''),
                    'heur_phase2_time': r.get('heur_phase2_time', ''),
                    'heur_core_checks': r.get('heur_core_checks', ''),
                    'heur_core_feasible': r.get('heur_core_feasible', ''),
                    'heur_coalitions': r.get('heur_coalitions', ''),
                    'heur_payoffs': r.get('heur_payoffs', ''),
                    'payoffs': r.get('payoffs', ''),
                    'timestamp': stamp,
                }
                append_row(path, row, need_header)
                need_header = False
                print(f"    -> obj={r.get('obj')}, "
                      f"time={r.get('runtime', 0):.1f}s, "
                      f"status={r.get('status')}", flush=True)
    print(f"#### {tag} complete -> {path} ####")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["exp1", "exp2"], default=None)
    ap.add_argument("--no-exp2-basic", action="store_true",
                    help="Skip the basic-cut ablation arm in exp2")
    args = ap.parse_args()

    t0 = time.time()
    if args.only in (None, "exp1"):
        run_grid("EXP1", EXP1_NS, METHODS_EXP1, EXP1_TIME_LIMIT,
                 "exp1_results.csv")
    if args.only in (None, "exp2"):
        methods2 = list(METHODS_EXP2_BASE)
        if args.no_exp2_basic:
            methods2 = [m for m in methods2
                        if m[0] != 'lazy_MIP_basic_McC']
        run_grid("EXP2", EXP2_NS, methods2, EXP2_TIME_LIMIT,
                 "exp2_results.csv")

    print(f"\nAll done in {(time.time()-t0)/3600:.1f} h.")
    print("Next: python run_refinement.py --max-n 30")
    print("      python run_root_lp_bounds.py --with-cuts")


if __name__ == "__main__":
    main()
