# CIPG — Cooperative Integer Programming Games

Code and computational results for *"Cooperative Integer Programming Games:
Core Stability and Optimal Coalition Structures"* (submitted to INFORMS
Journal on Optimization).

**Private repository — do not distribute.**

## Environment

- Python 3.10+, `gurobipy` 13.0.0 (requires a Gurobi license), `pandas`
- All reported experiments: Intel i9-13900F (24 cores), 64 GB RAM, 16 threads,
  enumeration threshold τ = 9

```
pip install gurobipy pandas
```

## Layout

| Path | Contents |
|---|---|
| `cipg/` | Library: instance generator, coalition-value IP (+cache), OCS master (McC/DL), lazy-constraint solver with exact separation, SCS-feasible heuristic, Shapley/nucleolus refinement |
| `run_full_experiments.py` | Main grid (Experiments 1 & 2, all configurations; resume-safe, per-row fsync) |
| `run_root_lp_bounds.py` | Root-LP bounds McC vs. DL, incl. a-priori-cuts inertness check |
| `run_bruteforce_baseline.py` | Exhaustive baseline: all 2^n−1 values, partition DP, core LPs (stage caps + geometric projections) |
| `run_refinement.py` | Per-coalition Shapley / nucleolus / least-core post-processing |
| `results/` | All raw results (CSV) |

The paper's LaTeX tables were produced from these CSVs and finalized in the
manuscript source.

## Results files → paper tables

| CSV | Feeds |
|---|---|
| `exp1_results.csv`, `exp2_results.csv` | Tables 3, 4; EC.2, EC.4, EC.5, EC.6, EC.7 |
| `root_lp_bounds.csv` | Root-LP columns of Table 3 |
| `bruteforce_baseline.csv` | Table EC.1 |
| `refinement_results.csv` | Tables EC.3, EC.8 |

## Data dictionary (main grid CSVs)

One row per (n, alpha, method). Key columns:

- `n, seed, alpha, method` — instance and configuration (`seed = 42 + n`;
  methods: `OCS/enum(iter)/lazy/lazy_MIP/lazy_MIP_ws` × `McC/DL`, plus the
  `lazy_MIP_basic_McC` ablation arm in exp1)
- `obj, runtime, status, mip_gap` — incumbent value, wall-clock seconds,
  `optimal`/`timeout`, Gurobi relative gap (fraction, not %)
- `n_cuts, n_callbacks, callback_time, n_lifted_added, n_basic_added` —
  separation statistics (lifted/basic split = synergy-fallback accounting)
- `n_value_evals` — distinct coalition-value IP solves (heuristic and
  solver share one value cache)
- `heur_*` — warm-start diagnostics: runtime, value, status, pool size,
  phase-1/2 times, core checks, core-feasible count, seeded partition/payoffs
- `coalitions, n_coalitions, avg_coalition_size, css_feasible, payoffs,
  timestamp`

Instance parameters (paper §4): |R| = 50, q = 0.5, p ~ U{10,100},
a ~ U{5,10}, b = 0.3·Σa over available items, u^i_j = 1 for every available
pair (0–1 knapsacks), α ∈ {0.5, 1.0}.

## Reproducing

```
python run_full_experiments.py         # main grid (long; resume-safe)
python run_root_lp_bounds.py
python run_bruteforce_baseline.py
python run_refinement.py --max-n 30
```

Gurobi logs were not persisted; the CSVs (with per-row timestamps) are the
record of each run.
