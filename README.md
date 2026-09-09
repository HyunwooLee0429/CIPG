# CIPG — Cooperative Integer Programming Games

Code and computational results for *"Cooperative Integer Programming Games:
Core Stability and Optimal Coalition Structures"* (submitted to INFORMS
Journal on Optimization).

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

Table numbers refer to the current version of the manuscript (main text
Tables 1–3, e-companion Tables EC.1–EC.9).

| CSV | Feeds |
|---|---|
| `exp1_results.csv`, `exp2_results.csv` | Tables 2, 3; EC.2 (pipeline columns), EC.3, EC.5, EC.6, EC.7, EC.8 |
| `root_lp_bounds.csv` | Root-LP bound columns of Table 2 |
| `bruteforce_baseline.csv` | Table EC.2 (brute-force columns) |
| `refinement_results.csv` | Tables EC.4, EC.9 |

Table EC.1 is the notation table and has no CSV behind it.

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

Instance parameters (paper §5): |R| = 50, q = 0.5, p ~ U{10,100},
a ~ U{5,10}, b = 0.3·Σa over available items, u^i_j = 1 for every available
pair (0–1 knapsacks), α ∈ {0.5, 1.0}.

The aggregate statistics quoted in the paper can be recomputed directly from
these files. For example, over the `lazy`, `lazy_MIP` and `lazy_MIP_ws` runs of
both experiments (that is, excluding the `lazy_MIP_basic_McC` ablation arm),
summing `n_basic_added` and `n_lifted_added` gives 25,954 fallback cuts out of
41,346 at α = 0.5 (62.8%) and 0 out of 342,585 at α = 1.0; and the
`lazy_MIP_ws` row at n = 30, α = 0.5 reports 766 coalition-value evaluations.

## Reproducing

```
python run_full_experiments.py         # main grid (long; resume-safe)
python run_root_lp_bounds.py
python run_bruteforce_baseline.py
python run_refinement.py --max-n 30
```

Gurobi logs were not persisted; the CSVs (with per-row timestamps) are the
record of each run.
