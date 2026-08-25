# Robust Belief-Space Games

> **Paper / citation:** _TBD — add context here later._

Julia implementation of a robust belief-space dynamic game solver, plus the experiment
harnesses used to produce the results (`hockey`, `senate`, `active surveillance`).

This README is a **how-to-run** reference: environment setup, launching single trials,
running parameter sweeps in parallel, and generating the analysis/plots from saved runs.

---

## 1. Repository layout

```
.
├── Project.toml                  # RobustBeliefGame — the core solver package
├── src/                          # Solver internals
│   ├── RobustBeliefGame.jl       #   module entry point
│   ├── BeliefSpaceUtils.jl       #   Belief, Beliefs, BeliefGame, BeliefCost, BeliefEnvironment
│   ├── EKF.jl                    #   ekf_update, ekf_update_with_observations
│   ├── RobustBeliefSpaceSolver.jl#   solve(::BeliefGame) — iLQG-style fwd/bwd pass w/ nature player
│   ├── MCPGame.jl / Solve.jl     #   MCP formulation + solve(::MCPGame)
│   ├── ProblemFormulation.jl
│   └── Plotting.jl
│
├── exp/
│   ├── TrajectoryAnalysis.jl     # Shared post-hoc analysis module (costs, yarnballs, summaries)
│   ├── KKTErrorTracker.jl        # KKT residual tracking + plots
│   │
│   ├── hockey/                   # ← main experiment (Attacker vs. Defender, 4D double integrator)
│   │   ├── Project.toml          #   separate Julia project, devs the root package by path
│   │   ├── src/                  #   Hockey package: params, dynamics, sensor, cost, runner, visuals
│   │   ├── versions/             #   entry-point scripts (sweeps, studies, analysis workflows)
│   │   ├── scripts/              #   one-off bash maintenance scripts
│   │   ├── outputs/              #   raw .jld2 trial results (gitignored)
│   │   └── analysis/             #   generated plots + significance reports
│   │
│   ├── senate/                   # Senate experiment (N senators, ellipsoidal costs, drift)
│   │   ├── Project.toml
│   │   ├── src/
│   │   ├── versions/
│   │   └── outputs/              #   .dat (Serialization) results
│   │
│   └── active surveillance/      # Standalone script, run against the root project
└── LICENSE
```

**Three separate Julia projects.** The root project is the solver package
(`RobustBeliefGame`). `exp/hockey` and `exp/senate` are their own packages
(`Hockey`, `Senate`) whose manifests `dev` the root package by absolute path
(`/home/ryan/codes/robust-belief-space-games/`). Pick the project matching the
experiment you want to run.

---

## 2. Setup

Julia **1.11.5** (that's what the manifests were resolved with).

```bash
# Core solver package
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Hockey experiments
julia --project=exp/hockey -e 'using Pkg; Pkg.instantiate()'

# Senate experiments
julia --project=exp/senate -e 'using Pkg; Pkg.instantiate()'
```

If the absolute dev-path in `exp/hockey/Manifest.toml` ever goes stale (e.g. the repo
was moved or cloned elsewhere), re-link it:

```bash
julia --project=exp/hockey -e 'using Pkg; Pkg.develop(path=".")'
```

Notes:

- `LocalPreferences.toml` sets `ForwardDiff.nansafe_mode = true`. This matters —
  the belief dynamics hit NaN-unsafe branches without it.
- `Makie` backend: the sweep/analysis paths use **CairoMakie** (headless, writes
  PNG/PDF). `GLMakie` is only needed for the interactive `HockeyVisuals` viewer;
  on a headless machine, skip anything that calls
  `visualize_receding_horizon_solutions_multi_figure`.
- Outputs are gitignored (`**/outputs/`, `**.png`, `**.jld2`, `**.csv`).

---

## 3. Hockey experiments

**Working directory matters.** `ExperimentRunner.setup_workers` loads code on workers
with a *relative* include (`include("./src/ExperimentRunner.jl")`), so anything that
spawns parallel workers must be launched from **`exp/hockey/`**:

```bash
cd exp/hockey
julia --project=.
```

### 3.1 Single trial (no parallelism)

```julia
using Hockey

params = HockeyParams(
    name       = "my_test",
    output_dir = "outputs/my_test",
    horizon    = 10,     # total receding-horizon steps executed
    planning_horizon = 5 # steps solved per RH iteration
)

# Defender (player 2) plays the robust (nature-augmented) game
params.player_configs[2].type = robust
params.player_configs[2].nature_control_cost_weight = 3000.0
params.player_configs[2].nature_bounds_cost_weight  = 0.25

params.sensor_model = h_noise_dict["high"]   # "low" | "medium" | "high"

result = run_receding_horizon_trial(params; override=true, trial_number=1)
# → outputs/my_test/my_test_trial_1.jld2
```

`run_receding_horizon_trials(params)` loops `trial_number = 1:params.trials` serially.

> ⚠️ Always pass `override=true`. The cache-hit branch in
> `run_receding_horizon_trial` tries to `@load` a `solutions` key that the save path
> never writes, so re-running against an existing file without `override` errors out.
> Delete the `.jld2` or override.

Per-trial RNG seed is `params.random_seed + trial_number - 1`, so trial indices are
the reproducibility handle — the same trial number always replays the same
process/sensor noise.

### 3.2 Parallel batch of trials

```julia
include("src/ExperimentRunner.jl"); using .ExperimentRunner

params_list = [p1, p2, p3]                    # Vector{HockeyParams}
run_experiment_batch(params_list; cores=40)   # flattens to Σ params.trials tasks, pmap'd
```

Useful kwargs:

| kwarg | meaning |
|---|---|
| `cores` | worker count; clamped to `min(cores, n_tasks)` |
| `trial_offset` | shift all trial indices (top up a directory without clobbering files) |
| `trial_offsets` | per-`params` offsets, same length as `params_list` |
| `auto_recycle` | kill/respawn workers when `n_tasks > n_workers` (default `true`) |

Workers are recycled + GC'd between batches; the solver holds large symbolic problem
objects and leaks memory across many trials otherwise.

### 3.3 Config-driven parameter sweep

`versions/param_sweep.jl` turns a list of override `Dict`s into `HockeyParams` and runs
the batch. Directory names are auto-generated from the config.

```julia
include("versions/param_sweep.jl")

configs = [
    Dict{Symbol,Any}(
        :name => "baseline_non_robust",
        :sensor_model => "high",                 # string key into h_noise_dict
        :player_configs => Dict(
            1 => Dict(:type => non_robust, :control_cost_weight => 0.05,
                      :terminal_cost_weight => 0.2, :steal_dist_weight => 0.01),
            2 => Dict(:type => non_robust, :control_cost_weight => 0.05,
                      :terminal_cost_weight => 0.2, :steal_dist_weight => 0.01),
        ),
        :planning_horizon => 5, :horizon => 10, :trials => 3,
    ),
]

run_param_sweep(configs; cores=40, output_subdir="sweep")
# → outputs/sweep/<generated-name>/<config-name>_trial_<n>.jld2
```

**Directory naming scheme** (`generate_name_from_config`) — abbreviations you'll see in
`outputs/sweep/` and `analysis/`:

| token | parameter |
|---|---|
| `p<i>_cc`  | `control_cost_weight` |
| `p<i>_tc`  | `terminal_cost_weight` |
| `p<i>_sd`  | `steal_dist_weight` |
| `p<i>_ncc` | `nature_control_cost_weight` (robust only) |
| `p<i>_nbc` | `nature_bounds_cost_weight` (robust only) |
| `sns`      | sensor model (`low`/`medium`/`high`) |

`name`, `trials`, `horizon`, `planning_horizon` are excluded from the directory name.
A directory **with** `_ncc`/`_nbc` is a robust run; **without** is the non-robust
baseline — that's exactly how the analysis code pairs them
(`is_robust_config` / `extract_base_config`).

### 3.4 The big factorial sweep

`versions/test_sweep.jl` defines the full-factorial sweep used for the paper-scale runs
(3-point log scale over `cc`, `tc`, `sd` per player × 3 sensor models × {non-robust, 3
`nccw` values}). It's **large** — ~8.7k configs × 3 trials — and prompts for `y`
before launching.

```julia
include("versions/test_sweep.jl")

run_sweep()                  # full factorial → outputs/sweepv2, then auto-analyzes
run_non_robust_baselines()   # only the non-robust baselines → outputs/sweep
```

Both prompt for confirmation and use `cores=40`.

### 3.5 Focused study

```julia
include("versions/control_cost_study.jl")
run_control_cost_study(weights=[0.01, 0.1, 0.5, 1.0], cores=4)
```

Sweeps the robust defender's `control_cost_weight` only.

---

## 4. Analysis & plots (hockey)

All of these live in `versions/test_sweep.jl` (loaded above) and lean on
`exp/TrajectoryAnalysis.jl`.

### 4.1 Rank configs by robust-vs-non-robust improvement

```julia
results = analyze_sweep_results(sweep_dir="outputs/sweep", top_k=10)
```

Pairs each robust directory with its non-robust baseline, then prints two leaderboards:
largest **defender cost improvement** and largest **trajectory difference**. Each entry
also carries per-arm mean/std cost, trial counts, and the fraction of KKT residuals
above `0.01`.

### 4.2 Interactive explorer

```julia
visualize_sweep(k=10, sweep_dir="outputs/sweep")
```

A REPL loop that lets you:

1. sort by cost improvement or trajectory difference,
2. pick a rank,
3. optionally **run more trials** for that config (`[1]`), or **run robust variants at
   new `nature_control_cost_weight` values** (`[2]`, comma-separated list),
4. then generate the comparison plots for that rank.

New trials are written into the existing directory with a `trial_offset` equal to the
current file count, so nothing is overwritten.

Programmatic equivalents:

```julia
run_extra_trials(entry, 20, "outputs/sweep")
run_nccw_variant_trials(entry, [100.0, 1000.0, 3000.0], 10, "outputs/sweep")
run_nccw_variant_trials_to_target(entry, [100.0, 1000.0, 3000.0], 50, "outputs/sweep")
```

`..._to_target` tops every variant up to N total trials in a **single** batch (one
worker pool across all `nccw` values) and skips variants already at target.

### 4.3 Comparison plots for one config

```julia
visualize_sweep_rank(3; sweep_dir="outputs/sweep", results=results)
```

Writes to `exp/hockey/analysis/<robust-dir-name>/`:

- `qq_plots.png` — normality check, with an **interactive outlier-trimming loop**
  (`[r]`/`[n]` to adjust how many highest-cost trials to drop per arm, `[c]` to accept)
- `significance_report.txt` — Welch's t-test, Mann-Whitney U, and a 10k-iteration
  bootstrap (seeded at 42) on defender total cost
- `defender_cost_comparison.{png,pdf}` — instantaneous + cumulative defender cost,
  mean ± 1σ bands, robust vs non-robust
- `rank_<n>_defender_yarnball.png`, `yarnball_*_cost_grid.{png,pdf}` — per-component
  cost yarnballs
- `control_trajectory_2d_*`, `control_differences_*`, `control_norm_angle_*` — per-trial
  control diagnostics
- `kkt.png` — KKT residual evolution

Note the outlier loop is interactive and defaults to removing 2 robust / 1 non-robust —
it will block waiting on stdin.

### 4.4 Two-pass `nccw` sweep analysis

`versions/nccw_comparison.jl` compares one base config across many
`nature_control_cost_weight` values against a shared non-robust baseline.

```julia
include("versions/test_sweep.jl")
include("versions/nccw_comparison.jl")

# Pass 1 — generate Q-Q plots so you can eyeball outliers per nccw variant
groups = generate_qq_plots_for_nccw_groups(sweep_dir="outputs/sweep", min_trials=50)
# → analysis/<base_config>/nccw_<value>/qq_plots.png

# Pass 2 — rerun with the outlier counts you chose
analyze_nccw_groups(
    robust_outliers = Dict(10.0=>4, 100.0=>4, 300.0=>1, 3000.0=>1, 5000.0=>1),
    non_robust_outliers = 1,
    min_trials = 50,
)
```

`min_trials` gates which variants qualify (default 50 per directory). The non-robust
baseline is loaded once per group and shared across all `nccw` variants.

### 4.5 Raw directory analysis

```julia
include("../TrajectoryAnalysis.jl"); using .TrajectoryAnalysis

load_and_analyze_solution_files(directory="outputs/sweep/<some-config-dir>")
get_trajectory_summary(directory="outputs/sweep/<some-config-dir>")
```

`load_and_analyze_solution_files` clears and repopulates the global
`TRAJECTORY_TRACKER` / KKT tracker from every `.jld2` in a directory;
`get_trajectory_summary` then emits the cost-component and action-comparison plots.
Both operate on globals — clear between directories
(`clear_trajectory_tracker!()`, `KKTErrorTracker.clear_rh_kkt_tracker!()`).

---

## 5. Hockey configuration reference

`HockeyParams` (`src/HockeyParams.jl`), all mutable, all keyword-constructible:

| field | default | notes |
|---|---|---|
| `name` | `"hockey_default"` | filename stem: `<name>_trial_<n>.jld2` |
| `output_dir` | `"outputs"` | relative to cwd unless absolute |
| `player_configs` | `Dict(1 => attacker, 2 => defender)` | see below |
| `goal_position` | `[[0.25,-1.5], [-0.25,-1.5]]` | |
| `planning_horizon` | `5` | steps solved per RH iteration |
| `horizon` | `10` | executes `horizon - 1` RH steps |
| `dt` | `0.3` | |
| `process_noise_distribution` | `MvNormal(zeros(8), 1e-3 I)` | 2 players × 4 states |
| `sensor_noise_distribution` | `MvNormal(zeros(8), 1e-3 I)` | |
| `sensor_model` | `h_noise_dict["low"]` | `low`/`medium`/`high` → noise gain `0.1`/`1.0`/`10.0` |
| `ground_truth_initial_states` | attacker `[0,5,0.5,0]`, defender `[0,1.5,0,0]` | `mortar`ed BlockVector |
| `initial_beliefs` | `nothing` | `nothing` ⇒ default 4-belief construction |
| `trials` | `1` | |
| `random_seed` | `1` | effective seed = `random_seed + trial - 1` |

`PlayerConfig`:

| field | default | notes |
|---|---|---|
| `type` | `non_robust` | `non_robust` \| `robust` \| `nature` \| `ground_truth_config` |
| `control_cost_weight` | `0.05` | |
| `terminal_cost_weight` | `2.0` | |
| `boundary_cost_weight` | `10.0` | |
| `steal_dist_weight` | `0.1` | |
| `shot_uncertainty_weight` | `20.0` | |
| `nature_control_cost_weight` | `300.0` | **robust only** — higher ⇒ weaker adversarial nature |
| `nature_bounds_cost_weight` | `10.0` | **robust only** |

State is `(x, y, vx, vy)` per player; control is `(ax, ay)`. Player 1 = attacker,
player 2 = defender. Only the **defender** is ever made robust in these experiments
(`robust_ids = [2]`).

Beliefs are indexed `[1..4]` = (attacker's belief of attacker, attacker's belief of
defender, defender's belief of attacker, defender's belief of defender).

### What's inside a saved `.jld2`

```julia
using JLD2, FileIO
d = load("outputs/sweep/<dir>/<name>_trial_1.jld2")
d["gt_state_history"]     # Vector{BlockVector}, length horizon
d["observation_history"]  # Vector of per-player observations
d["solution_history"]     # Dict(player_idx => Vector of (; beliefs, controls, kkt_error, costs))
d["params"]               # the full HockeyParams used
```

Loading requires the `Hockey` module to be available for JLD2 type resolution — hence
`using .TrajectoryAnalysis.Hockey` at the top of the analysis scripts.

---

## 6. Senate experiments

Senate paths are hardcoded **relative to the repo root**, so run these from the repo
root with the senate project active:

```bash
julia --project=exp/senate
```

```julia
using Senate

include("exp/senate/versions/base_experiment.jl")
base_experiment(override=false)          # → exp/senate/outputs/runs/base_experiment.dat
robust_base_experiment(override=false)

include("exp/senate/versions/underactuated_experiment.jl")
robust_underactuated_experiment()
non_robust_underactuated_attraction_experiment()

include("exp/senate/versions/mass_experiment.jl")
run_mass_experiment(p1_control_cost_weight=..., trials=5, experiment_name_prefix="mass_exp")

include("exp/senate/versions/assymetric_experiment.jl")
run_asymmetric_experiment(...)           # asymmetric-belief variants

include("exp/senate/versions/drift_experiment_runner.jl")
run_all_drift_experiments()              # batteries of asymmetric drift experiments
run_all_obstacle_cost_drift_experiments()
```

Results are `Serialization`-format `.dat` files under `exp/senate/outputs/runs/`.
Pass `override=true` to recompute an existing run.

Sweep-style runners (`run_mass_experiment`, `run_asymmetric_experiment`) accept any
parameter as a single value or a range spec — anything given multiple values becomes a
swept dimension in the cross-product.

To tidy `outputs/runs/` into a browsable tree (`per_param/`, `robustness/`,
`p2_wrong_tables/`):

```julia
include("exp/senate/src/organize_outputs.jl")
organize_outputs()          # or organize_outputs("asym") to filter by prefix
```

---

## 7. Active surveillance

Standalone script against the root project:

```bash
julia --project=. -e 'include("exp/active surveillance/ActiveSurveillance.jl"); belief_main()'
```

---

## 8. Maintenance scripts

Run from the repo root:

```bash
bash exp/hockey/scripts/rename_sweep_dirs.sh      # tag legacy dirs lacking _ncc/_nbc as ncc300/nbc10
bash exp/hockey/scripts/archive_fake_nonrobust.sh # move them out to outputs/sweep_archive_ncc300_nbcw10
```

Historical fix: early sweeps defaulted to `type=robust` with `nccw=300, nbcw=10` but
produced directory names *without* `_ncc`/`_nbc`, so the analysis code misread them as
non-robust baselines. These two scripts rename or quarantine those directories. Only
relevant to pre-existing data.

---

## 9. Gotchas

- **cwd for hockey sweeps must be `exp/hockey/`** — worker setup uses a relative
  `include("./src/ExperimentRunner.jl")`.
- **cwd for senate must be the repo root** — output paths are hardcoded as
  `exp/senate/outputs/...`.
- **`override=true`** on hockey trials; the cached-load path is broken (§3.1).
- **Interactive prompts**: `run_sweep`, `run_non_robust_baselines`, `visualize_sweep`,
  and the Q-Q outlier loop inside `generate_comparison_plots` all read stdin. Don't run
  them under `julia -e` in a non-interactive shell.
- **Memory**: long batches leak through the symbolic problem objects. `auto_recycle`
  handles this by killing workers between batches; keep it on.
- **KKT residuals** are stored per solve; `analyze_sweep_results` reports the fraction
  above `0.01`. High percentages mean the iLQG solve isn't converging for that config —
  treat those cost comparisons with suspicion.
- **Outputs are gitignored.** Nothing under `outputs/` and no `.png`/`.jld2`/`.csv`
  is tracked; `exp/hockey/analysis/` PDFs are the exception worth keeping around.
