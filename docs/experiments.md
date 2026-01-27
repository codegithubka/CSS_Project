# Metrics and measures
This is what should be measured each run. These runs can then be further aggregated for final metrics.
### Fixed parameter runs
- Population count (mean and std after warmup)
- Cluster size distribution (means and stds after warmup)
### Evolution runs
It is important to scrutenize whether these should be time-series or steady state values.
- Population count (time series after warmup)
- Cluster size distribution (time series after warmup)
- Prey death rate (time series mean and std after warmup)

# Experiments
These phases should be completed sequentially, deepening our understanding at each step. The different experiments in each phase should be completed with data from the same runs.
### Phase 1: finding the critical point
- Create bifurcation diagram of mean population count, varying prey death rate
	- Look for critical transition
- Create log-log plot of cluster size distribution, varying prey death rate
	- Look for power-law
### Phase 2: self-organization
- Measure final prey death rate after evolution
	- Look for self-organized criticality: an SOC-system should move towards the critical point
### Phase 3: finite-size scaling
- Sweep of grid sizes at critical point
	- Check for power-law cut-offs
### Phase 4: sensitivity analysis
- Show sensitivity of hydra effect varying other parameters
	- Investigate the ubiquity of the critical point across parameter regimes
- Show correlation between critical prey death rate and post-evolution prey death rate, varying other parameters
	- Again look for self-organized criticality: an SOC-system should move towards the critical point regardless of other parameters
### Phase 5: perturbation analysis
- Create autocorrelation plot of mean population count, following perturbations around the critical point
	- Look for critical slowing down: perturbations to states closer to the critical point should more slowly return to the steady state
	- This requires time series data
### Phase 6: model extensions
- Investigate whether hydra effect and SOC still occur with diffusion and directed reproduction

# Todo
The main functionality is all complete. Thus, the models folder should be relatively untouched.
However, it is important to standardize experiments and analysis. The following files should be used for this.
These files should contain very little (if any) functionality outside of what is listed here.
### experiments.py
This is the file that will be run on the cluster and should generate all experiment data.
- General config class to setup experiments (grid size, parameters, sweep, evolution, repetitions, etc.)
- Config objects for each phase (see phases above)
- Function that runs the experiment based on a config object (calls run_single_simulation in parallel)
	- Should save results to results folder (which can then be used by analysis.py)
- Function that runs a single simulation, saving all necessary results
	- This needs functionality to run a predetermined amount of time with a warmup
	- And needs functionality to dynamically run until it has found a steady state
- Should not contain any analysis (power-law fitting, bifurcation, etc.)
	- Exception to this is the PCF data
- Function to estimate runtime (already exists)
- Should have argparse functionality to choose which phase to execute
- Nice-to-have: argparse functionality to create new config object for arbitrary experiments
### analysis.py
This is the file that will generate our plots and statistics for the analysis.
- Function to create bifurcation diagram to find critical point
- Function to create log-log plot to check for power-law
	- Should also fit a power-function to the data (see scrips/experiments.fit_truncated_power_law)
- Function to calculate/ show similarity between post-evolution prey death rates and critical points
- Function for sensitivity analysis
- Function for perturbation analysis

---


## What we are currently collecting:

### 2D Parameter Sweep

We map the full phase space to find:
- Hydra regions
- Critical points
- Coexistence boundaries
- Evolutionary advantage zones

For now at least we sweep:

```
prey_birth in [0.10, 0.35]  
prey_death in [0.001, 0.10]
```

Metrics Collected (so far):

1. Population Dynamics

```

prey_mean: time-averaged prey pop
prey_std: variability in prey

# same for predator as above

prey_survived: did prey persist
pred_survived: did pred perist

```

2. Cluster structure

```

prey_n_clusters: total number of prey clusters
pred_n_clusters: total number of pred clusters
prey_tau: power law exp
prey_s_c: cutoff cluster sizes
pred_tau: pred cluster exp
pred_s_c: pred cutoff

```

3. Order Parameters

```
prey_largest_fraction_mean
prey_largest_fraction_std
pred_largest_fraction_mean
prey_percolation_prob: fraction of samples with spanning cluster
pred_percolation_prob: predator percolation prob

```


4. Spatial Correlations

```

pcf_distances: distance bins in lattice units
pcf_prey_prey_mean: prey-prey correlation function
pcf_pred_pred_mean
pcf_prey_pred_mean
segregation_index: measure spatial mixing
prey_clustering_index: short range prey clustering
pred_clustering_index

```

5. Evolutionary dynamics

```
evolved_prey_death_mean: time avg evolved mortality rate
evolved_prey_death_std
evolved_prey_death_final
evolve_sd: mutation strength used

```

---

### Finite-size scaling

We choose a fixed point identified in the main simulation run ```(target_prey_birth, target_prey_death)``` ideally near hydra boundary.


For selected grid sizes (TBD) we run independent reps for each size.


Metrics:

```
grid_size
prey_mean, prey_std
prey_survived: bool
prey_largest_fraction: order parameter
prey_percolation_prob
prey_tau: grid size dependent exponent
prey_tau_se: SE on tau
prey_s_c: cutoff scales
```

---

### Evo Sensitivity

How does mutation strength affect evolutionary advantage in Hydra regions, speed of adaptation and final evolved mortality rates.

Again. choose fixed point identified from main analysis.

Metrics Dict:

```
prey_mean: in cell units as the below metrics as well
prey_std
pred_mean
pred_std
prey_survived: bool

+ same cluster metrics and spatial correlation metrics


evolved_prey_death_mean: avg mortality across all prey
evolved_prey_death_std
evolved_prey_death_final
evolve_sd
```


## Additions Required:

1. Temporal dynamics for time series analysis. Needed to add critical slowing down effect near phase transitions.

```
result["prey_timeseries"] = prey_pops[::10]  # Subsample every 10 steps
result["pred_timeseries"] = pred_pops[::10]

```

```
def run_perturbation_experiment(...):
    # Save full time series only for these special runs
```

2. Snapshots of spatial configurations. This is a costly operation so we need to figure out how and when to do it in the sim.

3. Saving final grid states?

```
result["final_grid"] = model.grid.copy()
```













