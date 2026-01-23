# Metrics and measures
This is what should be measured each run. These runs can then be further aggregated for final metrics.
### Fixed parameter runs
- Population count (mean and std after warmup)
- Cluster size distribution (means and stds after warmup)
### Evolution runs
- Population count (over time after warmup)
- Cluster size distribution (over time after warmup)
- Prey death rate (mean and std over time after warmup)
# Experiments
These phases should be completed sequentially, deepening our understanding at each step. The different experiments in each phase should be completed with data from the same runs.
### Phase 1: finding the critical point
- Create bifurcation diagram of mean population count, varying prey death rate
	- Look for critical transition
- Create log-log plot of cluster size distribution, varying prey death rate
	- Look for power-law
### Phase 2: sensitivity analysis
- Show correlation between critical prey death rate and post-evolution prey death rate, varying other parameters
	- Look for self-organized criticality: an SOC-system should move towards the critical point regardless of other parameters
- Show sensitivity of hydra effect varying other parameters
### Phase 3: perturbation analysis
- Create autocorrelation plot of mean population count, following perturbations around the critical point
	- Look for critical slowing down: perturbations to states closer to the critical point should more slowly return to the steady state
### Phase 4: model extensions
- Investigate whether hydra effect and SOC still occur with diffusion and directed movement

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













