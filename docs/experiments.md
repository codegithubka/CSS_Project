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