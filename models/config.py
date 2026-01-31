#!/usr/bin/env python3
"""
Experiment Configuration
========================

This module provides the configuration dataclass and pre-defined phase
configurations for Predator-Prey Hydra Effect experiments.

Classes
-------
Config
    Central configuration dataclass with all experiment parameters.

Functions
---------
```python
get_phase_config: Retrieve configuration for a specific experimental phase.
````

Phase Configurations
--------------------
- ``PHASE1_CONFIG``: Parameter sweep to find critical point
- ``PHASE2_CONFIG``: Self-organization (evolution toward criticality)
- ``PHASE3_CONFIG``: Finite-size scaling at critical point
- ``PHASE4_CONFIG``: Sensitivity analysis (4D parameter sweep)
- ``PHASE5_CONFIG``: Directed hunting comparison

Example
-------
```python
from models.config import Config, get_phase_config

# Use predefined phase config
cfg = get_phase_config(1)

# Create custom config
cfg = Config(grid_size=200, n_replicates=10)

# Generate parameter sweep values
prey_deaths = cfg.get_prey_deaths()
```
"""
from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class Config:
    """
    Central configuration for Predator-Prey Hydra Effect experiments.

    Attributes
    ----------
    grid_size : int
        Side length of the square simulation grid.
    densities : Tuple[float, float]
        Initial population fractions for (prey, predator).
    grid_sizes : Tuple[int, ...]
        Grid dimensions for Finite-Size Scaling (FSS) analysis (Phase 3).
    prey_birth : float
        Global birth rate for prey species.
    prey_death : float
        Global death rate for prey species.
    predator_birth : float
        Global birth rate for predator species.
    predator_death : float
        Global death rate for predator species.
    critical_prey_birth : float
        Critical birth rate identified from Phase 1.
    critical_prey_death : float
        Critical death rate identified from Phase 1.
    prey_death_range : Tuple[float, float]
        Bounds for prey death rate sweep.
    n_prey_death : int
        Number of points in prey death rate sweep.
    n_replicates : int
        Independent stochastic runs per parameter set.
    warmup_steps : int
        Iterations before data collection begins.
    measurement_steps : int
        Iterations for collecting statistics.
    evolve_sd : float
        Standard deviation for parameter mutation.
    evolve_min : float
        Lower bound for evolving parameters.
    evolve_max : float
        Upper bound for evolving parameters.
    directed_hunting : bool
        Toggle for targeted predator movement.
    save_timeseries : bool
        Toggle for recording population time series.
    timeseries_subsample : int
        Subsample rate for time series data.
    collect_pcf : bool
        Toggle for Pair Correlation Function analysis.
    pcf_sample_rate : float
        Fraction of runs that compute PCFs.
    pcf_max_distance : float
        Maximum radial distance for PCF.
    pcf_n_bins : int
        Number of bins in PCF histogram.
    min_density_for_analysis : float
        Population threshold for spatial analysis.
    n_jobs : int
        CPU cores for parallelization (-1 = all).
    """

    # Grid settings
    grid_size: int = 1000
    densities: Tuple[float, float] = (0.30, 0.15)
    grid_sizes: Tuple[int, ...] = (50, 100, 250, 500, 1000, 2500)

    # Species parameters
    prey_birth: float = 0.2
    prey_death: float = 0.05
    predator_birth: float = 0.8
    predator_death: float = 0.05

    # Critical point (from Phase 1)
    critical_prey_birth: float = 0.20
    critical_prey_death: float = 0.0968

    # Parameter sweep settings
    prey_death_range: Tuple[float, float] = (0.0, 0.2)
    n_prey_death: int = 20

    # Replication
    n_replicates: int = 15

    # Simulation timing
    warmup_steps: int = 300
    measurement_steps: int = 500

    # Evolution settings
    evolve_sd: float = 0.10
    evolve_min: float = 0.0
    evolve_max: float = 0.10

    # Model variant
    directed_hunting: bool = False

    # Time series collection
    save_timeseries: bool = False
    timeseries_subsample: int = 10

    # PCF settings
    collect_pcf: bool = True
    pcf_sample_rate: float = 0.2
    pcf_max_distance: float = 20.0
    pcf_n_bins: int = 20

    # Analysis thresholds
    min_density_for_analysis: float = 0.002

    # Parallelization
    n_jobs: int = -1

    def get_prey_deaths(self) -> np.ndarray:
        """Generate array of prey death rates for parameter sweep."""
        return np.linspace(
            self.prey_death_range[0], self.prey_death_range[1], self.n_prey_death
        )

    def get_warmup_steps(self, L: int) -> int:
        """Get warmup steps (can be extended for size-dependent scaling)."""
        return self.warmup_steps

    def get_measurement_steps(self, L: int) -> int:
        """Get measurement steps (can be extended for size-dependent scaling)."""
        return self.measurement_steps

    def estimate_runtime(self, n_cores: int = 32) -> str:
        """
        Estimate wall-clock time for the experiment.

        Parameters
        ----------
        n_cores : int
            Number of available CPU cores.

        Returns
        -------
        str
            Human-readable runtime estimate.
        """
        ref_size = 100
        ref_steps_per_sec = 1182

        size_scaling = (self.grid_size / ref_size) ** 2
        actual_steps_per_sec = ref_steps_per_sec / size_scaling

        total_steps = self.warmup_steps + self.measurement_steps
        base_time_s = total_steps / actual_steps_per_sec

        pcf_time_s = (0.008 * size_scaling) if self.collect_pcf else 0

        n_sims = self.n_prey_death * self.n_replicates

        total_seconds = n_sims * (base_time_s + pcf_time_s * self.pcf_sample_rate)
        total_seconds /= n_cores

        hours = total_seconds / 3600
        core_hours = n_sims * (base_time_s + pcf_time_s * self.pcf_sample_rate) / 3600

        return f"{n_sims:,} sims, ~{hours:.1f}h on {n_cores} cores (~{core_hours:.0f} core-hours)"


# =============================================================================
# Phase Configurations
# =============================================================================

PHASE1_CONFIG = Config(
    grid_size=1000,
    n_prey_death=20,
    prey_birth=0.2,
    prey_death_range=(0.0963, 0.0973),
    predator_birth=0.8,
    predator_death=0.05,
    n_replicates=30,
    warmup_steps=1000,
    measurement_steps=1000,
    collect_pcf=False,
    save_timeseries=False,
    directed_hunting=False,
)

PHASE2_CONFIG = Config(
    grid_size=1000,
    n_prey_death=10,
    n_replicates=10,
    warmup_steps=1000,
    measurement_steps=10000,
    evolve_sd=0.01,
    evolve_min=0.0,
    evolve_max=0.20,
    collect_pcf=False,
    save_timeseries=False,
)

PHASE3_CONFIG = Config(
    grid_sizes=(50, 100, 250, 500, 1000, 2500),
    n_replicates=20,
    warmup_steps=1000,
    measurement_steps=1000,
    critical_prey_birth=0.20,
    critical_prey_death=0.0968,
    collect_pcf=True,
    pcf_sample_rate=1.0,
    save_timeseries=False,
    directed_hunting=False,
)

PHASE4_CONFIG = Config(
    grid_size=250,
    n_replicates=10,
    warmup_steps=500,
    measurement_steps=500,
    collect_pcf=False,
    save_timeseries=False,
    directed_hunting=False,
)

PHASE5_CONFIG = Config(
    grid_size=250,
    n_replicates=10,
    warmup_steps=500,
    measurement_steps=500,
    collect_pcf=False,
    save_timeseries=False,
    directed_hunting=True,
)

PHASE_CONFIGS = {
    1: PHASE1_CONFIG,
    2: PHASE2_CONFIG,
    3: PHASE3_CONFIG,
    4: PHASE4_CONFIG,
    5: PHASE5_CONFIG,
}


def get_phase_config(phase: int) -> Config:
    """
    Retrieve configuration for a specific experimental phase.

    Parameters
    ----------
    phase : int
        Phase number (1-5).

    Returns
    -------
    Config
        Configuration instance for the requested phase.

    Raises
    ------
    ValueError
        If phase number is invalid.
    """
    if phase not in PHASE_CONFIGS:
        raise ValueError(
            f"Unknown phase {phase}. Valid phases: {list(PHASE_CONFIGS.keys())}"
        )
    return PHASE_CONFIGS[phase]