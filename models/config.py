#!/usr/bin/env python3
"""
Configuration for Predator-Prey Hydra Effect Experiments

Single Config dataclass with pre-defined instances for each experimental phase.

Usage:
    from config import PHASE1_CONFIG, PHASE2_CONFIG, Config
    
    # Use pre-defined config
    cfg = PHASE1_CONFIG
    
    # Or create custom config
    cfg = Config(grid_size=150, n_replicates=20)
    
    # Or modify existing
    cfg = Config(**{**asdict(PHASE1_CONFIG), 'n_replicates': 30})
"""

from dataclasses import dataclass, field, asdict
from typing import Tuple, Optional
import numpy as np


@dataclass
class Config:
    """
    Central configuration for Predator-Prey Hydra Effect experiments.

    This dataclass aggregates all hyperparameters, grid settings, and 
    experimental phase definitions. It includes helper methods for 
    parameter sweep generation and runtime estimation.

    Attributes
    ----------
    grid_size : int, default 1000
        The side length of the square simulation grid.
    densities : Tuple[float, float], default (0.30, 0.15)
        Initial population fractions for (prey, predator).
    grid_sizes : Tuple[int, ...], default (50, 100, 250, 500, 1000, 2500)
        Grid dimensions used for Finite-Size Scaling (FSS) analysis.
    prey_birth : float, default 0.2
        Default global birth rate for the prey species.
    prey_death : float, default 0.05
        Default global death rate for the prey species.
    predator_birth : float, default 0.8
        Default global birth rate for the predator species.
    predator_death : float, default 0.05
        Default global death rate for the predator species.
    critical_prey_birth : float, default 0.20
        Identified critical birth rate for prey (Phase 1 result).
    critical_prey_death : float, default 0.947
        Identified critical death rate for prey (Phase 1 result).
    prey_death_range : Tuple[float, float], default (0.0, 0.2)
        Bounds for the prey death rate sweep in Phase 1.
    n_prey_birth : int, default 15
        Number of points along the prey birth rate axis for sweeps.
    n_prey_death : int, default 5
        Number of points along the prey death rate axis for sweeps.
    predator_birth_values : Tuple[float, ...]
        Discrete predator birth rates used for sensitivity analysis.
    predator_death_values : Tuple[float, ...]
        Discrete predator death rates used for sensitivity analysis.
    prey_death_offsets : Tuple[float, ...]
        Delta values applied to critical death rates for perturbation tests.
    n_replicates : int, default 15
        Number of independent stochastic runs per parameter set.
    warmup_steps : int, default 300
        Iterations to run before beginning data collection.
    measurement_steps : int, default 500
        Iterations spent collecting statistics after warmup.
    with_evolution : bool, default False
        Toggle for enabling per-cell parameter mutation.
    evolve_sd : float, default 0.10
        Standard deviation for parameter mutation (Gaussian).
    evolve_min : float, default 0.0
        Lower bound clamp for evolving parameters.
    evolve_max : float, default 0.10
        Upper bound clamp for evolving parameters.
    sensitivity_sd_values : Tuple[float, ...]
        Range of mutation strengths tested in sensitivity phases.
    synchronous : bool, default False
        If True, use synchronous grid updates (not recommended).
    directed_hunting : bool, default False
        Toggle for targeted predator movement logic.
    directed_hunting_values : Tuple[bool, ...]
        Options compared during Phase 6 extensions.
    save_timeseries : bool, default False
        Toggle for recording step-by-step population data.
    timeseries_subsample : int, default 10
        Frequency of temporal data points (e.g., every 10 steps).
    collect_pcf : bool, default True
        Toggle for Pair Correlation Function analysis.
    pcf_sample_rate : float, default 0.2
        Probability that a specific replicate will compute PCFs.
    pcf_max_distance : float, default 20.0
        Maximum radial distance for spatial correlation analysis.
    pcf_n_bins : int, default 20
        Number of bins in the PCF histogram.
    min_density_for_analysis : float, default 0.002
        Population threshold below which spatial analysis is skipped.
    perturbation_magnitude : float, default 0.1
        Strength of external shocks applied in Phase 5.
    n_jobs : int, default -1
        Number of CPU cores for parallelization (-1 uses all available).
    """

    # Grid settings
    grid_size: int = 1000  # FIXME: Decide default configuration
    densities: Tuple[float, float] = (
        0.30,
        0.15,
    )  # (prey, predator)  #FIXME: Default densities

    # For FSS experiments: multiple grid sizes
    grid_sizes: Tuple[int, ...] = (50, 100, 250, 500, 1000, 2500)

    # Default/fixed parameters
    prey_birth: float = 0.2
    prey_death: float = 0.05
    predator_birth: float = 0.8  # FIXME: Default predator death rate
    predator_death: float = 0.05  # FIXME: Default predator death rate

    # Critical point (UPDATE AFTER PHASE 1)
    critical_prey_birth: float = 0.20
    critical_prey_death: float = 0.947

    # Prey parameter sweep (Phase 1)
    prey_death_range: Tuple[float, float] = (0.0, 0.2)
    n_prey_birth: int = 15  # FIXME: Decide number of grid points along prey axes
    n_prey_death: int = 5

    # Predator parameter sweep (Phase 4 sensitivity)
    predator_birth_values: Tuple[float, ...] = (
        0.15,
        0.20,
        0.25,
        0.30,
    )  # FIXME: Bogus values for now
    predator_death_values: Tuple[float, ...] = (
        0.05,
        0.10,
        0.15,
        0.20,
    )  # FIXME: Bogus values for now

    # Perturbation offsets from critical point (Phase 5)
    prey_death_offsets: Tuple[float, ...] = (
        -0.02,
        -0.01,
        0.0,
        0.01,
        0.02,
    )  # FIXME: Bogus values for now

    # Number of replicates per parameter configuration
    n_replicates: int = 15  # FIXME: Decide number of indep. runs per parameter config

    # Simulation steps
    warmup_steps: int = 300  # FIXME: Steps to run before measuring
    measurement_steps: int = 500  # FIXME: Decide measurement steps

    # Evo
    with_evolution: bool = False
    evolve_sd: float = 0.10
    evolve_min: float = 0.0
    evolve_max: float = 0.10

    # Sensitivity: mutation strength values to test
    sensitivity_sd_values: Tuple[float, ...] = (
        0.02,
        0.05,
        0.10,
        0.15,
        0.20,
    )  # FIXME: Don't know if we use yet

    # Update mode
    synchronous: bool = False  # Always False for this model
    directed_hunting: bool = False

    # For Phase 6: compare model variants
    directed_hunting_values: Tuple[bool, ...] = (False, True)

    # Temporal data collection (time series)
    save_timeseries: bool = False
    timeseries_subsample: int = 10  # FIXME: Save every how many steps

    # PCF settings
    collect_pcf: bool = True
    pcf_sample_rate: float = 0.2  # Fraction of runs to compute PCF
    pcf_max_distance: float = 20.0
    pcf_n_bins: int = 20

    # Cluster analysis
    min_density_for_analysis: float = (
        0.002  # FIXME: Minimum prey density (fraction of grid) to analyze clusters/PCF
    )

    # Perturbation settings (Phase 5)
    perturbation_magnitude: float = (
        0.1  # FIXME: Fractional change to apply at perturbation time
    )

    # Parallelization
    n_jobs: int = -1  # Use all available cores by default

    # Helpers
    def get_prey_births(self) -> np.ndarray:
        """
        Generate a linear range of prey birth rates for experimental sweeps.

        Returns
        -------
        np.ndarray
            1D array of birth rates based on `prey_birth_range` and `n_prey_birth`.
        """
        return np.linspace(
            self.prey_birth_range[0], self.prey_birth_range[1], self.n_prey_birth
        )

    def get_prey_deaths(self) -> np.ndarray:
        """
        Generate a linear range of prey death rates for experimental sweeps.

        Returns
        -------
        np.ndarray
            1D array of death rates based on `prey_death_range` and `n_prey_death`.
        """
        return np.linspace(
            self.prey_death_range[0], self.prey_death_range[1], self.n_prey_death
        )

    def get_warmup_steps(
        self, L: int
    ) -> int:  # FIXME: This method will be updated depending on Sary's results.
        """
        Calculate the required warmup steps scaled by grid size.

        Parameters
        ----------
        L : int
            The side length of the current grid.

        Returns
        -------
        int
            The number of steps to discard before measurement.
        """
        return self.warmup_steps

    def get_measurement_steps(self, L: int) -> int:
        """
        Determine the number of measurement steps based on the grid side length.

        This method allows for dynamic scaling of data collection duration relative 
        to the system size. Currently, it returns a fixed value, but it is 
        designed to be overridden for studies where measurement time must 
        scale with the grid size (e.g., $L^z$ scaling in critical dynamics).

        Parameters
        ----------
        L : int
            The side length of the current simulation grid.

        Returns
        -------
        int
            The number of iterations to perform for statistical measurement.
        """
        return self.measurement_steps

    def estimate_runtime(self, n_cores: int = 32) -> str:
        """
        Estimate the wall-clock time required to complete the experiment.

        Calculations account for grid size scaling, PCF overhead, 
        replicate counts, and available parallel resources.

        Parameters
        ----------
        n_cores : int, default 32
            The number of CPU cores available for execution.

        Returns
        -------
        str
            A human-readable summary of simulation count and estimated hours.
        """
        # Benchmark: ~1182 steps/sec for 100x100 grid
        ref_size = 100
        ref_steps_per_sec = 1182

        size_scaling = (self.grid_size / ref_size) ** 2
        actual_steps_per_sec = ref_steps_per_sec / size_scaling

        total_steps = self.warmup_steps + self.measurement_steps
        base_time_s = total_steps / actual_steps_per_sec

        # PCF overhead (~8ms for 100x100)
        pcf_time_s = (0.008 * size_scaling) if self.collect_pcf else 0

        # Count simulations
        n_sims = self.n_prey_birth * self.n_prey_death * self.n_replicates
        if self.with_evolution:
            n_sims *= 2  # Both evo and non-evo runs

        total_seconds = n_sims * (base_time_s + pcf_time_s * self.pcf_sample_rate)
        total_seconds /= n_cores

        hours = total_seconds / 3600
        core_hours = n_sims * (base_time_s + pcf_time_s * self.pcf_sample_rate) / 3600

        return f"{n_sims:,} sims, ~{hours:.1f}h on {n_cores} cores (~{core_hours:.0f} core-hours)"


############################################################################################
# Experimental Phase Configurations
############################################################################################

# FIXME: These configs are arbitraty and should be finalized before running experiments.

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
    pcf_sample_rate=0.2,
    save_timeseries=False,
    directed_hunting=False,
)

# Phase 2: Self-organization (evolution toward criticality)
PHASE2_CONFIG = Config(
    grid_size=1000,
    n_prey_birth=1,  # Fixed at cfg.prey_birth (0.2)
    n_replicates=10,
    warmup_steps=1000,  # Shorter warmup (evolution starts immediately)
    measurement_steps=10000,  # Longer measurement to see convergence
    # Evolution settings
    with_evolution=True,
    evolve_sd=0.01,  # Smaller mutation rate for smoother convergence
    evolve_min=0.0,
    evolve_max=0.20,  # Allow full range
    collect_pcf=False,
    save_timeseries=False,  # Track evolution trajectory
)

# Phase 3: Finite-size scaling at critical point
PHASE3_CONFIG = Config(
    grid_sizes=(50, 100, 250, 500, 1000, 2500),
    n_replicates=20,
    warmup_steps=1000,
    measurement_steps=1000,
    critical_prey_birth=0.20,  # Add explicitly
    critical_prey_death=0.947,  # Add explicitly - verify from Phase 1!
    collect_pcf=True,
    pcf_sample_rate=1.0,
    save_timeseries=False,
    with_evolution=False,
    directed_hunting=False,
)

# Phase 4: Sensitivity analysis
PHASE4_CONFIG = Config(
    grid_size=250,  # As requested
    n_replicates=10,  # As requested
    warmup_steps=500,  # As requested
    measurement_steps=500,  # As requested
    with_evolution=False,
    collect_pcf=False,
    save_timeseries=False,
    timeseries_subsample=10,
    directed_hunting=False,
)


# Phase 5: Perturbation analysis (critical slowing down)
PHASE5_CONFIG = Config(
    grid_size=100,
    prey_death_offsets=(-0.02, -0.01, 0.0, 0.01, 0.02),  # FIXME: Is this what we vary?
    n_replicates=20,
    warmup_steps=500,
    measurement_steps=2000,
    perturbation_magnitude=0.1,
    collect_pcf=False,
    save_timeseries=True,
    timeseries_subsample=1,  # Full resolution for autocorrelation
)

# Phase 6: Model extensions (directed reproduction); same config as phase 4 but with directed reproduction
PHASE6_CONFIG = Config(
    grid_size=250,
    n_replicates=10,
    warmup_steps=500,
    measurement_steps=500,
    with_evolution=False,
    collect_pcf=False,
    save_timeseries=False,
    timeseries_subsample=10,
    directed_hunting=True,
)

PHASE_CONFIGS = {
    1: PHASE1_CONFIG,
    2: PHASE2_CONFIG,
    3: PHASE3_CONFIG,
    4: PHASE4_CONFIG,
    5: PHASE5_CONFIG,
    6: PHASE6_CONFIG,
}


def get_phase_config(phase: int) -> Config:
    """
    Retrieve the configuration object for a specific experimental phase.

    Parameters
    ----------
    phase : int
        The phase number (1 through 6) to retrieve.

    Returns
    -------
    Config
        The configuration instance associated with the requested phase.

    Raises
    ------
    ValueError
        If the phase number is not found in the pre-defined PHASE_CONFIGS.
    """
    if phase not in PHASE_CONFIGS:
        raise ValueError(
            f"Unknown phase {phase}. Valid phases: {list(PHASE_CONFIGS.keys())}"
        )
    return PHASE_CONFIGS[phase]
