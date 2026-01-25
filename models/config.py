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
    
    
    
NOTE: Saving snapshots of the grid can be implemented with the following logic:

    final_grid: cluster analysis verfication for every n_stps.
    
    For Phase 3, save fro all grif sizes
    
    Add to config:
        save_final_grid: bool = False
        save_grid_timeseries: bool = False  # Very costly, use sparingly
        grid_timeseries_subsample: int = N  # Save every N steps
        snapshot_sample_rate: float = 0.0X  # Only X% of runs save snapshots
        
    For run_single_simulation():
        # After cluster analysis
        if cfg.save_final_grid:
        # Only save for a sample of runs 
        if np.random.random() < cfg.snapshot_sample_rate:
            result["final_grid"] = model.grid.tolist()  # JSON-serializable

        # For grid timeseries (use very sparingly):
        if cfg.save_grid_timeseries:
            grid_snapshots = []
            
        # Inside measurement loop:
        if cfg.save_grid_timeseries and step % cfg.grid_timeseries_subsample == 0:
            grid_snapshots.append(model.grid.copy())

        # After loop:
        if cfg.save_grid_timeseries and grid_snapshots:
            # Save separately to avoid bloating JSONL
            snapshot_path = output_dir / f"snapshots_{seed}.npz"
            np.savez_compressed(snapshot_path, grids=np.array(grid_snapshots))
            result["snapshot_file"] = str(snapshot_path)
        
        
    OR create separate snapshot runs using some sort of SNAPSHOT_CONFIG.
"""

from dataclasses import dataclass, field, asdict
from typing import Tuple, Optional
import numpy as np


@dataclass
class Config:
    """Central configuration for all experiments."""
    
    # Grid settings
    grid_size: int = 100  #FIXME: Decide default configuration
    densities: Tuple[float, float] = (0.30, 0.15)  # (prey, predator)  #FIXME: Default densities
    
    # For FSS experiments: multiple grid sizes
    grid_sizes: Tuple[int, ...] = (100, 200, 500, 1000)
    
    # Default/fixed parameters
    prey_birth: float = 0.25
    prey_death: float = 0.05
    predator_birth: float = 0.8 # FIXME: Default predator death rate
    predator_death: float = 0.05 # FIXME: Default predator death rate
    
    # Critical point (UPDATE AFTER PHASE 1)
    critical_prey_birth: float = 0.22 # FIXME: Change after obtaining results
    critical_prey_death: float = 0.04 # FIXME; Change after obtaining results
    
    # Prey parameter sweep (Phase 1)
    prey_birth_range: Tuple[float, float] = (0.10, 0.35)
    prey_death_range: Tuple[float, float] = (0.0, 0.20)
    n_prey_birth: int = 15   # FIXME: Decide number of grid points along prey axes
    n_prey_death: int = 20
    
    # Predator parameter sweep (Phase 4 sensitivity)
    predator_birth_values: Tuple[float, ...] = (0.15, 0.20, 0.25, 0.30) #FIXME: Bogus values for now
    predator_death_values: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20) #FIXME: Bogus values for now
    
    # Perturbation offsets from critical point (Phase 5)
    prey_death_offsets: Tuple[float, ...] = (-0.02, -0.01, 0.0, 0.01, 0.02) #FIXME: Bogus values for now

    # Number of replicates per parameter configuration
    n_replicates: int = 15 # FIXME: Decide number of indep. runs per parameter config
    
    # Simulation steps
    warmup_steps: int = 300  # FIXME: Steps to run before measuring
    measurement_steps: int = 500 # FIXME: Decide measurement steps
    
    # Evo
    with_evolution: bool = False
    evolve_sd: float = 0.10
    evolve_min: float = 0.001
    evolve_max: float = 0.10
    
    # Sensitivity: mutation strength values to test
    sensitivity_sd_values: Tuple[float, ...] = (0.02, 0.05, 0.10, 0.15, 0.20) #FIXME: Don't know if we use yet
    
    # Update mode
    synchronous: bool = False  # Always False for this model
    directed_hunting: bool = True
    
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
    min_density_for_analysis: float = 0.002 # FIXME: Minimum prey density (fraction of grid) to analyze clusters/PCF
    
    # Perturbation settings (Phase 5)
    perturbation_magnitude: float = 0.1  # FIXME: Fractional change to apply at perturbation time
    
    # Parallelization
    n_jobs: int = -1 # Use all available cores by default
    
    # Helpers
    def get_prey_births(self) -> np.ndarray:
        """Generate prey birth rate sweep values."""
        return np.linspace(self.prey_birth_range[0], self.prey_birth_range[1], self.n_prey_birth)
    
    def get_prey_deaths(self) -> np.ndarray:
        """Generate prey death rate sweep values."""
        return np.linspace(self.prey_death_range[0], self.prey_death_range[1], self.n_prey_death)
    
    def get_warmup_steps(self, L: int) -> int: #FIXME: This method will be updated depending on Sary's results.
        """Scale warmup with grid size."""
        return self.warmup_steps
    
    def get_measurement_steps(self, L: int) -> int:
        """Scale measurement with grid size."""
        return self.measurement_steps
    
    def estimate_runtime(self, n_cores: int = 32) -> str:
        """Estimate total runtime based on benchmark data."""
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

#FIXME: These configs are arbitraty and should be finalized before running experiments.

# Phase 1: Parameter sweep to find critical point
PHASE1_CONFIG = Config(
    grid_size=100,
    n_prey_birth=15,
    n_prey_death=20,
    prey_birth_range=(0.10, 0.35),
    prey_death_range=(0.0, 0.20),
    n_replicates=30,
    warmup_steps=300,
    measurement_steps=500,
    collect_pcf=True,
    pcf_sample_rate=0.2,
    save_timeseries=False,
    directed_hunting = False,
)

# Phase 2: Self-organization (evolution toward criticality)
PHASE2_CONFIG = Config(
    grid_size=100,
    n_prey_birth=10,
    n_replicates=10,
    warmup_steps=100,
    measurement_steps=1000,
    with_evolution=True,
    evolve_sd=0.10,
    collect_pcf=False,  # Not needed for SOC analysis
    save_timeseries=True,
    timeseries_subsample=10,
)

# Phase 3: Finite-size scaling at critical point
PHASE3_CONFIG = Config(
    grid_sizes=(50, 75, 100, 150, 200),
    n_replicates=50,
    warmup_steps=200,
    measurement_steps=300,
    collect_pcf=False,
    save_timeseries=False,
)

# Phase 4: Sensitivity analysis
PHASE4_CONFIG = Config(
    grid_size=100,
    predator_birth_values=(0.15, 0.20, 0.25, 0.30),
    predator_death_values=(0.05, 0.10, 0.15, 0.20),
    n_prey_death=10,
    prey_death_range=(0.01, 0.10),
    n_replicates=20,
    warmup_steps=200,
    measurement_steps=1000,
    with_evolution=True,
    collect_pcf=False,
    save_timeseries=True,
    timeseries_subsample=10,
)

# Phase 5: Perturbation analysis (critical slowing down)
PHASE5_CONFIG = Config(
    grid_size=100,
    prey_death_offsets=(-0.02, -0.01, 0.0, 0.01, 0.02), #FIXME: Is this what we vary?
    n_replicates=20,
    warmup_steps=500,
    measurement_steps=2000,
    perturbation_magnitude=0.1,
    collect_pcf=False,
    save_timeseries=True,
    timeseries_subsample=1,  # Full resolution for autocorrelation
)

# Phase 6: Model extensions
PHASE6_CONFIG = Config() #FIXME: Will be defined later

PHASE_CONFIGS = {
    1: PHASE1_CONFIG,
    2: PHASE2_CONFIG,
    3: PHASE3_CONFIG,
    4: PHASE4_CONFIG,
    5: PHASE5_CONFIG,
    6: PHASE6_CONFIG,
}

def get_phase_config(phase: int) -> Config:
    """Get config for a specific phase."""
    if phase not in PHASE_CONFIGS:
        raise ValueError(f"Unknown phase {phase}. Valid phases: {list(PHASE_CONFIGS.keys())}")
    return PHASE_CONFIGS[phase]