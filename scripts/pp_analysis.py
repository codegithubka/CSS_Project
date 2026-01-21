"""
Prey-predator evo analysis - Snellius HPC Version

Focus: Prey Hydra effect - high prey death rates leading to higher prey density.


Includes:
- 2D parameter sweep
- Hydra effect quantification
- Critical point search
- Evolution sensitivity analysis
- Finite size scaling  
    
Usage:
    python pp_analysis.py --mode full          # Run everything
    python pp_analysis.py --mode sweep         # Only 2D sweep
    python pp_analysis.py --mode sensitivity   # Only evolution sensitivity
    python pp_analysis.py --mode fss           # Only finite-size scaling
    python pp_analysis.py --mode plot          # Only generate plots from saved data
    python pp_analysis.py --mode debug         # Interactive visualization (local only)
    python pp_analysis.py --dry-run            # Estimate runtime without running
    
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

project_root = str(Path(__file__).parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from scripts.numba_optimized import (
        compute_pcf_periodic_fast,
        compute_all_pcfs_fast,
        measure_cluster_sizes_fast,
        NUMBA_AVAILABLE
    )
    USE_NUMBA = NUMBA_AVAILABLE
except ImportError:
    USE_NUMBA = False

import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings("ignore")

# Config

@dataclass
class Config:
    """
    Central configuration; CPU budget adjustable.
    """
    
    # Grid setting
    default_grid: int = 100
    densities: Tuple[float, float] = (0.30, 0.15)
    
    # 2D sweep resolution
    n_prey_birth: int = 15
    n_prey_death: int = 15
    prey_birth_min: float = 0.10
    prey_birth_max: float = 0.35
    prey_death_min: float = 0.001
    prey_death_max: float = 0.10
    
    predator_death: float = 0.1
    predator_birth: float = 0.2

    # Replicates per parameter set
    n_replicates: int = 50
    
    # Simulation length
    warmup_steps: int = 200
    measurement_steps: int = 300
    cluster_samples: int = 3
    cluster_interval: int = 1000
    
    # Ecological parameters
    evolve_sd: float = 0.10
    evolve_min: float = 0.001
    evolve_max: float = 0.10
    
    # Finite size scaling
    fss_grid_sizes: Tuple[int, ...] = (50, 75, 100, 150)
    fss_replicates: int = 100
    
    # Evolution sensitivity analysis
    sensitivity_sd_values: Tuple[float, ...] = (0.02, 0.05, 0.10, 0.15, 0.20)
    sensitivity_replicates: int = 20
    
    # Additional metrics metrics (optional, adds overhead)
    collect_neighbor_stats: bool = False
    collect_timeseries: bool = False
    timeseries_interval: int = 10  # Collect every N steps
    
    synchronous: bool = False
    
    # Snapshot settings
    save_snapshots: bool = False
    snapshot_times: Tuple[int, ...] = (50, 100, 150, 200, 250)
    
    n_jobs: int = -1  # -1 = all available cores
    
    # Update mode (default async)
    collect_pcf: bool = True
    pcf_max_distance: float = 20.0
    pcf_n_bins: int = 20
    
    # Diagnostic snapshots (for representative runs)
    save_diagnostic_plots: bool = False
    diagnostic_param_sets: int = 5  # Number of parameter combinations to save diagnostics for
    
    
    
    def get_prey_deaths(self) -> np.ndarray:
        return np.linspace(self.prey_death_min, self.prey_death_max, self.n_prey_death)
    
    def get_prey_births(self) -> np.ndarray:
        return np.linspace(self.prey_birth_min, self.prey_birth_max, self.n_prey_birth)

    def estimate_runtime(self, n_cores: int = 32) -> str:
        """Estimate total runtime with grid-size scaling."""
        n_sweep = self.n_prey_birth * self.n_prey_death * self.n_replicates * 2
        n_sens = len(self.sensitivity_sd_values) * self.sensitivity_replicates
        
        # FSS with size-scaled equilibration
        n_fss = 0
        fss_time = 0
        base_time = 1.5
        for L in self.fss_grid_sizes:
            warmup_factor = L / self.default_grid
            time_factor = (L / 100) ** 2 * warmup_factor
            n_fss += self.fss_replicates
            fss_time += self.fss_replicates * base_time * time_factor
            
            
        # Time per sim scales as (L/100)^2
        grid_factor = (self.default_grid / 100) ** 2

        # Sweep and sensitivity use default_grid
        sweep_time = n_sweep * base_time * grid_factor
        sens_time = n_sens * base_time * grid_factor

        total_seconds = (sweep_time + sens_time + fss_time) / n_cores
        hours = total_seconds / 3600
        core_hours = (sweep_time + sens_time + fss_time) / 3600

        return f"{n_sweep + n_sens + n_fss:,} sims, ~{hours:.1f}h on {n_cores} cores (~{core_hours:.0f} core-hours)"



# Helpers
def count_populations(grid: np.ndarray) -> Tuple[int, int, int]:
    """Count empty, prey, predator cells."""
    return int(np.sum(grid == 0)), int(np.sum(grid == 1)), int(np.sum(grid == 2))


def analyze_neighbor_distribution(model)->Dict:
    """Analyze neighbor distributions using built-in CA methods."""
    neighbor_counts = model.count_neighbors()
    prey_neighbors = neighbor_counts[0]
    pred_neighbors = neighbor_counts[1]
    
    prey_mask = (model.grid == 1)
    pred_mask = (model.grid == 2)
    
    result = {}
    
    if np.any(prey_mask):
        prey_neighbor_vals = prey_neighbors[prey_mask]
        result["prey_mean_prey_neighbors"] = float(np.mean(prey_neighbor_vals))
        result["prey_std_prey_neighbors"] = float(np.std(prey_neighbor_vals))
        result["prey_neighbor_distribution"] = np.bincount(
            prey_neighbor_vals, minlength=9
        ).tolist()
    
    if np.any(pred_mask):
        pred_prey_neighbors = prey_neighbors[pred_mask]
        result["pred_mean_prey_neighbors"] = float(np.mean(pred_prey_neighbors))
        result["pred_std_prey_neighbors"] = float(np.std(pred_prey_neighbors))
    
    return result


def get_evolution_metadata(model)->Dict:
    metadata = {}
    
    for param_name, evolve_config in model._evolve_info.items():
        arr = model.cell_params.get(param_name)
        if arr is not None:
            valid_vals = arr[~np.isnan(arr)]
            metadata[param_name] = {
                "config": evolve_config,  # {sd, min, max}
                "current_mean": float(np.mean(valid_vals)) if len(valid_vals) > 0 else np.nan,
                "current_std": float(np.std(valid_vals)) if len(valid_vals) > 0 else np.nan,
                "current_min": float(np.min(valid_vals)) if len(valid_vals) > 0 else np.nan,
                "current_max": float(np.max(valid_vals)) if len(valid_vals) > 0 else np.nan,
                "n_cells": len(valid_vals),
            }
    
    return metadata



def collect_comprehensive_metrics(model, step: int) -> Dict:
    """Collect comprehensive metrics using built-in CA methods."""
    metrics = {
        "step": step,
        # Population counts
        "n_empty": int(np.sum(model.grid == 0)),
        "n_prey": int(np.sum(model.grid == 1)),
        "n_predator": int(np.sum(model.grid == 2)),
    }
    
    # Neighbor analysis using CA method
    neighbor_counts = model.count_neighbors()
    prey_mask = (model.grid == 1)
    pred_mask = (model.grid == 2)
    
    if np.any(prey_mask):
        metrics["prey_avg_prey_neighbors"] = float(np.mean(neighbor_counts[0][prey_mask]))
        metrics["prey_avg_pred_neighbors"] = float(np.mean(neighbor_counts[1][prey_mask]))
    
    if np.any(pred_mask):
        metrics["pred_avg_prey_neighbors"] = float(np.mean(neighbor_counts[0][pred_mask]))
        metrics["pred_avg_pred_neighbors"] = float(np.mean(neighbor_counts[1][pred_mask]))
    
    # Evolution statistics (if enabled)
    for param_name in model._evolve_info:
        arr = model.cell_params.get(param_name)
        if arr is not None:
            valid = arr[~np.isnan(arr)]
            if len(valid) > 0:
                metrics[f"{param_name}_mean"] = float(np.mean(valid))
                metrics[f"{param_name}_std"] = float(np.std(valid))
                metrics[f"{param_name}_min"] = float(np.min(valid))
                metrics[f"{param_name}_max"] = float(np.max(valid))
    
    return metrics

def measure_cluster_sizes(grid: np.ndarray, species: int) -> np.ndarray:
    """Extract cluster sizes using 4-connected component analysis."""
    if USE_NUMBA:
        return measure_cluster_sizes_fast(grid, species)
    
    # Fallback to scipy
    binary_mask = (grid == species).astype(int)
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    labeled, n = ndimage.label(binary_mask, structure=structure)
    if n == 0:
        return np.array([], dtype=int)
    return np.array(ndimage.sum(binary_mask, labeled, range(1, n + 1)), dtype=int)


def get_evolved_stats(model, param: str) -> Dict:
    """Get statistics of evolved parameter from model."""
    arr = model.cell_params.get(param)
    if arr is None:
        return {"mean": np.nan, "std": np.nan, "n": 0}
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return {"mean": np.nan, "std": np.nan, "n": 0}
    return {"mean": float(np.mean(valid)), "std": float(np.std(valid)), "n": len(valid)}
       
            
def truncated_power_law(s: np.ndarray, tau: float, s_c: float, A: float) -> np.ndarray:
    """
    Truncated power law: P(s) = A * s^(-tau) * exp(-s/s_c).
    
    This is based on the finite-size scaling assumption. 
    """
    return A * np.power(s, -tau) * np.exp(-s / s_c)



def fit_truncated_power_law(sizes: np.ndarray, s_min: int = 2) -> Dict:
    """Fit truncated power law to cluster size distribution."""
    sizes = sizes[sizes >= s_min]
    if len(sizes) < 100:
        return {"tau": np.nan, "s_c": np.nan, "valid": False, "n": len(sizes)}

    bins = np.logspace(np.log10(s_min), np.log10(sizes.max() + 1), 25)
    hist, edges = np.histogram(sizes, bins=bins, density=True)
    centers = np.sqrt(edges[:-1] * edges[1:])
    mask = hist > 0
    x, y = centers[mask], hist[mask]

    if len(x) < 5:
        return {"tau": np.nan, "s_c": np.nan, "valid": False, "n": len(sizes)}

    try:
        popt, pcov = curve_fit(
            lambda s, tau, s_c, A: np.log(truncated_power_law(s, tau, s_c, A) + 1e-20),
            x,
            np.log(y + 1e-20),
            p0=[2.0, 1000.0, y[0] * x[0] ** 2],
            bounds=([1.0, 10, 1e-15], [4.0, 50000, 1e10]),
            maxfev=5000,
        )
        perr = np.sqrt(np.diag(pcov))
        return {
            "tau": popt[0],
            "tau_se": perr[0],
            "s_c": popt[1],
            "valid": True,
            "n": len(sizes),
        }
    except Exception:
        return {"tau": np.nan, "s_c": np.nan, "valid": False, "n": len(sizes)}
    
    
def compute_pcf_periodic(
    positions_i: np.ndarray,
    positions_j: np.ndarray,
    grid_shape: Tuple[int, int],
    max_distance: float,
    n_bins: int = 50,
    self_correlation: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Compute pair correlation function (PCF) between two sets of positions.
    
    Uses periodic boundary conditions and proper normalization for spatial structure.
    
    Args:
        positions_i: (N, 2) array of (row, col) positions for species i
        positions_j: (M, 2) array of (row, col) positions for species j
        grid_shape: (rows, cols) shape of the grid
        max_distance: maximum distance to compute PCF
        n_bins: number of distance bins
        self_correlation: if True, computing C_ii (exclude self-pairs)
    
    Returns:
        bin_centers: array of distance bin centers
        pcf: C_ij(|ξ|) values (1.0 = random, >1 = clustering, <1 = segregation)
        n_pairs: total number of pairs counted
    
    Theory:
        PCF = (observed pairs at distance r) / (expected pairs at distance r in random config)
        - C_ij = 1: no spatial correlation (mean-field accurate)
        - C_ij > 1: species aggregate together (clustering)
        - C_ij < 1: species avoid each other (segregation)
    """
    rows, cols = grid_shape
    L_row, L_col = float(rows), float(cols)
    area = L_row * L_col
    
    # Handle empty position arrays
    if len(positions_i) == 0 or len(positions_j) == 0:
        bin_centers = np.linspace(0, max_distance, n_bins)
        return bin_centers, np.ones(n_bins), 0
    
    # Compute pairwise distances with periodic boundaries
    def periodic_distance(pos1, pos2):
        """Compute periodic distance between two position arrays."""
        dr = np.abs(pos1[:, None, 0] - pos2[None, :, 0])
        dc = np.abs(pos1[:, None, 1] - pos2[None, :, 1])
        
        # Apply periodic boundary
        dr = np.minimum(dr, L_row - dr)
        dc = np.minimum(dc, L_col - dc)
        
        return np.sqrt(dr**2 + dc**2)
    
    distances = periodic_distance(positions_i, positions_j)
    distances_flat = distances.ravel()
    
    # Exclude self-pairs for auto-correlation
    if self_correlation:
        mask = distances_flat > 1e-10
        distances_flat = distances_flat[mask]
    
    # Create distance bins
    bins = np.linspace(0, max_distance, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_widths = np.diff(bins)
    
    # Count observed pairs
    hist, _ = np.histogram(distances_flat, bins=bins)
    
    # Expected count for random distribution
    density_i = len(positions_i) / area
    
    if self_correlation:
        density_product = density_i * (len(positions_i) - 1) / area
    else:
        density_product = density_i * len(positions_j) / area
    
    # Expected count in each annulus: 2πr * dr * density * N_i
    expected = np.zeros(n_bins)
    for i in range(n_bins):
        r = bin_centers[i]
        dr = bin_widths[i]
        annulus_area = 2 * np.pi * r * dr
        expected[i] = density_product * annulus_area * len(positions_i)
    
    # PCF = observed / expected
    pcf = np.ones(n_bins)
    mask = expected > 0
    pcf[mask] = hist[mask] / expected[mask]
    
    # Handle noise from small expected values
    pcf[expected < 1.0] = 1.0
    
    return bin_centers, pcf, int(np.sum(hist))
    

def compute_all_pcfs(
    grid: np.ndarray,
    max_distance: Optional[float] = None,
    n_bins: int = 50,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, int]]:
    """
    Compute all three PCFs for a predator-prey grid.
    
    Returns:
        Dictionary with keys:
            'prey_prey': (distances, C_rr, n_pairs)
            'pred_pred': (distances, C_cc, n_pairs)
            'prey_pred': (distances, C_cr, n_pairs)
    """
    if USE_NUMBA:
        return compute_all_pcfs_fast(grid, max_distance, n_bins)
    
    rows, cols = grid.shape
    if max_distance is None:
        max_distance = min(rows, cols) / 4.0
    
    prey_pos = np.argwhere(grid == 1)
    pred_pos = np.argwhere(grid == 2)
    
    results = {}
    
    # Prey auto-correlation (C_rr)
    dist, pcf, n = compute_pcf_periodic(
        prey_pos, prey_pos, (rows, cols), max_distance, n_bins, self_correlation=True
    )
    results['prey_prey'] = (dist, pcf, n)
    
    # Predator auto-correlation (C_cc)
    dist, pcf, n = compute_pcf_periodic(
        pred_pos, pred_pos, (rows, cols), max_distance, n_bins, self_correlation=True
    )
    results['pred_pred'] = (dist, pcf, n)
    
    # Cross-correlation (C_cr)
    dist, pcf, n = compute_pcf_periodic(
        prey_pos, pred_pos, (rows, cols), max_distance, n_bins, self_correlation=False
    )
    results['prey_pred'] = (dist, pcf, n)
    
    return results

    
def average_pcfs(pcf_list: List[Tuple[np.ndarray, np.ndarray, int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average multiple PCF measurements with standard error."""
    if len(pcf_list) == 0:
        return np.array([]), np.array([]), np.array([])
    
    distances = pcf_list[0][0]
    pcfs = np.array([p[1] for p in pcf_list])
    
    pcf_mean = np.mean(pcfs, axis=0)
    pcf_se = np.std(pcfs, axis=0) / np.sqrt(len(pcfs))
    
    return distances, pcf_mean, pcf_se

def save_diagnostic_snapshot(
    model, 
    output_path, 
    run_id: str,
    prey_pops: List[int],
    pred_pops: List[int],
    evolved_vals: Optional[List[float]] = None,
):
    """
    Save CA-style diagnostic plots for a completed simulation.
    
    Creates a multi-panel figure similar to the interactive visualization
    but suitable for batch processing on HPC.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Color scheme matching CA visualization
    cmap_states = ListedColormap(['black', 'green', 'red'])
    
    # Panel 1: Final grid state
    ax = axes[0, 0]
    ax.imshow(model.grid, cmap=cmap_states, interpolation='nearest', vmin=0, vmax=2)
    ax.set_title('Final State')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    
    # Panel 2: Neighbor histogram
    ax = axes[0, 1]
    counts = model.count_neighbors()[0]
    prey_mask = model.grid == 1
    if np.any(prey_mask):
        vals = counts[prey_mask]
        n_neighbors = 8 if model.neighborhood == "moore" else 4
        ax.hist(vals, bins=np.arange(n_neighbors + 2) - 0.5, 
                align='mid', edgecolor='black', color='green', alpha=0.7)
        ax.set_xlim(-0.5, n_neighbors + 0.5)
    ax.set_xlabel('Prey Neighbor Count')
    ax.set_ylabel('Number of Prey')
    ax.set_title('Neighbor Distribution')
    
    # Panel 3: Evolved parameter map (if applicable)
    ax = axes[0, 2]
    prey_death_arr = model.cell_params.get('prey_death')
    if prey_death_arr is not None:
        meta = model._evolve_info.get('prey_death', {})
        vmin = float(meta.get('min', 0.0))
        vmax = float(meta.get('max', 0.1))
        im = ax.imshow(prey_death_arr, cmap='viridis', interpolation='nearest',
                       vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label='Prey Death Rate')
        ax.set_title('Evolved Prey Death Rate')
    else:
        ax.text(0.5, 0.5, 'No Evolution', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Evolved Parameters (N/A)')
    
    # Panel 4: Population time series
    ax = axes[1, 0]
    steps = np.arange(len(prey_pops))
    ax.plot(steps, prey_pops, 'g-', label='Prey', linewidth=1.5)
    ax.plot(steps, pred_pops, 'r-', label='Predator', linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Population')
    ax.set_title('Population Dynamics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 5: Evolution trajectory (if applicable)
    ax = axes[1, 1]
    if evolved_vals and len(evolved_vals) > 0:
        valid_evolved = [(i, v) for i, v in enumerate(evolved_vals) if not np.isnan(v)]
        if valid_evolved:
            idx, vals = zip(*valid_evolved)
            ax.plot(idx, vals, 'b-', linewidth=1.5)
            ax.axhline(model.params.get('prey_death', 0.05), color='k', 
                      linestyle='--', label='Initial')
            ax.set_xlabel('Step')
            ax.set_ylabel('Mean Evolved Prey Death')
            ax.set_title('Evolution Trajectory')
            ax.legend()
            ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No Evolution', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Evolution Trajectory (N/A)')
    
    # Panel 6: PCF snapshot
    ax = axes[1, 2]
    try:
        pcf_data = compute_all_pcfs(model.grid, max_distance=15.0, n_bins=15)
        dist = pcf_data['prey_prey'][0]
        ax.plot(dist, pcf_data['prey_prey'][1], 'g-o', markersize=4, label='C_rr (prey-prey)')
        ax.plot(dist, pcf_data['pred_pred'][1], 'r-s', markersize=4, label='C_cc (pred-pred)')
        ax.plot(dist, pcf_data['prey_pred'][1], 'b-^', markersize=4, label='C_cr (prey-pred)')
        ax.axhline(1.0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Distance')
        ax.set_ylabel('PCF')
        ax.set_title('Pair Correlation Functions')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    except Exception:
        ax.text(0.5, 0.5, 'PCF Error', ha='center', va='center', transform=ax.transAxes)
    
    plt.suptitle(f'Diagnostic: {run_id}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path / f'diagnostic_{run_id}.png', dpi=120, bbox_inches='tight')
    plt.close()
    

def save_sweep_binary(results: List[Dict], output_path: Path):
    """Saves high-volume sweep data to compressed .npz to avoid JSON overhead."""
    # We use a dictionary to map each run to its own key-space
    data_to_save = {}
    for i, res in enumerate(results):
        prefix = f"run_{i}_"
        for key, val in res.items():
            # np.savez handles numbers and arrays perfectly; 
            # objects/dicts are converted to array-wrappers
            data_to_save[f"{prefix}{key}"] = np.array(val)
            
    np.savez_compressed(output_path, **data_to_save)
    
def warmup_numba_kernels(grid_size: int):
    """Compiles kernels in the main thread so workers load from cache instantly."""
    logging.info(f"Warming up Numba kernels for {grid_size}x{grid_size} grid...")
    
    # Create dummy data structures matching real types
    dummy_grid = np.zeros((grid_size, grid_size), dtype=np.int32)
    p_death_arr = np.full((grid_size, grid_size), 0.05, dtype=np.float64)
    dr = np.array([-1, 1, 0, 0], dtype=np.int32)
    dc = np.array([0, 0, -1, 1], dtype=np.int32)
    
    # Trigger Simulation JIT
    from scripts.numba_optimized import _pp_async_kernel, _compute_distance_histogram
    _ = _pp_async_kernel(dummy_grid, p_death_arr, 0.2, 0.05, 0.2, 0.1, dr, dc, 0.1, 0.001, 0.1, False)

    # Trigger PCF JIT
    pos = np.ascontiguousarray(np.argwhere(dummy_grid == 0)[:10], dtype=np.float64)
    if len(pos) > 0:
        _ = _compute_distance_histogram(pos, pos, float(grid_size), float(grid_size), 20.0, 20, True)
 
    
##########################################################################
# Main Simulation Function
###########################################################################
def run_single_simulation(
    prey_birth: float,
    prey_death: float,
    grid_size: int,
    seed: int,
    with_evolution: bool,
    cfg,  # Config
    evolve_sd: Optional[float] = None,
    evolve_min: Optional[float] = None,
    evolve_max: Optional[float] = None,
    output_dir = None,  # Optional[Path]
    save_diagnostic: bool = False,  
) -> Dict:
    """
    Run a single PP simulation and collect all metrics.
    
    ENHANCED: Collects PCF data, evolution trajectories, and optional diagnostics.
    """
    from models.CA import PP

    if evolve_sd is None:
        evolve_sd = cfg.evolve_sd
    if evolve_min is None:
        evolve_min = cfg.evolve_min
    if evolve_max is None:
        evolve_max = cfg.evolve_max

    params = {
        "prey_birth": prey_birth,
        "prey_death": prey_death,
        "predator_death": cfg.predator_death,
        "predator_birth": cfg.predator_birth,
    }

    model = PP(
        rows=grid_size,
        cols=grid_size,
        densities=cfg.densities,
        neighborhood="moore",
        params=params,
        seed=seed,
        synchronous=cfg.synchronous
    )

    if with_evolution:
        model.evolve("prey_death", sd=evolve_sd, min_val=evolve_min, max_val=evolve_max)

    # Warmup
    model.run(cfg.warmup_steps)

    # Measurement phase
    prey_pops, pred_pops, evolved_vals = [], [], []
    prey_clusters, pred_clusters = [], []
    
    # Evolution trajectory tracking (min, mean, max per step)
    evo_trajectory = {"min": [], "mean": [], "max": []} if with_evolution else None
    
    # PCF storage
    pcf_samples = {
        'prey_prey': [],
        'pred_pred': [],
        'prey_pred': [],
    }
    
    sample_counter = 0
    timeseries = [] if cfg.collect_timeseries else None
    snapshots = [] if cfg.save_snapshots and output_dir else None

    for step in range(cfg.measurement_steps):
        model.update()
        _, prey, pred = count_populations(model.grid)
        prey_pops.append(prey)
        pred_pops.append(pred)

        # Track evolved parameter
        if with_evolution:
            stats = get_evolved_stats(model, "prey_death")
            evolved_vals.append(stats["mean"])
            
            # Track full trajectory
            arr = model.cell_params.get("prey_death")
            if arr is not None:
                valid = arr[~np.isnan(arr)]
                if len(valid) > 0:
                    evo_trajectory["min"].append(float(np.min(valid)))
                    evo_trajectory["mean"].append(float(np.mean(valid)))
                    evo_trajectory["max"].append(float(np.max(valid)))
                else:
                    evo_trajectory["min"].append(np.nan)
                    evo_trajectory["mean"].append(np.nan)
                    evo_trajectory["max"].append(np.nan)

        # Cluster and PCF sampling (periodic)
        if step % cfg.cluster_interval == 0 and sample_counter < cfg.cluster_samples:
            if prey > 10:
                prey_clusters.extend(measure_cluster_sizes(model.grid, 1))
                pred_clusters.extend(measure_cluster_sizes(model.grid, 2))
                
                # Compute PCFs if enabled and enough individuals
                if getattr(cfg, 'collect_pcf', True) and prey > 20 and pred > 5:
                    max_dist = getattr(cfg, 'pcf_max_distance', 20.0)
                    n_bins = getattr(cfg, 'pcf_n_bins', 20)
                    pcf_data = compute_all_pcfs(
                        model.grid,
                        max_distance=min(grid_size / 2, max_dist),
                        n_bins=n_bins,
                    )
                    pcf_samples['prey_prey'].append(pcf_data['prey_prey'])
                    pcf_samples['pred_pred'].append(pcf_data['pred_pred'])
                    pcf_samples['prey_pred'].append(pcf_data['prey_pred'])
            
            sample_counter += 1
        
        # Optional: time-series
        if cfg.collect_timeseries and step % cfg.timeseries_interval == 0:
            timeseries.append(collect_comprehensive_metrics(model, step))
        
        # Optional: snapshots
        if cfg.save_snapshots and output_dir and step in cfg.snapshot_times:
            snapshot = {
                "step": step,
                "grid": model.grid.copy().tolist(),
            }
            if with_evolution:
                prey_death_map = model.cell_params.get("prey_death")
                if prey_death_map is not None:
                    snapshot["prey_death_map"] = prey_death_map.copy().tolist()
            snapshots.append(snapshot)

    # Save diagnostic plot if requested
    if save_diagnostic and output_dir is not None:
        run_id = f"pb{prey_birth:.3f}_pd{prey_death:.3f}_evo{with_evolution}_s{seed}"
        try:
            save_diagnostic_snapshot(model, output_dir, run_id, 
                                    prey_pops, pred_pops, evolved_vals)
        except Exception as e:
            pass  # Don't fail simulation for diagnostic errors

    # Compile results
    result = {
        "prey_birth": prey_birth,
        "prey_death": prey_death,
        "grid_size": grid_size,
        "with_evolution": with_evolution,
        "seed": seed,
        "prey_mean": float(np.mean(prey_pops)),
        "prey_std": float(np.std(prey_pops)),
        "pred_mean": float(np.mean(pred_pops)),
        "pred_std": float(np.std(pred_pops)),
        "prey_survived": bool(np.mean(prey_pops) > 10),
        "pred_survived": bool(np.mean(pred_pops) > 10),
        "prey_n_clusters": len(prey_clusters),
        "pred_n_clusters": len(pred_clusters),
    }

    # Evolved parameter statistics
    if with_evolution and evolved_vals:
        valid_evolved = [v for v in evolved_vals if not np.isnan(v)]
        result["evolved_prey_death_mean"] = (
            float(np.mean(valid_evolved)) if valid_evolved else np.nan
        )
        result["evolved_prey_death_std"] = (
            float(np.std(valid_evolved)) if valid_evolved else np.nan
        )
        result["evolve_sd"] = evolve_sd
        
        # Final evolution state
        evo_meta = get_evolution_metadata(model)
        if "prey_death" in evo_meta:
            result["evolved_prey_death_final"] = evo_meta["prey_death"]
        
        # Evolution trajectory summary (for trajectory plots)
        if evo_trajectory and len(evo_trajectory["mean"]) > 0:
            result["evo_trajectory_final_min"] = evo_trajectory["min"][-1] if evo_trajectory["min"] else np.nan
            result["evo_trajectory_final_mean"] = evo_trajectory["mean"][-1] if evo_trajectory["mean"] else np.nan
            result["evo_trajectory_final_max"] = evo_trajectory["max"][-1] if evo_trajectory["max"] else np.nan

    # Cluster power-law fits
    if len(prey_clusters) > 50:
        fit = fit_truncated_power_law(np.array(prey_clusters))
        result["prey_tau"] = fit["tau"]
        result["prey_s_c"] = fit["s_c"]
    else:
        result["prey_tau"] = np.nan
        result["prey_s_c"] = np.nan

    if len(pred_clusters) > 50:
        fit = fit_truncated_power_law(np.array(pred_clusters))
        result["pred_tau"] = fit["tau"]
        result["pred_s_c"] = fit["s_c"]
    else:
        result["pred_tau"] = np.nan
        result["pred_s_c"] = np.nan
    
    # PCF statistics
    if len(pcf_samples['prey_prey']) > 0:
        dist, pcf_rr_mean, pcf_rr_se = average_pcfs(pcf_samples['prey_prey'])
        dist, pcf_cc_mean, pcf_cc_se = average_pcfs(pcf_samples['pred_pred'])
        dist, pcf_cr_mean, pcf_cr_se = average_pcfs(pcf_samples['prey_pred'])
        
        result["pcf_distances"] = dist.tolist()
        result["pcf_prey_prey_mean"] = pcf_rr_mean.tolist()
        result["pcf_pred_pred_mean"] = pcf_cc_mean.tolist()
        result["pcf_prey_pred_mean"] = pcf_cr_mean.tolist()
        result["pcf_prey_prey_se"] = pcf_rr_se.tolist()
        result["pcf_pred_pred_se"] = pcf_cc_se.tolist()
        result["pcf_prey_pred_se"] = pcf_cr_se.tolist()
        
        # Summary indices (short-range structure)
        short_dist_mask = dist < 3.0
        if np.any(short_dist_mask):
            result["segregation_index"] = float(np.mean(pcf_cr_mean[short_dist_mask]))
            result["prey_clustering_index"] = float(np.mean(pcf_rr_mean[short_dist_mask]))
            result["pred_clustering_index"] = float(np.mean(pcf_cc_mean[short_dist_mask]))
        else:
            result["segregation_index"] = 1.0
            result["prey_clustering_index"] = 1.0
            result["pred_clustering_index"] = 1.0
    else:
        result["pcf_distances"] = []
        result["pcf_prey_prey_mean"] = []
        result["pcf_pred_pred_mean"] = []
        result["pcf_prey_pred_mean"] = []
        result["pcf_prey_prey_se"] = []
        result["pcf_pred_pred_se"] = []
        result["pcf_prey_pred_se"] = []
        result["segregation_index"] = np.nan
        result["prey_clustering_index"] = np.nan
        result["pred_clustering_index"] = np.nan
    
    # Optional features
    if cfg.collect_neighbor_stats:
        result["neighbor_stats"] = analyze_neighbor_distribution(model)
    
    if timeseries:
        result["timeseries"] = timeseries
    
    if snapshots:
        result["snapshots"] = snapshots

    return result

# FSS specific runner

def run_single_simulation_fss(
    prey_birth: float,
    prey_death: float,
    grid_size: int,
    seed: int,
    with_evolution: bool,
    cfg: Config,
    warmup_steps: int,
    measurement_steps: int,
    output_dir: Optional[Path] = None,
) -> Dict:
    """FSS-specific simulation with size-scaled equilibration time.
    
    For finite-size scaling, equilibration time must scale with system size
    because relaxation dynamics slow down in larger systems.
    """
    
    from models.CA import PP
    
    params = {
        "prey_birth": prey_birth,
        "prey_death": prey_death,
        "predator_death": cfg.predator_death,
        "predator_birth": cfg.predator_birth,
    }

    model = PP(
        rows=grid_size,
        cols=grid_size,
        densities=cfg.densities,
        neighborhood="moore",
        params=params,
        seed = seed,
        synchronous=cfg.synchronous
    )
    
    # Note that evolution is disabled with FSS to isolate ecological dynamics
    # To examine SOC, evolution needs to be ON
    # We measure the properties of a static model at criticality
    model.run(warmup_steps)
    
    
    # Measurement
    prey_pops, pred_pops = [], []
    prey_clusters, pred_clusters = [], []
    
    # Scale cluster sampling interval with system size
    cluster_interval = max(1, int(cfg.cluster_interval * grid_size / cfg.default_grid))
    cluster_samples = cfg.cluster_samples
    sample_counter = 0

    for step in range(measurement_steps):
        model.update()
        _, prey, pred = count_populations(model.grid)
        prey_pops.append(prey)
        pred_pops.append(pred)
        
        if step % cluster_interval == 0 and sample_counter < cluster_samples:
            if prey > 10:
                prey_clusters.extend(measure_cluster_sizes(model.grid, 1))
                pred_clusters.extend(measure_cluster_sizes(model.grid, 2))
            sample_counter += 1
            
            
    # Compile results
    result = {
        "prey_birth": prey_birth,
        "prey_death": prey_death,
        "grid_size": grid_size,
        "with_evolution": with_evolution,
        "seed": seed,
        "warmup_steps": warmup_steps,  # Document actual warmup used
        "measurement_steps": measurement_steps,
        "prey_mean": float(np.mean(prey_pops)),
        "prey_std": float(np.std(prey_pops)),
        "pred_mean": float(np.mean(pred_pops)),
        "pred_std": float(np.std(pred_pops)),
        "prey_survived": bool(np.mean(prey_pops) > 10),
        "pred_survived": bool(np.mean(pred_pops) > 10),
        "prey_n_clusters": len(prey_clusters),
        "pred_n_clusters": len(pred_clusters),
    }
    
    # Cluster fits
    if len(prey_clusters) > 50:
        fit = fit_truncated_power_law(np.array(prey_clusters))
        result["prey_tau"] = fit["tau"]
        result["prey_tau_se"] = fit.get("tau_se", np.nan)
        result["prey_s_c"] = fit["s_c"]
    else:
        result["prey_tau"] = np.nan
        result["prey_tau_se"] = np.nan
        result["prey_s_c"] = np.nan

    if len(pred_clusters) > 50:
        fit = fit_truncated_power_law(np.array(pred_clusters))
        result["pred_tau"] = fit["tau"]
        result["pred_tau_se"] = fit.get("tau_se", np.nan)
        result["pred_s_c"] = fit["s_c"]
    else:
        result["pred_tau"] = np.nan
        result["pred_tau_se"] = np.nan
        result["pred_s_c"] = np.nan

    return result


##############################################################################
# Anslysis Runners
###############################################################################

def run_2d_sweep(cfg, output_dir, logger) -> List[Dict]:
    """Run full 2D parameter sweep with optional diagnostic snapshots."""
    from joblib import Parallel, delayed
    if USE_NUMBA:
        warmup_numba_kernels(cfg.default_grid)

    prey_births = cfg.get_prey_births()
    prey_deaths = cfg.get_prey_deaths()
    
    # Select representative parameter combinations for diagnostics
    diagnostic_params = set()
    if getattr(cfg, 'save_diagnostic_plots', False):
        n_diag = getattr(cfg, 'diagnostic_param_sets', 5)
        # Sample from corners and center of parameter space
        pb_indices = [0, len(prey_births)//2, len(prey_births)-1]
        pd_indices = [0, len(prey_deaths)//2, len(prey_deaths)-1]
        for pbi in pb_indices[:n_diag]:
            for pdi in pd_indices[:n_diag]:
                if len(diagnostic_params) < n_diag:
                    diagnostic_params.add((round(prey_births[min(pbi, len(prey_births)-1)], 4),
                                          round(prey_deaths[min(pdi, len(prey_deaths)-1)], 4)))
    
    jobs = []
    for pb in prey_births:
        for pd in prey_deaths:
            for rep in range(cfg.n_replicates):
                seed_base = int(pb * 1000) + int(pd * 10000) + rep
                # Only save diagnostic for first replicate of selected params
                save_diag = (round(pb, 4), round(pd, 4)) in diagnostic_params and rep == 0
                jobs.append((pb, pd, cfg.default_grid, seed_base, False, save_diag))
                jobs.append((pb, pd, cfg.default_grid, seed_base, True, save_diag))

    logger.info(f"2D Sweep: {len(jobs):,} simulations")
    logger.info(f"  Grid: {len(prey_births)}×{len(prey_deaths)} parameters")
    logger.info(f"  prey_birth: [{cfg.prey_birth_min:.3f}, {cfg.prey_birth_max:.3f}]")
    logger.info(f"  prey_death: [{cfg.prey_death_min:.3f}, {cfg.prey_death_max:.3f}]")
    logger.info(f"  Replicates: {cfg.n_replicates}")
    if diagnostic_params:
        logger.info(f"  Diagnostic snapshots: {len(diagnostic_params)} parameter sets")

    results = Parallel(n_jobs=cfg.n_jobs, verbose=10)(
        delayed(run_single_simulation)(
            pb, pd, gs, seed, evo, cfg, 
            output_dir=output_dir, 
            save_diagnostic=save_diag
        )
        for pb, pd, gs, seed, evo, save_diag in jobs
    )

    # REPLACE THE JSON BLOCK WITH THIS:
    output_file = output_dir / "sweep_results.npz"
    save_sweep_binary(results, output_file)
    
    # Save a tiny 'metadata' JSON just for quick human reading
    meta = {
        "n_sims": len(results),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "grid_size": cfg.default_grid
    }
    with open(output_dir / "sweep_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"SUCCESS: Saved binary sweep results to {output_file}")
    return results

def run_sensitivity(
    cfg: Config, output_dir: Path, logger: logging.Logger
) -> List[Dict]:
    """Run evolution parameter sensitivity analysis."""
    from joblib import Parallel, delayed

    # Fixed parameters in transition zone
    pb_test = 0.20
    pd_test = 0.05  # Mid-range prey death

    jobs = []
    for sd in cfg.sensitivity_sd_values:
        for rep in range(cfg.sensitivity_replicates):
            seed = int(sd * 100000) + rep
            jobs.append((pb_test, pd_test, cfg.default_grid, seed, True, sd))

    logger.info(f"Sensitivity: {len(jobs)} simulations")
    logger.info(f"  SD values: {cfg.sensitivity_sd_values}")

    results = Parallel(n_jobs=cfg.n_jobs, verbose=5)(
        delayed(run_single_simulation)(pb, pd, gs, seed, evo, cfg, evolve_sd=sd, output_dir=output_dir)
        for pb, pd, gs, seed, evo, sd in jobs
    )

    output_file = output_dir / "sensitivity_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f)
    logger.info(f"Saved to {output_file}")

    return results


def run_fss(cfg: Config, output_dir: Path, logger: logging.Logger) -> List[Dict]:
    """Run finite-size scaling analysis with proper equilibration scaling.
    
    NOTE: Evolution is DISABLED for FSS to study baseline spatial structure
    without evolutionary complications. FSS characterizes universal scaling
    properties of the ecological dynamics alone.
    """
    from joblib import Parallel, delayed

    # Fixed parameters - should be near critical point for interesting structure
    pb_test = 0.20 #FIXME: Adjust based on prior results
    pd_test = 0.03  # FIXME: Adjust based on prior results
    
    # VALIDATION: Quick test at default grid to check if we're near critical point
    logger.info("=" * 60)
    logger.info("FSS PARAMETER VALIDATION")
    logger.info("=" * 60)
    logger.info(f"Testing parameters: prey_birth={pb_test}, prey_death={pd_test}")
    logger.info(f"Predator params: death={cfg.predator_death}, birth={cfg.predator_birth}")
    
    test_results = []
    for rep in range(5):
        result = run_single_simulation(
            pb_test, pd_test, cfg.default_grid, 
            10000 + rep, False, cfg, output_dir=output_dir
        )
        test_results.append(result)
    
    tau_vals = [r["prey_tau"] for r in test_results if not np.isnan(r.get("prey_tau", np.nan))]
    if tau_vals:
        tau_test = np.mean(tau_vals)
        tau_std = np.std(tau_vals)
        logger.info(f"  Validation: τ = {tau_test:.3f} ± {tau_std:.3f} (target: ~2.05)")
        
        if abs(tau_test - 2.05) > 0.3:
            logger.warning(f" WARNING: Parameters may not be near critical point!")
            logger.warning(f"  Consider adjusting prey_birth or prey_death")
        else:
            logger.info(f"Parameters appear near critical point")
    else:
        logger.warning("WARNING: Could not measure τ in validation!")
    
    logger.info("=" * 60)
    # Generate jobs with size-scaled equilibration
    jobs = []
    for L in cfg.fss_grid_sizes:
        # Scale warmup and measurement times with system size
        # Dynamics slow down with system size
        warmup_factor = L / cfg.default_grid
        warmup_steps = int(cfg.warmup_steps * warmup_factor)
        measurement_steps = int(cfg.measurement_steps * warmup_factor)
        
        for rep in range(cfg.fss_replicates):
            seed = L * 1000 + rep
            jobs.append((pb_test, pd_test, L, seed, False, warmup_steps, measurement_steps))

    logger.info(f"FSS: {len(jobs)} simulations")
    logger.info(f"  Grid sizes: {cfg.fss_grid_sizes}")
    logger.info(f"  Warmup times: {[int(cfg.warmup_steps * L / cfg.default_grid) for L in cfg.fss_grid_sizes]}")
    logger.info(f"  Measurement times: {[int(cfg.measurement_steps * L / cfg.default_grid) for L in cfg.fss_grid_sizes]}")

    # Run with size-specific warmup
    results = Parallel(n_jobs=cfg.n_jobs, verbose=5)(
        delayed(run_single_simulation_fss)(
            pb, pd, gs, seed, evo, cfg, ws, ms, output_dir
        )
        for pb, pd, gs, seed, evo, ws, ms in jobs
    )

    output_file = output_dir / "fss_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f)
    logger.info(f"Saved to {output_file}")

    return results


# Debug Mode

def run_debug_mode(cfg: Config, logger: logging.Logger):
    """Run a single simulation with interactive visualization for debugging."""
    logger.info("=" * 60)
    logger.info("DEBUG MODE - Interactive Visualization")
    logger.info("=" * 60)
    logger.info("Running on LOCAL machine with matplotlib visualization")
    logger.info("Close the plot window to exit")
    
    try:
        from models.CA import PP
    except ImportError:
        logger.error("Cannot import PP model. Make sure models/CA.py is in path.")
        return
    
    # Smaller grid for faster visualization
    grid_size = 50
    params = {
        "prey_birth": 0.20,
        "prey_death": 0.05,
        "predator_death": cfg.predator_death,
        "predator_birth": cfg.predator_birth,
    }
    
    logger.info(f"Parameters: {params}")
    logger.info(f"Grid: {grid_size}×{grid_size}")
    logger.info(f"Evolution: prey_death (SD={cfg.evolve_sd})")
    model = PP(
        rows=grid_size,
        cols=grid_size,
        densities=cfg.densities,
        neighborhood="moore",
        params=params,
        seed=42,
        synchronous=cfg.synchronous,
    )
    
    # Enable evolution
    model.evolve("prey_death", sd=cfg.evolve_sd, 
                 min_val=cfg.evolve_min, max_val=cfg.evolve_max)
    
    # Enable visualization with evolved parameter display
    logger.info("Starting visualization...")
    model.visualize(
        interval=5,
        figsize=(16, 10),
        pause=0.01,
        show_cell_params=True,  # Show evolved prey_death map
        show_neighbors=True,     # Show neighbor statistics
        downsample=1,  # No downsampling for small grid
    )
    
    # Run simulation
    model.run(500)
    
    logger.info("Simulation complete. Press Enter to exit...")
    input()
    
# =============================================================================
# PLOTTING AND SUMMARY
# =============================================================================

def generate_plots(cfg, output_dir, logger):
    """Generate all analysis plots from saved data including PCF analysis."""
    import matplotlib.pyplot as plt
    from collections import defaultdict

    plt.rcParams["figure.figsize"] = (14, 10)
    plt.rcParams["font.size"] = 11

    prey_births = cfg.get_prey_births()
    prey_deaths = cfg.get_prey_deaths()
    n_pb, n_pd = len(prey_births), len(prey_deaths)
    extent = [prey_births[0], prey_births[-1], prey_deaths[0], prey_deaths[-1]]

    # Load sweep results
    sweep_file = output_dir / "sweep_results.json"
    if not sweep_file.exists():
        logger.error(f"Sweep results not found: {sweep_file}")
        return

    with open(sweep_file, "r") as f:
        results = json.load(f)

    logger.info(f"Loaded {len(results)} results from {sweep_file}")

    # Initialize grids (existing + new PCF grids)
    grids = {
        "prey_pop_no_evo": np.full((n_pd, n_pb), np.nan),
        "prey_pop_evo": np.full((n_pd, n_pb), np.nan),
        "pred_pop_no_evo": np.full((n_pd, n_pb), np.nan),
        "pred_pop_evo": np.full((n_pd, n_pb), np.nan),
        "survival_prey_no_evo": np.full((n_pd, n_pb), np.nan),
        "survival_prey_evo": np.full((n_pd, n_pb), np.nan),
        "survival_pred_no_evo": np.full((n_pd, n_pb), np.nan),
        "survival_pred_evo": np.full((n_pd, n_pb), np.nan),
        "tau_prey": np.full((n_pd, n_pb), np.nan),
        "tau_pred": np.full((n_pd, n_pb), np.nan),
        "evolved_prey_death": np.full((n_pd, n_pb), np.nan),
    }
    
    # NEW: PCF grids
    pcf_grids = {
        "segregation_index": np.full((n_pd, n_pb), np.nan),
        "prey_clustering_index": np.full((n_pd, n_pb), np.nan),
        "pred_clustering_index": np.full((n_pd, n_pb), np.nan),
        "segregation_index_evo": np.full((n_pd, n_pb), np.nan),
        "prey_clustering_index_evo": np.full((n_pd, n_pb), np.nan),
        "pred_clustering_index_evo": np.full((n_pd, n_pb), np.nan),
    }

    # Group by parameters
    grouped = defaultdict(list)
    for r in results:
        key = (
            round(r["prey_birth"], 4),
            round(r["prey_death"], 4),
            r["with_evolution"],
        )
        grouped[key].append(r)

    # Aggregate into grids
    for i, pd in enumerate(prey_deaths):
        for j, pb in enumerate(prey_births):
            pd_r, pb_r = round(pd, 4), round(pb, 4)

            # No evolution
            no_evo = grouped.get((pb_r, pd_r, False), [])
            if no_evo:
                grids["prey_pop_no_evo"][i, j] = np.mean([r["prey_mean"] for r in no_evo])
                grids["pred_pop_no_evo"][i, j] = np.mean([r["pred_mean"] for r in no_evo])
                grids["survival_prey_no_evo"][i, j] = np.mean([r["prey_survived"] for r in no_evo]) * 100
                grids["survival_pred_no_evo"][i, j] = np.mean([r["pred_survived"] for r in no_evo]) * 100
                
                taus = [r["prey_tau"] for r in no_evo if not np.isnan(r.get("prey_tau", np.nan))]
                if taus:
                    grids["tau_prey"][i, j] = np.mean(taus)
                taus = [r["pred_tau"] for r in no_evo if not np.isnan(r.get("pred_tau", np.nan))]
                if taus:
                    grids["tau_pred"][i, j] = np.mean(taus)
                
                # PCF metrics (no evolution)
                seg = [r.get("segregation_index", np.nan) for r in no_evo]
                seg = [s for s in seg if not np.isnan(s)]
                if seg:
                    pcf_grids["segregation_index"][i, j] = np.mean(seg)
                
                prey_clust = [r.get("prey_clustering_index", np.nan) for r in no_evo]
                prey_clust = [s for s in prey_clust if not np.isnan(s)]
                if prey_clust:
                    pcf_grids["prey_clustering_index"][i, j] = np.mean(prey_clust)
                
                pred_clust = [r.get("pred_clustering_index", np.nan) for r in no_evo]
                pred_clust = [s for s in pred_clust if not np.isnan(s)]
                if pred_clust:
                    pcf_grids["pred_clustering_index"][i, j] = np.mean(pred_clust)

            # With evolution
            evo = grouped.get((pb_r, pd_r, True), [])
            if evo:
                grids["prey_pop_evo"][i, j] = np.mean([r["prey_mean"] for r in evo])
                grids["pred_pop_evo"][i, j] = np.mean([r["pred_mean"] for r in evo])
                grids["survival_prey_evo"][i, j] = np.mean([r["prey_survived"] for r in evo]) * 100
                grids["survival_pred_evo"][i, j] = np.mean([r["pred_survived"] for r in evo]) * 100
                
                evolved = [r.get("evolved_prey_death_mean", np.nan) for r in evo]
                evolved = [e for e in evolved if not np.isnan(e)]
                if evolved:
                    grids["evolved_prey_death"][i, j] = np.mean(evolved)
                
                # PCF metrics (with evolution)
                seg = [r.get("segregation_index", np.nan) for r in evo]
                seg = [s for s in seg if not np.isnan(s)]
                if seg:
                    pcf_grids["segregation_index_evo"][i, j] = np.mean(seg)
                
                prey_clust = [r.get("prey_clustering_index", np.nan) for r in evo]
                prey_clust = [s for s in prey_clust if not np.isnan(s)]
                if prey_clust:
                    pcf_grids["prey_clustering_index_evo"][i, j] = np.mean(prey_clust)
                
                pred_clust = [r.get("pred_clustering_index", np.nan) for r in evo]
                pred_clust = [s for s in pred_clust if not np.isnan(s)]
                if pred_clust:
                    pcf_grids["pred_clustering_index_evo"][i, j] = np.mean(pred_clust)

    # Compute Hydra derivative
    dd = prey_deaths[1] - prey_deaths[0]
    dN_dd_no_evo = np.zeros_like(grids["prey_pop_no_evo"])
    dN_dd_evo = np.zeros_like(grids["prey_pop_evo"])
    for j in range(n_pb):
        from scipy.ndimage import gaussian_filter1d
        pop_smooth = gaussian_filter1d(grids["prey_pop_no_evo"][:, j], sigma=0.8)
        dN_dd_no_evo[:, j] = np.gradient(pop_smooth, dd)
        pop_smooth = gaussian_filter1d(grids["prey_pop_evo"][:, j], sigma=0.8)
        dN_dd_evo[:, j] = np.gradient(pop_smooth, dd)

    # =========================================================================
    # PLOT 1: Phase Diagrams (existing, unchanged)
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    ax = axes[0, 0]
    im = ax.imshow(grids["prey_pop_no_evo"], origin="lower", aspect="auto",
                   extent=extent, cmap="YlGn")
    ax.contour(prey_births, prey_deaths, grids["survival_prey_no_evo"],
               levels=[50], colors="black", linewidths=2)
    plt.colorbar(im, ax=ax, label="Population")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Prey Death Rate")
    ax.set_title("Prey Pop (No Evolution)\nBlack: 50% survival")

    ax = axes[0, 1]
    im = ax.imshow(grids["prey_pop_evo"], origin="lower", aspect="auto",
                   extent=extent, cmap="YlGn")
    ax.contour(prey_births, prey_deaths, grids["survival_prey_evo"],
               levels=[50], colors="black", linewidths=2)
    plt.colorbar(im, ax=ax, label="Population")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Prey Death Rate")
    ax.set_title("Prey Pop (With Evolution)\nBlack: 50% survival")

    ax = axes[0, 2]
    advantage = np.where(
        grids["prey_pop_no_evo"] > 10,
        (grids["prey_pop_evo"] - grids["prey_pop_no_evo"]) / grids["prey_pop_no_evo"] * 100,
        np.where(grids["prey_pop_evo"] > 10, 500, 0),
    )
    im = ax.imshow(np.clip(advantage, -50, 200), origin="lower", aspect="auto",
                   extent=extent, cmap="RdYlGn", vmin=-50, vmax=200)
    plt.colorbar(im, ax=ax, label="Advantage (%)")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Prey Death Rate")
    ax.set_title("Evolution Advantage (Prey)\n(Evo - NoEvo) / NoEvo")

    ax = axes[1, 0]
    im = ax.imshow(grids["tau_prey"], origin="lower", aspect="auto",
                   extent=extent, cmap="coolwarm", vmin=1.5, vmax=2.5)
    ax.contour(prey_births, prey_deaths, grids["tau_prey"],
               levels=[2.05], colors="green", linewidths=2)
    plt.colorbar(im, ax=ax, label="τ")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Prey Death Rate")
    ax.set_title("Prey τ (Green: τ=2.05 criticality)")

    ax = axes[1, 1]
    im = ax.imshow(grids["evolved_prey_death"], origin="lower", aspect="auto",
                   extent=extent, cmap="viridis")
    plt.colorbar(im, ax=ax, label="Evolved d")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Initial Prey Death Rate")
    ax.set_title("Evolved Prey Death Rate")

    ax = axes[1, 2]
    im = ax.imshow(dN_dd_no_evo, origin="lower", aspect="auto",
                   extent=extent, cmap="RdBu_r", vmin=-5000, vmax=5000)
    ax.contour(prey_births, prey_deaths, dN_dd_no_evo,
               levels=[0], colors="black", linewidths=2)
    plt.colorbar(im, ax=ax, label="dN/dd")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Prey Death Rate")
    ax.set_title("HYDRA EFFECT: dN/dd\nRed: Prey ↑ with mortality")

    plt.tight_layout()
    plt.savefig(output_dir / "phase_diagrams.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved phase_diagrams.png")

    # =========================================================================
    # PLOT 2: Hydra Effect Analysis (existing, unchanged)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    im = ax.imshow(dN_dd_no_evo, origin="lower", aspect="auto",
                   extent=extent, cmap="RdBu_r", vmin=-5000, vmax=5000)
    ax.contour(prey_births, prey_deaths, dN_dd_no_evo,
               levels=[0], colors="black", linewidths=2)
    plt.colorbar(im, ax=ax, label="dN/dd")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Prey Death Rate")
    ax.set_title("Hydra (No Evolution)")

    ax = axes[1]
    im = ax.imshow(dN_dd_evo, origin="lower", aspect="auto",
                   extent=extent, cmap="RdBu_r", vmin=-5000, vmax=5000)
    ax.contour(prey_births, prey_deaths, dN_dd_evo,
               levels=[0], colors="black", linewidths=2)
    plt.colorbar(im, ax=ax, label="dN/dd")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Prey Death Rate")
    ax.set_title("Hydra (With Evolution)")

    ax = axes[2]
    mid_pb_idx = n_pb // 2
    ax.plot(prey_deaths, grids["prey_pop_no_evo"][:, mid_pb_idx], 'b-o',
            label=f'No Evo (pb={prey_births[mid_pb_idx]:.2f})', markersize=4)
    ax.plot(prey_deaths, grids["prey_pop_evo"][:, mid_pb_idx], 'g-s',
            label=f'With Evo (pb={prey_births[mid_pb_idx]:.2f})', markersize=4)
    ax.set_xlabel("Prey Death Rate")
    ax.set_ylabel("Prey Population")
    ax.set_title("Prey Pop vs Death Rate Slice")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "hydra_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved hydra_analysis.png")

    # =========================================================================
    # NEW PLOT: PCF/Spatial Structure Analysis
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Row 1: No evolution
    ax = axes[0, 0]
    im = ax.imshow(pcf_grids["segregation_index"], origin="lower", aspect="auto",
                   extent=extent, cmap="RdBu", vmin=0.5, vmax=1.5)
    ax.contour(prey_births, prey_deaths, pcf_grids["segregation_index"],
               levels=[1.0], colors="black", linewidths=2)
    plt.colorbar(im, ax=ax, label="C_cr")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Prey Death Rate")
    ax.set_title("Segregation Index (No Evo)\nC_cr < 1: prey-pred avoid")

    ax = axes[0, 1]
    im = ax.imshow(pcf_grids["prey_clustering_index"], origin="lower", aspect="auto",
                   extent=extent, cmap="Greens", vmin=1.0, vmax=3.0)
    ax.contour(prey_births, prey_deaths, pcf_grids["prey_clustering_index"],
               levels=[1.5, 2.0], colors=["gray", "black"], linewidths=[1, 2])
    plt.colorbar(im, ax=ax, label="C_rr")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Prey Death Rate")
    ax.set_title("Prey Clustering (No Evo)\nC_rr > 1: prey cluster")

    ax = axes[0, 2]
    im = ax.imshow(pcf_grids["pred_clustering_index"], origin="lower", aspect="auto",
                   extent=extent, cmap="Reds", vmin=1.0, vmax=3.0)
    ax.contour(prey_births, prey_deaths, pcf_grids["pred_clustering_index"],
               levels=[1.5, 2.0], colors=["gray", "black"], linewidths=[1, 2])
    plt.colorbar(im, ax=ax, label="C_cc")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Prey Death Rate")
    ax.set_title("Predator Clustering (No Evo)\nC_cc > 1: predators cluster")

    # Row 2: With evolution
    ax = axes[1, 0]
    im = ax.imshow(pcf_grids["segregation_index_evo"], origin="lower", aspect="auto",
                   extent=extent, cmap="RdBu", vmin=0.5, vmax=1.5)
    ax.contour(prey_births, prey_deaths, pcf_grids["segregation_index_evo"],
               levels=[1.0], colors="black", linewidths=2)
    plt.colorbar(im, ax=ax, label="C_cr")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Prey Death Rate")
    ax.set_title("Segregation Index (With Evo)")

    ax = axes[1, 1]
    im = ax.imshow(pcf_grids["prey_clustering_index_evo"], origin="lower", aspect="auto",
                   extent=extent, cmap="Greens", vmin=1.0, vmax=3.0)
    ax.contour(prey_births, prey_deaths, pcf_grids["prey_clustering_index_evo"],
               levels=[1.5, 2.0], colors=["gray", "black"], linewidths=[1, 2])
    plt.colorbar(im, ax=ax, label="C_rr")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Prey Death Rate")
    ax.set_title("Prey Clustering (With Evo)")

    ax = axes[1, 2]
    im = ax.imshow(pcf_grids["pred_clustering_index_evo"], origin="lower", aspect="auto",
                   extent=extent, cmap="Reds", vmin=1.0, vmax=3.0)
    ax.contour(prey_births, prey_deaths, pcf_grids["pred_clustering_index_evo"],
               levels=[1.5, 2.0], colors=["gray", "black"], linewidths=[1, 2])
    plt.colorbar(im, ax=ax, label="C_cc")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Prey Death Rate")
    ax.set_title("Predator Clustering (With Evo)")

    plt.suptitle("Spatial Structure: Pair Correlation Functions", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "pcf_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved pcf_analysis.png")

    # =========================================================================
    # NEW PLOT: Hydra-PCF Correlation Analysis
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Scatter: Hydra strength vs segregation
    ax = axes[0]
    valid_mask = ~np.isnan(dN_dd_no_evo) & ~np.isnan(pcf_grids["segregation_index"])
    if np.any(valid_mask):
        ax.scatter(pcf_grids["segregation_index"][valid_mask].ravel(),
                   dN_dd_no_evo[valid_mask].ravel(), alpha=0.5, s=20, c='blue')
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.axvline(1.0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel("Segregation Index (C_cr)")
    ax.set_ylabel("Hydra Derivative (dN/dd)")
    ax.set_title("Hydra vs Prey-Pred Segregation")
    ax.grid(True, alpha=0.3)

    # Scatter: Hydra strength vs prey clustering
    ax = axes[1]
    valid_mask = ~np.isnan(dN_dd_no_evo) & ~np.isnan(pcf_grids["prey_clustering_index"])
    if np.any(valid_mask):
        ax.scatter(pcf_grids["prey_clustering_index"][valid_mask].ravel(),
                   dN_dd_no_evo[valid_mask].ravel(), alpha=0.5, s=20, c='green')
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.axvline(1.0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel("Prey Clustering Index (C_rr)")
    ax.set_ylabel("Hydra Derivative (dN/dd)")
    ax.set_title("Hydra vs Prey Clustering")
    ax.grid(True, alpha=0.3)

    # Overlay: Hydra region on segregation map
    ax = axes[2]
    im = ax.imshow(pcf_grids["segregation_index"], origin="lower", aspect="auto",
                   extent=extent, cmap="RdBu", vmin=0.5, vmax=1.5, alpha=0.8)
    # Overlay Hydra contour
    ax.contour(prey_births, prey_deaths, dN_dd_no_evo,
               levels=[0], colors="lime", linewidths=3, linestyles='-')
    ax.contour(prey_births, prey_deaths, grids["survival_prey_no_evo"],
               levels=[50], colors="black", linewidths=2, linestyles='--')
    plt.colorbar(im, ax=ax, label="C_cr")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Prey Death Rate")
    ax.set_title("Segregation + Hydra Boundary\nGreen: dN/dd=0, Black: 50% survival")

    plt.tight_layout()
    plt.savefig(output_dir / "hydra_pcf_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved hydra_pcf_correlation.png")

    # =========================================================================
    # Remaining plots (sensitivity, FSS, mean-field) - keep existing code
    # =========================================================================
    
    # [Include your existing sensitivity, FSS, and mean-field comparison plots here]
    # They remain unchanged from the original

    # =========================================================================
    # ENHANCED Summary statistics
    # =========================================================================
    
    # Mean-field comparison for spatial-only Hydra count
    def mean_field_equilibrium(prey_birth, prey_death, consumption=0.2,
                                predator_death=0.1, conversion=1.0,
                                prey_comp=0.1, pred_comp=0.05):
        r = prey_birth - prey_death
        c = consumption
        a = conversion * consumption
        d_c = predator_death
        e = prey_comp
        q = pred_comp
        
        if r <= 0:
            return 0.0, 0.0
        
        R_prey_only = r / e
        
        if a * R_prey_only - d_c <= 0:
            return R_prey_only, 0.0
        
        R_num = r * q + d_c * c
        R_den = c * a + e * q
        
        if R_den <= 0:
            return R_prey_only, 0.0
        
        R_star = R_num / R_den
        C_star = (a * R_star - d_c) / q
        
        if R_star < 0 or C_star < 0:
            return (R_prey_only, 0.0) if r > 0 else (0.0, 0.0)
        
        return R_star, C_star
    
    # Compute MF Hydra
    grid_cells = cfg.default_grid ** 2
    mf_dN_dd = np.zeros_like(dN_dd_no_evo)
    for j, pb in enumerate(prey_births):
        mf_prey = []
        for pd in prey_deaths:
            R_eq, _ = mean_field_equilibrium(pb, pd,
                                              consumption=cfg.predator_birth,
                                              predator_death=cfg.predator_death)
            mf_prey.append(R_eq)
        mf_dN_dd[:, j] = np.gradient(mf_prey, dd)
    
    hydra_spatial_only = int(np.sum((dN_dd_no_evo > 0) & (mf_dN_dd <= 0) & 
                                     (grids["prey_pop_no_evo"] > 50)))

    summary = {
        "coexistence_no_evo": int(np.sum((grids["survival_prey_no_evo"] > 80) & 
                                         (grids["survival_pred_no_evo"] > 80))),
        "coexistence_evo": int(np.sum((grids["survival_prey_evo"] > 80) & 
                                      (grids["survival_pred_evo"] > 80))),
        "hydra_region_size": int(np.sum((dN_dd_no_evo > 0) & 
                                        (grids["prey_pop_no_evo"] > 50))),
        "max_hydra_strength": float(np.nanmax(dN_dd_no_evo)),
        "hydra_region_size_evo": int(np.sum((dN_dd_evo > 0) & 
                                            (grids["prey_pop_evo"] > 50))),
        "hydra_spatial_only_count": hydra_spatial_only,
        
        # PCF-based metrics
        "mean_segregation_index": float(np.nanmean(pcf_grids["segregation_index"])),
        "mean_prey_clustering": float(np.nanmean(pcf_grids["prey_clustering_index"])),
        "mean_pred_clustering": float(np.nanmean(pcf_grids["pred_clustering_index"])),
        "mean_segregation_index_evo": float(np.nanmean(pcf_grids["segregation_index_evo"])),
        "mean_prey_clustering_evo": float(np.nanmean(pcf_grids["prey_clustering_index_evo"])),
        "mean_pred_clustering_evo": float(np.nanmean(pcf_grids["pred_clustering_index_evo"])),
    }
    
    # Find region with strongest segregation
    seg_grid = pcf_grids["segregation_index"]
    if not np.all(np.isnan(seg_grid)):
        min_seg_idx = np.unravel_index(np.nanargmin(seg_grid), seg_grid.shape)
        summary["max_segregation_prey_birth"] = float(prey_births[min_seg_idx[1]])
        summary["max_segregation_prey_death"] = float(prey_deaths[min_seg_idx[0]])
        summary["max_segregation_value"] = float(seg_grid[min_seg_idx])
    
    # Find critical point
    dist_crit = np.abs(grids["tau_prey"] - 2.05)
    if not np.all(np.isnan(dist_crit)):
        min_idx = np.unravel_index(np.nanargmin(dist_crit), dist_crit.shape)
        summary["critical_prey_birth"] = float(prey_births[min_idx[1]])
        summary["critical_prey_death"] = float(prey_deaths[min_idx[0]])
        summary["critical_tau_prey"] = float(grids["tau_prey"][min_idx])

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary.json")

    # Enhanced logging
    logger.info("=" * 60)
    logger.info("ANALYSIS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Coexistence region (prey & pred >80% survival):")
    logger.info(f"  Without evolution: {summary['coexistence_no_evo']} parameter combinations")
    logger.info(f"  With evolution: {summary['coexistence_evo']} parameter combinations")
    logger.info(f"HYDRA EFFECT (dN/dd > 0, prey pop > 50):")
    logger.info(f"  Without evolution: {summary['hydra_region_size']} combinations")
    logger.info(f"  With evolution: {summary['hydra_region_size_evo']} combinations")
    logger.info(f"  Max Hydra strength: {summary['max_hydra_strength']:.1f}")
    logger.info(f"  Spatial-only Hydra: {summary['hydra_spatial_only_count']} points")
    logger.info(f"SPATIAL STRUCTURE (PCF metrics):")
    logger.info(f"  No Evolution:")
    logger.info(f"    Mean segregation index: {summary['mean_segregation_index']:.3f}")
    logger.info(f"    Mean prey clustering: {summary['mean_prey_clustering']:.3f}")
    logger.info(f"    Mean pred clustering: {summary['mean_pred_clustering']:.3f}")
    logger.info(f"  With Evolution:")
    logger.info(f"    Mean segregation index: {summary['mean_segregation_index_evo']:.3f}")
    logger.info(f"    Mean prey clustering: {summary['mean_prey_clustering_evo']:.3f}")
    logger.info(f"    Mean pred clustering: {summary['mean_pred_clustering_evo']:.3f}")
    if "max_segregation_prey_birth" in summary:
        logger.info(f"Maximum segregation at:")
        logger.info(f"  prey_birth = {summary['max_segregation_prey_birth']:.3f}")
        logger.info(f"  prey_death = {summary['max_segregation_prey_death']:.3f}")
        logger.info(f"  C_cr = {summary['max_segregation_value']:.3f}")
    if "critical_prey_birth" in summary:
        logger.info(f"Closest to SOC criticality (τ=2.05):")
        logger.info(f"  prey_birth = {summary['critical_prey_birth']:.3f}")
        logger.info(f"  prey_death = {summary['critical_prey_death']:.3f}")
        logger.info(f"  τ_prey = {summary['critical_tau_prey']:.3f}")

# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="PP Evolutionary Analysis - Prey Hydra Effect (Enhanced & Corrected)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "sweep", "sensitivity", "fss", "plot", "debug"],
        help="Analysis mode",
    )
    parser.add_argument(
        "--output", type=str, default="results", help="Output directory"
    )
    parser.add_argument(
        "--cores", type=int, default=-1, help="Number of cores (-1 = all)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print estimated runtime and exit"
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        dest="synchronous",
        help="Run simulation in synchronous mode (default is asynchronous)"
    )
    parser.add_argument(
        "--neighbor-stats",
        action="store_true",
        help="Collect detailed neighbor statistics (adds overhead)"
    )
    parser.add_argument(
        "--timeseries",
        action="store_true",
        help="Collect detailed time-series data (adds overhead)"
    )
    parser.add_argument(
        "--snapshots",
        action="store_true",
        help="Save grid snapshots at key timepoints (adds storage)"
    )

    args = parser.parse_args()

    # Setup
    cfg = Config()
    cfg.synchronous = args.synchronous
    cfg.collect_neighbor_stats = args.neighbor_stats
    cfg.collect_timeseries = args.timeseries
    cfg.save_snapshots = args.snapshots
    cfg.n_jobs = (
        args.cores if args.cores > 0 else int(os.environ.get("SLURM_CPUS_PER_TASK", -1))
    )

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "analysis.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    # Header
    logger.info("=" * 60)
    logger.info("PP Evolutionary Analysis - PREY HYDRA EFFECT")
    logger.info("CORRECTED: PP defaults + proper FSS scaling")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Cores: {cfg.n_jobs}")
    logger.info(f"Update rule: {'synchronous' if cfg.synchronous else 'asynchronous'}")
    logger.info(f"Predator params: death={cfg.predator_death}, birth={cfg.predator_birth}")
    logger.info(f"Evolving: prey_death (SD={cfg.evolve_sd}, bounds=[{cfg.evolve_min}, {cfg.evolve_max}])")
    logger.info(f"Numba acceleration: {'ENABLED' if USE_NUMBA else 'DISABLED'}")
    
    if cfg.collect_neighbor_stats:
        logger.info("Enhanced: Collecting neighbor statistics")
    if cfg.collect_timeseries:
        logger.info("Enhanced: Collecting time-series data")
    if cfg.save_snapshots:
        logger.info("Enhanced: Saving grid snapshots")

    # Debug mode (local only)
    if args.mode == "debug":
        run_debug_mode(cfg, logger)
        return

    n_cores = cfg.n_jobs if cfg.n_jobs > 0 else os.cpu_count()
    logger.info(f"Estimated: {cfg.estimate_runtime(n_cores)}")

    if args.dry_run:
        logger.info("Dry run - exiting")
        return

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2, default=str)

    start_time = time.time()

    # Run analyses
    if args.mode in ["full", "sweep"]:
        run_2d_sweep(cfg, output_dir, logger)

    if args.mode in ["full", "sensitivity"]:
        run_sensitivity(cfg, output_dir, logger)

    if args.mode in ["full", "fss"]:
        run_fss(cfg, output_dir, logger)

    if args.mode in ["full", "plot"]:
        generate_plots(cfg, output_dir, logger)

    elapsed = time.time() - start_time
    logger.info(f"Total runtime: {elapsed/3600:.2f} hours")
    logger.info("Done!")


if __name__ == "__main__":
    main()